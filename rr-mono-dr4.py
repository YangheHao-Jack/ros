#!/usr/bin/env python3
"""
rr-mono-dr3.py

Vectorized per-frame camera→base extrinsic refinement via differentiable silhouette fitting.
Optimizes each frame’s extrinsics in parallel (one parameter per frame), still saving per-frame
results, static overlay PNGs, overlay videos, and computing the average extrinsic.
Retains original argparse and --display-progress behavior, and supports both torch optimizers
and SciPy’s L-BFGS-B via autodiff.
"""
import argparse
import importlib
import os
from enum import Enum
import time
import cv2
import numpy as np
import pytorch_kinematics as pk
import rich.progress
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from roboreg.io import find_files, parse_mono_data
from roboreg.losses import soft_dice_loss
from roboreg.util import (
    mask_distance_transform,
    mask_exponential_decay,
    overlay_mask,
)
from roboreg.util.factories import create_robot_scene, create_virtual_camera


class REGISTRATION_MODE(Enum):
    DISTANCE_FUNCTION = "distance-function"
    SEGMENTATION       = "segmentation"


def average_transforms(Hs):
    Ts = np.stack([H[:3, 3] for H in Hs], axis=0)
    t_avg = Ts.mean(axis=0)
    Rs = [H[:3, :3] for H in Hs]
    rots = Rotation.from_matrix(Rs)
    quats = rots.as_quat()
    q0 = quats[0]
    for i in range(1, len(quats)):
        if np.dot(q0, quats[i]) < 0:
            quats[i] = -quats[i]
    q_sum = quats.sum(axis=0)
    q_avg = q_sum / np.linalg.norm(q_sum)
    R_avg = Rotation.from_quat(q_avg).as_matrix()
    H = np.eye(4)
    H[:3, :3] = R_avg
    H[:3, 3]  = t_avg
    return H


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Optimizer to use, e.g. 'Adam' or 'SGD' or 'scipy_lbfgs'. Imported from torch.optim.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Number of epochs to optimize for.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=100,
        help="Step size for the learning rate scheduler.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma for the learning rate scheduler.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in REGISTRATION_MODE],
        default=REGISTRATION_MODE.DISTANCE_FUNCTION.value,
        help="Registration mode.",
    )
    parser.add_argument(
        "--display-progress",
        action="store_true",
        help="Display optimization progress.",
    )
    parser.add_argument(
        "--ros-package",
        type=str,
        default="lbr_description",
        help="Package where the URDF is located.",
    )
    parser.add_argument(
        "--xacro-path",
        type=str,
        default="urdf/med7/med7.xacro",
        help="Path to the xacro file, relative to --ros-package.",
    )
    parser.add_argument(
        "--root-link-name",
        type=str,
        default="",
        help="Root link name. If unspecified, the first link with mesh will be used, which may cause errors.",
    )
    parser.add_argument(
        "--end-link-name",
        type=str,
        default="",
        help="End link name. If unspecified, the last link with mesh will be used, which may cause errors.",
    )
    parser.add_argument(
        "--collision-meshes",
        action="store_true",
        help="If set, collision meshes will be used instead of visual meshes.",
    )
    parser.add_argument(
        "--camera-info-file",
        type=str,
        required=True,
        help="Full path to left camera parameters, <path_to>/left_camera_info.yaml.",
    )
    parser.add_argument(
        "--extrinsics-file",
        type=str,
        required=True,
        help="Full path to homogeneous transforms from base to left camera frame, <path_to>/HT_hydra_robust.npy.",
    )
    parser.add_argument("--path", type=str, required=True, help="Path to the data.")
    parser.add_argument(
        "--image-pattern",
        type=str,
        default="left_image_*.png",
        help="Left image file pattern.",
    )
    parser.add_argument(
        "--joint-states-pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="left_mask_*.png",
        help="Left mask file pattern.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="HT_left_dr.npy",
        help="Left output file name. Relative to --path.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=2,
        help="Number of concurrent compilation jobs for nvdiffrast. Only relevant on first run.",
    )
    return parser.parse_args()


def main():
    args = args_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MAX_JOBS"] = str(args.max_jobs)
    mode = REGISTRATION_MODE(args.mode)

    # --- Load & preprocess data ---
    image_files = find_files(args.path, args.image_pattern)
    joint_files = find_files(args.path, args.joint_states_pattern)
    mask_files  = find_files(args.path, args.mask_pattern)
    images, joint_list, masks_list = parse_mono_data(
        path=args.path,
        image_files=image_files,
        joint_states_files=joint_files,
        mask_files=mask_files,
    )
    N = len(images)

    joint_states = torch.tensor(
        np.stack(joint_list), dtype=torch.float32, device=device
    )  # [N,J]
    if mode == REGISTRATION_MODE.DISTANCE_FUNCTION:
        targ_np = [mask_distance_transform(m) for m in masks_list]
    else:
        targ_np = [mask_exponential_decay(m) for m in masks_list]
    targets = torch.tensor(
        np.stack(targ_np)[..., None], dtype=torch.float32, device=device
    )  # [N,H,W,1]

    # --- Build batched scene ---
    cam = {"camera": create_virtual_camera(
        camera_info_file=args.camera_info_file, device=device
    )}
    scene = create_robot_scene(
        batch_size=N,
        ros_package=args.ros_package,
        xacro_path=args.xacro_path,
        root_link_name=args.root_link_name or None,
        end_link_name=args.end_link_name or None,
        collision=args.collision_meshes,
        cameras=cam,
        device=device,
    )

    # --- Init per-frame extrinsics (cam→base) ---
    H_base_cam = np.load(args.extrinsics_file)
    H_cam_base  = np.linalg.inv(H_base_cam)
    init_9d     = pk.matrix44_to_se3_9d(
        torch.from_numpy(H_cam_base).float().to(device)
    )  # [9]
    extr_inv9d  = init_9d.unsqueeze(0).repeat(N,1).clone().requires_grad_(True)

    best_param = None
    best_loss  = float("inf")

    # --- Display window ---
    if args.display_progress:
        cv2.namedWindow("render ⎯ difference ⎯ segmentation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("render ⎯ difference ⎯ segmentation", 1200, 400)

    # --- Optimizer setup ---
    use_scipy = args.optimizer.lower() == "scipy_lbfgs"
    if not use_scipy:
        optim_mod  = importlib.import_module("torch.optim")
        optimizer  = getattr(optim_mod, args.optimizer)([extr_inv9d], lr=args.lr)
        scheduler  = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )

    # --- Optimization loop ---
    if use_scipy:
        # SciPy L-BFGS-B
        x0 = extr_inv9d.detach().cpu().numpy().ravel()
        iteration = 0
        last_loss = np.nan
        best_loss = float("inf")
               # how strongly to smooth across frames
        lambda_temporal = 5e-3

        import time

        # start timer
        start_time = time.time()

        def fun_and_grad(x):
            t = torch.from_numpy(x.reshape(N,9)).float().to(device).requires_grad_(True)
            mats = pk.se3_9d_to_matrix44(t)
            scene.robot.configure(joint_states, mats)
            sil = scene.observe_from("camera")
            loss_t = (torch.nn.functional.mse_loss(sil, targets)
                      if mode==REGISTRATION_MODE.DISTANCE_FUNCTION
                      else soft_dice_loss(sil, targets).mean())
            # temporal smoothness: sum squared difference between consecutive frames
            diffs = t[1:] - t[:-1]                # shape [N-1,9]
            reg   = (diffs*diffs).sum() * lambda_temporal
            loss_t = loss_t + reg
            loss = loss_t.item()
            loss_t.backward()
            nonlocal last_loss, best_loss
            last_loss = loss
            if loss < best_loss:
                best_loss = loss
            return loss, t.grad.detach().cpu().numpy().ravel()

        def callback_scipy(xk):
            nonlocal iteration
            iteration += 1
            # compute elapsed since start
            elapsed = time.time() - start_time
            # optional: ETA based on average per‐iter time
            avg_t = elapsed / iteration
            eta   = avg_t * (args.max_iterations - iteration)
            #  original rich.print format
            rich.print(
                f"Step [{iteration} / {args.max_iterations}], "
                f"loss: {np.round(last_loss, 3)}, "
                f"best loss: {np.round(best_loss, 3)}, "
                f"lr: {args.lr}, "
                f"elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
            )
            if args.display_progress:
                t = torch.from_numpy(xk.reshape(N,9)).float().to(device)
                mats = pk.se3_9d_to_matrix44(t)
                scene.robot.configure(joint_states, mats)
                sil0 = scene.observe_from("camera")[0, ..., 0].detach().cpu().numpy()
                ov = overlay_mask(images[0], (sil0*255).astype(np.uint8), scale=1.0)
                df = (cv2.cvtColor(
                    np.abs(sil0 - masks_list[0].astype(np.float32)/255.0),
                    cv2.COLOR_GRAY2BGR,
                )*255.0).astype(np.uint8)
                so = overlay_mask(images[0], masks_list[0], mode="b", scale=1.0)
                combo = np.hstack([ov, df, so])
                cv2.imshow("render ⎯ difference ⎯ segmentation", combo)
                cv2.waitKey(1)

        
        res = minimize(
            fun_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": args.max_iterations},
            callback=callback_scipy,
        )
        best_param = torch.from_numpy(res.x.reshape(N,9)).float().to(device)
        best_loss  = res.fun
    

    # ─── SAVE PER-FRAME OVERLAYS & VIDEOS ───────────────────────────────────────
    # --- After optimization: save per-frame extrinsics ---
    final_mats = pk.se3_9d_to_matrix44(best_param).detach().cpu().numpy()  # [N,4,4]
    per_dir = os.path.join(args.path, "per_frame")
    avg_dir = os.path.join(args.path, "average")
    os.makedirs(per_dir, exist_ok=True)
    os.makedirs(avg_dir, exist_ok=True)
    for i in range(N):
        outp = os.path.join(per_dir, f"camera_to_base_{i}.npy")
        np.save(outp, final_mats[i])
        print(f"Saved per-frame extrinsic → {outp}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps    = 30
    sil_vid  = os.path.join(args.path, "overlay_silhouette.mp4")
    mask_vid = os.path.join(args.path, "overlay_mask.mp4")
    diff_vid = os.path.join(args.path, "overlay_difference.mp4")
    sil_w = mask_w = diff_w = None

    # ** Here we detach, no_grad, then convert once to NumPy **
    with torch.no_grad():
        mats_tensor = torch.from_numpy(final_mats).float().to(device)
        scene.robot.configure(joint_states, mats_tensor)
        sil_np = scene.observe_from("camera").cpu().numpy()  # [N,H,W,1]

    for i in range(N):
        pm = (sil_np[i, ..., 0] * 255).astype(np.uint8)
        gm = (masks_list[i] * 255).astype(np.uint8)

        vis_s = overlay_mask(images[i], pm, mode="r", alpha=0.5)
        vis_m = overlay_mask(images[i], gm, mode="g", alpha=0.5)
        diff = cv2.absdiff(gm, pm)
        dv   = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(os.path.join(per_dir, f"overlay_silhouette_{i}.png"), vis_s)
        cv2.imwrite(os.path.join(per_dir, f"overlay_mask_{i}.png"),        vis_m)
        cv2.imwrite(os.path.join(per_dir, f"diff_{i}.png"),               dv)

        if sil_w is None:
            h, w = vis_s.shape[:2]
            sil_w  = cv2.VideoWriter(sil_vid,  fourcc, fps, (w,h))
            mask_w = cv2.VideoWriter(mask_vid, fourcc, fps, (w,h))
            diff_w = cv2.VideoWriter(diff_vid, fourcc, fps, (w,h))

        sil_w.write(vis_s)
        mask_w.write(vis_m)
        diff_w.write(dv)

    if sil_w:
        sil_w.release(); mask_w.release(); diff_w.release()
        print(f"Saved videos → {sil_vid}, {mask_vid}, {diff_vid}")

    # ─── SAVE AVERAGE & FINAL EXTRINSIC ────────────────────────────────────────
    
    
    H_avg = average_transforms(final_mats)
    np.save(os.path.join(avg_dir, "avg_camera_to_base.npy"), H_avg)
    print(f"Saved average extrinsic → {avg_dir}/avg_camera_to_base.npy")

    H_base_avg = np.linalg.inv(H_avg)
    outp = os.path.join(args.path, args.output_file)
    np.save(outp, H_base_avg)
    print(f"Saved final extrinsic → {outp}")

    if args.display_progress:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
