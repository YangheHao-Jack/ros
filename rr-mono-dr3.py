#!/usr/bin/env python3
"""
refine_cam2base_batch_perframe.py

Vectorized per‐frame camera→base extrinsic refinement via differentiable silhouette fitting.
Optimizes each frame’s extrinsics in parallel (one parameter per frame), still saving per-frame
results, static overlay images, overlay videos, and computing the average extrinsic.
Original argparse and display‐progress retained.
"""
import argparse
import importlib
import os
from enum import Enum

import cv2
import numpy as np
import pytorch_kinematics as pk
import rich.progress
import torch
from scipy.spatial.transform import Rotation

from roboreg.io import find_files, parse_mono_data
from roboreg.losses import soft_dice_loss
from roboreg.util import mask_distance_transform, mask_exponential_decay, overlay_mask
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
    q_avg = quats.sum(axis=0)
    q_avg /= np.linalg.norm(q_avg)
    R_avg = Rotation.from_quat(q_avg).as_matrix()
    H  = np.eye(4)
    H[:3, :3] = R_avg
    H[:3, 3] = t_avg
    return H


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Optimizer to use, e.g. 'Adam' or 'SGD'. Imported from torch.optim.",
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



def main() -> None:
    args = args_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MAX_JOBS"] = str(args.max_jobs)
    mode = REGISTRATION_MODE(args.mode)

    # --- Load and preprocess data ---
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

    # stack joints [N,J]
    joint_states = torch.tensor(
        np.stack(joint_list), dtype=torch.float32, device=device
    )
    # build targets [N,H,W,1]
    if mode == REGISTRATION_MODE.DISTANCE_FUNCTION:
        targ_np = [mask_distance_transform(m) for m in masks_list]
    else:
        targ_np = [mask_exponential_decay(m)   for m in masks_list]
    targets = torch.tensor(
        np.stack(targ_np)[..., None],
        dtype=torch.float32, device=device
    )

    # --- Build batched scene once ---
    cam = {"camera": create_virtual_camera(
        camera_info_file=args.camera_info_file,
        device=device
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

    # --- Initialize per-frame extrinsics parameters (camera→base) ---
    H_base_cam = np.load(args.extrinsics_file)    # base→cam
    H_cam_base = np.linalg.inv(H_base_cam)        # cam→base
    # single 9d vector for one extrinsic
    init_9d = pk.matrix44_to_se3_9d(
        torch.from_numpy(H_cam_base).float().to(device)
    )
    # replicate per frame: [N,9]
    extr_inv9d = init_9d.unsqueeze(0).repeat(N,1).clone().requires_grad_(True)

    # optimizer and scheduler
    optim_mod  = importlib.import_module("torch.optim")
    optimizer  = getattr(optim_mod, args.optimizer)(
        [extr_inv9d], lr=args.lr
    )
    scheduler  = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    best_loss  = float("inf")
    best_param = None

    # display-progress window
    if args.display_progress:
        cv2.namedWindow("render ⎯ difference ⎯ segmentation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("render ⎯ difference ⎯ segmentation", 1200, 400)

    # --- Batch optimization loop ---
    for it in rich.progress.track(
        range(1, args.max_iterations+1), "Optimizing..."
    ):
        optimizer.zero_grad()
        # convert all 9d→batch of 4×4 [N,4,4]
        extr_mats = pk.se3_9d_to_matrix44(extr_inv9d)
        scene.robot.configure(joint_states, extr_mats)
        sil_pred = scene.observe_from("camera")  # [N,H,W,1]

        # compute loss over all frames
        if mode == REGISTRATION_MODE.DISTANCE_FUNCTION:
            loss = torch.nn.functional.mse_loss(sil_pred, targets)
        else:
            loss = soft_dice_loss(sil_pred, targets).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()
        rich.print(
            f"Step [{it} / {args.max_iterations}], loss: {np.round(loss.item(), 3)}, best loss: {np.round(best_loss, 3)}, lr: {scheduler.get_last_lr().pop()}"
        )
        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_param = extr_inv9d.detach().clone()

        # original display-progress (first frame only)
        if args.display_progress:
            p0 = sil_pred[0, ..., 0].detach().cpu().numpy()
            img0 = images[0]; m0 = masks_list[0]
            render_overlay = overlay_mask(
                img0, (p0*255).astype(np.uint8), scale=1.0
            )
            diff = (
                cv2.cvtColor(
                    np.abs(p0 - m0.astype(np.float32)/255.0),
                    cv2.COLOR_GRAY2BGR
                )*255.0
            ).astype(np.uint8)
            seg_overlay = overlay_mask(img0, m0, mode="b", scale=1.0)
            combo = np.hstack([render_overlay, diff, seg_overlay])
            cv2.imshow("render ⎯ difference ⎯ segmentation", combo)
            cv2.waitKey(1)

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

    # compute & save average
    H_avg = average_transforms(final_mats)
    avg_out = os.path.join(avg_dir, "avg_camera_to_base.npy")
    np.save(avg_out, H_avg)
    print(f"Saved average extrinsic → {avg_out}")

    # --- Generate overlays & videos ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps    = 10
    sil_vid = os.path.join(args.path, "overlay_silhouette.mp4")
    mask_vid= os.path.join(args.path, "overlay_mask.mp4")
    diff_vid= os.path.join(args.path, "overlay_difference.mp4")
    sil_w = mask_w = diff_w = None

    for i in range(N):
        p = (final_mats.shape and None)  # dummy to illustrate per-frame loop

    # Actually render final silhouettes
    with torch.no_grad():
        scene.robot.configure(joint_states,
                              pk.se3_9d_to_matrix44(best_param))
        sil_np = scene.observe_from("camera").cpu().numpy()

    for i in range(N):
        # predicted silhouette mask (grayscale)
        pm = (sil_np[i, ..., 0] * 255).astype(np.uint8)
        # ground-truth mask
        gm = (masks_list[i]).astype(np.uint8)

        # 1) silhouette overlay: image + predicted
        vis_sil = overlay_mask(images[i], pm, mode="r", scale=1.0)
        # 2) mask overlay: image + ground truth
        vis_mask = overlay_mask(images[i], gm, mode="g", scale=1.0)
        # 3) difference: abs(predicted − gt)
        diff = cv2.absdiff(pm, gm)
        vis_diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

        # write static PNGs
        cv2.imwrite(os.path.join(per_dir, f"overlay_silhouette_{i}.png"), vis_sil)
        cv2.imwrite(os.path.join(per_dir, f"overlay_mask_{i}.png"),       vis_mask)
        cv2.imwrite(os.path.join(per_dir, f"diff_{i}.png"),               vis_diff)

        # init writers on first frame
        if sil_w is None:
            h, w = vis_sil.shape[:2]
            sil_w  = cv2.VideoWriter(sil_vid,  fourcc, fps, (w, h))
            mask_w = cv2.VideoWriter(mask_vid, fourcc, fps, (w, h))
            diff_w = cv2.VideoWriter(diff_vid, fourcc, fps, (w, h))

        # append to each video separately
        sil_w.write(vis_sil)
        mask_w.write(vis_mask)
        diff_w.write(vis_diff)

    if sil_w:
        sil_w.release(); mask_w.release(); diff_w.release()
        print(f"Saved videos → {sil_vid}, {mask_vid}, {diff_vid}")

    if args.display_progress:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 