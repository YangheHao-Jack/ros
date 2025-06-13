#!/usr/bin/env python3
"""
Compute per-frame and average camera→base transforms.
"""

import os
import glob
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

def load_H(path: str) -> np.ndarray:
    H = np.load(path)
    if H.shape != (4, 4):
        raise ValueError(f"{path} does not contain a 4×4 matrix")
    return H

def invert_H(H: np.ndarray) -> np.ndarray:
    R, t = H[:3, :3], H[:3, 3]
    R_inv, t_inv = R.T, -R.T @ t
    H_inv = np.eye(4, dtype=H.dtype)
    H_inv[:3, :3] = R_inv
    H_inv[:3, 3]  = t_inv
    return H_inv

def average_transforms(H_list: list[np.ndarray]) -> np.ndarray:
    Ts = np.stack([H[:3, 3] for H in H_list], axis=0)
    t_avg = Ts.mean(axis=0)
    Rs = [H[:3, :3] for H in H_list]
    rots  = Rotation.from_matrix(Rs)
    quats = rots.as_quat()
    # align hemisphere
    q0 = quats[0]
    for i in range(1, len(quats)):
        if np.dot(q0, quats[i]) < 0:
            quats[i] = -quats[i]
    q_sum = quats.sum(axis=0)
    q_avg = q_sum / np.linalg.norm(q_sum)
    R_avg = Rotation.from_quat(q_avg).as_matrix()
    H_avg = np.eye(4, dtype=np.float64)
    H_avg[:3, :3] = R_avg
    H_avg[:3, 3]  = t_avg
    return H_avg

def main(tf_dir: str, ht_init: str, output_dir: str):
    per_frame_dir = os.path.join(output_dir, "per_frame")
    average_dir   = os.path.join(output_dir, "average")
    os.makedirs(per_frame_dir, exist_ok=True)
    os.makedirs(average_dir,   exist_ok=True)

    # Load calibration and anchor
    H_cb0     = load_H(ht_init)
    H_wb0     = load_H(os.path.join(tf_dir, "tf_0.npy"))
    H_wb0_inv = invert_H(H_wb0)

    # Compute per-frame camera→base
    Hcbs = []
    for path in sorted(glob.glob(os.path.join(tf_dir, "tf_*.npy"))):
        seq   = os.path.basename(path).split("_")[1].split(".")[0]
        H_wbt = load_H(path)
        delta = H_wb0_inv @ H_wbt
        H_cb_t= H_cb0 @ delta
        Hcbs.append(H_cb_t)
        out_p = os.path.join(per_frame_dir, f"camera_to_base_{seq}.npy")
        np.save(out_p, H_cb_t)
        print(f"Saved per-frame transform: {out_p}")

    # Compute & save average
    H_avg = average_transforms(Hcbs)
    avg_p = os.path.join(average_dir, "avg_camera_to_base.npy")
    np.save(avg_p, H_avg)
    print(f"Saved average transform: {avg_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-frame + average camera→base transforms"
    )
    parser.add_argument("tf_dir",
                        help="Dir with tf_0.npy, tf_1.npy, …")
    parser.add_argument("ht_init",
                        help="Initial camera→base .npy from calibration")
    parser.add_argument("output_dir",
                        help="Where to write per_frame/ and average/")
    args = parser.parse_args()
    main(args.tf_dir, args.ht_init, args.output_dir)
