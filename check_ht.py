#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

def load_ht(path):
    """Load and validate a 4×4 homogeneous transform from .npy."""
    H = np.load(path)
    # Print raw data details
    print(f"--- Raw data from {path} ---")
    print(f"Shape: {H.shape}, dtype: {H.dtype}")
    print(H, "\n")
    # Basic sanity checks
    if H.shape != (4,4):
        raise ValueError(f"{path} has shape {H.shape}, expected (4,4)")
    if not np.allclose(H[3,:], [0,0,0,1], atol=1e-6):
        raise ValueError(f"{path} bottom row is {H[3,:]}, expected [0,0,0,1]")
    R = H[:3,:3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        raise ValueError(f"{path} rotation part is not orthonormal")
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(f"{path} rotation det = {det:.6f}, expected +1")
    return H

def decompose(H):
    """Return (translation, quaternion [x,y,z,w], euler XYZ degrees)."""
    t = H[:3,3]
    rot = Rotation.from_matrix(H[:3,:3])
    quat  = rot.as_quat()           # [x,y,z,w]
    euler = rot.as_euler('xyz', degrees=True)
    return t, quat, euler

def main():
    p = argparse.ArgumentParser(
        description="Validate, dump raw, and compare two homogeneous-transform .npy files"
    )
    p.add_argument('--ht-init',  required=True, help="Initial pose .npy")
    p.add_argument('--ht-final', required=True, help="Final pose .npy")
    args = p.parse_args()

    # load & validate, printing raw contents
    H_init  = load_ht(args.ht_init)
    H_final = load_ht(args.ht_final)
    print(f"Loaded and validated:\n • INIT  = {args.ht_init}\n • FINAL = {args.ht_final}\n")

    # decompose each
    ti, qi, ei = decompose(H_init)
    tf, qf, ef = decompose(H_final)

    print("Initial pose decomposition:")
    print(f"  Translation:            {ti}")
    print(f"  Quaternion [x,y,z,w]:   {qi}")
    print(f"  Euler XYZ (deg):        {ei}\n")

    print("Final pose decomposition:")
    print(f"  Translation:            {tf}")
    print(f"  Quaternion [x,y,z,w]:   {qf}")
    print(f"  Euler XYZ (deg):        {ef}\n")

    # compute relative:  H_rel = H_final * inv(H_init)
    H_rel = H_final @ np.linalg.inv(H_init)
    tr, qr, er = decompose(H_rel)
    print("Relative transform (H_final × H_init⁻¹):")
    print("  Full 4×4 matrix:\n", H_rel, "\n")
    print(f"  Translation:            {tr}")
    print(f"  Quaternion [x,y,z,w]:   {qr}")
    print(f"  Euler XYZ (deg):        {er}")

if __name__ == "__main__":
    main()
