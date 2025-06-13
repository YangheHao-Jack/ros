#!/usr/bin/env python3
import os
import numpy as np

def print_all_npy(root_dir):
    """
    Recursively find every .npy file under root_dir,
    load it, and print its full contents.
    """
    # Ensure numpy prints entire arrays
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if fname.lower().endswith(".npy"):
                full_path = os.path.join(dirpath, fname)
                rel_path  = os.path.relpath(full_path, root_dir)
                arr = np.load(full_path, allow_pickle=True)
                print(f"\n----- {rel_path} -----\n{arr}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Print full contents of every .npy under a directory"
    )
    parser.add_argument("root_dir", help="Root directory containing .npy files")
    args = parser.parse_args()
    print_all_npy(args.root_dir)
