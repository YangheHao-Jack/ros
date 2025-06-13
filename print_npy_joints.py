#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np

def print_npy(path):
    arr = np.load(path)
    print(f"\nFile: {os.path.basename(path)}")
    for i, val in enumerate(arr):
        print(f"  joint_{i}: {val:.6f}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file_or_directory>")
        sys.exit(1)
    arg = sys.argv[1]
    if os.path.isdir(arg):
        npy_files = sorted(glob.glob(os.path.join(arg, "*.npy")))
        for fn in npy_files:
            print_npy(fn)
    elif os.path.isfile(arg) and arg.endswith(".npy"):
        print_npy(arg)
    else:
        print("Please pass a .npy file or a directory containing .npy files.")
        sys.exit(1)

if __name__ == "__main__":
    main()
