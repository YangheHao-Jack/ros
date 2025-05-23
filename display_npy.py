#!/usr/bin/env python3
import numpy as np
import argparse

def display_npy(path):
    # Load with allow_pickle=True so we can handle object arrays (e.g. strings)
    data = np.load(path, allow_pickle=True)

    print(f"Loaded '{path}'")
    print(f"  dtype: {data.dtype}")
    print(f"  shape: {data.shape}")
    print()

    # Simply printing the array will show all elements
    print("Full contents:")
    print(data)

    # If it’s 2-D, you might also want to see it row by row:
    if data.ndim == 2:
        print("\nRow by row:")
        for i, row in enumerate(data):
            print(f"[{i}] {row}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Load a .npy file and print all its data"
    )
    parser.add_argument('file', help='Path to the .npy file')
    args = parser.parse_args()
    display_npy(args.file)
