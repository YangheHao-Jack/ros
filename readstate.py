#!/usr/bin/env python3
import numpy as np

def main(path):
    # 1) Load the file (allow_pickle in case it’s an object array)
    data = np.load(path, allow_pickle=True)

    # 2) Basic info
    print(f"Loaded '{path}' → dtype={data.dtype}, shape={data.shape}, ndim={data.ndim}")

    # 3) Branch on dimensionality
    if data.ndim == 1:
        # could be: a list of joint-names, or each element is one row
        print("\nThis is a 1-D array. First few elements:")
        for i, elem in enumerate(data[:5]):
            print(f"  [{i}] → {elem}")
    elif data.ndim == 2:
        # most likely: rows × columns
        print(f"\nThis is a 2-D array with {data.shape[0]} rows and {data.shape[1]} cols.")
        print("First 5 rows:")
        print(data[:10, :])
        # split timestamp vs values
        ts = data[:, 0]
        vals = data[:, 1:]
        print(f"\nTimestamps (first 5): {ts[:5]}")
        print(f"Values matrix shape: {vals.shape}")
    else:
        print("\nArray has ndim >", 2, "— you may need a custom loader.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 inspect_joint_states.py <path/to/joint_states_*.npy>")
        sys.exit(1)
    main(sys.argv[1])
