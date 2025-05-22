#!/usr/bin/env python3
import argparse
import pathlib
import numpy as np

from scipy.spatial.transform import Rotation
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage
from tf2_msgs.msg import TFMessage

def quaternion_to_homogeneous(x, y, z, w, tx, ty, tz):
    """Build a 4×4 homogeneous matrix from translation+quaternion."""
    R = Rotation.from_quat([x, y, z, w]).as_matrix()  # 3×3
    H = np.eye(4, dtype=np.float32)
    H[:3, :3] = R
    H[:3, 3] = [tx, ty, tz]
    return H

def main():
    p = argparse.ArgumentParser(
        description="Extract TF and check if homogeneous transform is unchanged between frames"
    )
    p.add_argument("bag_dir",
                   help="Path to rosbag2 (MCAP) folder")
    p.add_argument("--image-topic",
                   default="/zed/zed_node/left/image_rect_color/compressed",
                   help="CompressedImage topic for frames")
    p.add_argument("--tf-topic",
                   default="/tf_static",
                   help="TFMessage topic (usually /tf or /tf_static)")
    p.add_argument("--out-dir",
                   default="./tf_output",
                   help="Directory to save per-frame .npy transforms")
    p.add_argument("--tol",
                   type=float, default=1e-6,
                   help="Tolerance for considering two transforms equal")
    args = p.parse_args()

    out_tf_dir = pathlib.Path(args.out_dir) / "transforms"
    out_tf_dir.mkdir(parents=True, exist_ok=True)

    # Set up rosbag2 reader
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.bag_dir, storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"),
    )
    reader.set_filter(StorageFilter(topics=[args.image_topic, args.tf_topic]))

    last_tfs = []           # buffer TFMessage.transforms
    prev_H = None           # previous frame's homogeneous matrix
    frame_idx = 0

    while reader.has_next():
        topic, buf, ts = reader.read_next()
        if topic == args.tf_topic:
            # TFMessage contains a list of TransformStamped in .transforms
            tf_msg = deserialize_message(buf, TFMessage)
            last_tfs.extend(tf_msg.transforms)

        elif topic == args.image_topic:
            # On each image frame, compute & save the homogeneous transform
            if last_tfs:
                # pick the last transform seen
                tf = last_tfs[-1]
                tr = tf.transform.translation
                qt = tf.transform.rotation
                H  = quaternion_to_homogeneous(
                    qt.x, qt.y, qt.z, qt.w,
                    tr.x, tr.y, tr.z
                )
            else:
                # no TF yet: identity
                H = np.eye(4, dtype=np.float32)

            # check against previous
            if prev_H is None:
                status = "FIRST"
            else:
                if np.allclose(H, prev_H, atol=args.tol):
                    status = "UNCHANGED"
                else:
                    status = "CHANGED"

            name = f"{frame_idx:06d}"
            np.save(str(out_tf_dir / f"tf_{name}.npy"), H)
            print(f"Frame {name}: {status}")

            # prepare for next
            prev_H = H
            last_tfs.clear()
            frame_idx += 1

    print(f"\nDone! Saved {frame_idx} transforms to {out_tf_dir}")

if __name__ == "__main__":
    main()
