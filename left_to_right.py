#!/usr/bin/env python3
"""
convert_base_left_to_right.py

Reads a ROS2 bag (MCAP) to extract the static TFs that relate your left and right cameras
via a common parent frame, then composes that with your saved base→left‐camera extrinsic
to produce a base→right‐camera extrinsic.

Usage:
  ./convert_base_left_to_right.py \
    --bag-dir /path/to/bag \
    --extrinsics-left H_base_left.npy \
    --common-frame zed_camera_center \
    --left-frame  zed_left_camera_frame \
    --right-frame zed_right_camera_frame \
    --output-file H_base_right.npy
"""
import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


def tf_transform_to_matrix(tf_stamped: TransformStamped) -> np.ndarray:
    """
    Convert a geometry_msgs/TransformStamped to a 4×4 numpy matrix.
    """
    t = tf_stamped.transform.translation
    q = tf_stamped.transform.rotation
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3,  3] = [t.x, t.y, t.z]
    return M


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert base→left extrinsic to base→right using TFs from a ROS2 bag."
    )
    p.add_argument(
        "--bag-dir", required=True,
        help="Path to the rosbag2 (MCAP) folder."
    )
    p.add_argument(
        "--extrinsics-left", required=True,
        help="Path to the base→left‐camera .npy extrinsic file."
    )
    p.add_argument(
        "--common-frame", required=True,
        help="TF frame ID shared by both cameras (e.g. zed_camera_center)."
    )
    p.add_argument(
        "--left-frame", required=True,
        help="TF child frame for the left camera (e.g. zed_left_camera_frame)."
    )
    p.add_argument(
        "--right-frame", required=True,
        help="TF child frame for the right camera (e.g. zed_right_camera_frame)."
    )
    p.add_argument(
        "--output-file", required=True,
        help="Where to write the resulting base→right‐camera .npy extrinsic."
    )
    p.add_argument(
        "--tf-topic", default="/tf",
        help="Dynamic TF topic (default: /tf)."
    )
    p.add_argument(
        "--static-tf-topic", default="/tf_static",
        help="Static TF topic (default: /tf_static)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    # --- Open bag and collect the two needed static transforms ---
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.bag_dir, storage_id="mcap"),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        ),
    )
    reader.set_filter(StorageFilter(
        topics=[args.tf_topic, args.static_tf_topic]
    ))

    H_center_to_left  = None
    H_center_to_right = None

    while reader.has_next():
        topic, buf, _ = reader.read_next()
        msg = deserialize_message(buf, TFMessage)
        for tfs in msg.transforms:
            parent = tfs.header.frame_id
            child  = tfs.child_frame_id
            if parent == args.common_frame and child == args.left_frame:
                H_center_to_left = tf_transform_to_matrix(tfs)
            elif parent == args.common_frame and child == args.right_frame:
                H_center_to_right = tf_transform_to_matrix(tfs)
        # once we've got both, break
        if H_center_to_left is not None and H_center_to_right is not None:
            break

    if H_center_to_left is None or H_center_to_right is None:
        raise RuntimeError(
            f"Could not find static TF {args.common_frame}→{args.left_frame} "
            f"and/or {args.common_frame}→{args.right_frame}"
        )

    # Compute left→right
    H_left_to_center = np.linalg.inv(H_center_to_left)
    H_left_to_right = H_center_to_right @ H_left_to_center

    # Load base→left
    H_base_to_left = np.load(args.extrinsics_left)

    # Compose base→right = base→left @ left→right
    H_base_to_right = H_base_to_left @ H_left_to_right

    # Save result
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output_file, H_base_to_right)
    print(f"Saved base→right extrinsic → {args.output_file}")


if __name__ == "__main__":
    main()
