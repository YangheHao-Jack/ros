#!/usr/bin/env python3
"""
left_to_right_camera.py

Convert a left-camera_info YAML into a right-camera_info YAML by using the
stereo baseline between left and right camera frames as published in a ROS2 bag.

Usage:
  ./left_to_right_camera.py \
    --bag-dir /path/to/bag \
    --extrinsics-left base_to_left.npy \
    --camera-info-left left_camera_info.yaml \
    --common-frame zed_camera_center \
    --left-frame zed_left_camera_frame \
    --right-frame zed_right_camera_frame \
    --camera-info-right right_camera_info.yaml
"""
import argparse
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage


def tf_to_matrix(tf_stamped) -> np.ndarray:
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
        description="Convert left camera_info → right camera_info using TF baseline"
    )
    p.add_argument("--bag-dir",           required=True,
                   help="Path to rosbag2 (MCAP) folder")
    p.add_argument("--extrinsics-left",   required=True,
                   help="Path to base→left .npy file")
    p.add_argument("--camera-info-left",  required=True,
                   help="Path to left camera_info.yaml")
    p.add_argument("--common-frame",      required=True,
                   help="Common TF parent of both cameras")
    p.add_argument("--left-frame",        required=True,
                   help="TF frame ID of left camera")
    p.add_argument("--right-frame",       required=True,
                   help="TF frame ID of right camera")
    p.add_argument("--camera-info-right", required=True,
                   help="Output path for right camera_info.yaml")
    p.add_argument("--tf-topic",          default="/tf",
                   help="Dynamic TF topic")
    p.add_argument("--static-tf-topic",   default="/tf_static",
                   help="Static TF topic")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Read TFs from bag ---
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
                H_center_to_left = tf_to_matrix(tfs)
            elif parent == args.common_frame and child == args.right_frame:
                H_center_to_right = tf_to_matrix(tfs)
        if H_center_to_left is not None and H_center_to_right is not None:
            break

    if H_center_to_left is None or H_center_to_right is None:
        raise RuntimeError(
            f"Could not find static TF {args.common_frame}→{args.left_frame} "
            f"and/or {args.common_frame}→{args.right_frame}"
        )

    # --- Compute left→right baseline ---
    H_left_to_center = np.linalg.inv(H_center_to_left)
    H_left_to_right  = H_center_to_right @ H_left_to_center
    baseline = H_left_to_right[0, 3]  # x-axis offset (meters)

    # --- Load left camera_info YAML ---
    with open(args.camera_info_left, 'r') as f:
        info = yaml.safe_load(f)

    # --- Locate the projection vector in the YAML ---
    if 'projection_matrix' in info and 'data' in info['projection_matrix']:
        P = info['projection_matrix']['data']
        key_container = ('projection_matrix','data')
    elif 'P' in info and isinstance(info['P'], list) and len(info['P']) == 12:
        P = info['P']
        key_container = ('P',)
    elif 'p' in info and isinstance(info['p'], list) and len(info['p']) == 12:
        P = info['p']
        key_container = ('p',)
    elif 'camera_matrix' in info and 'data' in info['camera_matrix']:
        P = info['camera_matrix']['data']
        key_container = ('camera_matrix','data')
    else:
        raise KeyError(
            "No valid projection matrix found (keys 'projection_matrix.data', 'P', 'p', or 'camera_matrix.data')"
        )

    # --- Modify P[3] = -fx * baseline ---
    fx = float(P[0])
    P[3] = -fx * baseline

    # --- Cast every element of P back to native Python float ---
    P = [float(x) for x in P]

    # --- Write it back where it came from ---
    if key_container == ('projection_matrix','data'):
        info['projection_matrix']['data'] = P
    elif key_container == ('P',):
        info['P'] = P
    elif key_container == ('p',):
        info['p'] = P
    else:  # ('camera_matrix','data')
        info['camera_matrix']['data'] = P

    # --- Optionally update names/frame_id ---
    if 'camera_name' in info:
        info['camera_name'] = info['camera_name'].replace('left','right')
    if 'header' in info and 'frame_id' in info['header']:
        info['header']['frame_id'] = args.right_frame

    # --- Save new right camera_info YAML ---
    out_dir = os.path.dirname(args.camera_info_right)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.camera_info_right, 'w') as f:
        yaml.safe_dump(info, f, sort_keys=False)

    print(f"Wrote right camera_info → {args.camera_info_right}")


if __name__ == "__main__":
    main()
