#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo

def extract_camera_info(bag_dir: str, topic: str, output_yaml: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id='mcap'),
        ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
    )
    # only look for CameraInfo on the given topic
    reader.set_filter(StorageFilter(topics=[topic]))

    cam_msg = None
    while reader.has_next():
        _, buf, _ = reader.read_next()
        msg = deserialize_message(buf, CameraInfo)
        cam_msg = msg
        break

    if cam_msg is None:
        raise RuntimeError(f"No CameraInfo messages found on topic '{topic}'")

    # convert to ROS camera_info.yaml structure
    data = {
        'image_width':           cam_msg.width,
        'image_height':          cam_msg.height,
        'camera_name':           cam_msg.header.frame_id,
        'camera_matrix': {
            'rows': 3, 'cols': 3,
            'data': list(cam_msg.k)
        },
        'distortion_model':      cam_msg.distortion_model,
        'distortion_coefficients': {
            'rows': 1, 'cols': len(cam_msg.d),
            'data': list(cam_msg.d)
        },
        'rectification_matrix': {
            'rows': 3, 'cols': 3,
            'data': list(cam_msg.r)
        },
        'projection_matrix': {
            'rows': 3, 'cols': 4,
            'data': list(cam_msg.p)
        },
        'binning_x':             cam_msg.binning_x,
        'binning_y':             cam_msg.binning_y,
        'roi': {
            'x_offset':    cam_msg.roi.x_offset,
            'y_offset':    cam_msg.roi.y_offset,
            'height':      cam_msg.roi.height,
            'width':       cam_msg.roi.width,
            'do_rectify':  cam_msg.roi.do_rectify
        }
    }

    # write out YAML
    out_path = Path(output_yaml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"✅ Wrote camera info to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CameraInfo from a ros2 bag (MCAP) into a camera_info.yaml"
    )
    parser.add_argument("bag_dir",
        help="Path to rosbag2 (MCAP) directory")
    parser.add_argument("-t", "--topic",
        default="/zed/zed_node/left/camera_info",
        help="CameraInfo topic name in the bag")
    parser.add_argument("-o", "--output",
        default="camera_info.yaml",
        help="Where to write the YAML file")
    args = parser.parse_args()

    extract_camera_info(args.bag_dir, args.topic, args.output)
