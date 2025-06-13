#!/usr/bin/env python3
import argparse
import pathlib
import xml.etree.ElementTree as ET

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from std_msgs.msg import String

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract a fixed parent→child transform (xyz + rpy) from the URDF in /lbr/robot_description"
    )
    p.add_argument(
        "bag_dir",
        help="Path to rosbag2 MCAP folder"
    )
    p.add_argument(
        "--urdf-topic",
        default="/lbr/robot_description",
        help="Topic carrying the URDF XML (default: /lbr/robot_description)"
    )
    p.add_argument(
        "--parent-link",
        required=True,
        help="The parent link name in the URDF joint (e.g. lbr_link_0)"
    )
    p.add_argument(
        "--child-link",
        required=True,
        help="The child link name in the URDF joint (e.g. zed_camera_link)"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Open the bag and filter on the URDF topic
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.bag_dir, storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr")
    )
    reader.set_filter(StorageFilter(topics=[args.urdf_topic]))

    # Read the first robot_description message
    urdf_xml = None
    while reader.has_next():
        topic, buf, _ = reader.read_next()
        if topic == args.urdf_topic:
            msg = deserialize_message(buf, String)
            urdf_xml = msg.data
            break

    if urdf_xml is None:
        print(f"Error: No messages on topic {args.urdf_topic} in bag.")
        return

    # Parse the URDF XML
    root = ET.fromstring(urdf_xml)

    # Search for the joint with matching parent/child links
    found = False
    for joint in root.findall('joint'):
        parent = joint.find('parent').attrib.get('link')
        child  = joint.find('child').attrib.get('link')
        if parent == args.parent_link and child == args.child_link:
            origin = joint.find('origin')
            if origin is None:
                xyz = ["0","0","0"]
                rpy = ["0","0","0"]
            else:
                xyz = origin.attrib.get('xyz', "0 0 0").split()
                rpy = origin.attrib.get('rpy', "0 0 0").split()
            tx, ty, tz       = map(float, xyz)
            roll, pitch, yaw = map(float, rpy)

            print(f"Found joint '{joint.attrib.get('name')}' connecting")
            print(f"  parent link = {parent}")
            print(f"  child  link = {child}\n")
            print("Origin (xyz):")
            print(f"  tx = {tx:.6f}  ty = {ty:.6f}  tz = {tz:.6f}")
            print("Origin (rpy radians):")
            print(f"  roll  = {roll:.6f}")
            print(f"  pitch = {pitch:.6f}")
            print(f"  yaw   = {yaw:.6f}")
            found = True
            break

    if not found:
        print(f"No joint found in URDF with parent='{args.parent_link}' and child='{args.child_link}'.")

if __name__ == "__main__":
    main()
