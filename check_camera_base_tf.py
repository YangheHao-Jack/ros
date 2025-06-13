#!/usr/bin/env python3
import argparse

from rosbag2_py import (
    SequentialReader, StorageOptions, ConverterOptions, StorageFilter
)
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect TF topics in a rosbag and optionally check specific frame pairs"
    )
    p.add_argument("bag_dir",
                   help="Path to rosbag2 MCAP folder")
    p.add_argument("--tf-topic", default="/tf",
                   help="Dynamic TF topic (default: /tf)")
    p.add_argument("--static-tf-topic", default="/tf_static",
                   help="Static TF topic (default: /tf_static)")
    p.add_argument("--check", "-c", metavar="PARENT:CHILD", action="append",
                   help="Frame pair to check, e.g. base_link:zed_left_camera_frame; can repeat")
    p.add_argument("--list-only", "-l", action="store_true",
                   help="Only list all unique transforms, skip checks")
    return p.parse_args()

def main():
    args = parse_args()

    # Open rosbag2 reader filtering on both dynamic and static TF topics
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.bag_dir, storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"),
    )
    reader.set_filter(StorageFilter(
        topics=[args.tf_topic, args.static_tf_topic]
    ))

    # Collect dynamic vs static frame pairs
    dynamic_pairs = set()
    static_pairs  = set()

    # Parse any --check specs
    checks = []
    if args.check:
        for spec in args.check:
            if ":" not in spec:
                raise ValueError(f"Invalid --check spec '{spec}', must be PARENT:CHILD")
            parent, child = spec.split(":", 1)
            checks.append((parent, child))

    # Scan the bag
    while reader.has_next():
        topic, buf, _ = reader.read_next()
        tf_msg = deserialize_message(buf, TFMessage)
        for tfs in tf_msg.transforms:
            pair = (tfs.header.frame_id, tfs.child_frame_id)
            if topic == args.tf_topic:
                dynamic_pairs.add(pair)
            elif topic == args.static_tf_topic:
                static_pairs.add(pair)
        # early exit if we've checked all and not listing-only
        if checks and not args.list_only:
            all_seen = all(
                (p in dynamic_pairs or p in static_pairs or
                 (p[1], p[0]) in dynamic_pairs or (p[1], p[0]) in static_pairs)
                for p in checks
            )
            if all_seen:
                break

    # Combine
    all_pairs = dynamic_pairs | static_pairs

    # List all unique TF frame pairs, annotated by source
    print("Unique TF frame pairs observed:")
    for parent, child in sorted(all_pairs):
        tags = []
        if (parent, child) in dynamic_pairs:
            tags.append("dynamic")
        if (parent, child) in static_pairs:
            tags.append("static")
        tag_str = "/".join(tags)
        print(f"  • {parent} → {child} ({tag_str})")

    # If list-only, exit
    if args.list_only:
        return

    # If checks were specified, report their status with tags
    if checks:
        print("\nChecking requested frame pairs:")
        for parent, child in checks:
            statuses = []
            if (parent, child) in dynamic_pairs:
                statuses.append("DIRECT(dynamic)")
            if (parent, child) in static_pairs:
                statuses.append("DIRECT(static)")
            if (child, parent) in dynamic_pairs:
                statuses.append("INVERSE(dynamic)")
            if (child, parent) in static_pairs:
                statuses.append("INVERSE(static)")
            if not statuses:
                statuses = ["MISSING"]
            print(f"  • {parent}:{child} → {'/'.join(statuses)}")
    else:
        print("\nNo --check pairs specified; add -c PARENT:CHILD to verify specific transforms.")

if __name__ == "__main__":
    main()
