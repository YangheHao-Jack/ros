#!/usr/bin/env python3
import os
import csv
import argparse

import cv2
import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge

def extract_rosbag2_with_display(
    bag_dir: str,
    output_dir: str,
    image_topics: list,
    joint_topics: list,
):
    # 1) Open bag (MCAP)
    reader = SequentialReader()
    storage_opts = StorageOptions(uri=bag_dir, storage_id='mcap')
    conv_opts = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_opts, conv_opts)

    # 2) Topic→type map
    all_topics = reader.get_all_topics_and_types()
    topic_type = {t.name: t.type for t in all_topics}

    # 3) Check topics exist
    missing = [t for t in image_topics + joint_topics if t not in topic_type]
    if missing:
        raise RuntimeError(f"Topics not found: {missing}")

    # 4) Filter only our topics
    filt = StorageFilter(topics=image_topics + joint_topics)
    reader.set_filter(filt)

    # 5) Make output dirs
    img_out = os.path.join(output_dir, 'images')
    js_out  = os.path.join(output_dir, 'joint_states')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(js_out,  exist_ok=True)

    # 6) Display windows & bridge
    cv2.namedWindow('Left Camera',  cv2.WINDOW_NORMAL)
    cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
    bridge = CvBridge()

    # 7) Buffer for joint positions only
    joint_bufs = {topic: {'names': None, 'rows': []} for topic in joint_topics}

    # 8) Read & process
    while reader.has_next():
        topic, ser, ts = reader.read_next()
        mtype = topic_type[topic]

        if mtype == 'sensor_msgs/msg/CompressedImage':
            msg    = deserialize_message(ser, CompressedImage)
            img    = bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            # display
            if 'left' in topic:
                cv2.imshow('Left Camera', img)
            else:
                cv2.imshow('Right Camera', img)
            cv2.waitKey(1)
            # save
            side = 'left' if 'left' in topic else 'right'
            sub  = os.path.join(img_out, side)
            os.makedirs(sub, exist_ok=True)
            cv2.imwrite(os.path.join(sub, f"{side}_{ts}.png"), img)

        elif mtype == 'sensor_msgs/msg/JointState':
            msg = deserialize_message(ser, JointState)
            buf = joint_bufs[topic]
            if buf['names'] is None:
                buf['names'] = list(msg.name)
            # **positions only**
            buf['rows'].append([ts] + list(msg.position))

    # 9) Cleanup display
    cv2.destroyAllWindows()

    # 10) Dump joint positions to CSV + NPY
    for topic, buf in joint_bufs.items():
        safe  = topic.strip('/').replace('/', '_')
        names = buf['names'] or []
        rows  = buf['rows']
        if not rows:
            continue

        # CSV header: timestamp + one _pos per joint
        header = ['timestamp'] + [f"{n}_pos" for n in names]
        csv_path = os.path.join(js_out, f"{safe}.csv")
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

        # NPY
        data_arr  = np.array(rows, dtype=np.float64)  # shape (N,1+M)
        names_arr = np.array(names, dtype=object)     # shape (M,)
        np.save(os.path.join(js_out, f"{safe}.npy"),           data_arr)
        np.save(os.path.join(js_out, f"{safe}_names.npy"), names_arr)

    print(f"Done!\n Images → {img_out}\n Joint positions → {js_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Offline extract & display ZED compressed images + LBR joint positions"
    )
    parser.add_argument('bag_dir',    help='Path to MCAP-based rosbag2 folder')
    parser.add_argument('output_dir', help='Where to save PNGs, CSVs, NPYs')
    args = parser.parse_args()

    img_topics   = [
        '/zed/zed_node/left/image_rect_color/compressed',
        '/zed/zed_node/right/image_rect_color/compressed'
    ]
    joint_topics = ['/lbr/joint_states']

    extract_rosbag2_with_display(
        args.bag_dir,
        args.output_dir,
        img_topics,
        joint_topics,
    )
