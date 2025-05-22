#!/usr/bin/env python3
import os
import csv
import signal
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge
import cv2
import numpy as np

def live_extract_pos_only(output_dir: str, image_topics: list, joint_topics: list):
    # 1) Make output dirs
    img_left = os.path.join(output_dir, 'images', 'left')
    img_right= os.path.join(output_dir, 'images', 'right')
    js_dir   = os.path.join(output_dir, 'joint_states')
    for d in (img_left, img_right, js_dir):
        os.makedirs(d, exist_ok=True)

    # 2) Create display windows
    cv2.namedWindow('Left Camera',  cv2.WINDOW_NORMAL)
    cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)

    # 3) Bridge & buffer for positions only
    bridge = CvBridge()
    joint_buf = {'names': None, 'rows': []}

    # 4) Init ROS 2
    rclpy.init()
    node = Node('live_extractor')

    # 5) Image callback factory
    def make_img_cb(side, out_dir, win):
        def _cb(msg: CompressedImage):
            img = bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            ts  = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            cv2.imwrite(os.path.join(out_dir, f"{side}_{ts}.png"), img)
            cv2.imshow(win, img)
            cv2.waitKey(1)
        return _cb

    # 6) Joint position callback
    def on_js(msg: JointState):
        if joint_buf['names'] is None:
            joint_buf['names'] = list(msg.name)
        ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        joint_buf['rows'].append([int(ts)] + list(msg.position))

    # 7) Subscriptions
    node.create_subscription(
        CompressedImage, image_topics[0],
        make_img_cb('left',  img_left,  'Left Camera'),
        10
    )
    node.create_subscription(
        CompressedImage, image_topics[1],
        make_img_cb('right', img_right, 'Right Camera'),
        10
    )
    node.create_subscription(
        JointState, joint_topics[0],
        on_js,
        10
    )

    # 8) Shutdown handler: dump CSV+NPY & close windows
    def _shutdown(sig, frame):
        names = joint_buf['names'] or []
        rows  = joint_buf['rows']
        if names and rows:
            header   = ['timestamp'] + [f"{n}_pos" for n in names]
            csv_path = os.path.join(js_dir, 'lbr_joint_states.csv')
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
            data_arr  = np.array(rows, dtype=np.float64)
            names_arr = np.array(names, dtype=object)
            np.save(os.path.join(js_dir, 'lbr_joint_states.npy'),           data_arr)
            np.save(os.path.join(js_dir, 'lbr_joint_states_names.npy'), names_arr)
            node.get_logger().info('Wrote joint positions CSV & NPY')
        else:
            node.get_logger().warn('No joint data to write.')
        cv2.destroyAllWindows()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _shutdown)

    node.get_logger().info(f'LiveExtractor (pos only) → "{output_dir}"')
    rclpy.spin(node)
    # fallback
    _shutdown(None, None)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Live extract & display ZED images + LBR joint positions only'
    )
    p.add_argument('output_dir', help='Directory to save images & joint files')
    args = p.parse_args()

    img_topics   = [
        '/zed/zed_node/left/image_rect_color/compressed',
        '/zed/zed_node/right/image_rect_color/compressed'
    ]
    joint_topics = ['/lbr/joint_states']

    live_extract_pos_only(args.output_dir, img_topics, joint_topics)
