#!/usr/bin/env python3
import os
import argparse
import pathlib
import yaml
import csv
import re

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gdown
from scipy.spatial.transform import Rotation

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, JointState
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge

from kornia import image_to_tensor
from kornia.geometry import resize
from roboreg.util.viz import overlay_mask
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def quaternion_to_homogeneous(x, y, z, w, tx, ty, tz):
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = R
    H[:3, 3] = [tx, ty, tz]
    return H

class Rosbag2ExtractorWithSegmentation:
    def __init__(self, args):
        cudnn.benchmark = True
        self.device = args.device

        cam = yaml.safe_load(open(args.camera_info_file, 'r'))
        self.reg_w = int(cam.get('image_width', cam.get('width')))
        self.reg_h = int(cam.get('image_height', cam.get('height')))
        print(f"[Init] Registration size: {self.reg_w}×{self.reg_h}")

        self.num_components = args.num_components
        self.pth            = args.pth
        self.alpha          = args.alpha

        model_file = pathlib.Path.home() / args.model_path / args.model_name
        model_file.parent.mkdir(parents=True, exist_ok=True)
        if not model_file.exists():
            print(f"[Init] Downloading coarse model to {model_file}")
            gdown.download(args.model_url, str(model_file), quiet=False)
        print(f"[Init] Loading coarse model from {model_file}")
        self.coarse = torch.jit.load(str(model_file), map_location=self.device).eval().to(self.device)

        print(f"[Init] Building SAM2 with config={args.sam2_config}")
        sam = build_sam2(args.sam2_config, args.sam2_checkpoint).to(self.device).eval()
        sam = sam.to(memory_format=torch.channels_last)
        sam = torch.compile(sam)
        self.sam_pred = SAM2ImagePredictor(sam)

        self.args = args
        self.bridge = CvBridge()

        cv2.namedWindow('Left Camera',  cv2.WINDOW_NORMAL)
        cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Left Overlay',  cv2.WINDOW_NORMAL)
        cv2.namedWindow('Right Overlay', cv2.WINDOW_NORMAL)

        # NEW: only start extracting once we see a non-zero joint state
        self.started = False

    def inference_coarse(self, img):
        t = image_to_tensor(img)
        t = resize(t, (256, 480)).float() / 255.0
        t = t.unsqueeze(0).to(self.device)
        with torch.amp.autocast(self.device):
            logits = self.coarse(t)
        prob = torch.sigmoid(logits)
        mask = (prob > self.pth).float()
        h, w = img.shape[:2]
        full = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')[0, 0]
        return full.bool().cpu().numpy()

    def keep_largest_components(self, mask):
        labels, stats = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)[1:3]
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return labels, []
        topk = (np.argsort(areas)[-self.num_components:] + 1).tolist()
        return labels, topk

    def refine_with_sam2(self, img, labels, components):
        refined = np.zeros_like(labels, dtype=bool)
        for comp in components:
            ys, xs = np.where(labels == comp)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            box = np.array([[x1, y1, x2, y2]], dtype=np.int32)
            with torch.amp.autocast(self.device):
                self.sam_pred.set_image(img)
                masks, _, _ = self.sam_pred.predict(box=box, multimask_output=False)
            refined |= masks[0].astype(bool)
        return refined

    def run(self):
        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=self.args.bag_dir, storage_id='mcap'),
            ConverterOptions(input_serialization_format='cdr',
                             output_serialization_format='cdr'),
        )

        all_topics = reader.get_all_topics_and_types()
        topic_type = {t.name: t.type for t in all_topics}
        needed = self.args.image_topics + self.args.joint_topics + self.args.tf_topics
        missing = [t for t in needed if t not in topic_type]
        if missing:
            raise RuntimeError(f"Missing topics: {missing}")
        reader.set_filter(StorageFilter(topics=needed))

        base = self.args.output_dir
        img_out  = os.path.join(base, 'images')
        tf_out   = os.path.join(base, 'transforms')
        seg_base = os.path.join(base, 'segmentation')
        js_out   = os.path.join(base, 'joint_states')

        for d in [img_out, tf_out, seg_base, js_out]:
            os.makedirs(d, exist_ok=True)

        masks_dir  = os.path.join(seg_base, 'masks')
        over_dir   = os.path.join(seg_base, 'overlays')
        binary_dir = os.path.join(seg_base, 'binary_overlays')
        for parent in [masks_dir, over_dir, binary_dir]:
            for side in ['left', 'right']:
                os.makedirs(os.path.join(parent, side), exist_ok=True)

        left_img_dir  = os.path.join(img_out, 'left');  os.makedirs(left_img_dir,  exist_ok=True)
        right_img_dir = os.path.join(img_out, 'right'); os.makedirs(right_img_dir, exist_ok=True)

        js_left_dir  = os.path.join(js_out, 'left');  os.makedirs(js_left_dir,  exist_ok=True)
        js_right_dir = os.path.join(js_out, 'right'); os.makedirs(js_right_dir, exist_ok=True)

        left_seq  = 0
        right_seq = 0

        curr_joint_names = None
        curr_joint_positions = None
        last_world_base = np.eye(4, dtype=np.float64)

        while reader.has_next():
            topic, ser, _ts = reader.read_next()
            mtype = topic_type[topic]

            if mtype == 'tf2_msgs/msg/TFMessage':
                tfm = deserialize_message(ser, TFMessage)
                for tr in tfm.transforms:
                    if (tr.header.frame_id == self.args.world_frame
                        and tr.child_frame_id == self.args.base_frame):
                        t = tr.transform.translation
                        q = tr.transform.rotation
                        last_world_base = quaternion_to_homogeneous(
                            q.x, q.y, q.z, q.w, t.x, t.y, t.z)

            elif mtype == 'sensor_msgs/msg/JointState':
                js = deserialize_message(ser, JointState)
                curr_joint_names = list(js.name)
                curr_joint_positions = list(js.position)
                if not self.started and any(abs(p) > 1e-6 for p in curr_joint_positions):
                    self.started = True
                    print("[Extractor] joint_state non-zero detected, starting extraction.")

            elif mtype == 'sensor_msgs/msg/CompressedImage':
                if not self.started:
                    continue

                msg = deserialize_message(ser, CompressedImage)
                img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')

                if 'left' in topic:
                    win, side, save_dir, idx = 'Left Camera', 'left', left_img_dir, left_seq
                else:
                    win, side, save_dir, idx = 'Right Camera','right', right_img_dir, right_seq

                cv2.imshow(win, img); cv2.waitKey(1)
                img_name = f"camera_image_{side}_{idx}.png"
                cv2.imwrite(os.path.join(save_dir, img_name), img)

                if side == 'left':
                    np.save(os.path.join(tf_out, f"tf_{idx}.npy"), last_world_base)

                mask_coarse = self.inference_coarse(img)
                labels, comps = self.keep_largest_components(mask_coarse)
                mask_refined = self.refine_with_sam2(img, labels, comps)

                mask_u8  = (mask_refined.astype(np.uint8) * 255)
                mask_reg = cv2.resize(mask_u8, (self.reg_w, self.reg_h),
                                      interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(masks_dir, side, f"mask_{side}_{idx}.png"), mask_reg)

                overlay = overlay_mask(img, mask_u8, 'r', alpha=self.alpha, beta=0.5)
                win_overlay = 'Left Overlay' if side=='left' else 'Right Overlay'
                cv2.imshow(win_overlay, overlay); cv2.waitKey(1)
                cv2.imwrite(os.path.join(over_dir, side, f"overlay_{side}_{idx}.png"), overlay)

                binary = (mask_refined.astype(np.uint8) * 255)
                cv2.imwrite(os.path.join(binary_dir, side, f"binary_{side}_{idx}.png"), binary)

                # Save joint state sorted by numeric index
                js_dir = js_left_dir if side=='left' else js_right_dir
                if curr_joint_positions is not None:
                    names     = curr_joint_names
                    positions = curr_joint_positions
                    order = sorted(
                        range(len(names)),
                        key=lambda i: int(re.search(r'\d+', names[i]).group())
                    )
                    sorted_names     = [names[i]     for i in order]
                    sorted_positions = [positions[i] for i in order]

                    np.save(
                        os.path.join(js_dir, f"joint_state_{side}_{idx}.npy"),
                        np.array(sorted_positions, dtype=np.float64)
                    )
                    csv_path = os.path.join(js_dir, f"joint_state_{side}_{idx}.csv")
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(sorted_names)
                        writer.writerow(sorted_positions)

                if side == 'left':
                    left_seq  += 1
                else:
                    right_seq += 1

        cv2.destroyAllWindows()
        print("✅ Extraction complete once joint_state became valid.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract & segment; start once joint_state non-zero"
    )
    p.add_argument('bag_dir',    help='MCAP rosbag2 folder')
    p.add_argument('output_dir', help='Where to save outputs')
    p.add_argument('--camera-info-file','-cif', required=True,
                   help='YAML with image_width/height')
    p.add_argument('--sam2-config',    default='configs/sam2.1/sam2.1_hiera_l')
    p.add_argument('--sam2-checkpoint', required=True,
                   help='SAM2 checkpoint .pt')
    p.add_argument('--model-path', default='.cache/torch/hub/checkpoints/roboreg')
    p.add_argument('--model-name', default='model.pt')
    p.add_argument('--model-url',
                   default='https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro')
    p.add_argument('--num-components', type=int, default=1,
                   help='Connected components before refine')
    p.add_argument('--pth', type=float, default=0.5,
                   help='Threshold for coarse mask')
    p.add_argument('--alpha', type=float, default=0.3,
                   help='Overlay alpha')
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--image-topics', nargs=2,
                   default=[
                       '/zed/zed_node/left/image_rect_color/compressed',
                       '/zed/zed_node/right/image_rect_color/compressed'
                   ])
    p.add_argument('--joint-topics', nargs='+', default=['/lbr/joint_states'])
    p.add_argument('--tf-topics',    nargs='+', default=['/tf', '/tf_static'])
    p.add_argument('--world-frame', default='world')
    p.add_argument('--base-frame',  default='lbr_link_0')
    return p.parse_args()


if __name__=='__main__':
    args = parse_args()
    extractor = Rosbag2ExtractorWithSegmentation(args)
    extractor.run()
