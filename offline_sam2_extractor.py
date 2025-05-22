#!/usr/bin/env python3
import argparse
import pathlib
import yaml

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gdown

from kornia import image_to_tensor
from kornia.geometry import resize
from roboreg.util.viz import overlay_mask
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, JointState


class OfflineSegExtractor:
    def __init__(self, args):
        # speed up convolutions
        cudnn.benchmark = True

        # device & segmentation params
        self.device         = args.device
        self.num_pos        = args.num_pos
        self.num_neg        = args.num_neg
        self.num_components = args.num_components
        self.pth            = args.pth
        self.alpha          = args.alpha

        # sequence counter & base output dir
        self.frame_idx = 0
        self.base_dir  = pathlib.Path(args.out_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # parse target resolution from camera_info.yaml
        with open(args.camera_info_file, 'r') as f:
            cam_info = yaml.safe_load(f)
        # keys may be 'image_width'/'image_height' or 'width'/'height'
        self.reg_w = int(cam_info.get('image_width', cam_info.get('width')))
        self.reg_h = int(cam_info.get('image_height', cam_info.get('height')))
        print(f"[Extractor] will save reg-size files at {self.reg_w}×{self.reg_h}")

        # load coarse-segmentation model
        model_file = pathlib.Path.home() / args.model_path / args.model_name
        model_file.parent.mkdir(parents=True, exist_ok=True)
        if not model_file.exists():
            print(f"Downloading coarse model → {model_file}")
            gdown.download(args.model_url, str(model_file), quiet=False)
        print(f"Loading coarse model from {model_file}")
        self.coarse_model = torch.jit.load(str(model_file), map_location=self.device)
        self.coarse_model.eval().to(self.device)

        # build & compile SAM 2 predictor
        print(f"Building SAM2 with config {args.sam2_config}")
        sam = build_sam2(args.sam2_config, str(pathlib.Path(args.sam2_checkpoint))) \
                  .to(self.device).eval()
        sam = sam.to(memory_format=torch.channels_last)
        sam = torch.compile(sam)
        self.sam_pred = SAM2ImagePredictor(sam)

        # state for differential rendering
        self.prev_frame   = None
        self.prev_low_res = None

        # create display windows
        flags = cv2.WINDOW_NORMAL
        cv2.namedWindow('Original Image',            flags)
        cv2.namedWindow('Segmentation Overlay',      flags)
        cv2.namedWindow('Differential Mask',         flags)
        cv2.namedWindow('Differential Overlay Blue', flags)

    def inference_coarse(self, img: np.ndarray) -> np.ndarray:
        """Run coarse segmentation under AMP."""
        t = image_to_tensor(img)
        t = resize(t, (256, 480)).float() / 255.0
        t = t.unsqueeze(0).to(self.device)
        with torch.amp.autocast(self.device):
            logits = self.coarse_model(t)
            prob   = torch.sigmoid(logits)
            mask   = (prob > self.pth).float()
        h, w = img.shape[:2]
        mask_full = torch.nn.functional.interpolate(
            mask, size=(h, w), mode='nearest'
        )[0, 0]
        return mask_full.bool().cpu().numpy()

    def keep_largest_components(self, mask: np.ndarray):
        """Return labels map and IDs of the top-area components."""
        labels, stats = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )[1:3]
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return labels, []
        topk = (np.argsort(areas)[-self.num_components:] + 1).tolist()
        return labels, topk

    def refine_sam2(self, img: np.ndarray, labels_map, topk):
        """Refine each component with SAM 2 under AMP."""
        refined = np.zeros_like(labels_map, dtype=bool)
        for comp in topk:
            ys, xs = np.where(labels_map == comp)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            box = np.array([[x1, y1, x2, y2]], dtype=np.int32)
            with torch.amp.autocast(self.device):
                self.sam_pred.set_image(img)
                masks, _, low_res = self.sam_pred.predict(
                    box=box,
                    multimask_output=False,
                    mask_input=self.prev_low_res
                )
            refined |= masks[0].astype(bool)
            self.prev_low_res = low_res[0:1, ...]
        return refined

    def process_frame(self, ts, img, last_joints):
        """Segment + diff + save + display, saving reg-sized images & masks."""
        idx  = self.frame_idx
        name = f"{idx:06d}"
        bd   = self.base_dir

        # 1) resize & save original image for registration
        img_reg = cv2.resize(img, (self.reg_w, self.reg_h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(bd / f"left_image_reg_{name}.png"), img_reg)
        cv2.imshow('Original Image', img)

        # 2) coarse segmentation + cleanup
        m_coarse = self.inference_coarse(img)
        labels_map, topk = self.keep_largest_components(m_coarse)

        # 3) SAM 2 refinement
        refined = self.refine_sam2(img, labels_map, topk)

        # 4) resize & save mask for registration
        mask_u8  = (refined.astype(np.uint8) * 255)
        mask_reg = cv2.resize(mask_u8, (self.reg_w, self.reg_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(bd / f"mask_sam2_reg_{name}.png"), mask_reg)
        np.save(str(bd / f"mask_sam2_reg_{name}.npy"), (mask_reg > 0).astype(np.uint8))

        # 5) segmentation overlay (on original resolution)
        seg_over = overlay_mask(img, (refined * 255).astype(np.uint8), 'r', alpha=0.5, beta=0.5)
        cv2.imwrite(str(bd / f"overlay_{name}.png"), seg_over)
        cv2.imshow('Segmentation Overlay', seg_over)

        # 6) differential rendering
        if self.prev_frame is not None:
            diff      = cv2.absdiff(img, self.prev_frame)
            diff_mask = cv2.bitwise_and(diff, diff, mask=refined.astype(np.uint8))

            # 6a) grayscale diff mask
            dgray = cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(bd / f"diff_mask_{name}.png"), dgray)
            cv2.imshow('Differential Mask', dgray)

            # 6b) pure-blue overlay
            blue_mask = np.zeros_like(img)
            blue_mask[refined] = (255, 0, 0)
            blue_over = cv2.addWeighted(img, 1-self.alpha, blue_mask, self.alpha, 0)
            cv2.imwrite(str(bd / f"diff_overlay_{name}.png"), blue_over)
            cv2.imshow('Differential Overlay Blue', blue_over)

        # 7) save joint states
        if last_joints:
            positions, names = last_joints
            with open(bd / f"joint_states_{name}.csv", 'w') as f:
                f.write(','.join(names) + '\n')
                f.write(','.join(map(str, positions)) + '\n')
            np.save(str(bd / f"joint_states_{name}.npy"), np.array(positions))

        # 8) update state
        self.prev_frame = img.copy()
        self.frame_idx += 1

        # 9) refresh & quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit requested")
            cv2.destroyAllWindows()
            raise SystemExit()


def parse_args():
    p = argparse.ArgumentParser(
        description="Offline ZED2 bag → SAM2 segmentation + reg-sized masks"
    )
    p.add_argument('--bag-dir', help="Path to rosbag2 MCAP folder")
    p.add_argument('--camera-info-file', '-cif', required=True,
                   help="Path to left_camera_info.yaml")
    p.add_argument('--image-topic',
                   default='/zed/zed_node/left/image_rect_color/compressed')
    p.add_argument('--joint-topic', default='/lbr/joint_states')
    p.add_argument('--out-dir',    default='./output')
    p.add_argument('--num-pos',        type=int, default=30)
    p.add_argument('--num-neg',        type=int, default=30)
    p.add_argument('--num-components', type=int, default=2)
    p.add_argument('--pth',            type=float, default=0.5)
    p.add_argument('--alpha',          type=float, default=0.3)
    p.add_argument('--model-path',
                   default='.cache/torch/hub/checkpoints/roboreg')
    p.add_argument('--model-name',    default='model.pt')
    p.add_argument('--model-url',
                   default='https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro')
    p.add_argument('--sam2-config',   default='configs/sam2.1/sam2.1_hiera_l')
    p.add_argument('--sam2-checkpoint',
                   default=str(pathlib.Path.home()/'repos'/'sam2'/'checkpoints'/'sam2.1_hiera_large.pt'))
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    return p.parse_args()


def main():
    args = parse_args()

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.bag_dir, storage_id='mcap'),
        ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        ),
    )
    reader.set_filter(StorageFilter(topics=[args.image_topic, args.joint_topic]))

    extractor   = OfflineSegExtractor(args)
    last_joints = None

    while reader.has_next():
        topic, buf, ts = reader.read_next()
        if topic == args.joint_topic:
            msg = deserialize_message(buf, JointState)
            last_joints = (msg.position, msg.name)
        elif topic == args.image_topic:
            msg = deserialize_message(buf, CompressedImage)
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            extractor.process_frame(ts, img, last_joints)

    print("✅ Offline extraction & segmentation complete.")


if __name__ == "__main__":
    main()
