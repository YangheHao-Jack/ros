#!/usr/bin/env python3
import argparse
import pathlib
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv_bridge
import gdown

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from kornia import image_to_tensor
from kornia.geometry import resize
from roboreg.util.viz import overlay_mask

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class LiveSegDiffNode(Node):
    def __init__(self, args):
        super().__init__('live_seg_diff_node')

        # Speed optimizations
        cudnn.benchmark = True

        # OpenCV bridge
        self._bridge = cv_bridge.CvBridge()

        # Device for PyTorch
        self._device = args.device
        self.get_logger().info(f'Using device: {self._device}')

        # Parameters from command line
        self._video_out            = pathlib.Path(args.video_output_path)
        self._diff_out             = pathlib.Path(args.diff_overlay_output_path)
        self._diff_mask_out        = pathlib.Path(args.diff_mask_video_output_path)
        self._mask_dir             = pathlib.Path(args.mask_output_dir)
        self._num_pos              = args.num_pos
        self._num_neg              = args.num_neg
        self._pth                  = args.pth
        self._alpha                = args.alpha
        self._kernel_size          = args.kernel_size
        self._fps                  = args.fps
        self._topic                = args.topic
        self._model_path           = pathlib.Path(args.model_path)
        self._model_name           = args.model_name
        self._model_url            = args.model_url
        self._sam2_config          = args.sam2_config
        self._sam2_checkpoint      = pathlib.Path(args.sam2_checkpoint).expanduser()

        # Prepare mask directory
        self._mask_dir.mkdir(parents=True, exist_ok=True)

        # Video writers (initialized on first frame)
        self._video_writer     = None   # segmentation overlay
        self._diff_writer      = None   # diff overlay
        self._diff_mask_writer = None   # diff mask
        self._fourcc           = cv2.VideoWriter_fourcc(*'mp4v')

        # State for diff
        self._prev_frame   = None
        self._prev_low_res = None

        # Load coarse segmentation model
        self._download_model()
        self._load_model()

        # Build and compile SAM 2
        self.get_logger().info(f'Building SAM2 with config {self._sam2_config}')
        sam = build_sam2(self._sam2_config, str(self._sam2_checkpoint))\
                .to(self._device).eval()
        sam = sam.to(memory_format=torch.channels_last)
        sam = torch.compile(sam)
        self._sam_pred = SAM2ImagePredictor(sam)

        # Create display windows
        cv2.namedWindow('Segmentation Overlay',     cv2.WINDOW_NORMAL)
        cv2.namedWindow('Differential Mask',        cv2.WINDOW_NORMAL)
        cv2.namedWindow('Differential Overlay Blue',cv2.WINDOW_NORMAL)

        # Subscribe to the compressed image topic
        self.create_subscription(
            CompressedImage,
            self._topic,
            self._on_img,
            1
        )

    def _download_model(self):
        model_file = pathlib.Path.home()/self._model_path/self._model_name
        model_file.parent.mkdir(parents=True, exist_ok=True)
        if not model_file.exists():
            self.get_logger().info(f'Downloading model → {model_file}')
            gdown.download(self._model_url, str(model_file), quiet=False)
        self._model_file = model_file

    def _load_model(self):
        self.get_logger().info(f'Loading coarse model from {self._model_file}')
        t0 = time.time()
        self._model = torch.jit.load(str(self._model_file), map_location=self._device)
        self._model.eval().to(self._device)
        self.get_logger().info(f'Model loaded in {time.time()-t0:.1f}s')

    def _inference(self, img):
        # Convert and resize for coarse model
        t = image_to_tensor(img)
        t = resize(t, (256,480)).float()/255.0
        t = t.unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(t)
            prob   = torch.sigmoid(logits)
            mask   = (prob > self._pth).float()
        # Upsample back to original size
        h,w = img.shape[:2]
        mask_full = torch.nn.functional.interpolate(
            mask, size=(h,w), mode='nearest'
        )[0,0]
        return mask_full.bool().cpu().numpy()

    def _keep_largest_components(self, mask, k=1):
        labels, stats = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )[1:3]
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return mask
        topk = np.argsort(areas)[-k:] + 1
        clean = np.zeros_like(mask, dtype=bool)
        for lbl in topk:
            clean |= (labels==lbl)
        return clean

    def _sample_points(self, mask):
        ys, xs = np.nonzero(mask)
        coords = np.stack([xs, ys], axis=-1)
        if coords.size:
            idx = np.random.choice(len(coords), min(self._num_pos,len(coords)), replace=False)
            pos = coords[idx]
        else:
            pos = np.zeros((0,2),dtype=int)
        inv = ~mask
        ys0, xs0 = np.nonzero(inv)
        coords0  = np.stack([xs0, ys0], axis=-1)
        if coords0.size:
            idx0 = np.random.choice(len(coords0), min(self._num_neg,len(coords0)), replace=False)
            neg  = coords0[idx0]
        else:
            neg  = np.zeros((0,2),dtype=int)
        pts    = np.vstack([pos, neg]).astype(np.int32)
        labels = np.hstack([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int32)
        return pts, labels

    def _ensure_writers(self, w, h):
        # segmentation overlay writer
        if self._video_writer is None:
            self._video_out.parent.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'Writing segmentation video → {self._video_out}')
            self._video_writer = cv2.VideoWriter(
                str(self._video_out), self._fourcc, self._fps, (w,h)
            )
        # differential overlay writer
        if self._diff_writer is None:
            self._diff_out.parent.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'Writing diff-overlay video → {self._diff_out}')
            self._diff_writer = cv2.VideoWriter(
                str(self._diff_out), self._fourcc, self._fps, (w,h)
            )
        # differential mask writer (grayscale)
        if self._diff_mask_writer is None:
            self._diff_mask_out.parent.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'Writing diff-mask video → {self._diff_mask_out}')
            self._diff_mask_writer = cv2.VideoWriter(
                str(self._diff_mask_out), self._fourcc, self._fps, (w,h), isColor=False
            )

    def _on_img(self, msg: CompressedImage):
        # Decode compressed ROS image
        arr   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h,w   = frame.shape[:2]

        # 1) Coarse segmentation → boolean mask
        coarse = self._inference(frame)
        coarse = self._keep_largest_components(coarse, k=1)

        # 2) Random point prompts + SAM2 refine
        pts, lbls = self._sample_points(coarse)
        with torch.amp.autocast(self._device):
            self._sam_pred.set_image(frame)
            masks,_, low_res = self._sam_pred.predict(
                point_coords=pts,
                point_labels=lbls,
                mask_input=self._prev_low_res,
                multimask_output=False
            )
        self._prev_low_res = low_res[0:1,...]
        refined = masks[0].astype(bool)

        # 3) Save mask to disk (PNG + NPY)
        ts = int(msg.header.stamp.sec*1e9 + msg.header.stamp.nanosec)
        png_path = self._mask_dir / f"{ts}.png"
        npy_path = self._mask_dir / f"{ts}.npy"
        cv2.imwrite(str(png_path), (refined.astype(np.uint8)*255))
        np.save(str(npy_path), refined)

        # 4) Initialize video writers once we know frame size
        self._ensure_writers(w, h)

        # 5) Segmentation overlay (red)
        seg_overlay = overlay_mask(frame, (refined*255).astype(np.uint8),
                                   'r', alpha=0.5, beta=0.5)
        self._video_writer.write(seg_overlay)
        cv2.imshow('Segmentation Overlay', seg_overlay)

        # 6) Differential calculation if we have a previous frame
        if self._prev_frame is not None:
            diff       = cv2.absdiff(frame, self._prev_frame)
            diff_masked= cv2.bitwise_and(diff, diff, mask=refined.astype(np.uint8))

            # 6a) Differential Mask (grayscale)
            diff_gray = cv2.cvtColor(diff_masked, cv2.COLOR_BGR2GRAY)
            self._diff_mask_writer.write(diff_gray)
            cv2.imshow('Differential Mask', diff_gray)

            # 6b) Differential Overlay (pure-blue alpha blend)
            blue_mask    = np.zeros_like(frame)
            blue_mask[refined] = (255,0,0)  # BGR: pure blue
            blue_overlay = cv2.addWeighted(frame, 1-self._alpha,
                                           blue_mask, self._alpha, 0)
            self._diff_writer.write(blue_overlay)
            cv2.imshow('Differential Overlay Blue', blue_overlay)

        # store for next diff
        self._prev_frame = frame

        # 7) Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Quit requested; shutting down.")
            rclpy.shutdown()

    def destroy_node(self):
        # Release all writers and close windows
        if self._video_writer:      self._video_writer.release()
        if self._diff_writer:       self._diff_writer.release()
        if self._diff_mask_writer:  self._diff_mask_writer.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main():
    parser = argparse.ArgumentParser(
        description="Live SAM2 segmentation + differential rendering from ROS 2 bag play"
    )
    parser.add_argument('--video-output-path',
        default=str('/home/jack/experiments/segmentation_overlay.mp4'),
        help="Path to save red-overlay segmentation video")
    parser.add_argument('--diff-overlay-output-path',
        default=str('/home/jack/experiments/differential_overlay.mp4  '),
        help="Path to save blue-overlay differential video")
    parser.add_argument('--diff-mask-video-output-path',
        default=str('/home/jack/experiments/differential_mask.mp4  '),
        help="Path to save grayscale differential-mask video")
    parser.add_argument('--mask-output-dir',
        default=str('/home/jack/experiments/seg_masks  '),
        help="Directory to save per-frame PNG and NPY masks")
    parser.add_argument('--topic',
        default='/zed/zed_node/left/image_rect_color/compressed',
        help="ROS 2 CompressedImage topic to subscribe to")
    parser.add_argument('--num-pos', type=int, default=30,
        help="Number of positive prompt points sampled each frame")
    parser.add_argument('--num-neg', type=int, default=30,
        help="Number of negative prompt points sampled each frame")
    parser.add_argument('--pth', type=float, default=0.5,
        help="Threshold for coarse-model sigmoid output")
    parser.add_argument('--alpha', type=float, default=0.3,
        help="Alpha blend weight for blue-overlay (0–1)")
    parser.add_argument('--kernel-size', type=int, default=7,
        help="Size of morphological closing kernel")
    parser.add_argument('--fps', type=float, default=20.0,
        help="Framerate for output videos")
    parser.add_argument('--model-path', default='.cache/torch/hub/checkpoints/roboreg',
        help="Directory for coarse-segmentation model")
    parser.add_argument('--model-name', default='model.pt',
        help="Filename of the coarse-segmentation model")
    parser.add_argument('--model-url',
        default='https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro',
        help="Download URL for coarse-segmentation model if missing")
    parser.add_argument('--sam2-config',
        default='configs/sam2.1/sam2.1_hiera_l',
        help="Hydra config group/name for SAM2")
    parser.add_argument('--sam2-checkpoint',
        default=str(pathlib.Path.home()/ 'repos'/'sam2'/'checkpoints'/'sam2.1_hiera_large.pt'),
        help="Path to SAM2 checkpoint")
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda',
        help="Torch device for inference")
    args = parser.parse_args()

    rclpy.init()
    node = LiveSegDiffNode(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
