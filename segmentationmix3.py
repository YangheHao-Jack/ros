#!/usr/bin/env python3
import os
import cv2
import pathlib
import time

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
    def __init__(self):
        super().__init__('live_seg_diff_node')

        # Speedups
        cudnn.benchmark = True

        # CV bridge
        self._bridge = cv_bridge.CvBridge()

        # Device
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self._device}')

        # --- Output parameters ---
        self.declare_parameter(
            'video_output_path',
            str('/home/jack/experiments/segmentation_overlay.mp4')
        )
        self.declare_parameter(
            'diff_overlay_output_path',
            str('/home/jack/experiments/differential_overlay.mp4')
        )
        self.declare_parameter(
            'diff_mask_video_output_path',
            str('/home/jack/experiments/differential_mask.mp4')
        )
        self.declare_parameter(
            'mask_output_dir',
            str('/home/jack/experiments/seg_masks')
        )
        vid_out      = self.get_parameter('video_output_path').get_parameter_value().string_value
        diff_out     = self.get_parameter('diff_overlay_output_path').get_parameter_value().string_value
        diff_mask_out= self.get_parameter('diff_mask_video_output_path').get_parameter_value().string_value
        mask_dir     = self.get_parameter('mask_output_dir').get_parameter_value().string_value
        self._video_out       = pathlib.Path(vid_out)
        self._diff_out        = pathlib.Path(diff_out)
        self._diff_mask_out   = pathlib.Path(diff_mask_out)
        self._mask_dir        = pathlib.Path(mask_dir)
        self._mask_dir.mkdir(parents=True, exist_ok=True)

        # Writers and state
        self._video_writer      = None  # segmentation overlay video
        self._diff_writer       = None  # differential overlay video
        self._diff_mask_writer  = None  # differential mask video
        self._fourcc            = cv2.VideoWriter_fourcc(*'mp4v')
        self._fps               = 30.0
        self._prev_frame        = None
        self._prev_low_res      = None

        # Coarse‐segmentation model params
        self.declare_parameter('model_path', '.cache/torch/hub/checkpoints/roboreg')
        self.declare_parameter('model_name', 'model.pt')
        self.declare_parameter('model_url',
            'https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro'
        )
        self.declare_parameter('pth', 0.5)
        mp = pathlib.Path(self.get_parameter('model_path').value)
        mn = self.get_parameter('model_name').value
        self._model_file = pathlib.Path.home() / mp / mn
        self._model_url  = self.get_parameter('model_url').value
        self._pth        = self.get_parameter('pth').value

        self._download_model()
        self._load_model()

        # SAM 2 setup
        sam_cfg = self.declare_parameter(
            'sam2_config',
            'configs/sam2.1/sam2.1_hiera_l'
        ).get_parameter_value().string_value
        sam_ckpt = str(
            pathlib.Path.home() / 'repos' / 'sam2' / 'checkpoints' / 'sam2.1_hiera_large.pt'
        )
        self.get_logger().info(f'Building SAM2 with config {sam_cfg}')
        sam = build_sam2(sam_cfg, sam_ckpt).to(self._device).eval()
        sam = sam.to(memory_format=torch.channels_last)
        sam = torch.compile(sam)
        self._sam_pred = SAM2ImagePredictor(sam)

        # Prepare display windows
        cv2.namedWindow('Segmentation Overlay',      cv2.WINDOW_NORMAL)
        cv2.namedWindow('Differential Mask',           cv2.WINDOW_NORMAL)
        cv2.namedWindow('Differential Overlay Blue',   cv2.WINDOW_NORMAL)

        # Subscribe to the compressed image topic
        self.create_subscription(
            CompressedImage,
            '/zed/zed_node/left/image_rect_color/compressed',
            self._on_img,
            1
        )

    def _download_model(self):
        d = self._model_file.parent
        if not d.exists():
            d.mkdir(parents=True)
        if not self._model_file.exists():
            self.get_logger().info(f'Downloading coarse model → {self._model_url}')
            gdown.download(self._model_url, str(self._model_file), quiet=False)

    def _load_model(self):
        self.get_logger().info(f'Loading coarse model from {self._model_file}')
        t0 = time.time()
        self._model = torch.jit.load(str(self._model_file), map_location=self._device)
        self._model.eval().to(self._device)
        self.get_logger().info(f'Coarse model loaded in {time.time()-t0:.1f}s')

    def _inference(self, img: np.ndarray) -> np.ndarray:
        t = image_to_tensor(img)
        t = resize(t, (256,480)).float() / 255.0
        t = t.unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(t)
            prob   = torch.sigmoid(logits)
            mask   = (prob > self._pth).float()
        h, w = img.shape[:2]
        mask_full = torch.nn.functional.interpolate(
            mask, size=(h,w), mode='nearest'
        )[0,0]
        return mask_full.bool().cpu().numpy()

    def _keep_largest_components(self, mask: np.ndarray, k: int = 1) -> np.ndarray:
        labels, stats = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )[1:3]
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return mask
        topk = np.argsort(areas)[-k:] + 1
        clean = np.zeros_like(mask, dtype=bool)
        for lbl in topk:
            clean |= (labels == lbl)
        return clean

    def _sample_points(self, mask: np.ndarray, num_pos=30, num_neg=30):
        ys, xs = np.nonzero(mask)
        coords = np.stack([xs, ys], axis=-1)
        if coords.size:
            idx = np.random.choice(len(coords), min(num_pos,len(coords)), replace=False)
            pos = coords[idx]
        else:
            pos = np.zeros((0,2), dtype=int)
        inv = ~mask
        ys0, xs0 = np.nonzero(inv)
        coords0 = np.stack([xs0, ys0], axis=-1)
        if coords0.size:
            idx0 = np.random.choice(len(coords0), min(num_neg,len(coords0)), replace=False)
            neg  = coords0[idx0]
        else:
            neg = np.zeros((0,2), dtype=int)
        pts    = np.vstack([pos, neg]).astype(np.int32)
        labels = np.hstack([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int32)
        return pts, labels

    def _ensure_writers(self, w, h):
        if self._video_writer is None:
            self._video_out.parent.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'Writing segmentation video → {self._video_out}')
            self._video_writer = cv2.VideoWriter(
                str(self._video_out), self._fourcc, self._fps, (w,h)
            )
        if self._diff_writer is None:
            self._diff_out.parent.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'Writing differential overlay video → {self._diff_out}')
            self._diff_writer = cv2.VideoWriter(
                str(self._diff_out), self._fourcc, self._fps, (w,h)
            )
        if self._diff_mask_writer is None:
            self._diff_mask_out.parent.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'Writing differential mask video → {self._diff_mask_out}')
            self._diff_mask_writer = cv2.VideoWriter(
                str(self._diff_mask_out), self._fourcc, self._fps, (w,h), isColor=False
            )

    def _on_img(self, msg: CompressedImage):
        # Decode frame
        arr   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w  = frame.shape[:2]

        # 1) Coarse segmentation & cleanup
        coarse = self._inference(frame)
        coarse = self._keep_largest_components(coarse, k=1)

        # 2) Sample prompts & run SAM2
        pts, lbls = self._sample_points(coarse, 30, 30)
        with torch.amp.autocast('cuda'):
            self._sam_pred.set_image(frame)
            masks, _, low_res = self._sam_pred.predict(
                point_coords=pts,
                point_labels=lbls,
                mask_input=self._prev_low_res,
                multimask_output=False
            )
        self._prev_low_res = low_res[0:1,...]
        refined = masks[0].astype(bool)

        # 3) Save mask
        ts = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
        png_path = self._mask_dir / f"{ts}.png"
        np.save(str(self._mask_dir / f"{ts}.npy"), refined)
        cv2.imwrite(str(png_path), (refined.astype(np.uint8)*255))

        # 4) Ensure writers
        self._ensure_writers(w, h)

        # 5) Segmentation overlay (red)
        seg_overlay = overlay_mask(frame, (refined*255).astype(np.uint8),
                                   'r', alpha=0.5, beta=0.5)
        self._video_writer.write(seg_overlay)
        cv2.imshow('Segmentation Overlay', seg_overlay)

        # 6) Differential mask & overlay
        if self._prev_frame is not None:
            diff = cv2.absdiff(frame, self._prev_frame)
            diff_masked = cv2.bitwise_and(diff, diff, mask=refined.astype(np.uint8))

            # 6a) Differential mask (grayscale)
            diff_gray = cv2.cvtColor(diff_masked, cv2.COLOR_BGR2GRAY)
            self._diff_mask_writer.write(diff_gray)
            cv2.imshow('Differential Mask', diff_gray)

            # 6b) Differential overlay (blue)
            vis = cv2.normalize(diff_masked, None, 0,255,cv2.NORM_MINMAX)
            # new pure‐blue alpha blend:
            blue_mask = np.zeros_like(frame)           # same shape as frame, BGR
            blue_mask[refined] = (255, 0, 0)           # pure blue where mask==True
            alpha = 0.3                                # 30% blue, 70% original

# blend the original frame and the blue mask
            blue_overlay = cv2.addWeighted(frame, 1 - alpha,
                                           blue_mask, alpha,
                                           0)
            
            self._diff_writer.write(blue_overlay)
            cv2.imshow('Differential Overlay Blue', blue_overlay)

        self._prev_frame = frame

        # 7) Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Quit requested; shutting down.")
            rclpy.shutdown()

    def destroy_node(self):
        if self._video_writer:
            self._video_writer.release()
        if self._diff_writer:
            self._diff_writer.release()
        if self._diff_mask_writer:
            self._diff_mask_writer.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = LiveSegDiffNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
