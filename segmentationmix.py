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

class LBRSegmentationNode(Node):
    def __init__(self):
        super().__init__('lbr_segmentation_sam2')

        # Speedups
        cudnn.benchmark = True

        # OpenCV bridge
        self._bridge = cv_bridge.CvBridge()

        # Device
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device {self._device}')

        # Video writer
        self._video_writer = None
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._fps    = 20.0
        self._base_out = pathlib.Path.home() / 'lbr_output_refined'

        # Coarse model parameters
        self.declare_parameter('model_path', '.cache/torch/hub/checkpoints/roboreg')
        self.declare_parameter('model_name', 'model.pt')
        self.declare_parameter('model_url',
            'https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro'
        )
        self.declare_parameter('pth', 0.5)

        mp = pathlib.Path(self.get_parameter('model_path').value)
        mn = self.get_parameter('model_name').value
        self._model_file = pathlib.Path.home()/mp/mn
        self._model_url  = self.get_parameter('model_url').value
        self._pth        = self.get_parameter('pth').value

        # Download and load coarse segmentation model
        self._download_model()
        self._load_model()

        # Build and compile SAM 2 predictor
        sam_cfg  = self.declare_parameter('sam2_config','configs/sam2.1/sam2.1_hiera_l').value
        sam_ckpt = str(pathlib.Path.home()/ 'repos' / 'sam2' / 'checkpoints' / 'sam2.1_hiera_large.pt')
        self.get_logger().info(f'Building SAM2 with config {sam_cfg}')
        sam = build_sam2(sam_cfg, sam_ckpt).to(self._device).eval()
        sam = sam.to(memory_format=torch.channels_last)
        sam = torch.compile(sam)
        self._sam_pred = SAM2ImagePredictor(sam)

        # Propagation state
        self._first        = True
        self._prev_low_res = None
        self._last_ts      = None
        self._loop_count   = 0

        # Subscribe to compressed image topic
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
        self.get_logger().info(f'Loaded in {time.time()-t0:.1f}s')

    def _inference(self, img: np.ndarray) -> np.ndarray:
        # Returns boolean mask [H,W]
        t = image_to_tensor(img)
        t = resize(t, (256,480)).float() / 255.0
        t = t.unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(t)
            prob   = torch.sigmoid(logits)
            mask   = (prob > self._pth).float()
        # Upsample to original resolution
        h, w = img.shape[:2]
        mask_full = torch.nn.functional.interpolate(
            mask, size=(h,w), mode='nearest'
        )[0,0]
        return mask_full.bool().cpu().numpy()

    def _reset_for_new_loop(self):
        self.get_logger().info('** Playback loop detected – resetting state **')
        self._first        = True
        self._prev_low_res = None
        self._last_ts      = None
        if self._video_writer:
            out = str(self._base_out) + f'_loop{self._loop_count}.mp4'
            self.get_logger().info(f'Finalizing {out}')
            self._video_writer.release()
            self._video_writer = None
        self._loop_count += 1

    def _ensure_writer(self, w, h):
        if self._video_writer is None:
            out_path = str(self._base_out) + f'_loop{self._loop_count}.mp4'
            self.get_logger().info(f'Initializing VideoWriter → {out_path}')
            self._video_writer = cv2.VideoWriter(
                out_path, self._fourcc, self._fps, (w, h)
            )

    def _on_img(self, msg: CompressedImage):
        # Detect bag restart by timestamp
        ts = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
        if self._last_ts is not None and ts < self._last_ts:
            self._reset_for_new_loop()
        self._last_ts = ts

        # Decode image
        img_rgb = self._bridge.compressed_imgmsg_to_cv2(msg, 'rgb8')
        h, w     = img_rgb.shape[:2]

        # Coarse segmentation
        coarse = self._inference(img_rgb)  # [H,W] bool

        # SAM2 processing
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if self._first:
            ys, xs = np.where(coarse)
            box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.int32)
            with torch.amp.autocast('cuda'):
                self._sam_pred.set_image(img_bgr)
                masks, scores, low_res = self._sam_pred.predict(
                    box=box, multimask_output=False
                )
            refined = masks[0].astype(bool)
            self._prev_low_res = low_res[0:1,...]
            self._first = False
        else:
            with torch.amp.autocast('cuda'):
                self._sam_pred.set_image(img_bgr)
                masks, scores, low_res = self._sam_pred.predict(
                    mask_input=self._prev_low_res,
                    multimask_output=False
                )
            refined = masks[0].astype(bool)
            self._prev_low_res = low_res[0:1,...]

        # Overlay and write
        overlay = overlay_mask(
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            (refined*255).astype(np.uint8),
            'r', alpha=0.8, beta=0.5
        )
        self._ensure_writer(w, h)
        self._video_writer.write(overlay)
        cv2.imshow('Refined Segmentation', overlay)
        # Quit on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit requested via 'q'; shutting down.")
            rclpy.shutdown()

    def destroy_node(self):
        if self._video_writer:
            self.get_logger().info('Releasing VideoWriter')
            self._video_writer.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = LBRSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
