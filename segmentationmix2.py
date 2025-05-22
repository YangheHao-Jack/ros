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

        # Speed optimizations
        cudnn.benchmark = True

        # OpenCV bridge
        self._bridge = cv_bridge.CvBridge()

        # Device
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device {self._device}')

        # --- Output paths (separate) ---
        self.declare_parameter(
            'video_output_path',
            str(pathlib.Path.home() / 'lbr_segmented.mp4')
        )
        self.declare_parameter(
            'mask_output_dir',
            str('/home/jack/experiments/25_05_20_SAM2_Mixed_OutputMask/lbr_masks')
        )
        video_out_param = self.get_parameter('video_output_path').get_parameter_value().string_value
        mask_dir_param  = self.get_parameter('mask_output_dir').get_parameter_value().string_value
        self._video_out = pathlib.Path(video_out_param)
        self._mask_dir  = pathlib.Path(mask_dir_param)
        self._mask_dir.mkdir(parents=True, exist_ok=True)

        # Video writer placeholder
        self._video_writer = None
        self._fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
        self._fps          = 30.0

        # Coarse‐segmentation model parameters
        self.declare_parameter('model_path', '.cache/torch/hub/checkpoints/roboreg')
        self.declare_parameter('model_name', 'model.pt')
        self.declare_parameter('model_url',
            'https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro'
        )
        self.declare_parameter('pth', 0.9)

        model_path = pathlib.Path(self.get_parameter('model_path').value)
        model_name = self.get_parameter('model_name').value
        self._model_file = pathlib.Path.home() / model_path / model_name
        self._model_url  = self.get_parameter('model_url').value
        self._pth        = self.get_parameter('pth').value

        # Download & load coarse‐segmentation model
        self._download_model()
        self._load_model()

        # SAM 2 setup
        sam_cfg  = self.declare_parameter(
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

        # Propagation state
        self._prev_low_res = None

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
            self.get_logger().info(f'Downloading coarse model to {self._model_file}')
            gdown.download(self._model_url, str(self._model_file), quiet=False)

    def _load_model(self):
        self.get_logger().info(f'Loading coarse model from {self._model_file}')
        t0 = time.time()
        self._model = torch.jit.load(str(self._model_file), map_location=self._device)
        self._model.eval().to(self._device)
        self.get_logger().info(f'Coarse model loaded in {time.time() - t0:.1f}s')

    def _inference(self, img: np.ndarray) -> np.ndarray:
        """Run coarse‐segmentation model and upsample to original resolution."""
        t = image_to_tensor(img)                   # [C,H,W]
        t = resize(t, (256, 480)).float() / 255.0   # [C,256,480]
        t = t.unsqueeze(0).to(self._device)        # [1,C,256,480]
        with torch.no_grad():
            logits = self._model(t)                 # [1,1,256,480]
            prob   = torch.sigmoid(logits)
            mask   = (prob > self._pth).float()
        h, w = img.shape[:2]
        mask_full = torch.nn.functional.interpolate(
            mask, size=(h, w), mode='nearest'
        )[0, 0]                                     # [H,W]
        return mask_full.bool().cpu().numpy()       # bool ndarray

    def _keep_largest_components(self, mask: np.ndarray, k: int = 1) -> np.ndarray:
        """Keep only the k largest connected components in the binary mask."""
        labels, stats = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )[1:3]  # labels, stats
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return mask
        topk = np.argsort(areas)[-k:] + 1
        clean = np.zeros_like(mask, dtype=bool)
        for lbl in topk:
            clean |= (labels == lbl)
        return clean

    def _sample_points(self, mask: np.ndarray, num_pos=30, num_neg=30):
        """Randomly sample positive points inside mask and negative outside."""
        ys, xs = np.nonzero(mask)
        coords = np.stack([xs, ys], axis=-1)
        if len(coords) > 0:
            idx = np.random.choice(len(coords), min(num_pos, len(coords)), replace=False)
            pos = coords[idx]
        else:
            pos = np.zeros((0, 2), dtype=int)
        inv  = ~mask
        ys0, xs0 = np.nonzero(inv)
        coords0  = np.stack([xs0, ys0], axis=-1)
        if len(coords0) > 0:
            idx0 = np.random.choice(len(coords0), min(num_neg, len(coords0)), replace=False)
            neg  = coords0[idx0]
        else:
            neg = np.zeros((0, 2), dtype=int)
        pts    = np.vstack([pos, neg]).astype(np.int32)
        labels = np.hstack([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int32)
        return pts, labels

    def _ensure_writer(self, w: int, h: int):
        """Initialize video writer on first frame."""
        if self._video_writer is None:
            self._video_out.parent.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'Writing video to: {self._video_out}')
            self._video_writer = cv2.VideoWriter(
                str(self._video_out),
                self._fourcc,
                self._fps,
                (w, h)
            )

    def _on_img(self, msg: CompressedImage):
        # Decode image
        img_rgb = self._bridge.compressed_imgmsg_to_cv2(msg, 'rgb8')
        h, w     = img_rgb.shape[:2]

        # 1) Coarse segmentation & cleanup
        coarse = self._inference(img_rgb)
        coarse = self._keep_largest_components(coarse, k=1)

        # 2) Sample fresh prompts
        pts, lbls = self._sample_points(coarse, num_pos=30, num_neg=30)
        img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 3) SAM 2 prediction with points + history
        with torch.amp.autocast('cuda'):
            self._sam_pred.set_image(img_bgr)
            masks, scores, low_res = self._sam_pred.predict(
                point_coords=pts,
                point_labels=lbls,
                mask_input=self._prev_low_res,
                multimask_output=False
            )
        self._prev_low_res = low_res[0:1, ...]
        refined = masks[0].astype(bool)

        # 4) Morphological closing
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
        refined = cv2.morphologyEx((refined.astype(np.uint8)*255),
                                   cv2.MORPH_CLOSE, kernel) > 0

        # 5) Ensure video writer is ready
        self._ensure_writer(w, h)

        # 6) Save final mask per frame using timestamp
        ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        mask_png = self._mask_dir / f"{int(ts)}.png"
        mask_npy = self._mask_dir / f"{int(ts)}.npy"
        cv2.imwrite(str(mask_png), (refined.astype(np.uint8)*255))
        np.save(str(mask_npy), refined)

        # 7) Overlay & write video
        overlay = overlay_mask(
            img_bgr,
            (refined*255).astype(np.uint8),
            'r', alpha=0.8, beta=0.5
        )
        self._video_writer.write(overlay)

        # 8) Display & quit key
        cv2.imshow('Refined Segmentation', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
