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

class ImageFolderExtractor:
    def __init__(self, args):
        # 1) speed up
        cudnn.benchmark = True

        # 2) device & segmentation params
        self.device         = args.device
        self.num_pos        = args.num_pos
        self.num_neg        = args.num_neg
        self.num_components = args.num_components
        self.pth            = args.pth
        self.alpha          = args.alpha

        # 3) load camera_info.yaml for target resolution
        with open(args.camera_info_file, 'r') as f:
            cam = yaml.safe_load(f)
        self.reg_w = int(cam.get('image_width', cam.get('width')))
        self.reg_h = int(cam.get('image_height', cam.get('height')))
        print(f"[FolderExtractor] target registration size: {self.reg_w}×{self.reg_h}")

        # 4) input and output directories
        self.image_dir = pathlib.Path(args.image_dir)
        self.out_dir   = pathlib.Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # 5) prepare model paths
        model_file = pathlib.Path.home() / args.model_path / args.model_name
        model_file.parent.mkdir(parents=True, exist_ok=True)
        if not model_file.exists():
            print(f"Downloading coarse model → {model_file}")
            gdown.download(args.model_url, str(model_file), quiet=False)
        print(f"Loading coarse model from {model_file}")
        self.coarse = torch.jit.load(str(model_file), map_location=self.device)\
                          .eval().to(self.device)

        # 6) build SAM 2 predictor
        print(f"Building SAM2 with config={args.sam2_config}")
        sam = build_sam2(args.sam2_config, args.sam2_checkpoint)\
                  .to(self.device).eval()
        sam = sam.to(memory_format=torch.channels_last)
        sam = torch.compile(sam)
        self.sam_pred = SAM2ImagePredictor(sam)

        # 7) windows
        flags = cv2.WINDOW_NORMAL
        cv2.namedWindow('Original',            flags)
        cv2.namedWindow('Segmentation Overlay',flags)

    def inference_coarse(self, img: np.ndarray) -> np.ndarray:
        t = image_to_tensor(img)
        t = resize(t, (256, 480)).float() / 255.0
        t = t.unsqueeze(0).to(self.device)
        with torch.amp.autocast(self.device):
            logits = self.coarse(t)
            prob   = torch.sigmoid(logits)
            mask   = (prob > self.pth).float()
        h, w = img.shape[:2]
        full = torch.nn.functional.interpolate(
            mask, size=(h, w), mode='nearest'
        )[0, 0]
        return full.bool().cpu().numpy()

    def keep_largest(self, mask: np.ndarray):
        labels, stats = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )[1:3]
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size==0:
            return labels, []
        topk = (np.argsort(areas)[-self.num_components:] + 1).tolist()
        return labels, topk

    def refine_sam2(self, img: np.ndarray, labels_map, topk):
        refined = np.zeros_like(labels_map, dtype=bool)
        for comp in topk:
            ys, xs = np.where(labels_map==comp)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            box = np.array([[x1,y1,x2,y2]], dtype=np.int32)
            with torch.amp.autocast(self.device):
                self.sam_pred.set_image(img)
                masks,_,_ = self.sam_pred.predict(
                    box=box,
                    multimask_output=False
                )
            refined |= masks[0].astype(bool)
        return refined

    def run(self):
        img_paths = sorted(self.image_dir.glob("*.png"))
        for idx, img_p in enumerate(img_paths):
            name = f"{idx:06d}"
            img  = cv2.imread(str(img_p), cv2.IMREAD_COLOR)

            # a) save reg-sized original
            img_reg = cv2.resize(img, (self.reg_w, self.reg_h),
                                 interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(self.out_dir/f"left_image_reg_{name}.png"), img_reg)

            # b) coarse + cleanup
            coarse = self.inference_coarse(img)
            labels_map, topk = self.keep_largest(coarse)

            # c) SAM2 refine
            refined = self.refine_sam2(img, labels_map, topk)

            # d) save reg-sized mask
            mask_u8 = (refined.astype(np.uint8)*255)
            mask_reg= cv2.resize(mask_u8, (self.reg_w,self.reg_h),
                                 interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(self.out_dir/f"mask_sam2_reg_{name}.png"), mask_reg)
            np.save(     str(self.out_dir/f"mask_sam2_reg_{name}.npy"), (mask_reg>0).astype(np.uint8))

            # e) overlay on orig
            over = overlay_mask(img, (refined*255).astype(np.uint8),
                                'r', alpha=0.5, beta=0.5)
            cv2.imwrite(str(self.out_dir/f"overlay_{name}.png"), over)

            # f) binary overlay
            cv2.imwrite(str(self.out_dir/f"binary_overlay_{name}.png"),
                        refined.astype(np.uint8)*255)

            # g) display
            cv2.imshow('Original', img)
            cv2.imshow('Segmentation Overlay', over)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break

        cv2.destroyAllWindows()
        print("✅ Image‐folder extraction complete.")

def parse_args():
    p = argparse.ArgumentParser(
        description="Apply SAM2 segmentation to a folder of images"
    )
    p.add_argument('--image-dir',       required=True,
                   help="Directory of input .png images")
    p.add_argument('--camera-info-file','-cif', required=True,
                   help="YAML with image_width/height for registration")
    p.add_argument('--out-dir',         required=True,
                   help="Where to save registered images & masks")
    p.add_argument('--num-pos',        type=int, default=30)
    p.add_argument('--num-neg',        type=int, default=30)
    p.add_argument('--num-components', type=int, default=1)
    p.add_argument('--pth',            type=float, default=0.5)
    p.add_argument('--alpha',          type=float, default=0.3)
    p.add_argument('--model-path',     default='.cache/torch/hub/checkpoints/roboreg')
    p.add_argument('--model-name',     default='model.pt')
    p.add_argument('--model-url',
                   default='https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro')
    p.add_argument('--sam2-config',    default='configs/sam2.1/sam2.1_hiera_l')
    p.add_argument('--sam2-checkpoint',required=True,
                   help="Path to SAM2 checkpoint .pt")
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    return p.parse_args()

def main():
    args = parse_args()
    ext = ImageFolderExtractor(args)
    ext.run()

if __name__=='__main__':
    main()
