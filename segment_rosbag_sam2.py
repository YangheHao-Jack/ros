#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def collect_clicks(frame):
    title = "First Frame - left=pos, right=neg; press 'c' to confirm"
    vis = frame.copy()
    pts, lbls = [], []
    def mouse_cb(evt, x, y, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y]); lbls.append(1)
            cv2.circle(vis, (x, y), 5, (0,255,0), -1)
        elif evt == cv2.EVENT_RBUTTONDOWN:
            pts.append([x, y]); lbls.append(0)
            cv2.circle(vis, (x, y), 5, (0,0,255), -1)

    # Create & show window before callback
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, vis); cv2.waitKey(1)
    cv2.setMouseCallback(title, mouse_cb)
    while True:
        cv2.imshow(title, vis)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    cv2.destroyWindow(title)
    return np.array(pts, dtype=np.int32), np.array(lbls, dtype=np.int32)

def segment_rosbag_prompt(
    bag_dir: str,
    output_dir: str,
    topic_name: str,
    model_cfg: str,
    checkpoint: str,
    device: str,
    display: bool
):
    cudnn.benchmark = True

    # 1) Open bag
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id='mcap'),
        ConverterOptions(input_serialization_format='cdr',
                         output_serialization_format='cdr')
    )

    # 2) Filter to image topic
    tops = {t.name:t.type for t in reader.get_all_topics_and_types()}
    if topic_name not in tops:
        raise RuntimeError(f"Topic '{topic_name}' not in bag. Available: {list(tops)}")
    reader.set_filter(StorageFilter(topics=[topic_name]))

    # 3) Prepare outputs & display
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "segmented.mp4")
    if display:
        cv2.namedWindow("Segmented", cv2.WINDOW_NORMAL)

    # 4) Read first frame
    _, buf, ts0 = reader.read_next()
    msg = deserialize_message(buf, CompressedImage)
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    first_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 5) Collect clicks
    pts, lbls = collect_clicks(first_frame)

    # 6) Build and compile SAM2 model
    sam = build_sam2(model_cfg, checkpoint).to(device).eval()
    sam = sam.to(memory_format=torch.channels_last)
    sam = torch.compile(sam)

    predictor = SAM2ImagePredictor(sam)

    # 7) Initial mask under autocast (includes set_image and predict)
    with torch.amp.autocast('cuda'):
        predictor.set_image(first_frame)
        masks, scores, low_res_masks = predictor.predict(
            point_coords=pts,
            point_labels=lbls,
            multimask_output=False
        )

    prev_mask      = masks[0].astype(bool)
    prev_low_res   = low_res_masks[0:1, ...]

    # 8) Init video writer
    h, w = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,  # adjust if known
        (w, h)
    )

    def save_and_write(frame, mask_bool, timestamp):
        np.save(os.path.join(masks_dir, f"{timestamp}.npy"),
                mask_bool.astype(np.uint8))
        overlay = frame.copy()
        alpha = 0.6
        color = np.array([0,128,255], dtype=np.uint8)
        fg = overlay[mask_bool].astype(np.float32)
        blended = fg * (1 - alpha) + color * alpha
        overlay[mask_bool] = blended.astype(np.uint8)
        writer.write(overlay)
        if display:
            cv2.imshow("Segmented", overlay)
            cv2.waitKey(1)

    # 9) Save first frame
    save_and_write(first_frame, prev_mask, ts0)

    # 10) Process remaining frames
    print("▶ Propagating mask through video…")
    idx = 1
    while reader.has_next():
        _, buf, ts = reader.read_next()
        msg = deserialize_message(buf, CompressedImage)
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Propagation under autocast
        with torch.amp.autocast('cuda'):
            predictor.set_image(frame)
            masks, scores, low_res_masks = predictor.predict(
                mask_input=prev_low_res,
                multimask_output=False
            )

        prev_low_res = low_res_masks[0:1, ...]
        prev_mask    = masks[0].astype(bool)

        save_and_write(frame, prev_mask, ts)
        if idx % 50 == 0:
            print(f"   processed {idx} frames…")
        idx += 1

    # 11) Cleanup
    writer.release()
    if display:
        cv2.destroyAllWindows()

    print("✅ Done!")
    print(f"Raw masks saved to: {masks_dir}")
    print(f"Overlay video saved to: {video_path}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Prompt-based SAM2 segmentation on a rosbag2 CompressedImage topic"
    )
    parser.add_argument("bag_dir",    help="rosbag2 (MCAP) folder")
    parser.add_argument("output_dir", help="Directory to save outputs")
    parser.add_argument(
        "--topic", default="/zed/zed_node/left/image_rect_color/compressed"
    )
    parser.add_argument(
        "--model-config", required=True,
        help="Hydra config group/name, e.g. 'sam2.1/sam2.1_hiera_l'"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to SAM2 checkpoint, e.g. checkpoints/sam2.1_hiera_large.pt"
    )
    parser.add_argument(
        "--device", choices=["cuda","cpu"], default="cuda"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable live display"
    )
    args = parser.parse_args()

    segment_rosbag_prompt(
        args.bag_dir,
        args.output_dir,
        args.topic,
        args.model_config,
        args.checkpoint,
        args.device,
        not args.no_display
    )
