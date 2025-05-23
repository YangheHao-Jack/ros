import pathlib
import time

import cv2
import cv_bridge
import gdown
import numpy as np
import rclpy
import torch
from kornia import image_to_tensor, tensor_to_image
from kornia.geometry import resize
from rclpy.node import Node
from roboreg.util.viz import overlay_mask
from sensor_msgs.msg import CompressedImage


class LBRSegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__(node_name="lbr_segmentation")

        self._bridge = cv_bridge.CvBridge()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device {self._device}.")
        self._video_writer = None
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._output_path = str(pathlib.Path.home() / "lbr_output.mp4")
        self._fps = 20.0 

        # declare parameters
        self.declare_parameter(
            "model_path",
            ".cache/torch/hub/checkpoints/roboreg",
        )
        self.declare_parameter("model_name", "model.pt")
        self.declare_parameter(
            "model_url",
            "https://drive.google.com/uc?id=1_byUJRzTtV5FQbqvVRTeR8FVY6nF87ro",
        )
        self.declare_parameter("pth", 0.5)

        # get parameters
        self._model_path = pathlib.Path.home() / pathlib.Path(
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self._model_name = (
            self.get_parameter("model_name").get_parameter_value().string_value
        )
        self._model_url = (
            self.get_parameter("model_url").get_parameter_value().string_value
        )
        self._model_file = self._model_path / self._model_name
        self._pth = self.get_parameter("pth").get_parameter_value().double_value

        
        # subscriptions
        self._img_sub = self.create_subscription(
            CompressedImage, "/zed/zed_node/left/image_rect_color/compressed", self._on_img, 1
        )

        # model instantiation
        self._download_model()
        self._load_model()

    def _download_model(self) -> None:
        if not self._model_path.exists():
            self.get_logger().info(f"Creating {self._model_path} for download")
            self._model_path.mkdir(parents=True)

        if self._model_file.exists():
            self.get_logger().info(f"Model already available under {self._model_file}.")
            return

        self.get_logger().info(
            f"Downloading {self._model_url} to {self._model_file}..."
        )
        gdown.download(
            self._model_url,
            output=str(self._model_file),
            quiet=False,
        )
        self.get_logger().info(f"Finished download.")

    def _load_model(self) -> None:
        # load model
        self.get_logger().info(f"Attempt loading {self._model_file}...")
        start = time.time()
        self._model = torch.jit.load(self._model_file)
        self.get_logger().info(f"Loaded model in {time.time() - start:.2f} seconds.")
        self._model = self._model.to(self._device)
        self._model.eval()

    def _on_img(self, img_msg: CompressedImage) -> None:
        
        img = self._bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        

        height, width = img.shape[:2]
        self.get_logger().info(f"got image {height} x {width}")
        # run inference
        mask = self._inference(img)

        # imshow
        mask = tensor_to_image(mask)
        mask = (mask * 255.0).astype(np.uint8)

        # create an overlay
        mask = cv2.resize(
            mask,
            (img.shape[1], img.shape[0]),
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        overlay = overlay_mask(
            img,
            mask,
            "r",
            alpha=0.8,
            beta=0.5,
            scale=1.0,
        )
        if self._video_writer is None:
            self.get_logger().info(f"Initializing VideoWriter -> {self._output_path}")
            self._video_writer = cv2.VideoWriter(
                self._output_path,
                self._fourcc,
                self._fps,
                (width, height),
            )
        
        self._video_writer.write(overlay)
        cv2.imshow("Mask overlay", overlay)
        cv2.waitKey(1)

    def _inference(self, img: np.ndarray) -> np.ndarray:
        img = image_to_tensor(img)
        img_resized = resize(img, (256, 480))
        img_resized = (img_resized.float() / 255.0).unsqueeze(0)
        img_resized = img_resized.to(self._device)

        logits = self._model(img_resized)
        probabilities = torch.sigmoid(logits)
        mask = (probabilities > self._pth).float()
        return mask
    def destroy_node(self):
        
        if self._video_writer:
            self.get_logger().info("Releasing VideoWriter")
            self._video_writer.release()
        super().destroy_node()

def main() -> None:
    rclpy.init()
    segmentation_node = LBRSegmentationNode()
    try:
        rclpy.spin(segmentation_node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()