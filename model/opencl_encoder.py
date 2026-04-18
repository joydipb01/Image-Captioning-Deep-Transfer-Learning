import os
from typing import Tuple

import cv2
import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class OpenCLEncoder:
    """ResNet50 feature extractor using OpenCV DNN with OpenCL target."""

    def __init__(self, onnx_path: str, use_fp16: bool = False, prefer_opencl: bool = True):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"Encoder ONNX not found at: {onnx_path}. "
                "Export it with: python scripts/export_resnet50_onnx.py --output <path>"
            )

        self.onnx_path = onnx_path
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.backend = "opencv"

        self.target = "cpu"
        if prefer_opencl and cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            if use_fp16:
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
                self.target = "opencl_fp16"
            else:
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                self.target = "opencl"
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @staticmethod
    def _preprocess_bgr(img_bgr: np.ndarray, image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            img_bgr,
            scalefactor=1.0 / 255.0,
            size=image_size,
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )

        # Normalize with ImageNet stats after scaling to [0, 1].
        for c in range(3):
            blob[:, c, :, :] = (blob[:, c, :, :] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        return blob.astype(np.float32)

    def extract(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        blob = self._preprocess_bgr(img)
        self.net.setInput(blob)
        out = self.net.forward()

        # Export script emits a [1, 2048] vector.
        feature = out.reshape(-1).astype(np.float32)
        return feature
