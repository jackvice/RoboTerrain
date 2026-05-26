"""Pose-detection backend abstraction.

The original project ran on a Jetson and used ``jetson_inference.poseNet``
(Pose-ResNet18-Body) plus ``jetson_utils`` for image transfer to the GPU.  On a
regular Linux x86 machine those libraries aren't available, so this module
provides a thin compatibility layer that can run on either:

* the Jetson (using ``jetson_inference`` + ``jetson_utils`` if importable), or
* a desktop NVIDIA GPU (using Ultralytics YOLOv8-pose under the hood).

The rest of the pipeline (`person_keypoints.py`, `tracking.py`) only cares
about a small interface:

    poses = backend.process(bgr_or_rgba_numpy_image)
    for pose in poses:
        for kp in pose.Keypoints:
            kp.ID, kp.x, kp.y

So we expose exactly that surface here.  Keypoint IDs follow the COCO-17
ordering used by Pose-ResNet18-Body, with an extra synthetic ID ``17`` for the
neck (midpoint of the shoulders) to stay compatible with the existing
``person_keypoints`` logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import os

import numpy as np


@dataclass
class Keypoint:
    ID: int
    x: float
    y: float


@dataclass
class Pose:
    Keypoints: List[Keypoint] = field(default_factory=list)


def _try_import_jetson():
    try:
        import jetson_inference  # type: ignore
        import jetson_utils  # type: ignore
        return jetson_inference, jetson_utils
    except Exception:
        return None, None


class _JetsonBackend:
    """Original Pose-ResNet18-Body backend (used when running on a Jetson)."""

    name = "jetson"

    def __init__(self, network: str = "resnet18-body", threshold: float = 0.3):
        ji, ju = _try_import_jetson()
        assert ji is not None and ju is not None, "jetson_inference not available"
        self._ji = ji
        self._ju = ju
        self.net = ji.poseNet(network, [], threshold)

    def _to_cuda(self, rgba_image: np.ndarray):
        return self._ju.cudaFromNumpy(rgba_image)

    def process(self, image_bgra_or_bgr: np.ndarray) -> List[Pose]:
        import cv2
        if image_bgra_or_bgr.ndim == 3 and image_bgra_or_bgr.shape[2] == 3:
            rgba = cv2.cvtColor(image_bgra_or_bgr, cv2.COLOR_BGR2RGBA)
        else:
            rgba = cv2.cvtColor(image_bgra_or_bgr, cv2.COLOR_BGRA2RGBA)
        rgba = rgba.astype(np.float32)
        cuda_img = self._to_cuda(rgba)
        raw_poses = self.net.Process(cuda_img, overlay="none")
        out: List[Pose] = []
        for rp in raw_poses:
            kps = [Keypoint(int(k.ID), float(k.x), float(k.y)) for k in rp.Keypoints]
            out.append(Pose(Keypoints=kps))
        return out


class _UltralyticsBackend:
    """YOLOv8-pose backend that runs on any CUDA-capable x86 GPU (or CPU).

    Ultralytics emits the standard 17 COCO keypoints; we synthesize a neck
    keypoint (ID 17) as the midpoint of the shoulders to match what the
    downstream code originally got from Pose-ResNet18-Body.
    """

    name = "ultralytics"

    # COCO keypoint indices already match what person_keypoints expects:
    # 3=left_ear, 4=right_ear, 5=left_shoulder, 6=right_shoulder,
    # 11=left_hip, 12=right_hip.  Neck (17) is synthesised below.
    NECK_ID = 17

    def __init__(self, model_name: str = "yolov8n-pose.pt", threshold: float = 0.3,
                 device: Optional[str] = None, imgsz: int = 640):
        from ultralytics import YOLO  # imported lazily so the module is optional
        import torch

        self.threshold = threshold
        self.imgsz = imgsz
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = YOLO(model_name)
        self.model.to(device)

    def process(self, image_bgra_or_bgr: np.ndarray) -> List[Pose]:
        import cv2
        if image_bgra_or_bgr.ndim == 3 and image_bgra_or_bgr.shape[2] == 4:
            bgr = cv2.cvtColor(image_bgra_or_bgr, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image_bgra_or_bgr
        if bgr.dtype != np.uint8:
            bgr = np.clip(bgr, 0, 255).astype(np.uint8)

        results = self.model.predict(
            bgr, imgsz=self.imgsz, conf=self.threshold,
            device=self.device, verbose=False)

        out: List[Pose] = []
        if not results:
            return out
        r = results[0]
        if r.keypoints is None or r.keypoints.xy is None:
            return out

        xy = r.keypoints.xy.cpu().numpy()           # (N, 17, 2)
        confs = (r.keypoints.conf.cpu().numpy()      # (N, 17)
                 if r.keypoints.conf is not None else np.ones(xy.shape[:2]))

        for person_xy, person_conf in zip(xy, confs):
            kps: List[Keypoint] = []
            for kp_id in range(person_xy.shape[0]):
                if person_conf[kp_id] < self.threshold:
                    continue
                x, y = float(person_xy[kp_id, 0]), float(person_xy[kp_id, 1])
                if x <= 0.0 and y <= 0.0:
                    continue
                kps.append(Keypoint(kp_id, x, y))

            ls = next((k for k in kps if k.ID == 5), None)
            rs = next((k for k in kps if k.ID == 6), None)
            if ls is not None and rs is not None:
                kps.append(Keypoint(self.NECK_ID,
                                    (ls.x + rs.x) / 2.0,
                                    (ls.y + rs.y) / 2.0))
            if kps:
                out.append(Pose(Keypoints=kps))
        return out


def build_pose_backend(prefer: Optional[str] = None,
                       threshold: float = 0.3) -> object:
    """Return a usable pose backend.

    ``prefer`` may be ``"jetson"`` or ``"ultralytics"``; if ``None`` the
    backend is chosen from the ``POSE_BACKEND`` environment variable, falling
    back to auto-detection (Jetson first, then Ultralytics).
    """
    prefer = prefer or os.environ.get("POSE_BACKEND")

    if prefer == "jetson":
        return _JetsonBackend(threshold=threshold)
    if prefer == "ultralytics":
        model = os.environ.get("POSE_MODEL", "yolov8n-pose.pt")
        return _UltralyticsBackend(model_name=model, threshold=threshold)

    ji, ju = _try_import_jetson()
    if ji is not None and ju is not None:
        return _JetsonBackend(threshold=threshold)

    model = os.environ.get("POSE_MODEL", "yolov8n-pose.pt")
    return _UltralyticsBackend(model_name=model, threshold=threshold)
