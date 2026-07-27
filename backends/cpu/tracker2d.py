"""Stage A, CPU backend: FP32 2D detection (fine-tuned YOLOv8n) via
ultralytics. Detection only -- tracking-by-detection (IoU/Hungarian
matching) is shared and backend-independent, see pipeline/track2d.py.
"""
import numpy as np
from ultralytics import YOLO

from backends.cpu._gpu_compat import patch_nms_for_unsupported_gpu

patch_nms_for_unsupported_gpu()


def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def detect(image: np.ndarray, model: YOLO, conf_thres: float = 0.3, iou_thres: float = 0.5,
           device: "int | str" = "cpu") -> np.ndarray:
    """Returns (N, 6): [x1, y1, x2, y2, conf, class_id] -- same layout
    backends/hardware/tracker2d.py's detect() returns, so pipeline/track2d.py's
    Tracker2D can consume either backend's output identically."""
    result = model.predict(image, conf=conf_thres, iou=iou_thres, device=device, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    if len(boxes) == 0:
        return np.zeros((0, 6))
    return np.concatenate([boxes, conf[:, None], cls[:, None].astype(np.float64)], axis=1)
