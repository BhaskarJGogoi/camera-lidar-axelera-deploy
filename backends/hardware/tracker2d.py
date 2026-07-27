"""Stage A, hardware backend: 2D detection via the Axelera-compiled
tracker2d model on real Metis hardware. Detection only -- tracking-by
-detection is shared and backend-independent, see pipeline/track2d.py.

preprocess() letterboxes (aspect-ratio-preserving resize + pad to 640x640,
matching Ultralytics' default) rather than naively stretching to square --
the latter was the actual root cause of this model looking inaccurate on
hardware early in development (KITTI's ~3.3:1 images squished into a
square distort every object's shape far more than int8 quantization does).
See docs/HARDWARE_NOTES.md for the full story.

decode_boxes() filters anchors by class confidence (cheap, 3-channel
sigmoid) *before* running the DFL softmax box-decode, instead of after:
typically <20 of 8400 anchors survive the threshold, so the softmax only
runs on those -- see docs/HARDWARE_NOTES.md for why this matters (the
naive order made postprocessing ~10x slower for no accuracy difference).
"""
from typing import List

import cv2
import numpy as np
from axelera.runtime import TensorInfo

NUM_BOX_OUTPUTS = 3
FEATURE_SHAPES = [(80, 80), (40, 40), (20, 20)]
STRIDES = [8, 16, 32]
DFL_BINS = 16


# ============================================================================
# Axelera hardware quantize/pad helpers
# ============================================================================

def axelera_quantize(normalized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    quantized = np.round((normalized / scale) + zero_point)
    return quantized.clip(-128, 127).astype(np.int8)


def axelera_pad(unpadded: np.ndarray, padding: list, zero_point: int) -> np.ndarray:
    return np.pad(unpadded, padding, mode="constant", constant_values=zero_point)


def axelera_depad(padded: np.ndarray, padding: list) -> np.ndarray:
    slices = tuple(slice(start, -end if end else None) for start, end in padding)
    return padded[slices]


def axelera_dequantize(quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (quantized.astype(np.float32) - zero_point) * scale


# ============================================================================
# Preprocessing
# ============================================================================

def letterbox(image: np.ndarray, new_h: int, new_w: int, pad_value: int = 114):
    h, w = image.shape[:2]
    scale = min(new_h / h, new_w / w)
    unpad_h, unpad_w = round(h * scale), round(w * scale)
    resized = cv2.resize(image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    dh, dw = new_h - unpad_h, new_w - unpad_w
    top, bottom = round(dh / 2 - 0.1), round(dh / 2 + 0.1)
    left, right = round(dw / 2 - 0.1), round(dw / 2 + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    return padded, scale, left, top


def preprocess_image(image: np.ndarray, input_info: TensorInfo):
    batch, height, width, channels = input_info.unpadded_shape
    letterboxed, scale, left, top = letterbox(image, height, width)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    quantized = axelera_quantize(normalized, input_info.scale, input_info.zero_point)
    padded = axelera_pad(quantized, input_info.padding[1:], input_info.zero_point)
    return padded[np.newaxis, ...], scale, left, top


# ============================================================================
# Box decode (numpy reimplementation of postprocess_graph.onnx's DFL decode
# -- see docs/HARDWARE_NOTES.md for why: running it through onnxruntime cost
# 15-50ms/frame, almost entirely inside the Softmax kernel)
# ============================================================================

def _make_anchors_and_strides(feature_shapes, strides):
    anchor_points, stride_tensor = [], []
    for (h, w), s in zip(feature_shapes, strides):
        sx = np.arange(w, dtype=np.float32) + 0.5
        sy = np.arange(h, dtype=np.float32) + 0.5
        grid_y, grid_x = np.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2))
        stride_tensor.append(np.full((h * w,), s, dtype=np.float32))
    return np.concatenate(anchor_points, axis=0), np.concatenate(stride_tensor, axis=0)


_DFL_WEIGHT = np.arange(DFL_BINS, dtype=np.float32)
_ANCHOR_POINTS, _STRIDE_TENSOR = _make_anchors_and_strides(FEATURE_SHAPES, STRIDES)


def decode_boxes(box_maps: List[np.ndarray], cls_maps: List[np.ndarray], conf_threshold: float):
    num_box_ch, num_cls_ch = box_maps[0].shape[1], cls_maps[0].shape[1]
    box_flat = np.concatenate([b.reshape(1, num_box_ch, -1) for b in box_maps], axis=-1)
    cls_flat = np.concatenate([c.reshape(1, num_cls_ch, -1) for c in cls_maps], axis=-1)

    scores = 1.0 / (1.0 + np.exp(-cls_flat[0]))
    class_id = scores.argmax(axis=0)
    conf = scores.max(axis=0)
    idx = np.nonzero(conf > conf_threshold)[0]
    if idx.size == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

    box_re = box_flat.reshape(1, 4, DFL_BINS, -1)[..., idx].transpose(0, 2, 1, 3)
    box_re = box_re - box_re.max(axis=1, keepdims=True)
    exp = np.exp(box_re)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    dist = np.einsum("nchw,c->nhw", softmax, _DFL_WEIGHT)[0]

    lt, rb = dist[0:2, :], dist[2:4, :]
    anchor = _ANCHOR_POINTS[idx].T
    x1y1, x2y2 = anchor - lt, anchor + rb
    cxcy, wh = (x1y1 + x2y2) / 2.0, x2y2 - x1y1
    cx, cy = cxcy
    w, h = wh
    stride = _STRIDE_TENSOR[idx]
    x1, y1, x2, y2 = (cx - w / 2) * stride, (cy - h / 2) * stride, (cx + w / 2) * stride, (cy + h / 2) * stride
    return np.stack([x1, y1, x2, y2], axis=1), conf[idx], class_id[idx]


def apply_nms(boxes: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 6))
    final_boxes = []
    for cls_id in np.unique(boxes[:, 5].astype(int)):
        cls_boxes = boxes[boxes[:, 5] == cls_id]
        indices = np.argsort(cls_boxes[:, 4])[::-1]
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            if len(indices) == 1:
                break
            current_box = cls_boxes[current]
            remaining = cls_boxes[indices[1:]]
            x1 = np.maximum(current_box[0], remaining[:, 0])
            y1 = np.maximum(current_box[1], remaining[:, 1])
            x2 = np.minimum(current_box[2], remaining[:, 2])
            y2 = np.minimum(current_box[3], remaining[:, 3])
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_a = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            area_b = (remaining[:, 2] - remaining[:, 0]) * (remaining[:, 3] - remaining[:, 1])
            iou = np.where((area_a + area_b - inter) > 0, inter / (area_a + area_b - inter), 0)
            indices = indices[1:][iou < iou_threshold]
        final_boxes.append(cls_boxes[keep])
    return np.concatenate(final_boxes, axis=0) if final_boxes else np.zeros((0, 6))


def detect(image: np.ndarray, model_2d: dict, conf_thres: float = 0.3, iou_thres: float = 0.45) -> np.ndarray:
    """Returns (N, 6): [x1, y1, x2, y2, conf, class_id], in original image
    pixels -- same layout backends/cpu/tracker2d.py's detect() returns."""
    tensor, scale, left, top = preprocess_image(image, model_2d["input_infos"][0])
    model_2d["inputs"][0][:] = tensor
    model_2d["instance"].run(model_2d["inputs"], model_2d["outputs"])

    maps = []
    for raw, info in zip(model_2d["outputs"], model_2d["output_infos"]):
        depadded = axelera_depad(raw, info.padding)
        fp32 = axelera_dequantize(depadded, info.scale, info.zero_point)
        maps.append(fp32.transpose(0, 3, 1, 2))
    box_maps, cls_maps = maps[:NUM_BOX_OUTPUTS], maps[NUM_BOX_OUTPUTS:]

    xyxy, conf, class_id = decode_boxes(box_maps, cls_maps, conf_thres)
    if xyxy.shape[0] == 0:
        return np.zeros((0, 6))
    boxes = np.concatenate([xyxy, conf[:, None], class_id[:, None]], axis=1)
    boxes = apply_nms(boxes, iou_thres)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    return boxes
