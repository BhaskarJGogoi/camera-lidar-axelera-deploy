"""2D IoU/Hungarian tracker-by-detection. Backend-independent: takes
already-detected, already-NMS'd boxes and maintains track identity across
frames, regardless of which detector (FP32 YOLOv8n on CPU, or the AIPU
tracker2d model) produced those boxes.

Extracted from the original models/tracker2d.py's Tracker2D.update() method,
minus the ultralytics/YOLO.predict() call itself -- that stays backend
-specific (see backends/cpu/tracker2d.py and backends/hardware/tracker2d.py).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between box arrays a (N,4) and b (M,4), xyxy format."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # xyxy
    class_id: int
    conf: float
    missed: int = 0


@dataclass
class Tracker2D:
    """Track ID continuity via greedy-optimal (Hungarian) IoU matching
    between this frame's detections and the previous frame's tracks -- no
    motion model or re-id embedding, matching the project's Stage A design
    (see models/tracker2d.py in the original research repo for the
    rationale: acceptable as long as IoU-only matching doesn't prove too
    brittle under occlusion)."""

    match_iou_thres: float = 0.3
    max_missed: int = 5
    _tracks: list = field(default_factory=list, init=False)
    _next_id: int = field(default=0, init=False)

    def update(self, det_boxes: np.ndarray) -> list:
        """det_boxes: (N, 6) [x1,y1,x2,y2,conf,class_id], already NMS'd.
        Returns the tracks matched this frame (i.e. currently visible)."""
        if self._tracks and len(det_boxes):
            track_boxes = np.stack([t.bbox for t in self._tracks])
            iou_matrix = iou_xyxy(track_boxes, det_boxes[:, :4])
            rows, cols = linear_sum_assignment(-iou_matrix)
        else:
            rows, cols = np.array([], dtype=int), np.array([], dtype=int)

        matched_tracks, matched_dets = set(), set()
        for r, c in zip(rows, cols):
            if iou_matrix[r, c] >= self.match_iou_thres:
                t = self._tracks[r]
                t.bbox, t.class_id, t.conf, t.missed = det_boxes[c, :4], int(det_boxes[c, 5]), det_boxes[c, 4], 0
                matched_tracks.add(r)
                matched_dets.add(c)

        for i, t in enumerate(self._tracks):
            if i not in matched_tracks:
                t.missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]

        for c in range(len(det_boxes)):
            if c not in matched_dets:
                self._tracks.append(Track(
                    track_id=self._next_id, bbox=det_boxes[c, :4],
                    class_id=int(det_boxes[c, 5]), conf=det_boxes[c, 4],
                ))
                self._next_id += 1

        return [t for t in self._tracks if t.missed == 0]
