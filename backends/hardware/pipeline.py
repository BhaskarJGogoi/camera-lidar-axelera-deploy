"""Hardware backend: wires Stage A->B->C->D together over a full sequence,
using the Axelera-compiled models on real Metis hardware.

Three passes instead of one interleaved loop: switching which compiled
model is active on the AIPU has a real cost (~35-45ms), apparently
asymmetric -- tracker2d's L2-resident weights blob is ~2x pointnet_lite's,
so reloading it back after a run of pointnet_lite calls costs more than the
reverse direction. Since a frame's 2D detection would otherwise always
follow the previous frame's last 3D regression, nearly every frame would
pay that tax in a naive per-frame interleaved loop. Running each model to
completion across the whole clip (only ONE switch total, not ~2/frame)
avoids that -- see docs/HARDWARE_NOTES.md for the measurements behind this.

Despite the 3-pass internal structure, iter_sequence() yields one frame
dict at a time with the *same shape* backends.cpu.pipeline.iter_sequence
yields, so callers (scripts/demo.py) can treat both backends identically.
"""
from pathlib import Path

import cv2
import numpy as np
from axelera.runtime import Context

from pipeline.calibration import Calibration
from pipeline.frustum import extract_frustum_from_projection
from pipeline.track2d import Tracker2D
from pipeline.track3d import Tracker3D
from backends.hardware._runtime import load_model
from backends.hardware import tracker2d as tracker2d_backend
from backends.hardware import pointnet_lite as pointnet_backend

MIN_FRUSTUM_POINTS = 5
COMPILED_TRACKER2D = Path(__file__).resolve().parent / "compiled_tracker2d" / "model.json"
COMPILED_POINTNET_LITE = Path(__file__).resolve().parent / "compiled_pointnet_lite" / "model.json"


def iter_sequence(seq_dir, seq: str, conf_thres: float = 0.3, iou_thres: float = 0.45, num_frames: int = None):
    """Generator yielding one dict per frame, same shape as
    backends.cpu.pipeline.iter_sequence's (frame_idx, image, uv_all,
    depth_all, tracks, frustum_points, boxes_3d, track_ids_3d, tracker3d,
    calib) -- see that module's docstring for field meanings."""
    seq_dir = Path(seq_dir)
    calib = Calibration.from_file(str(seq_dir / "calib" / f"{seq}.txt"))

    image_dir = seq_dir / "image_2" / seq
    velo_dir = seq_dir / "velodyne" / seq
    frame_ids = sorted(int(p.stem) for p in image_dir.glob("*.png"))
    if num_frames:
        frame_ids = frame_ids[:num_frames]

    # Pass 0: load frames + project LiDAR (host-only, no AIPU)
    frames = []
    for frame_idx in frame_ids:
        image = cv2.imread(str(image_dir / f"{frame_idx:06d}.png"))
        points_velo = np.fromfile(str(velo_dir / f"{frame_idx:06d}.bin"), dtype=np.float32).reshape(-1, 4)
        uv_all, depth_all, mask_front = calib.project_velo_to_image(points_velo[:, :3])
        frames.append({
            "frame_idx": frame_idx, "image": image, "uv_all": uv_all, "depth_all": depth_all,
            "pts_front": points_velo[mask_front],
        })

    ctx = Context()
    try:
        # Pass 1: all 2D detection + tracking (tracker2d stays active throughout)
        model_2d = load_model(ctx, COMPILED_TRACKER2D)
        tracker2d = Tracker2D()
        for f in frames:
            det_boxes = tracker2d_backend.detect(f["image"], model_2d, conf_thres, iou_thres)
            f["tracks"] = tracker2d.update(det_boxes)

        # Pass 2: all 3D regression (pointnet_lite stays active throughout)
        model_3d = load_model(ctx, COMPILED_POINTNET_LITE)
        for f in frames:
            valid_tracks, frustum_points, detections_3d = [], [], []
            for track in f["tracks"]:
                frustum = extract_frustum_from_projection(f["pts_front"], f["uv_all"], f["depth_all"], track.bbox)
                if len(frustum) < MIN_FRUSTUM_POINTS:
                    continue
                xyz_cam = calib.velo_to_rect_cam(frustum[:, :3])
                points_cam4 = np.concatenate([xyz_cam, frustum[:, 3:4]], axis=1).astype(np.float32)
                result = pointnet_backend.regress(points_cam4, model_3d)
                valid_tracks.append(track)
                frustum_points.append(points_cam4)
                detections_3d.append((track.track_id, result["center"], result["dims"], result["heading"]))
            f["valid_tracks"] = valid_tracks
            f["frustum_points"] = frustum_points
            f["detections_3d"] = detections_3d
    finally:
        ctx.release()

    # Pass 3: 3D track relinking (host-only, no AIPU) -- yielded lazily so
    # callers can render/write frames as they come, same as the CPU backend
    tracker3d = Tracker3D()
    for f in frames:
        tracker3d.update(f["frame_idx"], f["detections_3d"])
        track_ids_3d = [tracker3d._id_map[t.track_id] for t in f["valid_tracks"]]
        boxes_3d = [
            (dims[0], dims[1], dims[2], center[0], center[1], center[2], heading)
            for (_, center, dims, heading) in f["detections_3d"]
        ]
        yield {
            "frame_idx": f["frame_idx"],
            "image": f["image"],
            "uv_all": f["uv_all"],
            "depth_all": f["depth_all"],
            "tracks": f["valid_tracks"],
            "frustum_points": f["frustum_points"],
            "boxes_3d": boxes_3d,
            "track_ids_3d": track_ids_3d,
            "tracker3d": tracker3d,
            "calib": calib,
        }
