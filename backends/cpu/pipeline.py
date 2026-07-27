"""CPU/FP32 backend: wires Stage A->B->C->D together over a full sequence,
single-pass/streaming (unlike the hardware backend's 3-pass design -- there's
no AIPU model-switching cost on CPU to batch around, see
backends/hardware/pipeline.py and docs/HARDWARE_NOTES.md).
"""
from pathlib import Path

import torch
import cv2
import numpy as np

from pipeline.calibration import Calibration
from pipeline.frustum import extract_frustum_from_projection
from pipeline.track2d import Tracker2D
from pipeline.track3d import Tracker3D
from backends.cpu.pointnet_lite import PointNetLite
from backends.cpu import tracker2d as tracker2d_backend

NUM_POINTS = 256
MIN_FRUSTUM_POINTS = 5


def iter_sequence(seq_dir, seq: str, tracker2d_weights: str, pointnet_weights: str,
                   conf_thres: float = 0.4, iou_thres: float = 0.5, num_frames: int = None):
    """Generator yielding one dict per frame: frame_idx, image, uv_all,
    depth_all (full-cloud projection, for background scatter), and
    per-valid-track parallel lists tracks (pipeline.track2d.Track),
    frustum_points (Nx4 xyzr, camera-rect frame), boxes_3d
    (h,w,l,x,y,z,ry tuples), track_ids_3d (post-relink id for stable
    coloring). `tracker3d` is the same mutating instance every iteration, so
    tracker3d.trajectories() at any point reflects frames seen so far.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_dir = Path(seq_dir)
    calib = Calibration.from_file(str(seq_dir / "calib" / f"{seq}.txt"))

    pointnet = PointNetLite().to(device)
    pointnet.load_state_dict(torch.load(pointnet_weights, map_location=device))
    pointnet.eval()

    detector = tracker2d_backend.load_model(tracker2d_weights)
    tracker2d = Tracker2D()
    tracker3d = Tracker3D()

    img_dir = seq_dir / "image_2" / seq
    velo_dir = seq_dir / "velodyne" / seq
    frame_ids = sorted(int(p.stem) for p in img_dir.glob("*.png"))
    if num_frames:
        frame_ids = frame_ids[:num_frames]

    for frame_idx in frame_ids:
        frame_str = f"{frame_idx:06d}"
        image = cv2.imread(str(img_dir / f"{frame_str}.png"))
        points_velo = np.fromfile(str(velo_dir / f"{frame_str}.bin"), dtype=np.float32).reshape(-1, 4)

        uv_all, depth_all, mask_front = calib.project_velo_to_image(points_velo[:, :3])
        pts_front = points_velo[mask_front]

        det_boxes = tracker2d_backend.detect(image, detector, conf_thres, iou_thres, device=device)
        tracks = tracker2d.update(det_boxes)

        frustum_points, valid_tracks = [], []
        for track in tracks:
            frustum = extract_frustum_from_projection(pts_front, uv_all, depth_all, track.bbox)
            if len(frustum) < MIN_FRUSTUM_POINTS:
                continue
            xyz_cam = calib.velo_to_rect_cam(frustum[:, :3])
            points = np.concatenate([xyz_cam, frustum[:, 3:4]], axis=1).astype(np.float32)
            frustum_points.append(points)
            valid_tracks.append(track)

        boxes_3d, track_ids_3d = [], []
        if valid_tracks:
            point_batches = []
            for points in frustum_points:
                replace = len(points) < NUM_POINTS
                sample_idx = np.random.choice(len(points), NUM_POINTS, replace=replace)
                point_batches.append(points[sample_idx])

            batch = torch.from_numpy(np.stack(point_batches)).to(device)
            with torch.no_grad():
                pred = pointnet(batch)

            centers = pred["center"].cpu().numpy()
            dims = pred["dims"].cpu().numpy()
            headings = torch.atan2(pred["heading_sin_cos"][:, 0], pred["heading_sin_cos"][:, 1]).cpu().numpy()

            detections_3d = [
                (track.track_id, centers[i], dims[i], float(headings[i]))
                for i, track in enumerate(valid_tracks)
            ]
            tracker3d.update(frame_idx, detections_3d)

            for i, track in enumerate(valid_tracks):
                h, w, l = dims[i]
                x, y, z = centers[i]
                boxes_3d.append((h, w, l, x, y, z, float(headings[i])))
                track_ids_3d.append(tracker3d._id_map[track.track_id])

        yield {
            "frame_idx": frame_idx,
            "image": image,
            "uv_all": uv_all,
            "depth_all": depth_all,
            "tracks": valid_tracks,
            "frustum_points": frustum_points,
            "boxes_3d": boxes_3d,
            "track_ids_3d": track_ids_3d,
            "tracker3d": tracker3d,
            "calib": calib,
        }
