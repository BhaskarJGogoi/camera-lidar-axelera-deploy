"""Sanity checks for pipeline/calibration.py + pipeline/frustum.py -- pure
numpy, no backend/model dependency, always runnable."""
import numpy as np

from pipeline.calibration import Calibration
from pipeline.frustum import extract_frustum_from_projection, extract_frustum_points


def test_calibration_loads(data_dir, seq):
    calib = Calibration.from_file(str(data_dir / "calib" / f"{seq}.txt"))
    assert calib.P2.shape == (3, 4)
    assert calib.R_rect.shape == (4, 4)
    assert calib.Tr_velo_to_cam.shape == (4, 4)


def _load_frame(data_dir, seq, frame_idx=0):
    velo_path = data_dir / "velodyne" / seq / f"{frame_idx:06d}.bin"
    return np.fromfile(str(velo_path), dtype=np.float32).reshape(-1, 4)


def test_project_velo_to_image_shapes(data_dir, seq):
    calib = Calibration.from_file(str(data_dir / "calib" / f"{seq}.txt"))
    points_velo = _load_frame(data_dir, seq)

    uv, depth, mask = calib.project_velo_to_image(points_velo[:, :3])

    assert mask.shape == (points_velo.shape[0],)
    assert uv.shape == (mask.sum(), 2)
    assert depth.shape == (mask.sum(),)
    assert np.all(depth > 0), "everything project_velo_to_image returns should be in front of the camera"


def test_velo_to_rect_cam_shape(data_dir, seq):
    calib = Calibration.from_file(str(data_dir / "calib" / f"{seq}.txt"))
    points_velo = _load_frame(data_dir, seq)

    points_cam = calib.velo_to_rect_cam(points_velo[:100, :3])
    assert points_cam.shape == (100, 3)


def test_extract_frustum_from_projection_filters_points(data_dir, seq):
    calib = Calibration.from_file(str(data_dir / "calib" / f"{seq}.txt"))
    points_velo = _load_frame(data_dir, seq)
    uv, depth, mask_front = calib.project_velo_to_image(points_velo[:, :3])
    pts_front = points_velo[mask_front]

    # a small centered box should select a strict subset of the front-facing cloud
    h, w = 375, 1242  # KITTI image_2 resolution
    box = (w * 0.4, h * 0.4, w * 0.6, h * 0.6)
    frustum = extract_frustum_from_projection(pts_front, uv, depth, box)

    assert frustum.shape[1] == 4  # xyzr
    assert 0 < len(frustum) < len(pts_front)


def test_extract_frustum_points_matches_from_projection(data_dir, seq):
    """extract_frustum_points (single-box convenience wrapper) should agree
    with extract_frustum_from_projection (the batched/precomputed-projection
    form) on the same input -- they're supposed to do the same filtering."""
    calib = Calibration.from_file(str(data_dir / "calib" / f"{seq}.txt"))
    points_velo = _load_frame(data_dir, seq)
    h, w = 375, 1242
    box = (w * 0.4, h * 0.4, w * 0.6, h * 0.6)

    direct = extract_frustum_points(points_velo, calib, box)

    uv, depth, mask_front = calib.project_velo_to_image(points_velo[:, :3])
    pts_front = points_velo[mask_front]
    via_projection = extract_frustum_from_projection(pts_front, uv, depth, box)

    assert direct.shape == via_projection.shape
    assert np.allclose(direct, via_projection)
