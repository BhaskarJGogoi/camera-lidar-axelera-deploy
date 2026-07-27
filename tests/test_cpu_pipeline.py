"""Runs the full CPU/FP32 pipeline over a few frames of the bundled KITTI
demo clip and checks the output is structurally and numerically sane.
Exercises Stage A (2D detect+track) -> B (frustum extract) -> C (3D
regress) -> D (3D relink) end to end, using the bundled weights/data --
runs anywhere with requirements-cpu.txt installed, no hardware needed.
"""
import math

import numpy as np
import pytest

from backends.cpu.pipeline import iter_sequence


@pytest.fixture(scope="module")
def frames(data_dir, seq, weights_dir):
    tracker2d_weights = str(weights_dir / "tracker2d_best.pt")
    pointnet_weights = str(weights_dir / "pointnet_lite_best.pt")
    return list(iter_sequence(data_dir, seq, tracker2d_weights, pointnet_weights, num_frames=3))


def test_yields_expected_frame_count(frames):
    assert len(frames) == 3


def test_frame_dict_has_expected_keys(frames):
    expected_keys = {
        "frame_idx", "image", "uv_all", "depth_all", "tracks",
        "frustum_points", "boxes_3d", "track_ids_3d", "tracker3d", "calib",
    }
    for f in frames:
        assert expected_keys.issubset(f.keys())


def test_image_and_projection_shapes(frames):
    for f in frames:
        assert f["image"].ndim == 3 and f["image"].shape[2] == 3
        assert f["uv_all"].shape[1] == 2
        assert f["depth_all"].shape[0] == f["uv_all"].shape[0]


def test_at_least_some_detections_across_frames(frames):
    """Not every frame needs a detection, but a 3-frame clip of a KITTI
    street scene with zero detections at all would indicate something is
    structurally broken (wrong weights, wrong preprocessing, ...)."""
    total_tracks = sum(len(f["tracks"]) for f in frames)
    assert total_tracks > 0


def test_3d_boxes_are_physically_sane(frames):
    for f in frames:
        assert len(f["boxes_3d"]) == len(f["track_ids_3d"])
        for h, w, l, x, y, z, ry in f["boxes_3d"]:
            assert 0 < h < 5 and 0 < w < 5 and 0 < l < 10, "car-scale dims (meters)"
            assert -math.pi <= ry <= math.pi, "heading should be a normalized angle"
            assert z > 0, "center should be in front of the camera (positive depth)"


def test_tracker3d_trajectories_reflect_seen_frames(frames):
    last = frames[-1]
    trajectories = last["tracker3d"].trajectories()
    for tid, t in trajectories.items():
        assert all(fr <= last["frame_idx"] for fr in t["frames"])


def test_frustum_points_are_xyzr(frames):
    for f in frames:
        for pts in f["frustum_points"]:
            assert pts.ndim == 2 and pts.shape[1] == 4
            assert pts.dtype == np.float32
