"""Runs the full hardware pipeline over a few frames of the bundled KITTI
demo clip and checks the output is structurally and numerically sane --
same checks as test_cpu_pipeline.py, but against the Axelera-compiled
models on real Metis hardware.

This can only actually run where axelera.runtime is importable, i.e. on
the device itself (after `make deploy` / scripts/deploy_to_device.sh) --
everywhere else it self-skips rather than pretending to be a normal
always-runnable test. Copy this repo to the device and run:
    python -m pytest tests/test_hardware_pipeline.py -v
"""
import math

import numpy as np
import pytest

pytest.importorskip("axelera.runtime")

from backends.hardware.pipeline import iter_sequence


@pytest.fixture(scope="module")
def frames(data_dir, seq):
    return list(iter_sequence(data_dir, seq, num_frames=3))


def test_yields_expected_frame_count(frames):
    assert len(frames) == 3


def test_frame_dict_has_expected_keys(frames):
    expected_keys = {
        "frame_idx", "image", "uv_all", "depth_all", "tracks",
        "frustum_points", "boxes_3d", "track_ids_3d", "tracker3d", "calib",
    }
    for f in frames:
        assert expected_keys.issubset(f.keys())


def test_at_least_some_detections_across_frames(frames):
    total_tracks = sum(len(f["tracks"]) for f in frames)
    assert total_tracks > 0


def test_3d_boxes_are_physically_sane(frames):
    for f in frames:
        assert len(f["boxes_3d"]) == len(f["track_ids_3d"])
        for h, w, l, x, y, z, ry in f["boxes_3d"]:
            assert 0 < h < 5 and 0 < w < 5 and 0 < l < 10, "car-scale dims (meters)"
            assert -math.pi <= ry <= math.pi, "heading should be a normalized angle"
            assert z > 0, "center should be in front of the camera (positive depth)"


def test_frustum_points_are_xyzr(frames):
    for f in frames:
        for pts in f["frustum_points"]:
            assert pts.ndim == 2 and pts.shape[1] == 4
            assert pts.dtype == np.float32
