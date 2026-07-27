"""Stage B: given a 2D box (from Stage A) and a raw LiDAR sweep, extracts the
"frustum" of points whose image projection falls inside the box, then drops
obvious non-object points with two cheap heuristics (no learning):

- ground plane: KITTI's Velodyne frame has z pointing up, mounted ~1.73m
  above the road, so ground points cluster tightly around a known height.
- depth outliers: points near a box's edges can land inside it while
  actually belonging to whatever is behind/around the object (e.g. a
  building sliver at a car's corner). Their depth stands far from the
  object's own depth, so a robust (median/MAD) depth filter removes them.
"""
import numpy as np

from pipeline.calibration import Calibration


def _filter_ground_and_depth_outliers(
    points: np.ndarray, depth: np.ndarray, ground_z_thresh: float, depth_outlier_mads: float
) -> np.ndarray:
    above_ground = points[:, 2] > ground_z_thresh
    points, depth = points[above_ground], depth[above_ground]

    if len(depth) > 0:
        median_depth = np.median(depth)
        mad = np.median(np.abs(depth - median_depth)) + 1e-6
        keep = np.abs(depth - median_depth) < depth_outlier_mads * mad * 1.4826  # MAD -> std-equivalent
        points = points[keep]

    return points


def extract_frustum_from_projection(
    points_front: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    box_xyxy,
    ground_z_thresh: float = -1.5,
    depth_outlier_mads: float = 3.0,
) -> np.ndarray:
    """Same filtering as extract_frustum_points, but takes an already-computed
    per-frame projection (points_front/uv/depth, as returned by
    Calibration.project_velo_to_image applied once for the whole cloud) so
    callers extracting frustums for many boxes in the same frame don't
    reproject the whole ~100k-point cloud once per box."""
    left, top, right, bottom = box_xyxy
    in_box = (uv[:, 0] >= left) & (uv[:, 0] < right) & (uv[:, 1] >= top) & (uv[:, 1] < bottom)
    pts_box, depth_box = points_front[in_box], depth[in_box]
    return _filter_ground_and_depth_outliers(pts_box, depth_box, ground_z_thresh, depth_outlier_mads)


def extract_frustum_points(
    points_velo: np.ndarray,
    calib: Calibration,
    box_xyxy,
    ground_z_thresh: float = -1.5,
    depth_outlier_mads: float = 3.0,
) -> np.ndarray:
    """points_velo: (N, 4) xyzr rows. box_xyxy: (left, top, right, bottom) in
    pixels. Returns the filtered (M, 4) xyzr subset.

    Convenience wrapper for the single-box case (demos/visualization). When
    extracting frustums for multiple boxes in the same frame, project once
    with calib.project_velo_to_image and call extract_frustum_from_projection
    per box instead.
    """
    uv, depth, mask_front = calib.project_velo_to_image(points_velo[:, :3])
    pts_front = points_velo[mask_front]
    return extract_frustum_from_projection(pts_front, uv, depth, box_xyxy, ground_z_thresh, depth_outlier_mads)
