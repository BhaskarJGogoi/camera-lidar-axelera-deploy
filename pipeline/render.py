"""Visualization shared by both backends: camera panel (2D boxes + LiDAR
points colored by frustum membership + projected 3D wireframes) and a
bird's-eye-view trajectory panel, matching demo.gif's style from the
original research repo.

Uses vectorized numpy pixel-writes (_draw_points_fast) instead of looping
cv2.circle() per point -- a KITTI frame has 15-25k visible LiDAR points,
and a Python-level cv2.circle() call per point (even ~15us of overhead
each) adds hundreds of ms/frame, dwarfing actual model inference time. See
docs/HARDWARE_NOTES.md for the measurement that found this.
"""
import cv2
import numpy as np

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]

PALETTE = [
    (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200),
]

BEV_WIDTH = 400
BEV_SCALE = 6  # pixels per meter
BEV_MAX_TRAIL = 60  # frames of trajectory history to draw (older points fade out)

BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def color_for(track_id: int):
    return PALETTE[track_id % len(PALETTE)]


def box_3d_corners(h: float, w: float, l: float, x: float, y: float, z: float, ry: float) -> np.ndarray:
    """8 corners of a KITTI-convention 3D box in the camera-rect frame.
    (x, y, z) is the bottom-center; y points down, so the box spans [y-h, y]."""
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    corners = R @ np.stack([x_corners, y_corners, z_corners])
    corners += np.array([[x], [y], [z]])
    return corners.T


def _draw_points_fast(image: np.ndarray, uv: np.ndarray, color, radius: int = 0):
    """Draws a (2*radius+1)^2 block per point, with one vectorized numpy
    write per pixel-offset (9-25 total) instead of one Python+OpenCV call
    per point (tens of thousands)."""
    h, w = image.shape[:2]
    u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu, vv = u + dx, v + dy
            valid = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            image[vv[valid], uu[valid]] = color


def draw_camera_panel(image, calib, uv_all, depth_all, tracks, frustum_points, boxes_3d, track_ids_3d) -> np.ndarray:
    """tracks: list of pipeline.track2d.Track. boxes_3d: list of
    (h, w, l, x, y, z, ry) tuples, one per track, same order as track_ids_3d."""
    image = image.copy()
    h, w = image.shape[:2]
    in_bounds = (uv_all[:, 0] >= 0) & (uv_all[:, 0] < w) & (uv_all[:, 1] >= 0) & (uv_all[:, 1] < h)
    _draw_points_fast(image, uv_all[in_bounds], (50, 50, 50), radius=0)

    for i, track in enumerate(tracks):
        color = color_for(track_ids_3d[i])
        points_cam = frustum_points[i][:, :3]
        uv_pts = calib.project_rect_cam_to_image(points_cam)
        _draw_points_fast(image, uv_pts, color, radius=2)

        x1, y1, x2, y2 = track.bbox.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

        box_h, box_w, box_l, bx, by, bz, ry = boxes_3d[i]
        corners_cam = box_3d_corners(box_h, box_w, box_l, bx, by, bz, ry)
        corners_uv = calib.project_rect_cam_to_image(corners_cam).astype(int)
        for a, b in BOX_EDGES:
            cv2.line(image, tuple(corners_uv[a]), tuple(corners_uv[b]), color, 1, cv2.LINE_AA)

        label = f"ID{track_ids_3d[i]} {CLASS_NAMES[track.class_id]}"
        cv2.putText(image, label, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return image


def draw_bev_panel(trajectories: dict, current_frame: int, height: int) -> np.ndarray:
    """trajectories: pipeline.track3d.Tracker3D.trajectories()'s return value."""
    canvas = np.full((height, BEV_WIDTH, 3), 25, dtype=np.uint8)
    origin_x, origin_y = BEV_WIDTH // 2, height - 20

    for meters in range(0, 80, 10):
        y = origin_y - meters * BEV_SCALE
        if y < 0:
            break
        cv2.line(canvas, (0, y), (BEV_WIDTH, y), (55, 55, 55), 1)
        cv2.putText(canvas, f"{meters}m", (4, max(y - 2, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    cv2.line(canvas, (origin_x, 0), (origin_x, height), (55, 55, 55), 1)
    cv2.drawMarker(canvas, (origin_x, origin_y), (200, 200, 200), cv2.MARKER_TRIANGLE_UP, 10, 2)

    for track_id, t in trajectories.items():
        color = color_for(track_id)
        pts = []
        for frame, (x, _, z) in zip(t["frames"], t["centers"]):
            if frame > current_frame or current_frame - frame > BEV_MAX_TRAIL:
                continue
            pts.append((origin_x + int(x * BEV_SCALE), origin_y - int(z * BEV_SCALE)))

        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts)], False, color, 2, cv2.LINE_AA)
        if pts:
            cv2.circle(canvas, pts[-1], 4, color, -1)
            cv2.putText(canvas, str(track_id), (pts[-1][0] + 6, pts[-1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return canvas
