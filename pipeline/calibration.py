"""Parses KITTI Tracking calibration files and projects Velodyne points into
the left color camera (image_2) image plane.

KITTI's projection chain for a point in the Velodyne frame is:
    y = P2 @ R_rect @ Tr_velo_to_cam @ x_velo_homogeneous
where P2 is the 3x4 projection matrix for camera 2 (already includes that
camera's rectifying rotation baked in), R_rect is the 3x3 reference-camera
rectification (padded to 4x4), and Tr_velo_to_cam is the 3x4 Velodyne->camera
rigid transform (padded to 4x4).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Calibration:
    P2: np.ndarray  # (3, 4) projection matrix, left color camera
    R_rect: np.ndarray  # (4, 4) rectifying rotation, padded to homogeneous
    Tr_velo_to_cam: np.ndarray  # (4, 4) Velodyne -> camera 0, padded to homogeneous

    @classmethod
    def from_file(cls, calib_path: str) -> "Calibration":
        rows = {}
        with open(calib_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, *values = line.split()
                rows[key.rstrip(":")] = np.array(values, dtype=np.float64)

        P2 = rows["P2"].reshape(3, 4)

        R_rect = np.eye(4)
        R_rect[:3, :3] = rows["R_rect"].reshape(3, 3)

        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[:3, :4] = rows["Tr_velo_cam"].reshape(3, 4)

        return cls(P2=P2, R_rect=R_rect, Tr_velo_to_cam=Tr_velo_to_cam)

    def velo_to_rect_cam(self, points_velo: np.ndarray) -> np.ndarray:
        """points_velo: (N, 3) xyz in the Velodyne frame.

        Returns (N, 3) xyz in the rectified camera-0 frame -- the frame
        KITTI's 3D box labels (center, dims, rotation_y) are defined in.
        Pure rigid transform, no projection.
        """
        n = points_velo.shape[0]
        points_h = np.hstack([points_velo, np.ones((n, 1))])  # (N, 4)
        points_cam = self.R_rect @ (self.Tr_velo_to_cam @ points_h.T)  # (4, N)
        return points_cam[:3, :].T

    def project_rect_cam_to_image(self, points_cam: np.ndarray) -> np.ndarray:
        """points_cam: (N, 3) xyz already in the rectified camera-0 frame
        (e.g. PointNet-lite's predicted box center/corners). Returns (N, 2)
        pixel coords -- just P2, since R_rect/Tr_velo_to_cam don't apply to
        points already in that frame.
        """
        n = points_cam.shape[0]
        points_h = np.hstack([points_cam, np.ones((n, 1))])  # (N, 4)
        points_img_h = self.P2 @ points_h.T  # (3, N)
        return (points_img_h[:2, :] / points_img_h[2, :]).T

    def project_velo_to_image(self, points_velo: np.ndarray):
        """points_velo: (N, 3) xyz in the Velodyne frame.

        Returns (uv, depth, mask) where uv is (M, 2) pixel coords, depth is
        (M,) camera-frame depth, and mask is the (N,) boolean selecting which
        input points are in front of the camera (depth > 0) and therefore
        valid for uv/depth.
        """
        n = points_velo.shape[0]
        points_h = np.hstack([points_velo, np.ones((n, 1))])  # (N, 4)

        points_cam = (self.Tr_velo_to_cam @ points_h.T)  # (4, N)
        points_cam = (self.R_rect @ points_cam)  # (4, N)

        depth = points_cam[2, :]
        mask = depth > 0

        points_img_h = self.P2 @ points_cam[:, mask]  # (3, M)
        uv = (points_img_h[:2, :] / points_img_h[2, :]).T  # (M, 2)

        return uv, depth[mask], mask
