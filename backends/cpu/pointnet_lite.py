"""Stage C: PointNet-lite regression head.

Takes a variable-size set of frustum points (x, y, z, reflectance) in the
rectified camera frame and regresses a 3D box: center, dimensions, heading.
A few shared per-point MLP layers (implemented as 1x1 Conv1d, i.e. the same
weights applied independently at every point) followed by a symmetric
max-pool make this permutation-invariant to point order -- the same trick
full PointNet uses, just with far fewer channels (this is a few tens of
thousands of parameters, not millions).

Heading is predicted as (sin, cos) rather than a raw angle so the loss
doesn't have to deal with the 0/2pi wraparound discontinuity; recover the
angle at inference time with atan2(sin, cos).

Center is predicted as an offset from the input points' centroid rather
than an absolute coordinate -- centroid-relative offsets are a much easier
regression target than raw scene coordinates.
"""
import torch
import torch.nn as nn


class PointNetLite(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(in_channels, 32, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8),  # center_offset(3) + dims(3) + heading_sin_cos(2)
        )

    def forward(self, points: torch.Tensor) -> dict:
        """points: (B, N, 4) xyz + reflectance, xyz already in camera frame."""
        centroid = points[:, :, :3].mean(dim=1)  # (B, 3)

        # functional (no in-place slice assignment) -- that traced as a
        # ScatterND/Transpose/Shape/Gather subgraph under torch's ONNX
        # exporter, which onnxruntime's quantizer's shape inference choked on
        centered_xyz = points[:, :, :3] - centroid.unsqueeze(1)
        centered = torch.cat([centered_xyz, points[:, :, 3:4]], dim=2)

        feat = self.point_mlp(centered.transpose(1, 2))  # (B, 128, N)
        global_feat = feat.max(dim=2).values  # (B, 128), symmetric max-pool

        out = self.head(global_feat)  # (B, 8)
        center_offset, dims, heading_sin_cos = out[:, :3], out[:, 3:6], out[:, 6:8]

        return {
            "center": centroid + center_offset,  # (B, 3) absolute center, camera frame
            "dims": dims,  # (B, 3) h, w, l (meters)
            "heading_sin_cos": heading_sin_cos,  # (B, 2), normalize before use if needed
        }
