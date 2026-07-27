"""v14: v13 compiled and ran correctly on hardware for center/dims, but
heading_sin_cos came back as the exact same value for every input sample.
Cause: the single combined (B,8,1,1) output shares ONE int8 quantization
scale across all 8 channels. Calibrated against center/dims' range
(roughly -21 to +80 m), that scale is ~0.396/step -- meaning heading_sin_cos's
true [-1,1] range spans only ~5 distinct representable values total, so it
collapses to a fixed quantization bin regardless of input. v14 scales
heading_sin_cos by HEADING_SCALE=15 before combining into the shared
output (safely within the existing representable range, so center/dims'
resolution is unaffected), and divides back by the same factor on the host
side after dequantizing.
"""
"""Recompiles pointnet_lite for Axelera Metis. Reproducibility reference,
not part of the normal test/demo flow -- the compiled artifact this
produces is already bundled at backends/hardware/compiled_pointnet_lite/.
Must run inside the Axelera Voyager SDK's own venv (not this project's).
See docs/HARDWARE_NOTES.md for the full six-failed-attempts-then-fix story
behind this file's PointNetLite4D reformulation.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import onnx
import torch
import torch.nn as nn
from onnx import helper

from backends.cpu.pointnet_lite import PointNetLite

NUM_POINTS = 255  # AIPU's iau_max_reduce hw instruction requires loop_len < 256; 256 hit that exactly.
                  # PointNet-style architectures (shared per-point MLP + symmetric pooling) are
                  # point-count invariant, so the same trained weights work unchanged at 255 points --
                  # no retraining needed, just a smaller pooling kernel and one fewer calibration point.
HEADING_SCALE = 15.0  # widens heading_sin_cos from [-1,1] to [-15,15] -- safely inside the existing
                      # representable output range (~[-21,80]) so center/dims resolution is unaffected,
                      # but gives heading ~76 quantization levels instead of ~5. Divide back out on the host.


class PointNetLite4D(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()

        self.select_xyz_for_sub = nn.Conv2d(in_channels, 3, (1, 1), bias=False)
        self.select_xyz_for_centroid1 = nn.Conv2d(in_channels, 3, (1, 1), bias=False)
        self.select_xyz_for_centroid2 = nn.Conv2d(in_channels, 3, (1, 1), bias=False)
        self.select_reflectance = nn.Conv2d(in_channels, 1, (1, 1), bias=False)
        with torch.no_grad():
            for conv in (self.select_xyz_for_sub, self.select_xyz_for_centroid1, self.select_xyz_for_centroid2):
                conv.weight.zero_()
                for c in range(3):
                    conv.weight[c, c, 0, 0] = 1.0
                conv.weight.requires_grad = False
            self.select_reflectance.weight.zero_()
            self.select_reflectance.weight[0, 3, 0, 0] = 1.0
        self.select_reflectance.weight.requires_grad = False

        # Scale heading_sin_cos (channels 6:8) up before it enters the
        # shared-quantization combined output -- see module docstring.
        # center_offset/dims (channels 0:6) are left at 1.0 (unaffected).
        self.register_buffer(
            "channel_scale", torch.tensor([1, 1, 1, 1, 1, 1, HEADING_SCALE, HEADING_SCALE], dtype=torch.float32)
            .reshape(1, 8, 1, 1)
        )

        # centroid computation: AvgPool2d over the point ("height") axis,
        # replacing .mean(dim=2, keepdim=True) -- same math, different op.
        self.centroid_pool = nn.AvgPool2d(kernel_size=(NUM_POINTS, 1))
        # global feature: MaxPool2d over the point axis, replacing
        # feat.max(dim=2).values (ReduceMax) -- same reasoning as AvgPool
        # above. ReduceMax was reported as not triggering the qtoolsv2 bug
        # in the original model, but v8 (AvgPool2d fix applied to the
        # centroid only) showed it now failing with the identical
        # Dequantize/"External op" signature, meaning the bug is general to
        # reduction-op boundaries, not specific to ReduceMean or to
        # Conv-vs-ReLU producers.
        self.global_pool = nn.MaxPool2d(kernel_size=(NUM_POINTS, 1))

        self.point_mlp = nn.Sequential(
            nn.Conv2d(in_channels, 32, (1, 1)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, (1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, (1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, (1, 1)), nn.ReLU(),
            nn.Conv2d(64, 32, (1, 1)), nn.ReLU(),
            nn.Conv2d(32, 8, (1, 1)),
        )

        # v11's 3-selector-conv split of the head output still hit the same
        # "Found 5 layout transforms, expected at most 4" bug -- it wasn't
        # about Slice-vs-Conv at all, it was about having 3 separate named
        # graph OUTPUTS. v12 avoids that entirely: instead of splitting the
        # head's (B,8,1,1) output apart to add the centroid to just channels
        # 0:3, zero-pad the centroid (B,3,1,1) out to (B,8,1,1) -- channels
        # 0:3 = centroid, 3:8 = zero -- and Add it directly to the full
        # 8-channel head output. One Add, one Concat-free (B,8,1,1) tensor,
        # ONE graph output. Host-side postprocessing does the trivial
        # channel-range split afterward (same pattern as tracker2d's
        # postprocess_graph.onnx box-decode output).
        self.pad_centroid_to_head = nn.Conv2d(3, 8, (1, 1), bias=False)
        with torch.no_grad():
            self.pad_centroid_to_head.weight.zero_()
            for c in range(3):
                self.pad_centroid_to_head.weight[c, c, 0, 0] = 1.0
        self.pad_centroid_to_head.weight.requires_grad = False

    def forward(self, points):
        """points: (B, 4, N, 1) -- xyz+reflectance already in the channel axis."""
        xyz_for_sub = self.select_xyz_for_sub(points)  # (B, 3, N, 1), only consumer: Sub
        reflectance = self.select_reflectance(points)  # (B, 1, N, 1)

        xyz_for_centroid1 = self.select_xyz_for_centroid1(points)  # only consumer: this AvgPool
        centroid_for_sub = self.centroid_pool(xyz_for_centroid1)  # (B, 3, 1, 1)
        centered_xyz = xyz_for_sub - centroid_for_sub  # (B, 3, N, 1)
        centered = torch.cat([centered_xyz, reflectance], dim=1)

        feat = self.point_mlp(centered)  # (B, 128, N, 1)
        global_feat = self.global_pool(feat)  # (B, 128, 1, 1)

        out = self.head(global_feat)  # (B, 8, 1, 1): [center_offset(3), dims(3), heading(2)]
        xyz_for_centroid2 = self.select_xyz_for_centroid2(points)  # only consumer: this AvgPool
        centroid_for_output = self.centroid_pool(xyz_for_centroid2)  # (B, 3, 1, 1)
        padded_centroid = self.pad_centroid_to_head(centroid_for_output)  # (B, 8, 1, 1), zero outside ch 0:3

        scaled = out * self.channel_scale  # only ch 6:8 (heading) actually change, scale=1 elsewhere
        combined = scaled + padded_centroid  # (B, 8, 1, 1): [center(3), dims(3), HEADING_SCALE*heading(2)]
        return combined


def transfer_weights(src: PointNetLite, dst: PointNetLite4D):
    src_sd, dst_sd = src.state_dict(), dst.state_dict()
    for key in dst_sd:
        if key not in src_sd:
            continue
        src_tensor = src_sd[key]
        gap = dst_sd[key].dim() - src_tensor.dim()
        for _ in range(gap):
            src_tensor = src_tensor.unsqueeze(-1)
        dst_sd[key] = src_tensor
    dst.load_state_dict(dst_sd)


class PointNetLite4DONNX(nn.Module):
    def __init__(self, model: PointNetLite4D):
        super().__init__()
        self.model = model

    def forward(self, points):
        return self.model(points)  # single (B, 8, 1, 1) tensor now


def downgrade_opset(model: onnx.ModelProto):
    """No ReduceMean/ReduceMax left to rewrite (both replaced by AvgPool2d/
    MaxPool2d, which have used attribute-based kernel_shape since opset 1 --
    no axes-as-input form to worry about). Still downgrade the declared
    opset to 13 to match every prior attempt's compiler-compatibility
    target and prune now-unused initializers."""
    used_inputs = {inp for n in model.graph.node for inp in n.input}
    keep = [init for init in model.graph.initializer if init.name in used_inputs]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)

    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = 13


def main():
    src = PointNetLite()
    src.load_state_dict(torch.load(
        str(PROJECT_ROOT / "weights" / "pointnet_lite_best.pt"), map_location="cpu"
    ))
    src.eval()

    dst = PointNetLite4D()
    transfer_weights(src, dst)
    dst.eval()

    torch.manual_seed(0)
    points_bn4 = torch.randn(2, NUM_POINTS, 4)
    points_4d = points_bn4.transpose(1, 2).unsqueeze(-1)  # (B,N,4) -> (B,4,N,1)

    with torch.no_grad():
        out_src = src(points_bn4)
        out_dst = dst(points_4d)  # (B, 8, 1, 1)

    expected = torch.cat(
        [out_src["center"], out_src["dims"], out_src["heading_sin_cos"] * HEADING_SCALE], dim=1
    )  # (B, 8), heading scaled to match what the compiled graph now outputs
    actual = out_dst[:, :, 0, 0]  # (B, 8)
    diff = (expected - actual).abs().max().item()
    print(f"combined output: max abs diff = {diff:.2e}")
    assert diff < 1e-4, "mismatch in combined output"
    print("4D (AvgPool2d + MaxPool2d + single combined output, 255 points, heading scale) "
          "reformulation verified numerically equivalent.\n")

    wrapped = PointNetLite4DONNX(dst)
    dummy = torch.randn(1, 4, NUM_POINTS, 1)
    out_path = str(HERE / "pointnet_lite_4d.onnx")
    torch.onnx.export(
        wrapped, dummy, out_path,
        input_names=["points"], output_names=["out"],
        opset_version=18, dynamo=False,
    )
    print(f"exported {out_path}")

    model = onnx.load(out_path)
    downgrade_opset(model)
    onnx.checker.check_model(model)
    onnx.save_model(model, out_path)
    print(f"fixed and re-saved {out_path}")

    ops = [n.op_type for n in model.graph.node]
    print("ReduceMean present:", "ReduceMean" in ops)
    print("ReduceMax present:", "ReduceMax" in ops)
    print("AveragePool present:", "AveragePool" in ops)
    print("MaxPool present:", "MaxPool" in ops)


if __name__ == "__main__":
    main()
