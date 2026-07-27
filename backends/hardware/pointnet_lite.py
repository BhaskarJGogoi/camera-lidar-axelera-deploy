"""Stage C, hardware backend: 3D regression via the Axelera-compiled
pointnet_lite model on real Metis hardware.

Model expects exactly 255 points (xyz + reflectance), NOT 256 -- the AIPU's
iau_max_reduce hardware instruction rejects loop_len >= 256, and this
PointNet-style architecture (shared per-point MLP + symmetric pooling) is
point-count invariant, so 255 points works with the same trained weights,
no retraining needed. See backends/hardware/compile/build_pointnet_onnx.py
and docs/HARDWARE_NOTES.md for the full six-failed-attempt compiler bug
story that preceded this working version.

Output is a single (1, 8, 1, 1) tensor: [center(3), dims(3),
heading_sin_cos(2)]. Center is already absolute (centroid + offset summed
inside the compiled graph). heading_sin_cos is scaled by HEADING_SCALE
inside the compiled graph -- the shared per-tensor output quantizer gave
heading's true [-1,1] range only ~5 usable int8 levels otherwise,
collapsing it to a fixed value regardless of input; scaling up before
quantizing and dividing back out here fixes that. center/dims are
unaffected (see docs/HARDWARE_NOTES.md).
"""
import numpy as np
from axelera.runtime import TensorInfo

NUM_POINTS = 255
HEADING_SCALE = 15.0


def axelera_quantize(normalized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    quantized = np.round((normalized / scale) + zero_point)
    return quantized.clip(-128, 127).astype(np.int8)


def axelera_pad(unpadded: np.ndarray, padding: list, zero_point: int) -> np.ndarray:
    return np.pad(unpadded, padding, mode="constant", constant_values=zero_point)


def axelera_depad(padded: np.ndarray, padding: list) -> np.ndarray:
    slices = tuple(slice(start, -end if end else None) for start, end in padding)
    return padded[slices]


def axelera_dequantize(quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (quantized.astype(np.float32) - zero_point) * scale


def preprocess(points: np.ndarray, input_info: TensorInfo) -> np.ndarray:
    """points: (255, 4) xyz+reflectance, camera frame, float32 (unnormalized,
    matching training -- no separate input normalization, unlike the image
    pipeline's /255.0)."""
    assert points.shape == (NUM_POINTS, 4), f"expected ({NUM_POINTS}, 4), got {points.shape}"
    hwc = points[:, np.newaxis, :]  # (255, 1, 4)
    quantized = axelera_quantize(hwc, input_info.scale, input_info.zero_point)
    padded = axelera_pad(quantized, input_info.padding[1:], input_info.zero_point)
    return padded[np.newaxis, ...]


def postprocess(raw_output: np.ndarray, output_info: TensorInfo) -> dict:
    depadded = axelera_depad(raw_output, output_info.padding)
    fp32 = axelera_dequantize(depadded, output_info.scale, output_info.zero_point)
    flat = fp32.reshape(-1)
    heading_sin_cos = flat[6:8] / HEADING_SCALE
    return {
        "center": flat[0:3],
        "dims": flat[3:6],
        "heading_sin_cos": heading_sin_cos,
        "heading": float(np.arctan2(heading_sin_cos[0], heading_sin_cos[1])),
    }


def regress(points: np.ndarray, model_3d: dict) -> dict:
    """points: (M, 4) real frustum points, M != NUM_POINTS in general --
    sampled down/up to exactly NUM_POINTS first (matches the CPU backend's
    Stage C sampling, just 255 instead of 256)."""
    m = len(points)
    idx = np.random.choice(m, NUM_POINTS, replace=(m < NUM_POINTS))
    sampled = points[idx].astype(np.float32)

    tensor = preprocess(sampled, model_3d["input_infos"][0])
    model_3d["inputs"][0][:] = tensor
    if len(model_3d["inputs"]) > 1:
        # model.json declares a 2nd input (same byte size, different offset
        # in a depth:2 DMA pool) -- a double-buffer slot the kernel actually
        # reads from, not an independent tensor; leaving it zeroed produced
        # near-constant output regardless of real input. See
        # docs/HARDWARE_NOTES.md.
        model_3d["inputs"][1][:] = tensor.reshape(model_3d["input_infos"][1].shape)
    model_3d["instance"].run(model_3d["inputs"], model_3d["outputs"])

    return postprocess(model_3d["outputs"][0], model_3d["output_infos"][0])
