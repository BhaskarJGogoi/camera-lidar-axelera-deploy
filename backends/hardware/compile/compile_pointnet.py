"""Quantizes + compiles the ONNX produced by build_pointnet_onnx.py. Must
run inside the Axelera Voyager SDK's own venv (not this project's)."""
import glob
import traceback
import numpy as np
from pathlib import Path
from axelera import compiler
from axelera.compiler import CompilerConfig

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "pointnet_lite_4d.onnx"  # produced by build_pointnet_onnx.py
CALIB_DIR = HERE / "assets" / "calib_pointnet_lite"
OUTPUT_DIR = HERE / "pointnet_lite_compiled"


def create_calibration_data():
    files = sorted(glob.glob(str(CALIB_DIR / "*.npy")))
    print(f"  Using {len(files)} calibration samples")
    for f in files:
        points = np.load(f).astype(np.float32)  # (4, 256, 1)
        points = points[:, :255, :]  # drop one point -- model now expects 255 (iau_max_reduce hw limit)
        yield points[np.newaxis, ...]  # (1, 4, 255, 1)


config = CompilerConfig(
    remove_output_dir=True,
    save_error_artifact=True,
    quantize_dw_channel_wise=True,
    quantization_scheme="per_tensor_histogram",
    graph_cleaner_split_pre_post_processing=False,
)

print(f"Model:  {MODEL_PATH}")
print(f"Calib:  {CALIB_DIR}")
print(f"Output: {OUTPUT_DIR}")

print("\n--- Quantization ---")
try:
    qmodel = compiler.quantize(
        model=str(MODEL_PATH),
        calibration_dataset=create_calibration_data(),
        config=config,
    )
    print("Quantization complete")
except Exception as e:
    print(f"Quantization failed: {type(e).__name__}: {e}")
    traceback.print_exc()
    raise SystemExit(1)

print("\n--- Compilation ---")
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compiler.compile(
        model=qmodel,
        config=config,
        output_dir=OUTPUT_DIR,
    )
    print(f"\nCompiled to {OUTPUT_DIR}/")
except Exception as e:
    print(f"Compilation failed: {type(e).__name__}: {e}")
    traceback.print_exc()
    raise SystemExit(1)
