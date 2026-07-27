"""Recompiles tracker2d for Axelera Metis. Reproducibility reference, not
part of the normal test/demo flow -- the compiled artifact this produces is
already bundled at backends/hardware/compiled_tracker2d/.

Must run inside the Axelera Voyager SDK's own venv (not this project's), and
needs CALIB_DIR populated first -- see make_calibration_set.py. See
docs/HARDWARE_NOTES.md for why letterbox preprocessing here (not a naive
resize) is what actually matters for accuracy, more than the quantization
scheme choice.
"""
import glob
import cv2
import numpy as np
from pathlib import Path
from axelera import compiler
from axelera.compiler import CompilerConfig

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "assets" / "tracker2d_opset13.onnx"
CALIB_DIR = HERE / "assets" / "calib_tracker2d"  # see make_calibration_set.py to (re)generate this
OUTPUT_DIR = HERE / "tracker2d_compiled"

INPUT_H = 640
INPUT_W = 640


def letterbox(image: np.ndarray, new_h: int, new_w: int, pad_value: int = 114) -> np.ndarray:
    """Matches Ultralytics' default LetterBox (scaleup=True, center=True,
    padding_value=114) -- must match run_on_hardware.py's preprocess() so
    calibration statistics reflect the same input distribution used at
    inference. A naive stretch-to-square here (the previous bug) distorted
    KITTI's ~3.3:1 aspect ratio images badly enough to be the dominant cause
    of this model's sparse/low-confidence on-hardware detections -- far more
    than any quantization scheme or calibration-diversity choice."""
    h, w = image.shape[:2]
    scale = min(new_h / h, new_w / w)
    unpad_h, unpad_w = round(h * scale), round(w * scale)
    resized = cv2.resize(image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    dh, dw = new_h - unpad_h, new_w - unpad_w
    top, bottom = round(dh / 2 - 0.1), round(dh / 2 + 0.1)
    left, right = round(dw / 2 - 0.1), round(dw / 2 + 0.1)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)


def preprocess_image(path: str) -> np.ndarray:
    img = cv2.imread(path)  # BGR
    img = letterbox(img, INPUT_H, INPUT_W)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]


def create_calibration_data():
    files = sorted(glob.glob(str(CALIB_DIR / "*.png")))
    print(f"  Using {len(files)} calibration samples")
    for f in files:
        yield preprocess_image(f)


config = CompilerConfig(
    remove_output_dir=True,
    save_error_artifact=True,
    quantize_dw_channel_wise=True,
    quantization_scheme="per_tensor_histogram",
)

print(f"Model:  {MODEL_PATH}")
print(f"Calib:  {CALIB_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Scheme: {config.ptq_scheme}")

print("\n--- Quantization ---")
try:
    qmodel = compiler.quantize(
        model=str(MODEL_PATH),
        calibration_dataset=create_calibration_data(),
        config=config,
    )
    print("Quantization complete")
except Exception as e:
    print(f"Quantization failed: {e}")
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
    elf = OUTPUT_DIR / "kernel_function.elf"
    print(f"kernel_function.elf: {'OK, ' + str(elf.stat().st_size) + ' bytes' if elf.exists() else 'MISSING'}")
except Exception as e:
    print(f"Compilation failed: {e}")
    raise SystemExit(1)
