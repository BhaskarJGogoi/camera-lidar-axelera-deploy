# Hardware deployment notes

The technical story of getting `tracker2d` and `pointnet_lite` running correctly
and reasonably fast on the Axelera Metis AIPU. Read this if you're debugging
the hardware backend, re-compiling either model, or curious why the code
looks the way it does in a few specific, commented spots.

## tracker2d: the letterbox bug

The single biggest accuracy problem on hardware had nothing to do with
quantization. `backends/hardware/tracker2d.py`'s `preprocess_image()`
letterboxes (aspect-ratio-preserving resize + pad to 640x640, matching
Ultralytics' own default) instead of naively stretching the image to
square. KITTI frames are ~1242x375 (a 3.3:1 aspect ratio); squashing that
into a square distorts every object's shape far more than int8
quantization does. This was confirmed by feeding the FP32 PyTorch model
the same naively-stretched input and seeing the identical sparse,
low-confidence detection pattern the compiled hardware model showed —
proving the compiler/quantizer weren't at fault. The compiled model's
calibration set must use the same letterboxed preprocessing (see
`backends/hardware/compile/compile_tracker2d.py`), or calibration
statistics won't match what inference actually feeds the model.

## tracker2d: quantization scheme findings (mostly negative results)

Three PTQ schemes were tried, plus a diverse (5-sequence) calibration set,
*before* the letterbox bug was found and fixed:

- `per_tensor_min_max`: more detections, lower confidence, genuinely
  different compiled weights.
- `per_tensor_histogram` (current default): fewer detections, slightly
  higher confidence. **Byte-for-byte identical compiled weights** to
  `hybrid_per_tensor_per_channel` — for this specific model, per-channel
  weight quantization granularity made zero difference to the final int8
  values. Not a bug; this model's weight distributions were apparently
  already well-served by per-tensor quantization.
- Diverse 5-sequence calibration (vs. 100 consecutive frames from one
  clip): shifted confidence slightly, **did not** recover the sparse
  detections. The scheme/calibration axis was a red herring; letterbox was
  the actual fix (see above), confirmed post-hoc by comparing to the
  FP32 model under the same broken preprocessing.

Lesson: don't assume a quantization-shaped problem is actually about
quantization. Check preprocessing parity with the FP32 reference first.

## pointnet_lite: the six-attempt compiler bug, then a fix

`pointnet_lite` originally failed to compile at all. The blocker: any
tensor from a `Conv` feeding directly into `ReduceMean` (the centroid
computation) crashed the quantizer (`qtoolsv2`) with `External op
..._branching_point_to_..._dre found in the model`. Six independent
workarounds — duplicate single-consumer Convs, `Identity`/`Mul-by-1`
passthroughs, `do_quantize_residuals=False`, alternate quantization
schemes — all failed identically, narrowing the cause to the
Conv→ReduceMean *adjacency itself*, not fan-out or numerics.

The fix chain (`backends/hardware/compile/build_pointnet_onnx.py`, in order):

1. **`ReduceMean` → `AvgPool2d`** for the centroid. Same math (mean over
   the point axis = global average pooling), different operator — pooling
   ops are first-class in a CNN-focused compiler in a way a generic
   `ReduceMean` apparently isn't. Got past the quantizer entirely for the
   first time.
2. **`ReduceMax` → `MaxPool2d`** for the global feature. `ReduceMax` had
   never triggered the bug in the original model, but after fix #1 it hit
   the *identical* error — proving the blocker was general to any
   reduction op, not specific to `ReduceMean`.
3. **Slice-before-squeeze**, not after, for the final `center_offset`/
   `dims`/`heading` split. Got past quantization entirely, hit a new
   *lowering* bug: `RewriteSliceToConv` asserted 4D input, and the
   original code squeezed to 2D before slicing.
4. **Selector-Conv instead of Slice** for that same split — turned out
   irrelevant; hit the *same* lowering error again (`AssertNoSpuriousLayoutTransforms:
   "Found 5 layout transforms, expected at most 4"`), proving it was never
   about Slice vs. Conv, it was about having 3 separate named graph
   outputs.
5. **Single combined output**: zero-pad the centroid to 8 channels and
   `Add` it directly to the full head output, instead of splitting output
   channels apart at all. One tensor, one graph output. Finally reached
   real AIPU code generation.
6. **256 → 255 points**: hit a genuine hardware limit, not a bug —
   `iau_max_reduce` (the reduction instruction backing both pooling ops)
   rejects `loop_len >= 256`. PointNet-style architectures (shared
   per-point MLP + symmetric pooling) are point-count invariant, so the
   same trained weights work unchanged at 255 points; no retraining
   needed.

Also: compiling via `minimal-compiler`-style tooling produces an artifact
*missing* `kernel_function.elf` even when the toolchain reports success —
only compiling inside the full Axelera Voyager SDK's own venv (which wires
up the complete CMake/RISC-V baremetal toolchain) produces a complete
artifact. Same lesson applies to `tracker2d`.

## pointnet_lite: the heading-quantization bug

After the model finally ran on hardware, `center`/`dims` were accurate but
`heading` came back as the *exact same value* for every input. Cause: the
single combined `(1,8,1,1)` output shares one int8 quantization scale
across all 8 channels. Calibrated against `center`/`dims`'s range (~-21 to
+80 m), that scale is ~0.396/step — meaning `heading_sin_cos`'s true
`[-1,1]` range spans only ~5 distinct representable int8 values total, so
it collapses to a fixed bin regardless of input.

Fix: scale `heading_sin_cos` by `HEADING_SCALE=15` before it enters the
shared-quantization output (`[-15,15]`, safely inside the existing
representable range, so `center`/`dims` resolution is unaffected), and
divide back out on the host after dequantizing
(`backends/hardware/pointnet_lite.py:postprocess`). Gives heading ~76
levels instead of ~5.

## pointnet_lite: the mystery second input

`model.json` declares **two** inputs, `var_input_ifd0` (the real
`(1,255,16,4)` tensor) and `var_input_ifd0_1` (same byte size, flat, a
different offset in a `depth:2` DMA pool). Leaving the second one zeroed
produced near-constant output regardless of real input data — it's a
double-buffer slot the kernel actually reads from, not an independent
tensor. `backends/hardware/pointnet_lite.py:regress` fills both with the
same data.

## Performance: what's actually slow, and why

Building the full `Stage A->D` hardware pipeline (`backends/hardware/pipeline.py`,
`scripts/demo.py`) surfaced several distinct bottlenecks — worth
distinguishing because they have different causes and different fixes:

- **Per-point `cv2.circle()` drawing loops.** A KITTI frame has
  15-25k visible LiDAR points; looping a Python+OpenCV call per point
  costs hundreds of ms/frame. Fixed by vectorizing to vectorized numpy
  pixel writes (`pipeline/render.py:_draw_points_fast` — 9-25 numpy
  operations total regardless of point count, instead of one call per
  point).
- **`tracker2d`'s box-decode running the expensive DFL softmax on all
  8400 anchors before filtering by confidence.** Reordering to filter by
  the cheap 3-channel class-confidence sigmoid *first* (typically <20
  anchors survive) cuts the softmax work ~400-800x regardless of how fast
  `exp()` itself is on a given CPU (`backends/hardware/tracker2d.py:decode_boxes`).
- **AIPU model-switching cost.** Alternating between `tracker2d` and
  `pointnet_lite` on the same device every frame cost ~35-45ms extra per
  switch, apparently asymmetric (`tracker2d`'s L2-resident weights blob is
  ~2x `pointnet_lite`'s, so reloading it back costs more than the reverse
  direction). Fixed by the 3-pass design in `backends/hardware/pipeline.py`
  (run all of `tracker2d` across the whole clip, then all of
  `pointnet_lite`, only one switch total).
- **Generic numpy elementwise ops (quantize, pad, LiDAR projection) being
  slow on the target ARM board specifically**, not on a normal x86 dev
  machine. Hit repeatedly (the DFL softmax's `exp()`, the LiDAR projection
  matmul, image quantization/padding) — the working hypothesis is that
  this board's numpy build lacks ARM NEON/SIMD dispatch for elementwise
  ufuncs and falls back to scalar loops, but this was never actually
  confirmed by checking `np.show_config()` on-device (worth doing if you
  pick this back up). Nothing to fix at the application level short of
  installing a better numpy wheel on the device; the early-confidence
  -filter and 3-pass fixes above are what's actually within this
  project's control.
