"""Workaround for a real hardware/software gap on this dev machine: the RTX
5070 (Blackwell, sm_120) is newer than the CUDA kernels torchvision ships in
its compiled ops (nms, roi_align, ...), even in the latest cu129 wheel. Plain
torch tensor ops (matmul, conv) work fine on this GPU; only torchvision's
custom CUDA kernels are missing sm_120 builds. NMS is cheap relative to the
YOLO backbone forward pass, so falling it back to CPU costs little.
"""
import torchvision

_original_nms = torchvision.ops.nms
_patched = False


def _cpu_fallback_nms(boxes, scores, iou_threshold):
    if not boxes.is_cuda:
        return _original_nms(boxes, scores, iou_threshold)
    device = boxes.device
    keep = _original_nms(boxes.cpu(), scores.cpu(), iou_threshold)
    return keep.to(device)


def patch_nms_for_unsupported_gpu():
    global _patched
    if not _patched:
        torchvision.ops.nms = _cpu_fallback_nms
        _patched = True
