#!/usr/bin/env python3
"""Runs the full Stage A->D pipeline (2D detect -> LiDAR frustum -> 3D
regress -> 2D/3D track) over a KITTI sequence clip and writes one mp4:
camera panel (2D boxes + LiDAR points colored by frustum membership +
projected 3D wireframes) plus a bird's-eye-view trajectory panel -- same
visual style as the original project's demo.gif.

Works identically for either backend: both backends' iter_sequence()
generators yield the same frame-dict shape (see
backends/cpu/pipeline.py's or backends/hardware/pipeline.py's docstring),
so this script never needs to know which one it's driving.

Usage:
    python scripts/demo.py --backend cpu --num-frames 50
    python scripts/demo.py --backend hardware --data-dir kitti --num-frames 50
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from pipeline.render import draw_camera_panel, draw_bev_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["cpu", "hardware"], required=True)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "kitti_demo_clip"),
                         help="dir containing image_2/<seq>, velodyne/<seq>, calib/<seq>.txt")
    parser.add_argument("--seq", default="0011")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tracker2d-weights", default=str(PROJECT_ROOT / "weights" / "tracker2d_best.pt"),
                         help="CPU backend only")
    parser.add_argument("--pointnet-weights", default=str(PROJECT_ROOT / "weights" / "pointnet_lite_best.pt"),
                         help="CPU backend only")
    args = parser.parse_args()

    out_path = args.out or str(PROJECT_ROOT / f"demo_{args.backend}.mp4")

    if args.backend == "cpu":
        from backends.cpu.pipeline import iter_sequence
        frames_iter = iter_sequence(
            args.data_dir, args.seq, args.tracker2d_weights, args.pointnet_weights,
            conf_thres=args.conf, iou_thres=args.iou, num_frames=args.num_frames,
        )
    else:
        from backends.hardware.pipeline import iter_sequence
        frames_iter = iter_sequence(
            args.data_dir, args.seq, conf_thres=args.conf, iou_thres=args.iou, num_frames=args.num_frames,
        )

    writer = None
    n_frames = 0
    for frame_data in frames_iter:
        calib = frame_data["calib"]
        cam_panel = draw_camera_panel(
            frame_data["image"], calib, frame_data["uv_all"], frame_data["depth_all"],
            frame_data["tracks"], frame_data["frustum_points"], frame_data["boxes_3d"], frame_data["track_ids_3d"],
        )
        bev_panel = draw_bev_panel(frame_data["tracker3d"].trajectories(), frame_data["frame_idx"], cam_panel.shape[0])
        combined = np.hstack([cam_panel, bev_panel])

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, args.fps, (combined.shape[1], combined.shape[0]))
        writer.write(combined)
        n_frames += 1
        print(f"frame {frame_data['frame_idx']}: {len(frame_data['tracks'])} tracked, "
              f"{len(frame_data['boxes_3d'])} with 3D box")

    if writer is not None:
        writer.release()
    print(f"wrote {n_frames} frames to {out_path}")


if __name__ == "__main__":
    main()
