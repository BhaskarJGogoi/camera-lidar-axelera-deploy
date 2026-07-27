"""Regenerates the tracker2d calibration image set used by compile_tracker2d.py.

The original compile (this session) sampled 40 evenly-spaced frames from
each of 5 KITTI sequences (0004, 0005, 0011, 0017, 0020) for calibration
diversity -- using consecutive frames from a single short clip badly
under-samples the true activation distribution compared to varied scenes.
That 153MB image set isn't bundled in this repo (it's a reproducibility
aid, not something the main test/demo flow needs), so regenerate it here
from wherever you have KITTI data. Works with just one sequence (e.g. the
bundled data/kitti_demo_clip) if that's all you have -- less diverse than
the original 5-sequence version, but still functional.

Usage:
    python make_calibration_set.py --kitti-root /path/to/kitti --out ./assets/calib_tracker2d
"""
import argparse
import glob
import os
import shutil
from pathlib import Path

PER_SEQUENCE = 40


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti-root", required=True, help="dir containing image_2/<seq>/*.png subdirs")
    parser.add_argument("--out", default=str(Path(__file__).parent / "assets" / "calib_tracker2d"))
    parser.add_argument("--per-sequence", type=int, default=PER_SEQUENCE)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_2 = Path(args.kitti_root) / "image_2"
    sequences = sorted(p.name for p in image_2.iterdir() if p.is_dir())
    if not sequences:
        raise SystemExit(f"no sequences found under {image_2}")

    count = 0
    for seq in sequences:
        files = sorted(glob.glob(str(image_2 / seq / "*.png")))
        n = len(files)
        stride = max(1, n // args.per_sequence)
        picked = files[::stride][: args.per_sequence]
        for f in picked:
            shutil.copy(f, out_dir / f"{seq}_{os.path.basename(f)}")
            count += 1

    print(f"wrote {count} calibration images ({len(sequences)} sequence(s)) to {out_dir}")


if __name__ == "__main__":
    main()
