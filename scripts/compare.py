"""
Compare every tracking run in runs/baseline/ side by side.

    uv run python scripts/compare.py
    uv run python scripts/compare.py --clip clip02_carpark   # only runs on one clip

Reads the *_tracks.csv files that baseline.py writes. No video needed.
"""
import argparse, csv, statistics as st
from pathlib import Path


def summarise(path, fps):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    frames = {int(r["frame"]) for r in rows}
    lengths = {}
    for r in rows:
        lengths[r["track_id"]] = lengths.get(r["track_id"], 0) + 1
    lens = list(lengths.values())
    conf = [float(r["conf"]) for r in rows]
    n_frames = max(frames) + 1
    return {
        "per_frame": len(rows) / n_frames,
        "ids": len(lens),
        "flicker": sum(n < 10 for n in lens),          # tracks under 10 frames (1/6 s at 60 fps)
        "long": sum(n >= fps for n in lens),           # tracks lasting 1 s or more
        "median_len_s": st.median(lens) / fps,
        "median_conf": st.median(conf),
        "low_conf": sum(c < 0.4 for c in conf) / len(conf),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/baseline")
    ap.add_argument("--clip", default=None, help="only show runs whose name starts with this")
    ap.add_argument("--fps", type=float, default=59.94)
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*_tracks.csv"))
    if args.clip:
        files = [f for f in files if f.name.startswith(args.clip)]
    head = f"{'run':<44}{'cars/frame':>11}{'IDs':>6}{'<10 fr':>8}{'>=1 s':>7}{'median s':>10}{'conf':>7}{'conf<0.4':>10}"
    print(head); print("-" * len(head))
    for f in files:
        s = summarise(f, args.fps)
        name = f.name.removesuffix("_tracks.csv")
        if s is None:
            print(f"{name:<44}  (no detections)"); continue
        print(f"{name:<44}{s['per_frame']:>11.1f}{s['ids']:>6}{s['flicker']:>8}{s['long']:>7}"
              f"{s['median_len_s']:>10.1f}{s['median_conf']:>7.2f}{s['low_conf']:>10.0%}")
    print("\nIDs: fewer is better if the real car count is fixed. <10 fr: flicker tracks, fewer is better.")
    print(">=1 s: stable tracks. conf<0.4: share of unsure detections, lower is better.")


if __name__ == "__main__":
    main()
