"""
Run two (or more) models over every clip in a folder and compare them side by side.
Same tracker and settings for every model, so the model is the only thing that changes.

    uv run python scripts/compare_models.py
    uv run python scripts/compare_models.py --clips-dir more_recordings_for_training \\
        --models yolo26s.pt models/visdrone_v1.pt

Safe to stop and restart: runs that already have results are skipped.
Results: runs/compare/<clip>_<model>_... CSVs and runs/compare/summary.csv
"""
import argparse, csv, subprocess, sys, time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from compare import summarise  # same metrics as compare.py

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
COLS = ["cars/frame", "IDs", "<10 fr", ">=1 s", "median s", "conf", "conf<0.4"]
KEYS = ["per_frame", "ids", "flicker", "long", "median_len_s", "median_conf", "low_conf"]


def model_label(m):
    p = Path(m)
    return p.parent.parent.name if p.parent.name == "weights" else p.stem


def fmt(key, v):
    if key in ("ids", "flicker", "long"):
        return f"{v:.0f}"
    if key == "low_conf":
        return f"{v:.0%}"
    return f"{v:.2f}" if key == "median_conf" else f"{v:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", default="more_recordings_for_training")
    ap.add_argument("--models", nargs="+", default=["yolo26s.pt", "models/visdrone_v1.pt"])
    ap.add_argument("--tracker", default="configs/botsort_60fps.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out-dir", default="runs/compare")
    args = ap.parse_args()

    clips = sorted(p for p in Path(args.clips_dir).iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not clips:
        sys.exit(f"No videos found in {args.clips_dir}")
    out = Path(args.out_dir)
    total_s = 0
    for c in clips:
        cap = cv2.VideoCapture(str(c)); n = cap.get(cv2.CAP_PROP_FRAME_COUNT); fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release(); total_s += n / fps
    print(f"{len(clips)} clips, {total_s:.0f} s of video, {len(args.models)} models "
          f"-> roughly {total_s * 60 * len(args.models) / 10 / 60:.0f} min at ~10 frames/s (a guess; your Mac may differ)\n")

    results = []  # (clip, model, summary)
    t0 = time.perf_counter()
    for i, clip in enumerate(clips, 1):
        cap = cv2.VideoCapture(str(clip)); fps = cap.get(cv2.CAP_PROP_FPS) or 30; cap.release()
        for m in args.models:
            tag = f"{clip.stem}_{model_label(m)}_{Path(args.tracker).stem}_{args.imgsz}"
            tracks = out / f"{tag}_tracks.csv"
            if tracks.exists():
                print(f"[{i}/{len(clips)}] {clip.name} x {model_label(m)}: already done, skipping")
            else:
                print(f"[{i}/{len(clips)}] {clip.name} x {model_label(m)}: running "
                      f"({(time.perf_counter() - t0) / 60:.0f} min elapsed)", flush=True)
                r = subprocess.run([sys.executable, str(Path(__file__).parent / "baseline.py"),
                                    "--video", str(clip), "--model", m, "--tracker", args.tracker,
                                    "--imgsz", str(args.imgsz), "--no-video", "--out-dir", str(out)],
                                   capture_output=True, text=True)
                if r.returncode != 0 or not tracks.exists():
                    print(r.stdout[-1500:], r.stderr[-1500:])
                    sys.exit(f"Run failed on {clip.name} with {m}. Fix and re-run; finished runs are kept.")
            results.append((clip.stem, model_label(m), summarise(tracks, fps)))

    labels = [model_label(m) for m in args.models]
    width = 34
    print("\n" + f"{'clip':<{width}}{'model':<14}" + "".join(f"{c:>11}" for c in COLS))
    print("-" * (width + 14 + 11 * len(COLS)))
    last = None
    for clip, model, s in results:
        name = clip if clip != last else ""
        if last is not None and clip != last:
            print()
        last = clip
        vals = "".join(f"{fmt(k, s[k]):>11}" if s else f"{'-':>11}" for k in KEYS)
        print(f"{name[:width - 2]:<{width}}{model:<14}{vals}")

    print("\nAcross all clips")
    print(f"{'':<{width}}{'model':<14}{'total IDs':>11}{'<10 fr':>11}{'>=1 s':>11}{'mean conf':>11}{'conf<0.4':>11}{'wins*':>11}")
    by_clip = {}
    for clip, model, s in results:
        by_clip.setdefault(clip, {})[model] = s
    for lab in labels:
        ss = [by_clip[c][lab] for c in by_clip if by_clip[c].get(lab)]
        wins = sum(1 for c in by_clip if all(by_clip[c].get(x) for x in labels)
                   and min(labels, key=lambda x: by_clip[c][x]["flicker"]) == lab)
        print(f"{'':<{width}}{lab:<14}{sum(s['ids'] for s in ss):>11}{sum(s['flicker'] for s in ss):>11}"
              f"{sum(s['long'] for s in ss):>11}{sum(s['median_conf'] for s in ss) / len(ss):>11.2f}"
              f"{sum(s['low_conf'] for s in ss) / len(ss):>11.0%}{wins:>11}")
    print("* wins = clips where this model had the fewest flicker tracks")

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "model"] + KEYS)
        for clip, model, s in results:
            w.writerow([clip, model] + ([round(s[k], 4) for k in KEYS] if s else [""] * len(KEYS)))
    print(f"\nSaved {out / 'summary.csv'}  ({(time.perf_counter() - t0) / 60:.0f} min this session)")


if __name__ == "__main__":
    main()
