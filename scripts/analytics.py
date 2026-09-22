"""
Turn saved tracking results into per-frame analytics: camera motion removed,
each vehicle marked moving or parked, and running totals for the panel.

    uv run python scripts/analytics.py --video data/processed/clip.mp4 --model visdrone_v1

Writes runs/analytics/<clip>_<model>.csv (one row per vehicle per frame) and
<clip>_<model>_frames.csv (one row per frame). render.py calls this automatically
when the files are missing, so you rarely run it by hand.

How "moving" is decided, in plain terms:
  1. The drone drifts, so everything appears to move. We measure that drift by
     following fixed ground features (road markings, roofs) between frames and
     taking the median shift, ignoring anything inside a vehicle box.
  2. Each vehicle's position is corrected by that drift.
  3. A vehicle counts as moving if it travels more than 25% of its own body
     length over half a second. That is roughly 8 km/h for a typical car, and it
     needs no distance calibration, so it works at any altitude.
"""
import argparse, csv, statistics as st
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("runs/analytics")
SEARCH_DIRS = [Path("runs/compare"), Path("runs/baseline")]
MOVE_FRACTION = 0.25      # of body length
WINDOW_S = 0.5            # seconds to measure movement over
FLICKER_FRAMES = 10       # tracks shorter than this are counted as flicker


def find_tracks(stem, model, tracker):
    for d in SEARCH_DIRS:
        hits = sorted(d.glob(f"{stem}_{model}_{tracker}_*_tracks.csv"))
        if hits:
            return hits[0]
        if model == "yolo26s":   # early runs left the model out of the file name
            hits = sorted(d.glob(f"{stem}_{tracker}_*_tracks.csv"))
            if hits:
                return hits[0]
    return None


def read_tracks(path, sx, sy):
    per_frame, per_track = {}, {}
    for r in csv.DictReader(open(path)):
        f, tid = int(r["frame"]), int(r["track_id"])
        x1, y1 = float(r["x1"]) * sx, float(r["y1"]) * sy
        x2, y2 = float(r["x2"]) * sx, float(r["y2"]) * sy
        det = {"id": tid, "conf": float(r["conf"]), "box": (x1, y1, x2, y2),
               "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
               "len": max(x2 - x1, y2 - y1)}
        per_frame.setdefault(f, []).append(det)
        per_track.setdefault(tid, []).append((f, det))
    return per_frame, per_track


def camera_motion(video, per_frame, w, h, scale=0.5):
    """Median shift of static ground features between consecutive frames, in panel pixels."""
    cap = cv2.VideoCapture(str(video))
    prev_gray, offsets, f = None, [(0.0, 0.0)], 0
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            mask = np.full(gray.shape, 255, np.uint8)
            for d in per_frame.get(f - 1, []):      # don't track features on vehicles
                x1, y1, x2, y2 = [int(v * scale) for v in d["box"]]
                cv2.rectangle(mask, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 0, -1)
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01,
                                          minDistance=12, mask=mask)
            dx = dy = 0.0
            if pts is not None and len(pts) >= 8:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **lk)
                good = status.ravel() == 1
                if good.sum() >= 8:
                    delta = (nxt[good] - pts[good]).reshape(-1, 2) / scale
                    dx, dy = float(np.median(delta[:, 0])), float(np.median(delta[:, 1]))
            offsets.append((dx, dy))
        prev_gray = gray
        f += 1
        if f % 600 == 0:
            print(f"    camera motion: {f} frames", flush=True)
    cap.release()
    cum, ax, ay = [], 0.0, 0.0
    for dx, dy in offsets:
        ax += dx; ay += dy
        cum.append((ax, ay))
    return cum


def analyse(video, model, tracker, panel_w, out_dir=OUT_DIR):
    stem = Path(video).stem
    tracks_csv = find_tracks(stem, model, tracker)
    if tracks_csv is None:
        return None
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    pw = panel_w; ph = int(H * pw / W) // 2 * 2
    sx, sy = pw / W, ph / H

    per_frame, per_track = read_tracks(tracks_csv, sx, sy)
    print(f"  {stem} / {model}: measuring camera motion", flush=True)
    cam = camera_motion(video, per_frame, pw, ph)

    def corrected(f, d):
        ox, oy = cam[min(f, len(cam) - 1)]
        return d["cx"] - ox, d["cy"] - oy

    win = max(int(round(fps * WINDOW_S)), 2)
    state = {}          # (frame, id) -> ("moving"/"parked", speed in body-lengths per second)
    for tid, items in per_track.items():
        items.sort()
        pos = {f: corrected(f, d) for f, d in items}
        body = st.median([d["len"] for _, d in items]) or 1.0
        frames = [f for f, _ in items]
        for f in frames:
            past = [g for g in frames if f - win <= g <= f]
            if len(past) < 2:
                state[(f, tid)] = ("parked", 0.0); continue
            (x0, y0), (x1, y1) = pos[past[0]], pos[past[-1]]
            dist = float(np.hypot(x1 - x0, y1 - y0))
            secs = max((past[-1] - past[0]) / fps, 1e-6)
            state[(f, tid)] = ("moving" if dist > MOVE_FRACTION * body else "parked",
                               dist / body / secs)

    lengths = {tid: len(v) for tid, v in per_track.items()}
    first_seen = {tid: min(f for f, _ in v) for tid, v in per_track.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{stem}_{model}"
    with open(f"{base}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "track_id", "state", "bodylen_per_s"])
        for (f, tid), (s, v) in sorted(state.items()):
            w.writerow([f, tid, s, round(v, 3)])

    n_frames = (max(per_frame) if per_frame else 0) + 1
    seen, conf_sum, conf_n = set(), 0.0, 0
    rows = []
    for f in range(n_frames):
        dets = per_frame.get(f, [])
        seen.update(d["id"] for d in dets)
        conf_sum += sum(d["conf"] for d in dets); conf_n += len(dets)
        moving = sum(1 for d in dets if state.get((f, d["id"]), ("parked",))[0] == "moving")
        flicker = sum(1 for tid in seen if lengths[tid] < FLICKER_FRAMES and first_seen[tid] <= f)
        rows.append([f, len(dets), moving, len(dets) - moving, len(seen), flicker,
                     round(conf_sum / max(conf_n, 1), 4)])
    with open(f"{base}_frames.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "on_screen", "moving", "parked", "unique_ids", "flicker", "mean_conf"])
        w.writerows(rows)
    print(f"  {stem} / {model}: {len(per_track)} tracks, "
          f"{sum(1 for v in lengths.values() if v < FLICKER_FRAMES)} flicker -> {base}.csv")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="visdrone_v1")
    ap.add_argument("--tracker", default="botsort_60fps")
    ap.add_argument("--panel-width", type=int, default=1280)
    args = ap.parse_args()
    if analyse(args.video, args.model, args.tracker, args.panel_width) is None:
        raise SystemExit(f"No saved tracking results for {args.video} with {args.model}")


if __name__ == "__main__":
    main()
