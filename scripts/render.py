"""
Build the comparison video: footage on the left, an analytics panel on the right,
one row per model (untrained on top, trained below).

    uv run python scripts/render.py --video data/processed/clip02_carpark.mp4
    uv run python scripts/render.py --all

Nothing is re-detected here. It redraws the boxes from the saved *_tracks.csv files
and the moving/parked analysis from scripts/analytics.py (run automatically if missing).
Output: runs/videos/<clip>.mp4
"""
import argparse, csv, shutil, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analytics as an

VIDEO_W, PANEL_W, GAP, HEADER_H = 1280, 560, 8, 76
LABELS = {"yolo26s": "Untrained Model (YOLO26s)",
          "visdrone_v1": "Trained Model (VisDrone + Hand Labeling)"}
ACCENT = {0: (70, 160, 240), 1: (175, 200, 45)}          # BGR: row 0 amber, row 1 teal
BG, PANEL_BG = (20, 20, 22), (30, 30, 33)
FG, MUTED, RULE = (240, 240, 240), (150, 150, 155), (58, 58, 63)
PARKED = (125, 125, 130)
F_VAL, F_LBL = cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale=0.5, colour=FG, font=F_LBL, thick=1, right=False):
    if right:
        (tw, _), _ = cv2.getTextSize(s, font, scale, thick)
        org = (org[0] - tw, org[1])
    cv2.putText(img, s, org, font, scale, colour, thick, cv2.LINE_AA)


def read_frames_csv(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[int(r["frame"])] = {k: float(v) for k, v in r.items() if k != "frame"}
    return rows


def read_states(path):
    return {(int(r["frame"]), int(r["track_id"])): r["state"] for r in csv.DictReader(open(path))}


def draw_boxes(img, dets, states, frame_i, accent):
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        moving = states.get((frame_i, d["id"])) == "moving"
        colour = accent if moving else PARKED
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2 if moving else 1, cv2.LINE_AA)
        tag = str(d["id"])
        (tw, th), _ = cv2.getTextSize(tag, F_LBL, 0.42, 1)
        if moving:
            cv2.rectangle(img, (x1, y1 - th - 7), (x1 + tw + 8, y1), colour, -1)
            text(img, tag, (x1 + 4, y1 - 5), 0.42, (15, 15, 15))
        else:
            text(img, tag, (x1 + 2, y1 - 4), 0.4, PARKED)


def sparkline(panel, series, i, box, accent, caption):
    x, y, w, h = box
    cv2.rectangle(panel, (x, y), (x + w, y + h), (24, 24, 27), -1)
    top = max(max(series) if series else 1, 1)
    pts = []
    for f, v in enumerate(series):
        px = x + int(f * w / max(len(series) - 1, 1))
        py = y + h - int(v / top * (h - 8)) - 4
        pts.append((px, py))
    if len(pts) > 1:
        fill = np.array([[(x, y + h)] + pts + [(x + w, y + h)]], np.int32)
        shade = panel.copy(); cv2.fillPoly(shade, fill, accent)
        cv2.addWeighted(shade, 0.18, panel, 0.82, 0, panel)
        cv2.polylines(panel, [np.array(pts, np.int32)], False, accent, 1, cv2.LINE_AA)
    if 0 <= i < len(pts):
        cv2.line(panel, (pts[i][0], y), (pts[i][0], y + h), (90, 90, 95), 1)
        cv2.circle(panel, pts[i], 3, FG, -1, cv2.LINE_AA)
    text(panel, caption, (x, y - 9), 0.38, MUTED)
    text(panel, f"max {top:.0f}", (x + w, y - 9), 0.38, MUTED, right=True)


def build_panel(h, title, stats, series, ids_series, i, accent):
    p = np.full((h, PANEL_W, 3), PANEL_BG, np.uint8)
    m = 28                                        # margin
    cv2.rectangle(p, (0, 0), (4, h), accent, -1)  # accent spine

    y = 46
    for line in wrap(title, 26):
        text(p, line, (m, y), 0.62, FG, F_VAL); y += 28
    y += 6
    cv2.line(p, (m, y), (PANEL_W - m, y), RULE, 1); y += 44

    text(p, "NOW", (m, y), 0.42, accent); y += 26
    text(p, "VEHICLES ON SCREEN", (m, y), 0.4, MUTED); y += 46
    text(p, f"{stats['on_screen']:.0f}", (m, y), 1.5, FG, F_VAL); y += 34

    bar_y = y
    total = max(stats["on_screen"], 1)
    bw = PANEL_W - 2 * m
    mv = int(bw * stats["moving"] / total)
    cv2.rectangle(p, (m, bar_y), (m + bw, bar_y + 8), (46, 46, 50), -1)
    cv2.rectangle(p, (m, bar_y), (m + mv, bar_y + 8), accent, -1)
    y = bar_y + 30
    text(p, f"MOVING  {stats['moving']:.0f}", (m, y), 0.44, accent)
    text(p, f"{stats['parked']:.0f}  PARKED", (PANEL_W - m, y), 0.44, PARKED, right=True)
    y += 30
    cv2.line(p, (m, y), (PANEL_W - m, y), RULE, 1); y += 40

    text(p, "SINCE START", (m, y - 14), 0.42, accent); y += 16
    for label, value in [("UNIQUE IDS", f"{stats['unique_ids']:.0f}"),
                         ("FLICKER TRACKS (<10 FRAMES)", f"{stats['flicker']:.0f}"),
                         ("MEAN CONFIDENCE", f"{stats['mean_conf']:.2f}")]:
        text(p, label, (m, y), 0.4, MUTED)
        text(p, value, (PANEL_W - m, y + 2), 0.72, FG, F_VAL, right=True)
        y += 24
        cv2.line(p, (m, y), (PANEL_W - m, y), (40, 40, 44), 1)
        y += 34

    avail = h - y - 78                       # space left above the footnote
    sh = int(min(92, max(26, (avail - 46) / 2)))
    sparkline(p, series, i, (m, y + 16, PANEL_W - 2 * m, sh), accent, "COUNT OVER TIME")
    y2 = y + 16 + sh + 28
    sparkline(p, ids_series, i, (m, y2, PANEL_W - 2 * m, sh), accent, "UNIQUE IDS OVER TIME")
    foot = "Moving = travels >25% of its own body length in 0.5 s, after camera drift is removed."
    yy = min(y2 + sh + 24, h - 32)
    for line in wrap(foot, 52):
        text(p, line, (m, yy), 0.36, (110, 110, 115)); yy += 15
    return p


def wrap(s, n):
    out, line = [], ""
    for word in s.split():
        if len(line) + len(word) + 1 > n and line:
            out.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    return out + ([line] if line else [])


def header(w, clip, frame_i, fps, n):
    bar = np.full((HEADER_H, w, 3), BG, np.uint8)
    text(bar, "AERIAL VEHICLE DETECTION & TRACKING", (28, 32), 0.56, FG, F_VAL)
    text(bar, f"{clip}   detector comparison", (28, 56), 0.42, MUTED)
    t = frame_i / fps
    text(bar, f"{int(t // 60):01d}:{t % 60:05.2f}", (w - 28, 36), 0.62, FG, F_VAL, right=True)
    text(bar, f"frame {frame_i} / {n}   {fps:.0f} fps", (w - 28, 58), 0.4, MUTED, right=True)
    return bar


def render(video, models, tracker, out_dir, labels):
    stem = Path(video).stem
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30; n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    vh = int(H * VIDEO_W / W) // 2 * 2
    sx, sy = VIDEO_W / W, vh / H

    rows = []
    for m in models:
        tracks_csv = an.find_tracks(stem, m, tracker)
        if tracks_csv is None:
            print(f"{stem}: no results for {m} yet, skipping clip"); return
        base = an.OUT_DIR / f"{stem}_{m}"
        if not Path(f"{base}_frames.csv").exists():
            an.analyse(video, m, tracker, VIDEO_W)
        per_frame, _ = an.read_tracks(tracks_csv, sx, sy)
        rows.append({"model": m, "dets": per_frame, "states": read_states(f"{base}.csv"),
                     "stats": read_frames_csv(f"{base}_frames.csv")})
    n = max(max(r["stats"]) for r in rows) + 1
    series = [[r["stats"].get(f, {"on_screen": 0})["on_screen"] for f in range(n)] for r in rows]
    ids_series = [[r["stats"].get(f, {"unique_ids": 0})["unique_ids"] for f in range(n)] for r in rows]

    out_w = VIDEO_W + GAP + PANEL_W
    out_h = HEADER_H + len(rows) * vh + (len(rows) - 1) * GAP
    out_w += out_w % 2; out_h += out_h % 2
    out = out_dir / f"{stem}.mp4"

    hw = ["-hwaccel", "videotoolbox"] if sys.platform == "darwin" else []
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", *hw, "-i", str(video),
                            "-vf", f"scale={VIDEO_W}:{vh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{out_w}x{out_h}", "-r", f"{fps}", "-i", "-",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)
    nbytes, i = VIDEO_W * vh * 3, 0
    blank = {"on_screen": 0, "moving": 0, "parked": 0, "unique_ids": 0, "flicker": 0, "mean_conf": 0}
    while True:
        buf = dec.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        base_img = np.frombuffer(buf, np.uint8).reshape(vh, VIDEO_W, 3)
        canvas = np.full((out_h, out_w, 3), BG, np.uint8)
        canvas[:HEADER_H] = header(out_w, stem, i, fps, n_total)
        for k, r in enumerate(rows):
            top = HEADER_H + k * (vh + GAP)
            img = base_img.copy()
            draw_boxes(img, r["dets"].get(i, []), r["states"], i, ACCENT[k % 2])
            canvas[top:top + vh, :VIDEO_W] = img
            canvas[top:top + vh, VIDEO_W + GAP:] = build_panel(
                vh, labels.get(r["model"], r["model"]), r["stats"].get(i, blank),
                series[k], ids_series[k], i, ACCENT[k % 2])
        enc.stdin.write(canvas.tobytes())
        i += 1
        if i % 300 == 0:
            print(f"  {stem}: {i}/{n_total} frames", flush=True)
    enc.stdin.close(); enc.wait(); dec.wait()
    print(f"{stem}: saved {out}  ({i} frames, {out_w}x{out_h})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--clips-dir", default="more_recordings_for_training")
    ap.add_argument("--models", nargs="+", default=["yolo26s", "visdrone_v1"])
    ap.add_argument("--tracker", default="botsort_60fps")
    ap.add_argument("--labels", nargs="+", default=None, help="panel titles, one per model")
    args = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found: brew install ffmpeg")
    out_dir = Path("runs/videos"); out_dir.mkdir(parents=True, exist_ok=True)
    labels = dict(LABELS)
    if args.labels:
        labels.update(zip(args.models, args.labels))
    if args.all:
        vids = sorted(p for p in Path(args.clips_dir).iterdir() if p.suffix.lower() in {".mp4", ".mov"})
    elif args.video:
        vids = [Path(args.video)]
    else:
        sys.exit("give --video PATH or --all")
    for v in vids:
        render(v, args.models, args.tracker, out_dir, labels)


if __name__ == "__main__":
    main()
