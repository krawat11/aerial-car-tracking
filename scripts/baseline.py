"""
Baseline test: pretrained YOLO (COCO weights) + BoT-SORT on your own drone clip.
No training. The point is to get a "before" number that later steps must beat.

Usage:
    python baseline.py --video clip.mp4
    python baseline.py --video clip.mp4 --tracker bytetrack.yaml --imgsz 640
"""
import argparse, csv, time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# Keep only vehicle classes, looked up by name so this works for both the
# COCO model (car, bus, truck, motorcycle) and our one-class "vehicle" model.
VEHICLE_NAMES = {"car", "van", "truck", "bus", "motorcycle", "vehicle"}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():   # Apple Silicon GPU
        return "mps"
    return "cpu"


def load_model(name):
    try:
        return YOLO(name)
    except Exception as e:
        print(f"Couldn't load {name} ({e}); falling back to yolo11s.pt")
        return YOLO("yolo11s.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)  # higher = better for small cars, slower
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out-dir", default="runs/baseline")
    ap.add_argument("--no-video", action="store_true", help="skip writing the annotated video (saves ~1 GB per 30 s of 4K)")
    args = ap.parse_args()

    device = pick_device()
    model = load_model(args.model)
    vehicle_ids = [i for i, n in model.names.items() if n in VEHICLE_NAMES]
    print(f"Device: {device} | model: {args.model} | tracker: {args.tracker} | imgsz: {args.imgsz}")

    cap = cv2.VideoCapture(args.video)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    stem = Path(args.video).stem
    mp = Path(args.model)
    model_name = mp.parent.parent.name if mp.parent.name == "weights" else mp.stem
    tag = f"{stem}_{model_name}_{Path(args.tracker).stem}_{args.imgsz}"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    writer = None if args.no_video else cv2.VideoWriter(
        str(out_dir / f"{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    rows, all_ids, infer_ms, dets = [], set(), [], []
    track_len = {}  # track ID -> number of frames it appeared in
    t0 = time.perf_counter()

    results = model.track(source=args.video, tracker=args.tracker, classes=vehicle_ids,
                          conf=args.conf, imgsz=args.imgsz, device=device,
                          stream=True, persist=True, verbose=False)

    for i, r in enumerate(results):
        n = len(r.boxes)
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else []
        all_ids.update(ids)
        for tid in ids:
            track_len[tid] = track_len.get(tid, 0) + 1
        if r.boxes.id is not None:
            for tid, c, cf, b in zip(ids, r.boxes.cls.int().tolist(),
                                     r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                dets.append([i, tid, c, round(cf, 3)] + [round(v, 1) for v in b])
        mean_conf = float(r.boxes.conf.mean()) if n else 0.0
        infer_ms.append(sum(r.speed.values()))
        rows.append([i, n, len(ids), round(mean_conf, 3)])
        if writer:
            writer.write(r.plot())
        if i % 100 == 0:
            print(f"frame {i}: {n} vehicles")

    if writer:
        writer.release()
    wall = time.perf_counter() - t0
    frames = len(rows)

    with open(out_dir / f"{tag}_per_frame.csv", "w", newline="") as f:
        csv.writer(f).writerows([["frame", "detections", "tracked", "mean_conf"]] + rows)

    with open(out_dir / f"{tag}_tracks.csv", "w", newline="") as f:
        csv.writer(f).writerows([["frame", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"]] + dets)

    lens = sorted(track_len.values())
    one_sec = int(round(fps_in))
    short = sum(1 for n in lens if n < one_sec)
    median = lens[len(lens) // 2] if lens else 0

    avg_det = sum(r[1] for r in rows) / max(frames, 1)
    avg_conf = sum(r[3] for r in rows if r[1]) / max(sum(1 for r in rows if r[1]), 1)
    print("\n===== BASELINE SUMMARY =====")
    print(f"Frames processed:          {frames}")
    print(f"Avg vehicles per frame:    {avg_det:.1f}")
    print(f"Avg confidence:            {avg_conf:.2f}")
    print(f"Unique track IDs:          {len(all_ids)}")
    print(f"  lasting >= 1 s:          {len(lens) - short}")
    print(f"  lasting < 1 s:           {short}  (fragments / flicker)")
    print(f"Median track length:       {median} frames ({median / fps_in:.1f} s)")
    print(f"Model speed (per frame):   {sum(infer_ms)/max(frames,1):.1f} ms "
          f"(~{1000/max(sum(infer_ms)/max(frames,1),1e-6):.0f} FPS)")
    print(f"End-to-end incl. drawing:  {frames/wall:.1f} FPS")
    print(f"Saved to {out_dir}/: {tag}" + ("" if args.no_video else ".mp4,") + " _per_frame.csv, _tracks.csv")


if __name__ == "__main__":
    main()
