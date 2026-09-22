"""
Watch the model track vehicles live in a window.

    uv run python scripts/watch.py --video data/processed/clip02_carpark.mp4
    uv run python scripts/watch.py --video data/processed/clip.mp4 --save    # also save a 1080p copy

Keys (click the video window first):  q = quit   space = pause/resume   s = save a screenshot
"""
import argparse, time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

VEHICLE_NAMES = {"car", "van", "truck", "bus", "motorcycle", "vehicle"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="models/visdrone_v1.pt")
    ap.add_argument("--tracker", default="configs/botsort_60fps.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=1600, help="window width in pixels")
    ap.add_argument("--save", action="store_true", help="also save the annotated video at 1080p")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(args.model)
    classes = [i for i, n in model.names.items() if n in VEHICLE_NAMES]
    name = f"{Path(args.video).stem} | {Path(args.model).stem}"

    writer = None
    out_dir = Path("runs/watch"); out_dir.mkdir(parents=True, exist_ok=True)
    fps_in = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 30

    ids_seen, paused, t_last = set(), False, time.perf_counter()
    results = model.track(source=args.video, tracker=args.tracker, classes=classes, conf=args.conf,
                          imgsz=args.imgsz, device=device, stream=True, persist=True, verbose=False)
    for i, r in enumerate(results):
        if r.boxes.id is not None:
            ids_seen.update(r.boxes.id.int().tolist())
        frame = r.plot(line_width=3, font_size=1.2)
        now = time.perf_counter(); fps = 1 / max(now - t_last, 1e-6); t_last = now
        text = f"frame {i}   vehicles {len(r.boxes)}   IDs so far {len(ids_seen)}   {fps:.0f} FPS"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

        if args.save:
            small = cv2.resize(frame, (1920, int(frame.shape[0] * 1920 / frame.shape[1])))
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / f"{Path(args.video).stem}_{Path(args.model).stem}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps_in, small.shape[1::-1])
            writer.write(small)

        h = int(frame.shape[0] * args.width / frame.shape[1])
        cv2.imshow(name, cv2.resize(frame, (args.width, h)))
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                shot = out_dir / f"{Path(args.video).stem}_{Path(args.model).stem}_f{i}.jpg"
                cv2.imwrite(str(shot), frame); print(f"Saved {shot}")
            if key == ord(" "):
                paused = not paused
            if not paused:
                break
        if key == ord("q"):
            break

    if writer:
        writer.release(); print(f"Saved video to {out_dir}/")
    cv2.destroyAllWindows()
    print(f"Distinct track IDs seen: {len(ids_seen)}")


if __name__ == "__main__":
    main()
