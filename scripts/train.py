"""
Fine-tune YOLO26s to detect "vehicle" from above.

Speed test first (a slice of one epoch), to decide Mac vs cloud GPU:
    uv run python scripts/train.py --name speedtest --epochs 1 --fraction 0.05

Full run:
    uv run python scripts/train.py --name visdrone_v1

Results land in runs/train/<name>/ : weights/best.pt is the model to use,
results.csv / results.png show how accuracy changed per epoch.
"""
import argparse, time
from pathlib import Path

import torch
from ultralytics import YOLO


def pick_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/visdrone_vehicle_balanced.yaml")
    ap.add_argument("--model", default="yolo26s.pt")   # start from COCO weights, not from scratch
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1280)  # VisDrone images are ~1360-2000 px wide
    ap.add_argument("--batch", type=int, default=None,
                    help="images per step; default: auto-size on NVIDIA GPUs, 4 on Mac/CPU")
    ap.add_argument("--fraction", type=float, default=1.0)  # use part of the training set
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hours", type=float, default=None,
                    help="train for this many hours instead of a fixed epoch count (fits cloud session limits)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    if args.batch is None:
        args.batch = -1 if device == 0 else 4   # -1 = let Ultralytics fill ~60% of GPU memory
    print(f"Device: {device} | model: {args.model} | data: {args.data} | "
          f"imgsz {args.imgsz} | batch {args.batch} | epochs {args.epochs} | fraction {args.fraction}")

    project = Path("runs/train").resolve()
    if args.resume:
        model = YOLO(project / args.name / "weights" / "last.pt")
        model.train(resume=True)
        return

    model = YOLO(args.model)
    # time only the training part of each epoch (validation is timed separately)
    times = {"train": 0.0}
    model.add_callback("on_train_epoch_start", lambda t: times.__setitem__("t", time.perf_counter()))
    model.add_callback("on_train_epoch_end", lambda t: times.__setitem__("train", times["train"] + time.perf_counter() - times["t"]))
    t0 = time.perf_counter()
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        fraction=args.fraction, device=device, workers=args.workers,
        project=str(project), name=args.name, exist_ok=True,
        time=args.hours,    # if set, overrides epochs and fits the learning-rate schedule to the time
        patience=15,        # stop early if val accuracy hasn't improved for 15 epochs
        cos_lr=True,
        close_mosaic=10,
        plots=True,
    )
    mins = (time.perf_counter() - t0) / 60
    print(f"\nTraining took {mins:.1f} min (fraction {args.fraction})")
    if args.hours is None and (args.fraction < 1.0 or args.epochs == 1):
        train_min = times["train"] / 60 / args.epochs / args.fraction   # scales with dataset size
        val_min = (mins - times["train"] / 60) / args.epochs              # val set size is fixed
        per_epoch = train_min + val_min
        print(f"Estimate for one FULL epoch: {train_min:.0f} min training + {val_min:.1f} min validation "
              f"= {per_epoch:.0f} min -> 50 epochs = {per_epoch * 50 / 60:.1f} h (a guide, not a promise)")
    print(f"Best weights: {project / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
