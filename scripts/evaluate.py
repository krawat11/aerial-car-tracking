"""
Score a trained model on VisDrone validation, split by camera angle.

    uv run python scripts/evaluate.py --model runs/train/visdrone_v1/weights/best.pt

mAP50 = accuracy counting a detection as right if it overlaps the true box by 50%+.
mAP50-95 = the stricter average over overlap thresholds 50%..95% (rewards tight boxes).
Precision = share of detections that are real vehicles (low -> false alarms).
Recall = share of real vehicles that were found (low -> missed cars).
"""
import argparse
import torch
from ultralytics import YOLO

SETS = [("all val", "configs/visdrone_vehicle_balanced.yaml"),
        ("straight-down", "configs/visdrone_val_nadir.yaml"),
        ("angled", "configs/visdrone_val_angled.yaml")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(args.model)
    rows = []
    for name, cfg in SETS:
        m = model.val(data=cfg, imgsz=args.imgsz, batch=args.batch, device=device,
                      plots=False, verbose=False).box
        rows.append((name, m.map50, m.map, m.mp, m.mr))
    print(f"\n{'subset':<15}{'mAP50':>8}{'mAP50-95':>10}{'precision':>11}{'recall':>8}")
    for name, a, b, p, r in rows:
        print(f"{name:<15}{a:>8.3f}{b:>10.3f}{p:>11.3f}{r:>8.3f}")


if __name__ == "__main__":
    main()
