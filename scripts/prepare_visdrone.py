"""
Download VisDrone2019-DET and convert it to a single-class "vehicle" YOLO dataset.

Why one class: the COCO model kept flipping between "car" and "truck" (14% of
clip 1 detections were "truck"). For tracking we only care that it's a vehicle,
so car, van, truck and bus are merged. Pedestrians, bikes, tricycles and
motorbikes are dropped (they become background).

Usage:
    uv run python scripts/prepare_visdrone.py            # train + val (~1.5 GB download)
    uv run python scripts/prepare_visdrone.py --splits val   # small test of the script
"""
import argparse, shutil, urllib.request, zipfile
from pathlib import Path
from PIL import Image

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{}.zip"
SPLIT_NAMES = {"train": "train", "val": "val", "test": "test-dev"}
# VisDrone category IDs: 4 car, 5 van, 6 truck, 9 bus  -> our class 0 "vehicle"
VEHICLE_CATS = {4, 5, 6, 9}

ROOT = Path("data/visdrone")


def download(split):
    zip_path = ROOT / "downloads" / f"VisDrone2019-DET-{SPLIT_NAMES[split]}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        url = URL.format(SPLIT_NAMES[split])
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, zip_path)
    extract_dir = ROOT / "downloads" / split
    if not extract_dir.exists():
        print(f"Extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    # the zip contains one top-level folder with images/ and annotations/
    return next(p for p in extract_dir.iterdir() if (p / "annotations").exists())


def convert(split, src):
    img_out = ROOT / split / "images"
    lbl_out = ROOT / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n_img = n_box = n_empty = 0
    for ann in sorted((src / "annotations").glob("*.txt")):
        img = src / "images" / f"{ann.stem}.jpg"
        if not img.exists():
            continue
        W, H = Image.open(img).size
        lines = []
        for row in ann.read_text().strip().splitlines():
            v = row.strip().rstrip(",").split(",")
            if len(v) < 6:
                continue
            x, y, w, h, score, cat = map(int, v[:6])
            if score == 0 or cat not in VEHICLE_CATS or w <= 1 or h <= 1:
                continue  # score 0 = "ignored region" in VisDrone
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        (lbl_out / ann.name).write_text("\n".join(lines) + ("\n" if lines else ""))
        dst = img_out / img.name
        if not dst.exists():
            shutil.move(str(img), dst)
        n_img += 1; n_box += len(lines); n_empty += not lines
    print(f"{split}: {n_img} images, {n_box} vehicle boxes, {n_empty} images with no vehicles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=list(SPLIT_NAMES))
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()
    for split in args.splits:
        convert(split, download(split))
    if not args.keep_zips:
        shutil.rmtree(ROOT / "downloads", ignore_errors=True)

    yaml = Path("configs/visdrone_vehicle.yaml")
    yaml.parent.mkdir(exist_ok=True)
    yaml.write_text(
        f"# VisDrone2019-DET, merged to one class. Made by scripts/prepare_visdrone.py\n"
        f"path: {ROOT.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"names:\n  0: vehicle\n")
    print(f"Wrote {yaml}")


if __name__ == "__main__":
    main()
