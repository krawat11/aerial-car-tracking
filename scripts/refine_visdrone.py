"""
Run after prepare_visdrone.py. Three improvements to the VisDrone vehicle dataset:

1. Grey out "ignored regions": areas the VisDrone annotators marked as too crowded or
   unclear to label. Left alone, a correct detection there gets punished as a false
   positive. Labelled vehicles that overlap a region are kept visible.
2. Tag every image by camera angle, using perspective: in a tilted shot, cars at the
   top of the frame look much smaller than cars at the bottom; straight down, they
   are about the same size. ratio = median box area (bottom third) / (top third).
3. Oversample straight-down images (VisDrone is mostly tilted) by listing them more
   than once in the training list, and write per-angle validation subsets.

Usage:
    uv run python scripts/refine_visdrone.py
"""
import argparse, csv, shutil, statistics as st, urllib.request, zipfile
from pathlib import Path
from PIL import Image, ImageDraw

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{}.zip"
ROOT = Path("data/visdrone")
GREY = (114, 114, 114)  # same grey YOLO uses for padding
BUCKETS = [("nadir", 0, 1.5), ("slight", 1.5, 3), ("angled", 3, 6), ("steep", 6, float("inf"))]


# ---------- 1. ignored regions ----------
def mask_ignored(split):
    marker = ROOT / split / ".masked"
    if marker.exists():
        print(f"{split}: ignored regions already masked, skipping")
        return
    zip_path = ROOT / "downloads" / f"VisDrone2019-DET-{split}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading {split} zip again for its annotation files (images already converted)")
        urllib.request.urlretrieve(URL.format(split), zip_path)
    n_img = n_reg = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if "/annotations/" not in name or not name.endswith(".txt"):
                continue
            stem = Path(name).stem
            img_path = ROOT / split / "images" / f"{stem}.jpg"
            if not img_path.exists():
                continue
            ignored, keep = [], []
            for row in z.read(name).decode().strip().splitlines():
                v = row.strip().rstrip(",").split(",")
                if len(v) < 6:
                    continue
                x, y, w, h, score, cat = map(int, v[:6])
                box = (x, y, x + w, y + h)
                if cat == 0:              # category 0 = ignored region
                    ignored.append(box)
                elif score == 1:
                    keep.append(box)      # any labelled object: don't paint over it
            if not ignored:
                continue
            im = Image.open(img_path).convert("RGB")
            original = im.copy()
            d = ImageDraw.Draw(im)
            for b in ignored:
                d.rectangle(b, fill=GREY)
            for b in keep:                # restore labelled objects on top
                im.paste(original.crop(b), b[:2])
            im.save(img_path, quality=95)
            n_img += 1; n_reg += len(ignored)
    marker.touch()
    for c in (ROOT / split).glob("*.cache"):
        c.unlink()
    print(f"{split}: greyed out {n_reg} ignored regions in {n_img} images")


# ---------- 2. angle tags ----------
def angle_ratio(label_file):
    boxes = [list(map(float, l.split()[1:])) for l in label_file.read_text().splitlines() if l.strip()]
    if len(boxes) < 6:
        return None
    top = [w * h for x, y, w, h in boxes if y < 0.4]
    bot = [w * h for x, y, w, h in boxes if y > 0.6]
    if len(top) < 2 or len(bot) < 2:
        return None
    return st.median(bot) / st.median(top)


def bucket(r):
    if r is None:
        return "unknown"
    return next(name for name, lo, hi in BUCKETS if lo <= r < hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nadir-x", type=int, default=3, help="times each straight-down image is listed")
    ap.add_argument("--slight-x", type=int, default=2, help="times each slightly-tilted image is listed")
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()

    for split in ("train", "val"):
        mask_ignored(split)
    if not args.keep_zips:
        shutil.rmtree(ROOT / "downloads", ignore_errors=True)

    repeats = {"nadir": args.nadir_x, "slight": args.slight_x}
    rows, train_list, val_subsets = [], [], {"nadir": [], "angled": []}
    for split in ("train", "val"):
        for img in sorted((ROOT / split / "images").glob("*.jpg")):
            r = angle_ratio(ROOT / split / "labels" / f"{img.stem}.txt")
            b = bucket(r)
            rows.append([split, img.name, "" if r is None else round(r, 2), b])
            rel = f"./{split}/images/{img.name}"
            if split == "train":
                train_list += [rel] * repeats.get(b, 1)
            elif b in ("nadir", "slight"):
                val_subsets["nadir"].append(rel)
            elif b in ("angled", "steep"):
                val_subsets["angled"].append(rel)

    with open(ROOT / "angles.csv", "w", newline="") as f:
        csv.writer(f).writerows([["split", "image", "ratio", "bucket"]] + rows)
    (ROOT / "train_balanced.txt").write_text("\n".join(train_list) + "\n")
    for k, v in val_subsets.items():
        (ROOT / f"val_{k}.txt").write_text("\n".join(v) + "\n")

    for split in ("train", "val"):
        counts = {}
        for s, _, _, b in rows:
            if s == split:
                counts[b] = counts.get(b, 0) + 1
        print(f"{split} by angle: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in
              ["nadir", "slight", "angled", "steep", "unknown"]))
    n_train = sum(1 for r in rows if r[0] == "train")
    nadir_names = {r[1] for r in rows if r[0] == "train" and r[3] == "nadir"}
    n_nadir = sum(1 for p in train_list if Path(p).name in nadir_names)
    print(f"Balanced training list: {len(train_list)} entries from {n_train} images "
          f"(straight-down share {n_nadir / len(train_list):.0%}, "
          f"was {len(nadir_names) / n_train:.0%})")
    print(f"Validation subsets: {len(val_subsets['nadir'])} straight-down/slight, "
          f"{len(val_subsets['angled'])} angled (rest have too few vehicles to judge)")

    cfg = Path("configs")
    header = f"path: {ROOT.resolve()}\nnames:\n  0: vehicle\n"
    (cfg / "visdrone_vehicle_balanced.yaml").write_text(
        "# Training config: masked images, straight-down oversampled\n"
        + header + "train: train_balanced.txt\nval: val/images\n")
    (cfg / "visdrone_val_nadir.yaml").write_text(
        "# Eval only: straight-down + slight-tilt validation images\n"
        + header + "train: train_balanced.txt\nval: val_nadir.txt\n")
    (cfg / "visdrone_val_angled.yaml").write_text(
        "# Eval only: tilted validation images\n"
        + header + "train: train_balanced.txt\nval: val_angled.txt\n")
    print("Wrote configs/visdrone_vehicle_balanced.yaml, visdrone_val_nadir.yaml, visdrone_val_angled.yaml")


if __name__ == "__main__":
    main()
