"""L2's tile set, re-exposed for L3 without touching L2's folder.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.build_datasets

Nothing is re-tiled or re-labelled: every image is a **hard link** to L2's
JPEG tile (same volume, no copy) and every label file is L2's YOLO box label,
byte for byte.  Two layouts:

* ``dataset_yolo/`` -- L2's own layout (``images/<view>``, ``labels/<view>``)
  and L2's fold lists (``holdN_train.txt`` / ``holdN_val.txt``) rewritten to
  the L3 paths, so ultralytics' label cache lands here, not in L2's folder;
* ``dataset_rfdetr/<fold>/`` -- the Roboflow-YOLO layout RF-DETR reads:
  ``train/`` = the fold's training tiles, ``valid/`` = L2's 150 random
  *training* tiles (RF-DETR's per-epoch COCO eval needs one; nothing is
  selected on it -- the last epoch's weights are the model).  L2's
  ``build_rfdetr`` wrote polygons for the segmentation preview; L3 fine-tunes
  RF-DETR's *detection* checkpoints, so the boxes are the labels -- the same
  ones YOLO26 trains on.

Fold ``all`` (every tile; ``valid`` = 150 of them, seeded as L2) is the
production candidate's training set.

``tiles_hash.json``: SHA-256 over the sorted (tile name, image bytes, label
bytes) of each fold's training tiles.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402
from experiments.l2_detector import data as D  # noqa: E402

L2DS = env.L2OUT / "dataset"
YDS = env.OUT / "dataset_yolo"
RF = env.OUT / "dataset_rfdetr"
FOLDS = ("hold13", "hold24", "hold33")


def label_of(img: Path) -> Path:
    # .../images/<view>/<name>.jpg -> .../labels/<view>/<name>.txt
    return img.parents[2] / "labels" / img.parent.name / f"{img.stem}.txt"


def tiles_hash(images: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(images, key=lambda q: q.name):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
        h.update(hashlib.sha256(label_of(p).read_bytes()).digest())
    return h.hexdigest()


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def to_l3(p: Path) -> Path:
    return YDS / "images" / p.parent.name / p.name


def yolo_fold(fold: str, train: list[Path], val: list[Path]) -> None:
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(D.DET_CLASSES))
    (YDS / f"{fold}_train.txt").write_text(
        "\n".join(to_l3(p).as_posix() for p in train) + "\n", encoding="utf-8")
    (YDS / f"{fold}_val.txt").write_text(
        "\n".join(to_l3(p).as_posix() for p in val) + "\n", encoding="utf-8")
    (YDS / f"{fold}.yaml").write_text(
        f"path: {YDS.as_posix()}\ntrain: {fold}_train.txt\nval: {fold}_val.txt\n"
        f"names:\n{names}", encoding="utf-8")


def rf_fold(root: Path, train: list[Path], val: list[Path]) -> None:
    for split, items in (("train", train), ("valid", val)):
        for src in items:
            link(src, root / split / "images" / src.name)
            lab = root / split / "labels" / f"{src.stem}.txt"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_bytes(label_of(src).read_bytes())
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(D.DET_CLASSES))
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: train/images\nval: valid/images\n"
        f"nc: {len(D.DET_CLASSES)}\nnames:\n{names}", encoding="utf-8")


def main() -> int:
    manifest = json.loads((L2DS / "manifest.json").read_text(encoding="utf-8"))
    all_imgs = [Path(m["image"]) for m in manifest]
    missing = [p for p in all_imgs if not p.exists() or not label_of(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} L2 tiles or labels missing, e.g. {missing[:3]}")
    # the YOLO layout: hard-linked images, copied labels
    for p in all_imgs:
        link(p, to_l3(p))
        lab = YDS / "labels" / p.parent.name / f"{p.stem}.txt"
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_bytes(label_of(p).read_bytes())
    splits = {}
    for fold in FOLDS:
        train = [Path(p) for p in (L2DS / f"{fold}_train.txt").read_text(encoding="utf-8").split()]
        val = [Path(p) for p in (L2DS / f"{fold}_val.txt").read_text(encoding="utf-8").split()]
        splits[fold] = (train, val)
    rng = random.Random(0)
    splits["all"] = (all_imgs, rng.sample(all_imgs, 150))
    hashes = {}
    for fold, (train, val) in splits.items():
        yolo_fold(fold, train, val)
        rf_fold(RF / fold, train, val)
        hashes[fold] = {"tiles": len(train), "sha256": tiles_hash(train)}
        print(f"  {fold}: {len(train)} train tiles, {len(val)} valid", flush=True)
    (env.OUT / "tiles_hash.json").write_text(json.dumps(hashes, indent=1), encoding="utf-8")
    print(json.dumps(hashes, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
