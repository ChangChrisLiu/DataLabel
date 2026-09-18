"""Task 4 -- is a DINOv3 backbone worth pursuing for the trained assist model?

Deliberately small: extract dense patch features from DINOv3 ViT-L/16 on a
handful of frames, fit a plain multinomial logistic regression on the patch
features of five desktops, and test on a held-out sixth. If frozen DINOv3
features already separate motherboard / screw / RAM / PSU / connector with a
linear probe, a DINOv3-backbone segmentation head is worth a real experiment;
if they do not, YOLO26 / RF-DETR remain the better bet.

This is a *feasibility signal*, not a segmentation benchmark. The probe is
per-patch (16 px cells at the model's input scale), there is no decoder, no
fine-tuning and no CRF, so the absolute mIoU is a floor, not a forecast.

Patch labels come from the reference masks: a patch takes the class whose mask
covers most of it, and ``background`` when no reference mask covers >=50 %.
Because the reference is incomplete, "background" really means "not drawn",
which is the single biggest caveat on these numbers.

Run::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.sam_compare.run_dinov3
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import numpy as np

from experiments.sam_compare import _env, common
from experiments.sam_compare.store import Ref, load_refs

#: Probe classes, **most specific first**. A screw sits on the motherboard, so a
#: patch covered by both must be labelled "screw" -- a plain argmax over
#: coverage would hand every small part to the big part underneath it and the
#: probe would never see a screw at all.
CLASSES = {
    "screw": lambda s: "screw" in s and "cover" not in s and "bracket" not in s,
    "connector": lambda s: "connector" in s,
    "ram module": lambda s: s == "ram module",
    "psu": lambda s: "power supply" in s,
    "motherboard": lambda s: s == "motherboard",
}
CLASS_NAMES = ["background", *CLASSES]
#: Coverage a class needs to claim a patch. Small parts get a lower bar: at the
#: native scale a screw head is 20-30 px, i.e. barely two 16 px patches wide.
COVER = {"screw": 0.30, "connector": 0.30, "ram module": 0.40,
         "psu": 0.50, "motherboard": 0.50}

#: Input side fed to DINOv3 (multiple of the 16 px patch size). Kept at the
#: frame's native 1600 px: at 1024 a screw is smaller than one patch.
INPUT_SIDE = 1600
FALLBACK_SIDE = 1024
PATCH = 16
#: Frames per desktop, and how many desktops take part.
FRAMES_PER_DESKTOP = 4
N_DESKTOPS = 6
SEED = 20260918


def class_of(label: str) -> Optional[str]:
    low = label.lower().strip()
    for name, pred in CLASSES.items():
        if pred(low):
            return name
    return None


def _frame_classes(refs: list[Ref]) -> dict[tuple[int, int], set[str]]:
    out: dict[tuple[int, int], set[str]] = {}
    for ref in refs:
        name = class_of(ref.label)
        if name:
            out.setdefault(ref.frame, set()).add(name)
    return out


def pick_frames(refs: list[Ref]) -> tuple[dict[int, list[tuple[int, int]]], int]:
    """Frames per desktop chosen for *class diversity*, plus the test desktop.

    Ranking by raw annotation count picked frames full of one big part, which
    left the held-out desktop with no screw or RAM patches at all and made those
    rows of the probe empty. Frames are ranked by how many distinct probe
    classes they carry instead, and the test desktop is the one whose selection
    covers the most classes.
    """
    per_frame = _frame_classes(refs)
    by_desktop: dict[int, list[tuple[int, int]]] = {}
    for frame, classes in per_frame.items():
        by_desktop.setdefault(frame[0], []).append(frame)

    chosen: dict[int, list[tuple[int, int]]] = {}
    coverage: dict[int, int] = {}
    for desktop, frames in by_desktop.items():
        ranked = sorted(frames, key=lambda f: (-len(per_frame[f]), f))
        picked: list[tuple[int, int]] = []
        seen: set[str] = set()
        # greedy set cover, then fill up with the next richest frames
        for frame in ranked:
            if len(picked) >= FRAMES_PER_DESKTOP:
                break
            if per_frame[frame] - seen:
                picked.append(frame)
                seen |= per_frame[frame]
        for frame in ranked:
            if len(picked) >= FRAMES_PER_DESKTOP:
                break
            if frame not in picked:
                picked.append(frame)
        chosen[desktop] = sorted(picked)
        coverage[desktop] = len(seen)

    top = sorted(chosen, key=lambda d: (-coverage[d], d))[:N_DESKTOPS]
    test_desktop = max(top, key=lambda d: (coverage[d], -d))
    return {d: chosen[d] for d in sorted(top)}, test_desktop


def patch_labels(refs_of_frame: list[Ref], grid: int) -> np.ndarray:
    """``grid x grid`` class ids: most specific class whose coverage clears its bar."""
    out = np.zeros((grid, grid), np.int64)
    claimed = np.zeros((grid, grid), bool)
    for name in CLASSES:  # most specific first
        cover = np.zeros((grid, grid), np.float32)
        for ref in refs_of_frame:
            if class_of(ref.label) != name:
                continue
            small = common.cv2.resize(ref.mask.astype(np.float32), (grid, grid),
                                      interpolation=common.cv2.INTER_AREA)
            cover = np.maximum(cover, small)
        hit = (cover >= COVER[name]) & ~claimed
        out[hit] = CLASS_NAMES.index(name)
        claimed |= hit
    return out


class Dinov3Features:
    """Frozen DINOv3 ViT-L/16 dense patch features via ``transformers``."""

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.device = device
        t0 = time.perf_counter()
        self._proc = AutoImageProcessor.from_pretrained(_env.DINOV3_REPO)
        self._model = AutoModel.from_pretrained(_env.DINOV3_REPO).to(device).eval()
        self.load_s = time.perf_counter() - t0
        self.n_register = int(getattr(self._model.config, "num_register_tokens", 0) or 0)

    def dense(self, image_rgb: np.ndarray, side: int = INPUT_SIDE) -> tuple[np.ndarray, int]:
        """Return ``(grid*grid, dim)`` patch features and the grid side.

        Falls back to a smaller input on CUDA OOM: the GPU is shared, and a
        coarser grid is far better than losing the frame.
        """
        try:
            return self._dense(image_rgb, side)
        except self._torch.cuda.OutOfMemoryError:
            self._torch.cuda.empty_cache()
            return self._dense(image_rgb, FALLBACK_SIDE)

    def _dense(self, image_rgb: np.ndarray, side: int) -> tuple[np.ndarray, int]:
        img = common.cv2.resize(image_rgb, (side, side),
                                interpolation=common.cv2.INTER_AREA)
        inputs = self._proc(images=img, return_tensors="pt", do_resize=False,
                            do_center_crop=False).to(self.device)
        with self._torch.inference_mode():
            out = self._model(**inputs)
        tokens = out.last_hidden_state[0]
        grid = side // PATCH
        # drop CLS + register tokens; what is left is the patch grid
        tokens = tokens[tokens.shape[0] - grid * grid:]
        return tokens.float().cpu().numpy(), grid


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-patches-per-frame", type=int, default=6000)
    ap.add_argument("--max-iter", type=int, default=1200,
                    help="lbfgs iterations; 300 did not converge on 1024-d features")
    args = ap.parse_args(argv)

    refs = load_refs()
    chosen, test_desktop = pick_frames(refs)
    frames = [f for fs in chosen.values() for f in fs]
    print(f"[dinov3] desktops {sorted(chosen)}, {len(frames)} frames, "
          f"test on D{test_desktop}", flush=True)

    by_frame: dict[tuple[int, int], list[Ref]] = {}
    for ref in refs:
        if ref.frame in set(frames):
            by_frame.setdefault(ref.frame, []).append(ref)

    extractor = Dinov3Features()
    print(f"[dinov3] ViT-L/16 loaded in {extractor.load_s:.1f} s "
          f"({extractor.n_register} register tokens)", flush=True)

    rng = np.random.default_rng(SEED)
    feats: dict[int, list[np.ndarray]] = {}
    labels: dict[int, list[np.ndarray]] = {}
    t0 = time.perf_counter()
    for desktop, fs in chosen.items():
        for frame in fs:
            image = common.load_frame_rgb(*frame)
            if image is None:
                continue
            x, grid = extractor.dense(image)
            y = patch_labels(by_frame.get(frame, []), grid).reshape(-1)
            # Subsample, but keep every non-background patch: small parts are
            # rare enough that uniform sampling would delete most screws.
            fg = np.flatnonzero(y != 0)
            bg = np.flatnonzero(y == 0)
            n_bg = max(0, args.max_patches_per_frame - len(fg))
            idx = np.concatenate([fg, rng.permutation(bg)[:n_bg]])
            feats.setdefault(desktop, []).append(x[idx])
            labels.setdefault(desktop, []).append(y[idx])
    print(f"[dinov3] features in {time.perf_counter() - t0:.0f} s", flush=True)

    desktops = sorted(feats)
    train_desktops = [d for d in desktops if d != test_desktop]
    xtr = np.concatenate([np.concatenate(feats[d]) for d in train_desktops])
    ytr = np.concatenate([np.concatenate(labels[d]) for d in train_desktops])
    xte = np.concatenate(feats[test_desktop])
    yte = np.concatenate(labels[test_desktop])
    print(f"[dinov3] train {xtr.shape} on D{train_desktops}, "
          f"test {xte.shape} on D{test_desktop}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=args.max_iter, C=1.0,
                             class_weight="balanced", n_jobs=-1)
    t0 = time.perf_counter()
    clf.fit(scaler.transform(xtr), ytr)
    fit_s = time.perf_counter() - t0
    pred = clf.predict(scaler.transform(xte))

    acc = float((pred == yte).mean())
    per_class = {}
    ious = []
    for i, name in enumerate(CLASS_NAMES):
        inter = int(((pred == i) & (yte == i)).sum())
        union = int(((pred == i) | (yte == i)).sum())
        support = int((yte == i).sum())
        iou_i = inter / union if union else float("nan")
        per_class[name] = {"iou": iou_i, "support": support,
                           "recall": inter / support if support else float("nan")}
        if support:
            ious.append(iou_i)
    miou = float(np.nanmean(ious))

    result = {
        "model": _env.DINOV3_REPO,
        "input_side": INPUT_SIDE, "patch": PATCH,
        "train_desktops": train_desktops, "test_desktop": test_desktop,
        "n_train_patches": int(len(ytr)), "n_test_patches": int(len(yte)),
        "fit_seconds": round(fit_s, 1),
        "converged": bool(np.all(np.asarray(clf.n_iter_) < args.max_iter)),
        "patch_accuracy": round(acc, 4),
        "patch_mIoU": round(miou, 4),
        "per_class": {k: {kk: (None if vv != vv else round(float(vv), 4))
                          if isinstance(vv, float) else vv for kk, vv in v.items()}
                      for k, v in per_class.items()},
    }
    out = _env.OUT_ROOT / "dinov3_probe.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[dinov3] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
