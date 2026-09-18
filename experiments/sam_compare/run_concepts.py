"""Task 3 -- SAM 3 open-vocabulary *text* prompts against the reference masks.

Answers the practical question "can SAM 3 propose all the screws on a frame by
itself?", which SAM 2.1 cannot do at all.

Protocol per frame and concept:

* run the text prompt, keep detections above a low score floor,
* greedily match detections (highest score first) to the reference masks whose
  Label Studio label maps onto that concept, at IoU >= 0.5, one-to-one,
* report recall and precision at several score thresholds.

**Precision here is a lower bound.** The reference only covers the parts the
annotators chose to draw, so a correct detection of an undrawn screw counts as
a false positive. Recall is the trustworthy number.

Two framings are measured, because a 20 px screw in a 1600 px frame is only
~12 px once SAM 3 resizes to its 1008 px input:

* ``full``  -- the whole frame in one shot,
* ``tile2`` -- a 2x2 grid of overlapping tiles, detections merged and de-duped,
  which is what the annotation tool could actually afford to run.

Every mask is carried as a bbox-local :class:`Inst`, never as a 1600x1600
array: matching is O(detections x references) and full-frame IoU at that count
would dominate the runtime by orders of magnitude.

Run::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.sam_compare.run_concepts
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from experiments.sam_compare import _env, common
from experiments.sam_compare.store import Ref, load_refs

#: Text prompt -> predicate deciding which Label Studio labels count as truth.
#: Kept as predicates (not a hard-coded name list) because the export's label
#: names are long and inconsistently suffixed; the resolved sets are printed
#: and saved to ``concept_labels.json`` so the mapping is auditable.
CONCEPTS: dict[str, object] = {
    "screw": lambda s: "screw" in s and "cover" not in s and "bracket" not in s,
    "RAM module": lambda s: s.strip() == "ram module",
    "connector": lambda s: "connector" in s,
    "fan": lambda s: "fan" in s and not any(
        w in s for w in ("screw", "connector", "bracket")),
    "power supply": lambda s: "power supply" in s,
    "hard drive": lambda s: "hard disk drive" in s or "solid state drive" in s,
    "cable": lambda s: False,  # no cable label exists in the export
}

#: Score floor for detections we keep at all; thresholds are applied afterwards.
SCORE_FLOOR = 0.25
THRESHOLDS = (0.25, 0.4, 0.55)
#: Cap on detections kept per prompt (per tile, for the tiled framing).
MAX_DETECTIONS = 120
#: IoU above which two detections from different tiles are the same object.
DEDUP_IOU = 0.6
N_FRAMES = 60
SEED = 20260918

CSV_FIELDS = [
    "desktop", "step", "concept", "framing", "threshold",
    "n_ref", "n_pred", "n_match", "recall", "precision_lb", "ms",
]


# --------------------------------------------------------------------------- #
# bbox-local instances
# --------------------------------------------------------------------------- #
@dataclass
class Inst:
    """One mask stored as ``bbox`` + a bbox-sized boolean crop."""

    x0: int
    y0: int
    x1: int
    y1: int
    mask: np.ndarray
    area: int
    score: float = 0.0


def inst_from_mask(mask: np.ndarray, score: float = 0.0,
                   off: tuple[int, int] = (0, 0)) -> Optional[Inst]:
    """Build an :class:`Inst` from a (possibly tile-local) boolean mask."""
    bb = common.mask_bbox(mask)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    crop = np.ascontiguousarray(mask[y0:y1, x0:x1])
    return Inst(x0 + off[0], y0 + off[1], x1 + off[0], y1 + off[1],
                crop, int(crop.sum()), float(score))


def inst_iou(a: Inst, b: Inst) -> float:
    """IoU of two instances, evaluated only on the overlapping region."""
    ox0, oy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ox1, oy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if ox0 >= ox1 or oy0 >= oy1:
        return 0.0
    sa = a.mask[oy0 - a.y0 : oy1 - a.y0, ox0 - a.x0 : ox1 - a.x0]
    sb = b.mask[oy0 - b.y0 : oy1 - b.y0, ox0 - b.x0 : ox1 - b.x0]
    inter = int(np.count_nonzero(sa & sb))
    if inter == 0:
        return 0.0
    return inter / (a.area + b.area - inter)


# --------------------------------------------------------------------------- #
# label mapping and frame choice
# --------------------------------------------------------------------------- #
def resolve_labels(labels: set[str]) -> dict[str, set[str]]:
    return {c: {lab for lab in labels if pred(lab.lower())}
            for c, pred in CONCEPTS.items()}


def pick_frames(refs: list[Ref], mapped: dict[str, set[str]], n: int) -> list[tuple[int, int]]:
    """``n`` frames that actually carry mapped labels, spread over desktops."""
    interesting: set[str] = set().union(*mapped.values())
    by_desktop: dict[int, set[tuple[int, int]]] = {}
    for ref in refs:
        if ref.label in interesting:
            by_desktop.setdefault(ref.desktop, set()).add(ref.frame)
    rng = np.random.default_rng(SEED)
    queues = []
    for desktop in sorted(by_desktop):
        frames = sorted(by_desktop[desktop])
        rng.shuffle(frames)
        queues.append(list(frames))
    picked: list[tuple[int, int]] = []
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop())
    return sorted(picked)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def detect_full(svc, frame: np.ndarray, concept: str) -> tuple[list[Inst], float]:
    out = svc.detect_text(frame, concept, threshold=SCORE_FLOOR)
    insts = []
    for m, s in zip(out["masks"][:MAX_DETECTIONS], out["scores"][:MAX_DETECTIONS]):
        inst = inst_from_mask(m, float(s))
        if inst is not None:
            insts.append(inst)
    return insts, out["ms"]


def detect_tiled(svc, frame: np.ndarray, concept: str, grid: int = 2,
                 overlap: int = 128) -> tuple[list[Inst], float]:
    """Run the prompt on an overlapping ``grid x grid`` tiling and de-duplicate."""
    h, w = frame.shape[:2]
    th, tw = h // grid, w // grid
    insts: list[Inst] = []
    total_ms = 0.0
    for gy in range(grid):
        for gx in range(grid):
            y0 = max(0, gy * th - overlap // 2)
            x0 = max(0, gx * tw - overlap // 2)
            y1 = min(h, y0 + th + overlap)
            x1 = min(w, x0 + tw + overlap)
            tile = np.ascontiguousarray(frame[y0:y1, x0:x1])
            out = svc.detect_text(tile, concept, threshold=SCORE_FLOOR)
            total_ms += out["ms"]
            for m, s in zip(out["masks"][:MAX_DETECTIONS], out["scores"][:MAX_DETECTIONS]):
                inst = inst_from_mask(m, float(s), off=(x0, y0))
                if inst is not None:
                    insts.append(inst)
    kept: list[Inst] = []
    for inst in sorted(insts, key=lambda i: -i.score):
        if all(inst_iou(inst, k) < DEDUP_IOU for k in kept):
            kept.append(inst)
    return kept, total_ms


def greedy_match(preds: list[Inst], gts: list[Inst], thr: float) -> int:
    """One-to-one matches at IoU >= 0.5, highest-scoring detection first."""
    if not gts:
        return 0
    taken: set[int] = set()
    matched = 0
    for pred in sorted((p for p in preds if p.score >= thr), key=lambda p: -p.score):
        best_j, best_iou = -1, 0.5
        for j, gt in enumerate(gts):
            if j in taken:
                continue
            score = inst_iou(pred, gt)
            if score >= best_iou:
                best_j, best_iou = j, score
        if best_j >= 0:
            taken.add(best_j)
            matched += 1
    return matched


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--framings", default="full,tile2")
    args = ap.parse_args(argv)

    refs = load_refs()
    mapped = resolve_labels({r.label for r in refs})
    print("[concepts] label mapping:")
    for concept, labels in mapped.items():
        print(f"  {concept:<14} <- {sorted(labels) if labels else '(no reference label)'}")

    frames = pick_frames(refs, mapped, args.frames)
    want = set(frames)
    by_frame: dict[tuple[int, int], list[Ref]] = {}
    for ref in refs:
        if ref.frame in want:
            by_frame.setdefault(ref.frame, []).append(ref)
    print(f"[concepts] {len(frames)} frames", flush=True)

    from experiments.sam_compare.sam3_service import Sam3Service

    svc = Sam3Service(device="cuda", autocast=True)
    print(f"[concepts] SAM 3 loaded in {svc.load_ms / 1000:.1f} s", flush=True)

    out_path = _env.OUT_ROOT / "concepts.csv"
    framings = args.framings.split(",")
    t0 = time.perf_counter()
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for n, key in enumerate(frames, 1):
            image = common.load_frame_rgb(*key)
            if image is None:
                continue
            gts_by_concept = {
                concept: [i for i in (inst_from_mask(r.mask)
                                      for r in by_frame.get(key, [])
                                      if r.label in labels) if i is not None]
                for concept, labels in mapped.items()
            }
            # framing outermost: the tiled pass reuses each tile's embedding
            # across all seven concepts instead of recomputing it seven times
            for framing in framings:
                fn = detect_full if framing == "full" else detect_tiled
                for concept in mapped:
                    try:
                        preds, ms = fn(svc, image, concept)
                    except Exception as exc:
                        print(f"[concepts] {key} {concept} {framing}: "
                              f"{type(exc).__name__}: {exc}", flush=True)
                        continue
                    gts = gts_by_concept[concept]
                    for thr in THRESHOLDS:
                        n_pred = sum(1 for p in preds if p.score >= thr)
                        n_match = greedy_match(preds, gts, thr)
                        writer.writerow({
                            "desktop": key[0], "step": key[1], "concept": concept,
                            "framing": framing, "threshold": thr,
                            "n_ref": len(gts), "n_pred": n_pred, "n_match": n_match,
                            "recall": round(n_match / len(gts), 5) if gts else "",
                            "precision_lb": round(n_match / n_pred, 5) if n_pred else "",
                            "ms": round(ms, 1),
                        })
            if n % 5 == 0:
                el = time.perf_counter() - t0
                print(f"[concepts] {n}/{len(frames)} frames, {el:.0f} s, "
                      f"ETA {el / n * (len(frames) - n):.0f} s", flush=True)

    (_env.OUT_ROOT / "concept_labels.json").write_text(
        json.dumps({k: sorted(v) for k, v in mapped.items()}, indent=1), encoding="utf-8")
    print(f"[concepts] wrote {out_path} in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
