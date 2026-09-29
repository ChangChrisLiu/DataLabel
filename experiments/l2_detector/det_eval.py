"""Detector metrics on the held-out desktops, against the drafts.

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l2_detector.det_eval

Per class x view x held-out desktop, on every frame of that desktop-view that
has drafts (the frames the fold never saw):

* **AP50** -- all-point interpolated, detections at conf >= 0.001 greedily
  matched (by confidence) to same-class draft boxes at IoU >= 0.5;
* **R@0.10** -- recall of the drafts at conf >= 0.10, IoU >= 0.5;
* **hit@0.10** -- recall where "found" means a same-class detection whose box
  centre lies inside the draft's box (what the guess needs: a point on it);
* **FP/frame@0.10** -- detections at conf >= 0.10 that match no same-class
  draft (IoU < 0.3 and centre outside every draft box), per frame.  The drafts
  are incomplete (visible, unlabeled connectors and latches were seen in the
  tiles), so part of this is missing labels, not detector error.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l2_detector import env  # noqa: E402
from experiments.l2_detector import data as D  # noqa: E402

DETS = env.OUT / "detections"
LOW = 0.10


def iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-6)


def centre_in(dets: np.ndarray, gts: np.ndarray) -> np.ndarray:
    if len(dets) == 0 or len(gts) == 0:
        return np.zeros((len(dets), len(gts)), bool)
    cx = 0.5 * (dets[:, 0] + dets[:, 2])
    cy = 0.5 * (dets[:, 1] + dets[:, 3])
    return ((cx[:, None] >= gts[None, :, 0]) & (cx[:, None] < gts[None, :, 2])
            & (cy[:, None] >= gts[None, :, 1]) & (cy[:, None] < gts[None, :, 3]))


def ap_all_point(scores: list[float], tps: list[int], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(dets: dict, held, by) -> list[dict]:
    """``dets``: ``{"d:view:step": rows}`` of one fold, or of all three pooled."""
    fold = f"hold{held}"
    acc = defaultdict(lambda: {"scores": [], "tps": [], "n_gt": 0, "hit": 0,
                               "r50": 0, "fp": 0, "frames": 0, "gt_w": []})
    for key, rows in dets.items():
        d, view, step = key.split(":")
        d, step = int(d), int(step)
        gts_all = [b for b in by.get((d, view, step), []) if b.cls in D.CLS_ID]
        for cls in D.DET_CLASSES:
            a = acc[(cls, view)]
            a["frames"] += 1
            g = np.asarray([b.box for b in gts_all if b.cls == cls], float).reshape(-1, 4)
            a["gt_w"] += [b.width for b in gts_all if b.cls == cls]
            dr = [r for r in rows if r[4] == cls]
            dr.sort(key=lambda r: -r[5])
            db = np.asarray([r[:4] for r in dr], float).reshape(-1, 4)
            dc = np.asarray([r[5] for r in dr], float)
            a["n_gt"] += len(g)
            iou = iou_mat(db, g)
            used = np.zeros(len(g), bool)
            for i in range(len(db)):
                tp = 0
                if len(g):
                    cand = np.where(used, -1.0, iou[i])
                    j = int(np.argmax(cand))
                    if cand[j] >= 0.5:
                        used[j] = True
                        tp = 1
                a["scores"].append(float(dc[i]))
                a["tps"].append(tp)
            low = dc >= LOW
            if len(g):
                a["r50"] += int(np.sum(np.any(iou[low] >= 0.5, axis=0))) if low.any() else 0
                cin = centre_in(db[low], g)
                a["hit"] += int(np.sum(np.any(cin, axis=0))) if low.any() else 0
            if low.any():
                cin_d = centre_in(db[low], g).any(axis=1) if len(g) else np.zeros(int(low.sum()), bool)
                iou_d = iou[low].max(axis=1) if len(g) else np.zeros(int(low.sum()))
                a["fp"] += int(np.sum(~cin_d & (iou_d < 0.3)))
    rows_out = []
    for (cls, view), a in sorted(acc.items()):
        if a["frames"] == 0:
            continue
        rows_out.append({
            "fold": fold, "desktop": held, "view": view, "cls": cls,
            "frames": a["frames"], "n_gt": a["n_gt"],
            "gt_w_med": round(float(np.median(a["gt_w"])), 1) if a["gt_w"] else "",
            "AP50": round(ap_all_point(a["scores"], a["tps"], a["n_gt"]) * 100, 1)
            if a["n_gt"] else "",
            "R50@0.10": round(100 * a["r50"] / a["n_gt"], 1) if a["n_gt"] else "",
            "hit@0.10": round(100 * a["hit"] / a["n_gt"], 1) if a["n_gt"] else "",
            "FP/frame@0.10": round(a["fp"] / a["frames"], 2),
        })
    return rows_out


def main() -> int:
    import pandas as pd

    boxes = D.load_boxes()
    by = D.index(boxes)
    rows = []
    pooled: dict = {}
    for fold in ("hold13", "hold24", "hold33"):
        p = DETS / f"{fold}.json"
        if p.exists():
            dets = json.loads(p.read_text(encoding="utf-8"))
            pooled.update(dets)
            rows += evaluate(dets, int(fold[4:]), by)
    rows += evaluate(pooled, "all", by)
    df = pd.DataFrame(rows)
    df.to_csv(env.OUT / "det_metrics.csv", index=False)
    lines = ["# L2 detector metrics, YOLO26n (held-out desktops, drafts as GT)", ""]
    for cls in D.DET_CLASSES:
        s = df[(df.cls == cls) & (df.n_gt.astype(str) != "0")]
        if s.empty:
            continue
        lines += [f"## {cls}", "", D.md_table(s.drop(columns=["fold", "cls"])), ""]
    (env.OUT / "det_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
