"""Tiled inference for every L3 model, through L2's tiling and merge code.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.infer --model yolo26s --fold hold13

Writes ``<OUT>/<model>/detections/<fold>.json`` in L2's format
(``{"<desktop>:<view>:<step>": [[x0, y0, x1, y1, cls, conf], ...]}``, conf >=
0.001) for every frame with drafts of the held-out desktop -- exactly what
L2's ``infer`` wrote for YOLO26n, so L2's ``det_eval`` / ``guess_eval`` read it
unchanged.

The YOLO models use L2's ``Detector`` as is (fp16, one batched forward over the
ROI's tiles).  RF-DETR plugs into the same class: only the per-tile forward
(``_run``) differs -- RF-DETR's own ``predict`` on the 640 tiles at the
training resolution (640), fp16 via ``inference(compile=False,
dtype=float16)``, its top-300 query x class outputs (DETR has no NMS; L2's
cross-tile class-wise NMS still runs in ``merge``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402
from experiments.l2_detector import data as D  # noqa: E402
from experiments.l2_detector import tiles as T  # noqa: E402
from experiments.l2_detector.infer import Detector  # noqa: E402

YOLO_MODELS = ("yolo26n", "yolo26s", "yolo26m", "yolo26l")
RF_MODELS = ("rfdetr_small", "rfdetr_medium", "rfdetr_base")


def weights_of(model: str, fold: str) -> Path:
    if model == "yolo26n":
        return env.L2OUT / "runs" / fold / "weights" / "last.pt"
    if model in YOLO_MODELS:
        return env.OUT / model / "runs" / fold / "weights" / "last.pt"
    return env.OUT / model / "runs" / fold / "last_ema.pth"


class RFDetector(Detector):
    """L2's tiled detector with RF-DETR's forward in place of YOLO's."""

    def __init__(self, weights: Path, half: bool = True, conf: float = 0.001):
        env.limit_threads()
        import torch
        from rfdetr import RFDETR

        self.torch = torch
        self.model = RFDETR.from_checkpoint(str(weights))
        res = self.model.model.resolution
        if res != T.TILE:
            raise SystemExit(f"{weights}: resolution {res}, expected {T.TILE}")
        names = list(self.model.class_names)
        if names != list(D.DET_CLASSES):
            raise SystemExit(f"{weights}: class names {names} != {list(D.DET_CLASSES)}")
        self.names = dict(enumerate(names))
        self.half = half
        self.conf = conf
        if half:
            self.model.inference(compile=False, dtype=torch.float16)
        dummy = [np.full((T.TILE, T.TILE, 3), 114, np.uint8)] * 4
        for _ in range(2):
            self.model.predict(dummy, threshold=self.conf, include_source_image=False)

    def _run(self, tiles):
        res = self.model.predict([t.rgb for t in tiles], threshold=self.conf,
                                 include_source_image=False)
        if not isinstance(res, list):
            res = [res]
        out = []
        for r in res:
            if r is None or len(r) == 0:
                out.append(None)
                continue
            # the head has num_classes + 1 logits; the extra slot is never a
            # target (it only ever scores near zero) and is not a class
            k = np.asarray(r.class_id, int)
            ok = (k >= 0) & (k < len(self.names))
            out.append((np.asarray(r.xyxy, np.float64)[ok],
                        np.asarray(r.confidence, np.float64)[ok], k[ok]))
        return out


def make_detector(model: str, fold: str, half: bool = True, conf: float = 0.001):
    """``conf`` is the score floor applied *before* L2's merge (0.001 = L2's
    evaluation protocol; 0.10 = the guess's own threshold, what an app would use)."""
    w = weights_of(model, fold)
    if not w.exists():
        raise SystemExit(f"missing weights {w}")
    if model in YOLO_MODELS:
        return Detector(w, half=half, conf=conf)
    return RFDetector(w, half=half, conf=conf)


def tag_of(model: str, conf: float) -> str:
    return model if conf == 0.001 else f"{model}_c{int(round(conf * 100)):02d}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=YOLO_MODELS + RF_MODELS)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="score floor before the merge; != 0.001 writes to <model>_cNN/")
    args = ap.parse_args(argv)
    out_dir = env.bind_l2(tag_of(args.model, args.conf))
    held = int(args.fold.replace("hold", ""))
    boxes = D.load_boxes()
    by = D.index(boxes)
    det = make_detector(args.model, args.fold, conf=args.conf)
    con = D.connect()
    out: dict[str, list] = {}
    tim = []
    t_all = time.perf_counter()
    for view in D.VIEWS:
        steps = sorted(s for (d, v, s) in by if d == held and v == view)
        if not steps:
            continue
        roi = D.roi_of(boxes, held, view)
        for step in steps:
            path = D.frame_path(con, held, view, step)
            img = D.read_rgb(path) if path else None
            if img is None:
                print(f"  missing D{held} {view} s{step}")
                continue
            t = {}
            dets = det.detect(img, roi, view, timings=t)
            tim.append({"view": view, **t})
            out[f"{held}:{view}:{step}"] = [d.row() for d in dets]
        print(f"  D{held} {view}: {len(steps)} frames", flush=True)
    con.close()
    dd = out_dir / "detections"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / f"{args.fold}.json").write_text(json.dumps(out), encoding="utf-8")
    (dd / f"{args.fold}_timing.json").write_text(json.dumps(tim), encoding="utf-8")
    print(f"wrote {dd / (args.fold + '.json')} ({len(out)} frames, "
          f"{time.perf_counter() - t_all:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
