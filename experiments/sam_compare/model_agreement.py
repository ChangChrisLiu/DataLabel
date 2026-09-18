"""How much of the IoU gap is the models' fault and how much is the reference?

Runs both models on the same subsample and reports three numbers per size
bucket: IoU(SAM 2.1, reference), IoU(SAM 3, reference) and IoU(SAM 2.1, SAM 3).

If the two independently-trained models agree with *each other* far more than
either agrees with the human polygon, the measured ceiling is mostly reference
noise rather than model error -- which is exactly what the overlays suggest for
screws and RAM clips. This is the honesty check on Task 2's headline numbers.

Run::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.sam_compare.model_agreement
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

import numpy as np

from experiments.sam_compare import _env, common
from experiments.sam_compare.run_interactive import Sam2Backend, Sam3Backend, prompts_for
from experiments.sam_compare.store import load_refs, load_sample

SEED = 20260918


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--prompt", default="point_box", choices=common.PROMPTS)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(SEED)
    indices = load_sample()
    pick = sorted(rng.permutation(len(indices))[: args.n])
    refs = load_refs([indices[i] for i in pick])
    by_frame: dict[tuple[int, int], list] = {}
    for ref in refs:
        by_frame.setdefault(ref.frame, []).append(ref)

    backends = [Sam2Backend(), Sam3Backend()]
    print(f"[agree] {len(refs)} masks, prompt={args.prompt}", flush=True)

    rows: list[dict] = []
    for frame in sorted(by_frame):
        image = common.load_frame_rgb(*frame)
        if image is None:
            continue
        for ref in by_frame[frame]:
            win = common.crop_window(ref.bbox, ref.frame_hw)
            crop = common.crop_image(image, win)
            gt = common.crop_mask(ref.mask, win)
            pts, box, multimask = prompts_for(ref, win)[args.prompt]
            preds = []
            for backend in backends:
                cands, scores = backend.predict(crop, pts, box, multimask)
                preds.append(common.paste_mask(cands[int(np.argmax(scores))], win))
            rows.append({
                "label": ref.label,
                "bucket": common.size_bucket(ref.area),
                "iou_21_ref": common.iou(preds[0], gt),
                "iou_3_ref": common.iou(preds[1], gt),
                "iou_21_3": common.iou(preds[0], preds[1]),
            })

    out: dict[str, dict] = {}
    for bucket in (*common.SIZE_NAMES, "ALL"):
        sub = rows if bucket == "ALL" else [r for r in rows if r["bucket"] == bucket]
        if not sub:
            continue
        out[bucket] = {
            "n": len(sub),
            "IoU_sam2.1_vs_reference": round(float(np.mean([r["iou_21_ref"] for r in sub])), 4),
            "IoU_sam3_vs_reference": round(float(np.mean([r["iou_3_ref"] for r in sub])), 4),
            "IoU_sam2.1_vs_sam3": round(float(np.mean([r["iou_21_3"] for r in sub])), 4),
        }
    result = {"prompt": args.prompt, "n_masks": len(rows), "by_bucket": out}
    path = _env.OUT_ROOT / "model_agreement.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[agree] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
