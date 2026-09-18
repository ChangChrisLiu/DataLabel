"""Task 2 -- the main result: interactive prompts, SAM 2.1 vs SAM 3.

For every sampled reference mask the same production-like protocol runs on both
models: crop a 512-1024 px window at native resolution around the object, embed
it once, then issue three prompts off that one embedding --

* ``box``       -- the reference bbox padded by 2 px,
* ``point``     -- one positive click at the distance-transform argmax
                   (``multimask_output=True``, best of three, which is what an
                   annotation tool does with an ambiguous single click),
* ``point_box`` -- both together.

Per prompt it records IoU, boundary F at 2 px and the decoder time, with the
one-off embedding time recorded separately per crop. One model per process, so
the two never share VRAM and a crash in one does not lose the other's rows.

Run::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.sam_compare.run_interactive --model sam2.1
    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.sam_compare.run_interactive --model sam3
"""
from __future__ import annotations

import argparse
import csv
import time
from typing import Optional

import numpy as np

from experiments.sam_compare import _env
from experiments.sam_compare import common
from experiments.sam_compare.store import Ref, load_refs, load_sample

CSV_FIELDS = [
    "index", "desktop", "step", "label", "area", "size_bucket",
    "win_size", "win_scale", "oversize", "prompt", "model",
    "iou", "boundary_f1", "iou_oracle", "n_cand", "pred_area", "score",
    "decode_ms", "embed_ms", "status",
]


class Sam2Backend:
    """Adapter over the production ``SamService``'s predictor.

    ``SamService.predict`` now returns every candidate, but it also returns them
    *ranked by SAM's own score*, and the experiment needs the raw decoder order
    to report both "what the tool would pick" and the oracle best (see
    :func:`run_one_frame`) against the reference mask. So the underlying
    predictor is called directly; the embedding cache and the autocast
    behaviour are still the service's.
    """

    name = "sam2.1"

    def __init__(self) -> None:
        from tda.models.sam_service import SamService

        t0 = time.perf_counter()
        self._svc = SamService(checkpoint=str(_env.SAM2_CKPT), device="cuda")
        self.load_s = time.perf_counter() - t0
        self._torch = self._svc._torch

    def set_image(self, crop: np.ndarray) -> float:
        self._svc.set_image(crop)
        return self._svc.last_set_image_ms

    def predict(self, crop, points, box, multimask):
        self._svc.set_image(crop)
        labels = None if points is None else np.array([1], dtype=np.int32)
        with self._torch.inference_mode(), self._svc._autocast():
            masks, scores, _ = self._svc._predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=None if box is None else np.asarray(box, np.float32).reshape(1, 4),
                multimask_output=bool(multimask),
                normalize_coords=True,
            )
        return np.asarray(masks) > 0.5, np.asarray(scores, np.float32).reshape(-1)


class Sam3Backend:
    """Adapter over :class:`experiments.sam_compare.sam3_service.Sam3Service`."""

    name = "sam3"

    def __init__(self) -> None:
        from experiments.sam_compare.sam3_service import Sam3Service

        t0 = time.perf_counter()
        self._svc = Sam3Service(device="cuda", autocast=True)
        self.load_s = time.perf_counter() - t0

    def set_image(self, crop: np.ndarray) -> float:
        self._svc.set_image(crop)
        return self._svc.last_set_image_ms

    def predict(self, crop, points, box, multimask):
        self._svc.set_image(crop)
        labels = None if points is None else np.array([1], dtype=np.int32)
        with self._svc._autocast():
            masks, scores, _ = self._svc._model.predict_inst(
                self._svc._state,
                point_coords=points,
                point_labels=labels,
                box=None if box is None else np.asarray(box, np.float32),
                multimask_output=bool(multimask),
                normalize_coords=True,
            )
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None]
        return masks > 0.5, np.asarray(scores, np.float32).reshape(-1)


def build_backend(name: str):
    return Sam2Backend() if name == "sam2.1" else Sam3Backend()


def prompts_for(ref: Ref, win: common.Window) -> dict[str, tuple]:
    """``{prompt_name: (points, box, multimask)}`` in model-input coordinates."""
    box = common.box_prompt(ref.bbox, win)
    pts = common.point_prompt(ref.point, win)
    return {
        "box": (None, box, False),
        # a lone click is ambiguous: ask for 3 candidates and keep the best,
        # which is what the annotation tool would do on a single click
        "point": (pts, None, True),
        "point_box": (pts, box, False),
    }


def _timed(torch, fn, *args, **kwargs):
    """Call ``fn`` and return ``(result, wall_ms)`` with CUDA work drained.

    CUDA launches are asynchronous, so timing without the two synchronisations
    charges the image-encoder's work to whichever later call happens to pull a
    tensor back to the host -- which made the SAM 3 embedding look faster than
    SAM 2.1's. Both syncs are needed: the first drains earlier work so it is not
    billed here, the second waits for this call's own kernels.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) * 1000.0


def run_one_frame(backend, refs: list[Ref], writer: csv.DictWriter, torch) -> int:
    """Evaluate every sampled mask of one frame; returns rows written."""
    desktop, step = refs[0].frame
    frame = common.load_frame_rgb(desktop, step)
    if frame is None:
        return 0
    rows = 0
    for ref in refs:
        win = common.crop_window(ref.bbox, ref.frame_hw)
        crop = common.crop_image(frame, win)
        gt = common.crop_mask(ref.mask, win)
        base = {
            "index": ref.index, "desktop": desktop, "step": step, "label": ref.label,
            "area": ref.area, "size_bucket": common.size_bucket(ref.area),
            "win_size": win.size[0], "win_scale": round(win.scale, 4),
            "oversize": int(win.oversize), "model": backend.name,
        }
        try:
            embed_ms = _timed(torch, backend.set_image, crop)[1]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            try:  # retry smaller: halve the crop, same protocol otherwise
                small = common.cv2.resize(crop, (common.MIN_WINDOW, common.MIN_WINDOW),
                                          interpolation=common.cv2.INTER_AREA)
                embed_ms = _timed(torch, backend.set_image, small)[1]
                crop = small
            except Exception:
                for prompt in common.PROMPTS:
                    writer.writerow({**base, "prompt": prompt, "status": "oom"})
                    rows += 1
                continue

        for prompt, (pts, box, multimask) in prompts_for(ref, win).items():
            row = {**base, "prompt": prompt, "embed_ms": round(embed_ms, 2)}
            try:
                (cands, scores), decode_ms = _timed(
                    torch, backend.predict, crop, pts, box, multimask
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                writer.writerow({**row, "status": "oom"})
                rows += 1
                continue
            except Exception as exc:  # a bad prompt must not lose the whole run
                writer.writerow({**row, "status": f"error:{type(exc).__name__}"})
                rows += 1
                continue
            # What the tool would ship is the top-*scoring* candidate; the oracle
            # over all candidates separates "the model cannot segment this" from
            # "the model can, but its own IoU head picks the wrong candidate".
            ious = [common.iou(common.paste_mask(m, win), gt) for m in cands]
            best = int(np.argmax(scores))
            pred = common.paste_mask(cands[best], win)
            writer.writerow({
                **row,
                "iou": round(ious[best], 5),
                "boundary_f1": round(common.boundary_f1(pred, gt), 5),
                "iou_oracle": round(max(ious), 5),
                "n_cand": len(ious),
                "pred_area": int(pred.sum()),
                "score": round(float(scores[best]), 5),
                "decode_ms": round(decode_ms, 2),
                "status": "ok",
            })
            rows += 1
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=common.MODELS, required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N masks")
    args = ap.parse_args(argv)

    import torch

    indices = load_sample()
    if args.limit:
        indices = indices[: args.limit]
    refs = load_refs(indices)
    # group by frame so each 1600^2 PNG is decoded exactly once
    by_frame: dict[tuple[int, int], list[Ref]] = {}
    for ref in refs:
        by_frame.setdefault(ref.frame, []).append(ref)

    backend = build_backend(args.model)
    print(f"[run] {backend.name} loaded in {backend.load_s:.1f} s; "
          f"{len(refs)} masks over {len(by_frame)} frames", flush=True)

    out = _env.OUT_ROOT / f"interactive_{args.model.replace('.', '')}.csv"
    t0 = time.perf_counter()
    rows = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for n, frame in enumerate(sorted(by_frame), 1):
            rows += run_one_frame(backend, by_frame[frame], writer, torch)
            if n % 20 == 0:
                el = time.perf_counter() - t0
                print(f"[run] {n}/{len(by_frame)} frames, {rows} rows, "
                      f"{el:.0f} s elapsed, ETA {el / n * (len(by_frame) - n):.0f} s",
                      flush=True)
    print(f"[run] {rows} rows in {time.perf_counter() - t0:.0f} s -> {out}")
    print(f"[run] peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
