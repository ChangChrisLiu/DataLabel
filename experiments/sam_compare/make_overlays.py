"""Render side-by-side overlays for typical and worst cases.

Picks ~40 cases from the per-mask CSVs -- weighted towards the small hardware
the tool is judged on (screws, RAM retention clips, connectors) -- re-runs both
models on exactly those prompts, and writes one PNG per case:

    [ crop + reference ] [ crop + SAM 2.1 ] [ crop + SAM 3 ]

Reference outline is green, prediction fill is red, and the prompt (click dot /
box) is drawn in yellow so the failure mode is readable at a glance.

Run::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.sam_compare.make_overlays
"""
from __future__ import annotations

import argparse
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from experiments.sam_compare import _env, common
from experiments.sam_compare.run_interactive import Sam2Backend, Sam3Backend, prompts_for
from experiments.sam_compare.store import load_refs

OUT_DIR = _env.OUT_ROOT / "overlays"

#: Families to over-sample, in priority order; ``None`` means "everything else".
FAMILIES = ("Screw", "Retention Clip", "Connector", None)
GREEN = (0, 220, 0)
RED = (0, 0, 235)
YELLOW = (0, 235, 235)
WHITE = (255, 255, 255)


def select_cases(n_extra: int = 4) -> pd.DataFrame:
    """Typical + worst cases per family and prompt, plus biggest disagreements."""
    a = pd.read_csv(_env.OUT_ROOT / "interactive_sam21.csv")
    b = pd.read_csv(_env.OUT_ROOT / "interactive_sam3.csv")
    keys = ["index", "prompt", "label", "area", "size_bucket", "desktop", "step"]
    merged = a[a.status == "ok"].merge(
        b[b.status == "ok"], on=keys, suffixes=("_21", "_3"))
    merged["iou_min"] = merged[["iou_21", "iou_3"]].min(axis=1)
    merged["iou_gap"] = (merged["iou_3"] - merged["iou_21"]).abs()

    picks = []
    used_families = pd.Series(False, index=merged.index)
    for family in FAMILIES:
        if family is None:
            sub = merged[~used_families]
        else:
            hit = merged["label"].str.contains(family, case=False, na=False)
            used_families |= hit
            sub = merged[hit]
        if sub.empty:
            continue
        for prompt in common.PROMPTS:
            g = sub[sub["prompt"] == prompt]
            if g.empty:
                continue
            picks.append(g.nsmallest(2, "iou_min").assign(kind="worst",
                                                          family=family or "other"))
            median = g["iou_min"].median()
            typical = g.iloc[(g["iou_min"] - median).abs().argsort()[:1]]
            picks.append(typical.assign(kind="typical", family=family or "other"))
    picks.append(merged.nlargest(n_extra, "iou_gap").assign(kind="disagree",
                                                            family="any"))
    out = pd.concat(picks).drop_duplicates(subset=["index", "prompt"])
    return out.reset_index(drop=True)


def _blend(crop: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45) -> np.ndarray:
    """BGR copy of ``crop`` with ``mask`` tinted ``color``."""
    out = crop.copy()
    if mask.any():
        out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def _outline(img: np.ndarray, mask: np.ndarray, color, thickness: int = 1) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)


def _draw_prompt(img: np.ndarray, pts, box) -> None:
    if box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        cv2.rectangle(img, (x0, y0), (x1, y1), YELLOW, 1)
    if pts is not None:
        cv2.circle(img, (int(pts[0][0]), int(pts[0][1])), 4, YELLOW, -1)
        cv2.circle(img, (int(pts[0][0]), int(pts[0][1])), 5, (0, 0, 0), 1)


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((22, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def render_case(row, ref, backends, frame) -> Optional[np.ndarray]:
    win = common.crop_window(ref.bbox, ref.frame_hw)
    crop_rgb = common.crop_image(frame, win)
    gt = common.crop_mask(ref.mask, win)
    pts, box, multimask = prompts_for(ref, win)[row["prompt"]]

    # zoom into the object so a 20 px screw is actually visible
    x0, y0, x1, y1 = ref.bbox
    pad = max(48, int(0.6 * max(x1 - x0, y1 - y0)))
    vx0 = max(0, x0 - win.x0 - pad)
    vy0 = max(0, y0 - win.y0 - pad)
    vx1 = min(win.size[0], x1 - win.x0 + pad)
    vy1 = min(win.size[1], y1 - win.y0 + pad)

    base = cv2.cvtColor(crop_rgb if not win.oversize else
                        cv2.resize(crop_rgb, win.size, interpolation=cv2.INTER_LINEAR),
                        cv2.COLOR_RGB2BGR)
    panels = []
    ref_panel = base.copy()
    _outline(ref_panel, gt, GREEN, 2)
    _draw_prompt(ref_panel, None if pts is None else
                 [(pts[0][0] / win.scale, pts[0][1] / win.scale)],
                 None if box is None else [v / win.scale for v in box])
    panels.append(("reference", ref_panel, None))

    for backend in backends:
        cands, scores = backend.predict(crop_rgb, pts, box, multimask)
        pred = common.paste_mask(cands[int(np.argmax(scores))], win)
        panel = _blend(base, pred, RED)
        _outline(panel, gt, GREEN, 1)
        panels.append((backend.name, panel, common.iou(pred, gt)))

    tiles = []
    for name, panel, score in panels:
        view = panel[vy0:vy1, vx0:vx1]
        if view.size == 0:
            return None
        scale = min(3.0, max(1.0, 320.0 / max(1, max(view.shape[:2]))))
        view = cv2.resize(view, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
        text = name if score is None else f"{name}  IoU {score:.3f}"
        tiles.append(_caption(view, text))
    height = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((height - t.shape[0], t.shape[1], 3), np.uint8)])
             for t in tiles]
    grid = np.hstack(tiles)
    head = (f"{row['kind']}  {row['label'][:38]}  area={int(row['area'])}px "
            f"({row['size_bucket']})  prompt={row['prompt']}  "
            f"D{int(row['desktop']):02d}/s{int(row['step']):03d}  win={win.size[0]}")
    return _caption(grid, head)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = select_cases()
    if args.limit:
        cases = cases.head(args.limit)
    refs = {r.index: r for r in load_refs(sorted(set(cases["index"].tolist())))}
    backends = [Sam2Backend(), Sam3Backend()]
    print(f"[overlays] {len(cases)} cases", flush=True)

    written = 0
    frame_cache: dict[tuple[int, int], np.ndarray] = {}
    for _, row in cases.sort_values(["desktop", "step"]).iterrows():
        ref = refs[row["index"]]
        if ref.frame not in frame_cache:
            frame_cache.clear()  # one 1600^2 frame in memory at a time
            loaded = common.load_frame_rgb(*ref.frame)
            if loaded is None:
                continue
            frame_cache[ref.frame] = loaded
        image = render_case(row, ref, backends, frame_cache[ref.frame])
        if image is None:
            continue
        name = (f"{row['family'].replace(' ', '')}_{row['kind']}_{row['prompt']}_"
                f"D{int(row['desktop']):02d}s{int(row['step']):03d}_"
                f"i{int(row['index'])}.png")
        cv2.imwrite(str(OUT_DIR / name), image)
        written += 1
    print(f"[overlays] wrote {written} PNGs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
