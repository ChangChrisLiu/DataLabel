"""Derive per-class mask-area priors from an annotation database.

::

    D:\\Anaconda\\envs\\tda\\python.exe experiments/area_priors_from_db.py ^
        --db D:\\DataSet\\.cache\\tmp\\tda_copy.sqlite --out configs/area_priors.yaml

What it reads: every ``shape_keyframe`` with a mask, its instance's class, and
the ROI of the pose segment the keyframe belongs to.  What it writes: per class
``{min_frac, max_frac}`` as **fractions of the ROI area**, which is the only
view-independent way to compare a screw on a 1600x1600 scan with the same screw
on a 4032x3040 OAK frame.

The bounds are deliberately generous -- ``median / MARGIN_LOW`` and
``median * MARGIN_HIGH``, clipped to the observed extremes -- because they only
raise a *warning bar* that a second ``Enter`` overrides (spec 4.3 has no notion
of a refused area).  A prior that cries wolf is worse than no prior: it teaches
the annotator to press ``Enter`` twice without reading.

**Never point ``--db`` at the live database.**  Take a copy first; this script
opens it read-only but the rule is the rule.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: The lower bound is ``p05 / MARGIN_LOW``: tiny-but-valid happens (a screw head
#: half behind a bracket), so there is slack on this side.
MARGIN_LOW = 4.0
#: The upper bound is ``p95 * K``, **capped**, and ``K`` depends on the size
#: regime.  It used to be ``max(observed) * 6`` and the window then multiplied
#: by another 10, which put the ceiling for a screw at 1.9 ROIs and for a
#: motherboard at 10 -- a mask of the whole frame is 3.5 ROIs, so the rule could
#: not fire at all.
#:
#: One K for every class cannot work, because the two ends fail in opposite
#: directions.  Small parts are *under-drawn* in the draft masks these
#: percentiles come from -- a screw's p95 is 0.055 % of the ROI, about 20x20 px,
#: while a hand-drawn 60x60 screw is plausible and nine times that -- so they
#: need a wide factor.  Large parts do not: with a wide factor a psu's ceiling
#: reaches the cap and the 619,923-pixel mask the rehearsal committed as one
#: (0.86 of the ROI) stops being remarkable.
#:
#: So: ``K = 10`` for a class whose p95 is under :data:`SMALL_P95` of the ROI,
#: ``K = 4`` for the rest.
SMALL_P95 = 0.005
MARGIN_HIGH_SMALL = 10.0
MARGIN_HIGH_LARGE = 4.0
#: No class may exceed one ROI ...
CAP_FRAC = 1.0
#: ... except the ones that *are* the machine, which can fill it and spill over
#: the detected box a little.
STRUCTURE_CAP_FRAC = 1.2
STRUCTURE_CLASSES = ("chassis", "motherboard")
#: Classes with fewer masks than this are not worth a prior of their own.
MIN_SAMPLES = 8


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, help="a COPY of the annotation database")
    ap.add_argument("--out", default=str(REPO / "configs" / "area_priors.yaml"))
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    return ap.parse_args(argv)


def mask_area(rle_json: str) -> int:
    """Pixel count of one COCO RLE, without decoding it into an array."""
    from tda.core import masks

    rle = json.loads(rle_json)
    counts = rle.get("counts")
    if isinstance(counts, str):
        return int(masks.decode_rle(rle).sum())
    # uncompressed RLE: the odd runs are the set pixels
    return int(sum(int(n) for n in counts[1::2]))


#: Share of a frame the chassis ROI takes up, for views with no stored ROI yet.
#: Measured on D13/scan, where the detector gives [596, 284, 1423, 1160] of a
#: 1600x1600 frame: 0.283.  It is only the denominator of a prior that is then
#: widened by a factor of six either way, so it does not have to be exact -- but
#: it does have to be written down, and the generated file says how many masks
#: were measured against it.
DEFAULT_ROI_SHARE = 0.28


def _roi_area_fallback(view: str) -> Optional[float]:
    """``ROI area`` for a view with no stored ROI, from its nominal frame size."""
    from tda.core.truth_inputs import infer_hw

    try:
        h, w = infer_hw(str(view))
    except ValueError:
        return None
    return float(h) * float(w) * DEFAULT_ROI_SHARE


def collect(db_path: str) -> tuple[dict[str, list[float]], dict[str, int]]:
    """``({class: [area / roi_area, ...]}, {"stored", "inferred", "skipped"})``."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        source = {"stored": 0, "inferred": 0, "skipped": 0}
        rois: dict[tuple, float] = {}
        for row in conn.execute("SELECT desktop, view, seg, roi_json FROM pose_segment"):
            if not row["roi_json"]:
                continue
            try:
                x0, y0, x1, y1 = json.loads(row["roi_json"])
            except Exception:  # noqa: BLE001 - a malformed ROI is not a prior
                continue
            area = max(1.0, float(x1 - x0) * float(y1 - y0))
            rois[(row["desktop"], row["view"], row["seg"])] = area

        classes = {
            (row["desktop"], row["key"]): row["cls"]
            for row in conn.execute("SELECT desktop, key, cls FROM instance")
        }

        out: dict[str, list[float]] = defaultdict(list)
        query = (
            "SELECT k.desktop, k.view, k.pose_segment, k.instance, p.rle_json "
            "FROM shape_keyframe k JOIN shape_part p ON p.keyframe_id = k.id "
            "WHERE k.geom_type = 'mask' AND p.rle_json IS NOT NULL"
        )
        for row in conn.execute(query):
            cls = classes.get((row["desktop"], row["instance"]))
            if not cls:
                source["skipped"] += 1
                continue
            roi = rois.get((row["desktop"], row["view"], row["pose_segment"]))
            if roi:
                source["stored"] += 1
            else:
                roi = _roi_area_fallback(row["view"])
                if not roi:
                    source["skipped"] += 1
                    continue
                source["inferred"] += 1
            try:
                area = mask_area(row["rle_json"])
            except Exception:  # noqa: BLE001 - one unreadable mask is not fatal
                source["skipped"] += 1
                continue
            if area > 0:
                out[str(cls)].append(area / roi)
        return dict(out), source
    finally:
        conn.close()


def percentile(values: list[float], q: float) -> float:
    """The ``q``-th percentile, nearest rank; no numpy needed for 11k floats."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(round(q / 100.0 * (len(ordered) - 1)))))
    return float(ordered[index])


def margin_high(p95: float) -> float:
    """``K`` for this class's size regime (see :data:`SMALL_P95`)."""
    return MARGIN_HIGH_SMALL if p95 < SMALL_P95 else MARGIN_HIGH_LARGE


def priors(samples: dict[str, list[float]], min_samples: int) -> dict[str, dict]:
    """Robust bounds per class: ``p05 / 4`` to ``min(p95 * K, cap)``.

    Percentiles rather than the extremes, because the draft masks these come
    from include the mistakes this warning is meant to catch: one 1.5-million
    pixel "screw" in the sample pushed the old ``max(observed)`` ceiling past
    anything a screw could ever be, and the rule then never fired.
    """
    out: dict[str, dict] = {}
    for cls, fracs in sorted(samples.items()):
        if len(fracs) < min_samples:
            continue
        p05, median, p95 = (percentile(fracs, 5), statistics.median(fracs),
                            percentile(fracs, 95))
        cap = STRUCTURE_CAP_FRAC if cls in STRUCTURE_CLASSES else CAP_FRAC
        out[cls] = {
            "min_frac": round(float(p05 / MARGIN_LOW), 6),
            "max_frac": round(float(min(p95 * margin_high(p95), cap)), 6),
            "p05_frac": round(float(p05), 6),
            "median_frac": round(float(median), 6),
            "p95_frac": round(float(p95), 6),
            "samples": len(fracs),
        }
    return out


def write_yaml(path: str, table: dict[str, dict], source: dict) -> None:
    import yaml

    header = (
        "# Per-class mask-area priors, as fractions of the pose segment's ROI area.\n"
        "# Generated by experiments/area_priors_from_db.py from a COPY of the\n"
        "# annotation database; they only raise the non-blocking warning bar\n"
        "# (a second Enter commits anyway), so they are deliberately generous.\n"
        f"# masks measured: {source.get('stored', 0)} against a stored ROI, "
        f"{source.get('inferred', 0)} against {DEFAULT_ROI_SHARE:.2f} x the\n"
        "# nominal frame area (no ROI had been confirmed for that segment yet).\n"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump({"classes": table}, fh, allow_unicode=True, sort_keys=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    samples, source = collect(args.db)
    table = priors(samples, int(args.min_samples))
    write_yaml(args.out, table, source)
    print(f"{len(table)} class priors from {sum(len(v) for v in samples.values())} masks"
          f" -> {args.out}")
    print(f"denominators: {source}")
    print(f"  {'class':<24} {'n':>5} {'p05':>9} {'median':>9} {'p95':>9} "
          f"{'low':>9} {'high':>9}")
    for cls, row in sorted(table.items()):
        print(f"  {cls:<24} {row['samples']:>5} {row['p05_frac']:>9.5f} "
              f"{row['median_frac']:>9.5f} {row['p95_frac']:>9.5f} "
              f"{row['min_frac']:>9.5f} {row['max_frac']:>9.5f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
