"""Build the reference-mask cache and draw the stratified evaluation sample.

Two outputs in ``experiments_out/sam_compare/``:

* ``refs.npz`` -- every usable scanner-view reference mask (bit-packed) with its
  bbox, area and click point. Written once; the three evaluation scripts read it.
* ``sample.json`` -- the indices of the ~1,500 masks to run the models on, with
  the per-label cap applied.

The cap is water-filled: rare labels are taken whole, and only the labels that
would otherwise dominate (screws, cables, the motherboard) get truncated, so
small hardware stays well represented without any label disappearing.

Run::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.sam_compare.prepare_sample
"""
from __future__ import annotations

import collections
import json

import numpy as np

from experiments.sam_compare import _env
from experiments.sam_compare.common import iter_reference_masks, size_bucket

#: Total masks we aim to evaluate (Task 2 budget).
TARGET_TOTAL = 1500
#: Reproducible sampling.
SEED = 20260918


def build_refs() -> list[dict]:
    """Rasterise and cache every usable reference mask."""
    records = list(iter_reference_masks())
    print(f"[refs] {len(records)} usable masks", flush=True)
    return records


def save_refs(records: list[dict]) -> None:
    out = _env.OUT_ROOT / "refs.npz"
    meta = [
        {k: r[k] for k in ("desktop", "step", "label", "bbox", "area", "point", "frame_hw")}
        for r in records
    ]
    np.savez_compressed(
        out,
        meta=np.array(json.dumps(meta), dtype=object),
        **{f"m{i}": r["packed"] for i, r in enumerate(records)},
    )
    print(f"[refs] wrote {out} ({out.stat().st_size / 1024**2:.1f} MB)", flush=True)


def water_fill(counts: dict[str, int], total: int) -> dict[str, int]:
    """Per-label quota: equal share, with unused headroom of rare labels re-spread.

    Labels with fewer masks than the current share are taken whole and their
    leftover budget is redistributed over the remaining labels, repeating until
    the allocation is stable. That is what "cap per label" has to mean here --
    a flat cap would either starve the rare labels or blow past the budget.
    """
    quota = {k: 0 for k in counts}
    remaining_labels = set(counts)
    budget = total
    while remaining_labels and budget > 0:
        share = max(1, budget // len(remaining_labels))
        done = [k for k in remaining_labels if counts[k] <= share]
        if not done:
            for k in remaining_labels:
                quota[k] = share
            break
        for k in done:
            quota[k] = counts[k]
            budget -= counts[k]
            remaining_labels.discard(k)
    return quota


def draw_sample(records: list[dict], total: int = TARGET_TOTAL) -> list[int]:
    """Pick indices: per-label quota, spread over size buckets and desktops."""
    rng = np.random.default_rng(SEED)
    by_label: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(records):
        by_label[r["label"]].append(i)

    quota = water_fill({k: len(v) for k, v in by_label.items()}, total)
    chosen: list[int] = []
    for label, idxs in sorted(by_label.items()):
        want = quota[label]
        if want >= len(idxs):
            chosen.extend(idxs)
            continue
        # stratify inside the label by size bucket, then round-robin desktops
        buckets: dict[str, list[int]] = collections.defaultdict(list)
        for i in idxs:
            buckets[size_bucket(records[i]["area"])].append(i)
        per_bucket = water_fill({k: len(v) for k, v in buckets.items()}, want)
        for bucket, bidx in buckets.items():
            take = min(per_bucket[bucket], len(bidx))
            order = rng.permutation(len(bidx))
            # round-robin over desktops so one machine cannot dominate a stratum
            by_desktop: dict[int, list[int]] = collections.defaultdict(list)
            for j in order:
                by_desktop[records[bidx[j]]["desktop"]].append(bidx[j])
            queues = list(by_desktop.values())
            picked: list[int] = []
            while len(picked) < take and any(queues):
                for q in queues:
                    if q and len(picked) < take:
                        picked.append(q.pop())
            chosen.extend(picked)
    chosen.sort()
    return chosen


def census(records: list[dict], chosen: list[int]) -> dict:
    pick = set(chosen)
    rows: dict[str, dict] = {}
    for i, r in enumerate(records):
        row = rows.setdefault(
            r["label"], {"total": 0, "sampled": 0, **{b: 0 for b in
                         ("<400", "400-2k", "2k-20k", ">20k")}}
        )
        row["total"] += 1
        if i in pick:
            row["sampled"] += 1
            row[size_bucket(r["area"])] += 1
    return rows


def main() -> int:
    records = build_refs()
    save_refs(records)
    chosen = draw_sample(records)
    rows = census(records, chosen)

    out = _env.OUT_ROOT / "sample.json"
    out.write_text(json.dumps({"seed": SEED, "indices": chosen}, indent=1), encoding="utf-8")

    print(f"\n[sample] {len(chosen)} masks from {len(records)} refs")
    print(f"{'label':<34}{'total':>7}{'samp':>7}{'<400':>7}{'400-2k':>8}"
          f"{'2k-20k':>8}{'>20k':>7}")
    for label, row in sorted(rows.items(), key=lambda kv: -kv[1]["total"]):
        print(f"{label[:33]:<34}{row['total']:>7}{row['sampled']:>7}{row['<400']:>7}"
              f"{row['400-2k']:>8}{row['2k-20k']:>8}{row['>20k']:>7}")
    frames = {(r["desktop"], r["step"]) for r in records}
    desktops = sorted({r["desktop"] for r in records})
    print(f"\n[sample] {len(frames)} frames, desktops {desktops}")
    print(f"[sample] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
