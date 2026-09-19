"""Turn the per-instance CSVs into the tables the report quotes.

Every rate is printed with the ``n`` it was computed from; a cell with fewer
than :data:`MIN_N` observations prints its count and a dash instead of a
percentage, because a rate over five instances is not a measurement.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.plan_b_probe.transfer.common import OUT, TARGETS  # noqa: E402

MIN_N = 6
GROUPS = ("flat", "tall", "rim", "other")


def read(name: str) -> list[dict]:
    path = OUT / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(row: dict, key: str, default=float("nan")) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else default


def rate(hits: int, n: int) -> str:
    if n == 0:
        return "   -  (0)"
    if n < MIN_N:
        return f"   -  ({n})"
    return f"{100.0 * hits / n:5.1f}% ({n})"


def med(values: list[float]) -> str:
    vals = [v for v in values if v == v]
    return f"{np.median(vals):6.2f}" if len(vals) >= MIN_N else f"  -({len(vals)})"


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    rows = read("m1_per_instance.csv")
    inchassis = [r for r in rows if r["outside_chassis"] == "0"]
    uniq = [r for r in inchassis if r["unique_label"] == "1"]
    multi = [r for r in inchassis if r["unique_label"] == "0"]
    outside = [r for r in rows if r["outside_chassis"] == "1"]

    head("population")
    print(f"instance-step pairs present in scan and target: {len(rows)}")
    print(f"  in the clicked scan chassis quad : {len(inchassis)}")
    print(f"  annotated outside it (on bench)  : {len(outside)}"
          f"  ({100.0 * len(outside) / max(1, len(rows)):.1f}%)")
    print(f"  of the in-chassis rows, unique-label: {len(uniq)}, "
          f"multi-instance label: {len(multi)}")

    head("M1/M2 - HIT rate of the projected scan centroid, by view and plane")
    print(f"{'view':6s} {'subset':10s} {'rim HIT':>14s} {'floor HIT':>14s} "
          f"{'baseline':>14s} {'rim d/size':>9s} {'floor d/size':>11s} "
          f"{'rim IoU':>8s} {'floor IoU':>9s}")
    for view in TARGETS:
        for name, subset in (("unique", uniq), ("multi-ord", multi), ("all", inchassis)):
            sel = [r for r in subset if r["view"] == view]
            if not sel:
                continue
            print(f"{view:6s} {name:10s} "
                  f"{rate(sum(int(r['rim_hit']) for r in sel), len(sel)):>14s} "
                  f"{rate(sum(int(r['floor_hit']) for r in sel), len(sel)):>14s} "
                  f"{rate(sum(int(r['centre_baseline_hit']) for r in sel), len(sel)):>14s} "
                  f"{med([num(r, 'rim_dist_norm') for r in sel]):>9s} "
                  f"{med([num(r, 'floor_dist_norm') for r in sel]):>11s} "
                  f"{med([num(r, 'rim_iou') for r in sel]):>8s} "
                  f"{med([num(r, 'floor_iou') for r in sel]):>9s}")
        print()

    head("M2 - plane comparison per class group (unique-label rows only)")
    print(f"{'view':6s} {'group':7s} {'rim HIT':>14s} {'floor HIT':>14s} "
          f"{'rim d px':>9s} {'floor d px':>11s} {'rim d/size':>11s} "
          f"{'floor d/size':>12s}")
    for view in TARGETS:
        for group in GROUPS:
            sel = [r for r in uniq if r["view"] == view and r["group"] == group]
            if not sel:
                continue
            print(f"{view:6s} {group:7s} "
                  f"{rate(sum(int(r['rim_hit']) for r in sel), len(sel)):>14s} "
                  f"{rate(sum(int(r['floor_hit']) for r in sel), len(sel)):>14s} "
                  f"{med([num(r, 'rim_dist_px') for r in sel]):>9s} "
                  f"{med([num(r, 'floor_dist_px') for r in sel]):>11s} "
                  f"{med([num(r, 'rim_dist_norm') for r in sel]):>11s} "
                  f"{med([num(r, 'floor_dist_norm') for r in sel]):>12s}")
        print()

    head("M2 - plane comparison per class group, pooled over views")
    print(f"{'group':7s} {'rim HIT':>14s} {'floor HIT':>14s} {'rim d/size':>11s} "
          f"{'floor d/size':>12s}  better")
    for group in GROUPS:
        sel = [r for r in uniq if r["group"] == group]
        if not sel:
            continue
        rim = sum(int(r["rim_hit"]) for r in sel)
        flr = sum(int(r["floor_hit"]) for r in sel)
        better = "rim" if rim > flr else ("floor" if flr > rim else "tie")
        print(f"{group:7s} {rate(rim, len(sel)):>14s} {rate(flr, len(sel)):>14s} "
              f"{med([num(r, 'rim_dist_norm') for r in sel]):>11s} "
              f"{med([num(r, 'floor_dist_norm') for r in sel]):>12s}  {better}")

    head("M1 - per desktop (unique-label, in chassis)")
    print(f"{'desktop':8s} {'view':6s} {'rim HIT':>14s} {'floor HIT':>14s}")
    for desktop in ("13", "24", "33"):
        for view in TARGETS:
            sel = [r for r in uniq if r["desktop"] == desktop and r["view"] == view]
            if not sel:
                continue
            print(f"{desktop:8s} {view:6s} "
                  f"{rate(sum(int(r['rim_hit']) for r in sel), len(sel)):>14s} "
                  f"{rate(sum(int(r['floor_hit']) for r in sel), len(sel)):>14s}")

    head("M1 - what the unique-label subset is made of (rim plane)")
    print("A hit on the motherboard is nearly free: its polygon covers most of the\n"
          "chassis. Per-label counts say how much of the headline that is.")
    print(f"{'label':34s} {'n':>5s} {'med size px':>11s} {'rim HIT':>14s} "
          f"{'rim d/size':>10s}")
    per_label = defaultdict(list)
    for r in uniq:
        per_label[r["label"]].append(r)
    for label, sel in sorted(per_label.items(), key=lambda kv: -len(kv[1])):
        print(f"{label[:34]:34s} {len(sel):5d} "
              f"{np.median([num(r, 'tgt_size_px') for r in sel]):11.0f} "
              f"{rate(sum(int(r['rim_hit']) for r in sel), len(sel)):>14s} "
              f"{med([num(r, 'rim_dist_norm') for r in sel]):>10s}")

    head("M1 - unique-label rows by part size (rim plane; motherboard excluded)")
    print("sqrt(target area) is the part's own length scale in target pixels.")
    bands = ((0, 40, "tiny  <40px"), (40, 100, "small 40-100"),
             (100, 250, "medium 100-250"), (250, 1e9, "large >250"))
    print(f"{'view':6s} {'band':16s} {'rim HIT':>14s} {'floor HIT':>14s} "
          f"{'best-plane HIT':>15s} {'rim d/size':>10s}")
    nomb = [r for r in uniq if r["cls"] != "motherboard"]
    for view in TARGETS:
        for lo, hi, name in bands:
            sel = [r for r in nomb if r["view"] == view
                   and lo <= num(r, "tgt_size_px") < hi]
            if not sel:
                continue
            best = max(("rim", "floor"),
                       key=lambda p: sum(int(r[f"{p}_hit"]) for r in sel))
            print(f"{view:6s} {name:16s} "
                  f"{rate(sum(int(r['rim_hit']) for r in sel), len(sel)):>14s} "
                  f"{rate(sum(int(r['floor_hit']) for r in sel), len(sel)):>14s} "
                  f"{rate(sum(int(r[best + '_hit']) for r in sel), len(sel)):>15s} "
                  f"{med([num(r, 'rim_dist_norm') for r in sel]):>10s}")
        print()

    head("M2 - the policy Plan B would actually ship: pick the plane by class group")
    print("flat parts (board, RAM, CPU, connectors, screws) -> floor homography;\n"
          "tall parts and rim parts -> rim homography. Unique-label rows.")
    print(f"{'view':6s} {'policy HIT':>14s} {'rim only':>14s} {'floor only':>14s} "
          f"{'oracle best':>14s}")
    for view in TARGETS:
        sel = [r for r in uniq if r["view"] == view]
        if not sel:
            continue
        pol = sum(int(r["floor_hit" if r["group"] == "flat" else "rim_hit"])
                  for r in sel)
        orc = sum(max(int(r["rim_hit"]), int(r["floor_hit"])) for r in sel)
        print(f"{view:6s} {rate(pol, len(sel)):>14s} "
              f"{rate(sum(int(r['rim_hit']) for r in sel), len(sel)):>14s} "
              f"{rate(sum(int(r['floor_hit']) for r in sel), len(sel)):>14s} "
              f"{rate(orc, len(sel)):>14s}")

    head("M1 - headline without the motherboard (unique-label, in chassis)")
    print(f"{'view':6s} {'rim HIT':>14s} {'floor HIT':>14s} {'baseline':>14s}")
    for view in TARGETS:
        sel = [r for r in nomb if r["view"] == view]
        print(f"{view:6s} "
              f"{rate(sum(int(r['rim_hit']) for r in sel), len(sel)):>14s} "
              f"{rate(sum(int(r['floor_hit']) for r in sel), len(sel)):>14s} "
              f"{rate(sum(int(r['centre_baseline_hit']) for r in sel), len(sel)):>14s}")

    # ---------------- identity ---------------- #
    ident = read("m1_identity.csv")
    head("M1/M2 - how far the projected point actually lands, in target pixels")
    print("Absolute error is what decides a HIT: a part is hit when it is bigger\n"
          "than the error. Quartiles over the unique-label rows, motherboard out.")
    print(f"{'view':6s} {'frame w':>8s} {'plane':6s} {'p25':>7s} {'median':>7s} "
          f"{'p75':>7s} {'p90':>7s}  {'median as % of frame width':>10s}")
    width = {"oak1": 4032, "oak2": 4032, "rs": 1280}
    for view in TARGETS:
        sel = [r for r in nomb if r["view"] == view]
        for plane in ("rim", "floor"):
            d = np.asarray([num(r, f"{plane}_dist_px") for r in sel])
            d = d[~np.isnan(d)]
            if d.size < MIN_N:
                continue
            q = np.percentile(d, [25, 50, 75, 90])
            print(f"{view:6s} {width[view]:8d} {plane:6s} {q[0]:7.0f} {q[1]:7.0f} "
                  f"{q[2]:7.0f} {q[3]:7.0f}  {100 * q[1] / width[view]:>10.1f}%")

    head("calibration check - the motherboard, per (desktop, view)")
    print("The board is one big unambiguous object on the floor plane, present in\n"
          "every view, so its projection error is a clean read on how good the four\n"
          "clicks were. Error is also given as a fraction of the clicked rim quad's\n"
          "longer side, which is the chassis's own size in that view.")
    try:
        import yaml

        from experiments.plan_b_probe.transfer.common import HERE
        quads = (yaml.safe_load((HERE / "corners.yaml").read_text(encoding="utf-8"))
                 or {}).get("views", {})
    except Exception:
        quads = {}
    print(f"{'desktop':8s} {'view':6s} {'n':>4s} {'rim px':>8s} {'floor px':>9s} "
          f"{'chassis px':>11s} {'rim %':>7s} {'floor %':>8s}")
    for desktop in ("13", "24", "33"):
        for view in TARGETS:
            sel = [r for r in rows if r["desktop"] == desktop and r["view"] == view
                   and r["cls"] == "motherboard"]
            if not sel:
                continue
            q = quads.get(f"{desktop}/{view}", {}).get("rim")
            span = float("nan")
            if q:
                pts = np.asarray([q[c] for c in ("s_tl", "s_tr", "s_br", "s_bl")],
                                 dtype=float)
                span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
            rim_d = float(np.median([num(r, "rim_dist_px") for r in sel]))
            flr_d = float(np.median([num(r, "floor_dist_px") for r in sel]))
            print(f"{desktop:8s} {view:6s} {len(sel):4d} {rim_d:8.0f} {flr_d:9.0f} "
                  f"{span:11.0f} {100 * rim_d / span:6.1f}% {100 * flr_d / span:7.1f}%")

    head("M1 - identity: does a Hungarian assignment agree with the #ordinal?")
    print(f"{'view':6s} {'agree':>14s}")
    for view in TARGETS:
        sel = [r for r in ident if r["view"] == view]
        print(f"{view:6s} {rate(sum(int(r['agrees']) for r in sel), len(sel)):>14s}")
    sel = ident
    print(f"{'all':6s} {rate(sum(int(r['agrees']) for r in sel), len(sel)):>14s}")

    # ---------------- M3 ---------------- #
    jit = read("m3_jitter.csv")
    if jit:
        head("M3 - HIT rate under N(0, sigma) jitter of the four target clicks")
        print(f"{'view':6s} {'plane':6s} {'sigma=0':>14s} {'sigma=3':>14s} "
              f"{'sigma=6':>14s} {'sigma=12':>14s}")
        base = defaultdict(lambda: [0, 0])
        for r in uniq:
            for plane in ("rim", "floor"):
                b = base[(r["view"], plane)]
                b[0] += int(r[f"{plane}_hit"])
                b[1] += 1
        agg = defaultdict(lambda: [0, 0])
        for r in jit:
            if r["unique_label"] != "1":
                continue
            a = agg[(r["view"], r["plane"], r["sigma"])]
            a[0] += int(r["hits"])
            a[1] += int(r["n"])
        for view in TARGETS:
            for plane in ("rim", "floor"):
                cells = [rate(*base[(view, plane)])]
                for sigma in ("3.0", "6.0", "12.0"):
                    cells.append(rate(*agg[(view, plane, sigma)]))
                print(f"{view:6s} {plane:6s} " + " ".join(f"{c:>14s}" for c in cells))

        head("M3 - how far the jitter actually moves the projected point")
        print("HIT is flat in sigma only if the jitter displacement is small next to\n"
              "the systematic error. This measures the displacement itself: 200 jitters\n"
              "of the four target clicks, applied to a grid of points inside the scan\n"
              "quad, median |projected - unjittered| in target pixels.")
        from experiments.plan_b_probe.transfer.m1_m2_m3_homography import (
            homography as _hom, load_quads as _lq, project as _proj)
        qd = _lq()
        rng = np.random.default_rng(7)
        print(f"{'view':6s} {'desktop':8s} {'sigma=3':>9s} {'sigma=6':>9s} "
              f"{'sigma=12':>9s}")
        for view in TARGETS:
            for desktop in (13, 24, 33):
                if (desktop, view, "rim") not in qd:
                    continue
                src, dst = qd[(desktop, "scan", "rim")], qd[(desktop, view, "rim")]
                u, v = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
                grid = np.stack([
                    src[0] * (1 - u).ravel()[:, None] * (1 - v).ravel()[:, None]
                    + src[1] * u.ravel()[:, None] * (1 - v).ravel()[:, None]
                    + src[3] * (1 - u).ravel()[:, None] * v.ravel()[:, None]
                    + src[2] * u.ravel()[:, None] * v.ravel()[:, None]
                ]).reshape(-1, 2)
                ref = _proj(_hom(src, dst), grid)
                cells = []
                for sigma in (3.0, 6.0, 12.0):
                    d = []
                    for _ in range(200):
                        h = _hom(src, dst + rng.normal(0, sigma, size=dst.shape))
                        d.append(np.linalg.norm(_proj(h, grid) - ref, axis=1))
                    cells.append(f"{np.median(np.concatenate(d)):9.1f}")
                print(f"{view:6s} {desktop:<8d} " + " ".join(cells))

    # ---------------- M4 / M5 ---------------- #
    ev = read("m4_per_event.csv")
    done = [r for r in ev if r.get("resolved") == "1"]
    head("M4 - per-view difference map between frames k-1 and k")
    print(f"events with a removal action: {len(ev)}; "
          f"resolved to exactly one draft: {len(done)}")
    print(f"\n{'view':6s} {'top-1 blob':>14s} {'top-3 blob':>14s} "
          f"{'top-1 overlap':>14s} {'top-3 overlap':>14s}")
    for view in ("scan",) + TARGETS:
        sel = [r for r in done if r["view"] == view]
        if not sel:
            continue
        print(f"{view:6s} "
              f"{rate(sum(int(r['top1_hit']) for r in sel), len(sel)):>14s} "
              f"{rate(sum(int(r['top3_hit']) for r in sel), len(sel)):>14s} "
              f"{rate(sum(int(r['top1_overlap']) for r in sel), len(sel)):>14s} "
              f"{rate(sum(int(r['top3_overlap']) for r in sel), len(sel)):>14s}")

    head("M4 - by class group (all views pooled)")
    print(f"{'group':7s} {'top-1':>14s} {'top-3':>14s}")
    for group in GROUPS:
        sel = [r for r in done if r.get("group") == group]
        if not sel:
            continue
        print(f"{group:7s} {rate(sum(int(r['top1_hit']) for r in sel), len(sel)):>14s} "
              f"{rate(sum(int(r['top3_hit']) for r in sel), len(sel)):>14s}")

    head("M4 - step-alignment probe (top-3 hit with the polygon taken at k-1+off)")
    print(f"{'view':6s} {'off=-1':>14s} {'off=0':>14s} {'off=+1':>14s}")
    for view in ("scan",) + TARGETS:
        sel = [r for r in done if r["view"] == view]
        if not sel:
            continue
        cells = []
        for off in ("-1", "+0", "+1"):
            col = f"top3_hit_off{off}"
            vals = [int(r[col]) for r in sel if r.get(col) not in ("", None)]
            cells.append(rate(sum(vals), len(vals)))
        print(f"{view:6s} " + " ".join(f"{c:>14s}" for c in cells))

    head("M5 - homography point snapped to the nearest top-3 diff blob")
    print(f"{'view':6s} {'M1 point':>14s} {'M4 top-1':>14s} {'M5 combined':>14s}")
    for view in TARGETS:
        sel = [r for r in done if r["view"] == view and r.get("m5_hit") not in ("", None)]
        if not sel:
            continue
        print(f"{view:6s} "
              f"{rate(sum(int(r['m1_hit']) for r in sel), len(sel)):>14s} "
              f"{rate(sum(int(r['top1_hit']) for r in sel), len(sel)):>14s} "
              f"{rate(sum(int(r['m5_hit']) for r in sel), len(sel)):>14s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
