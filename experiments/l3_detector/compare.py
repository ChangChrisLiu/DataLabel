"""Side-by-side tables: YOLO26n (L2) against the larger models.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.compare

Reads each model's ``det_metrics.csv``, ``guess_events.csv`` (L2's code, see
``evaluate``) and ``timing_l3.json`` from ``<OUT>/<model>/`` and writes
``<OUT>/compare.md``:

1. detector metrics (AP50, hit@0.10, FP/frame) per class x view x held-out
   desktop for screw, connector, ram_latch;
2. the guess (``L2small``: detector for the six small classes, pre-declared in
   L2) on GT-B: top-1 / top-3 for parts < 40 px per view, and paired bootstrap
   CIs of each model minus YOLO26n on scan and oak1;
3. by what was removed (motherboard screw, connector, cooler screw);
4. the candidate ceiling (removed part among the conf >= 0.10 candidates after
   the skip; conf >= 0.01 in brackets), small-class events;
5. runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402
from experiments.l2_detector.data import md_table  # noqa: E402

ORDER = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "rfdetr_small", "rfdetr_medium",
         "rfdetr_base"]
SMALL = ["screw", "connector", "ram_latch", "psu_latch", "cpu_socket_lever", "drive_latch"]
VIEWS = ["scan", "oak1", "oak2", "rs"]


def models() -> list[str]:
    return [m for m in ORDER if (env.OUT / m / "guess_events.csv").exists()]


def pct(s) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return 100 * s.mean() if len(s) else float("nan")


def f1(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"


def det_tables(ms: list[str]) -> list[str]:
    frames = {m: pd.read_csv(env.OUT / m / "det_metrics.csv") for m in ms}
    out = ["## 1. Detector metrics on held-out desktops (drafts as GT)", "",
           "Cell = AP50 / hit@0.10 / FP per frame@0.10. `all` pools the three folds.", ""]
    for cls in ("screw", "connector", "ram_latch"):
        rows = []
        base = frames[ms[0]]
        sel = base[(base.cls == cls) & (base.n_gt > 0)]
        for _, r in sel.iterrows():
            row = {"view": r["view"], "held-out": r["desktop"], "n": int(r["n_gt"]),
                   "width p50": r["gt_w_med"]}
            for m in ms:
                d = frames[m]
                x = d[(d.cls == cls) & (d["view"] == r["view"])
                      & (d.desktop.astype(str) == str(r["desktop"]))]
                if len(x):
                    x = x.iloc[0]
                    row[m] = f"{x['AP50']} / {x['hit@0.10']} / {x['FP/frame@0.10']}"
                else:
                    row[m] = "-"
            rows.append(row)
        df = pd.DataFrame(rows)
        df["_v"] = df["view"].map({v: i for i, v in enumerate(VIEWS)})
        df["_d"] = df["held-out"].astype(str).map(lambda s: 99 if s == "all" else int(s))
        df = df.sort_values(["_v", "_d"]).drop(columns=["_v", "_d"])
        out += [f"### {cls}", "", md_table(df), ""]
    return out


def boot_ci(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = (b - a).astype(float)
    boots = [100 * d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return 100 * d.mean(), lo, hi


def guess_tables(ms: list[str]) -> list[str]:
    ev = {m: pd.read_csv(env.OUT / m / "guess_events.csv") for m in ms}
    base = ev[ms[0]]
    for m in ms[1:]:
        if list(ev[m]["ev"]) != list(base["ev"]):
            raise SystemExit(f"{m}: event list differs from {ms[0]}")
    out = ["## 2. The guess on GT-B events (L2small policy), parts < 40 px", "",
           "Top-1 / top-3 point-in-part %.  M0app = today's guess.", ""]
    rows = []
    for v in VIEWS + ["all"]:
        x = base if v == "all" else base[base["view"] == v]
        x = x[x.bucketB == "<40"]
        idx = x.index
        row = {"view": v, "n": len(x),
               "M0app": f"{f1(pct(x.M0app_in1))} / {f1(pct(x.M0app_in3))}"}
        for m in ms:
            y = ev[m].loc[idx]
            row[m] = f"{f1(pct(y.L2small_in1))} / {f1(pct(y.L2small_in3))}"
        rows.append(row)
    out += [md_table(pd.DataFrame(rows)), ""]
    # all buckets, top-1 (no regression on large parts: they use M0app anyway)
    rows = []
    for v in VIEWS + ["all"]:
        x = base if v == "all" else base[base["view"] == v]
        row = {"view": v, "n": len(x), "M0app": f1(pct(x.M0app_in1))}
        for m in ms:
            row[m] = f1(pct(ev[m].loc[x.index].L2small_in1))
        rows.append(row)
    out += ["All sizes, top-1 %:", "", md_table(pd.DataFrame(rows)), ""]
    # paired differences vs the first model
    rows = []
    for v in ("scan", "oak1", "oak2"):
        x = base[(base["view"] == v) & (base.bucketB == "<40")]
        for m in ms[1:]:
            y = ev[m].loc[x.index]
            d, lo, hi = boot_ci(x.L2small_in1.to_numpy(), y.L2small_in1.to_numpy())
            win = int(((y.L2small_in1 == 1) & (x.L2small_in1 == 0)).sum())
            loss = int(((y.L2small_in1 == 0) & (x.L2small_in1 == 1)).sum())
            rows.append({"view": v, "model": m, "n": len(x),
                         f"top-1 minus {ms[0]}": f"{d:+.1f}",
                         "95 % CI": f"{lo:+.1f} .. {hi:+.1f}", "wins / losses": f"{win} / {loss}"})
    out += [f"Paired change in top-1 (< 40 px) against {ms[0]}, bootstrap over events:", "",
            md_table(pd.DataFrame(rows)), ""]
    # per held-out desktop
    rows = []
    for dk in (13, 24, 33):
        for v in ("scan", "oak1", "oak2"):
            x = base[(base.desktop == dk) & (base["view"] == v) & (base.bucketB == "<40")]
            if not len(x):
                continue
            row = {"desktop": dk, "view": v, "n": len(x), "M0app": f1(pct(x.M0app_in1))}
            for m in ms:
                row[m] = f1(pct(ev[m].loc[x.index].L2small_in1))
            rows.append(row)
    out += ["Per held-out desktop, < 40 px, top-1 %:", "", md_table(pd.DataFrame(rows)), ""]

    # 3. by what was removed
    out += ["## 3. By what was removed (small-class events), top-1 % (ceiling %)", ""]
    s = base[base.cls.isin(SMALL)].copy()
    s["what"] = np.where(s.cls == "screw", s.target.str.split(".").str[:2].str.join("."), s.cls)
    rows = []
    for (v, w), x in s.groupby([s["view"], "what"]):
        if w not in ("screw.motherboard", "connector", "screw.cpu_cooler"):
            continue
        row = {"view": v, "removed": w, "n": len(x), "M0app": f1(pct(x.M0app_in1))}
        for m in ms:
            y = ev[m].loc[x.index]
            row[m] = f"{f1(pct(y.L2small_in1))} ({f1(pct(y.ceilsup_c10))})"
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_v"] = df["view"].map({v: i for i, v in enumerate(VIEWS)})
    out += [md_table(df.sort_values(["_v", "removed"]).drop(columns="_v")), ""]

    # 4. ceiling
    out += ["## 4. Candidate ceiling: removed part among the candidates (small classes)", "",
            "conf >= 0.10 after the skip (conf >= 0.01 in brackets); `rank-1 | in pool` = "
            "of the events whose part was a candidate, how often dE ranked it first.", ""]
    for title, pop in (("small-class events", base[base.cls.isin(SMALL)]),
                       ("small-class events, parts < 40 px (the L2 report's ceiling)",
                        base[base.cls.isin(SMALL) & (base.bucketB == "<40")])):
        rows = []
        for v in VIEWS + ["all"]:
            x = pop if v == "all" else pop[pop["view"] == v]
            row = {"view": v, "n": len(x)}
            for m in ms:
                y = ev[m].loc[x.index]
                inpool = y[y.ceilsup_c10 == 1]
                r1 = f"{int((inpool.L2small_true_rank == 1).sum())}/{len(inpool)}"
                row[m] = f"{f1(pct(y.ceilsup_c10))} ({f1(pct(y.ceilsup_c01))}); {r1}"
            rows.append(row)
        out += [f"{title}:", "", md_table(pd.DataFrame(rows)), ""]
    return out


def timing_table(ms: list[str]) -> list[str]:
    rows = []
    for m in ms:
        p = env.OUT / m / "timing_l3.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))["summary"]
        v = s["views"]

        def mp(view, k):
            return f"{v[view][k][0]:.3f} / {v[view][k][1]:.3f}" if view in v else "-"
        rows.append({
            "model": m,
            "scan det (6 tiles)": mp("scan", "det"),
            "scan guess": mp("scan", "guess"),
            "oak1 ROI det (20 tiles)": mp("oak1", "det"),
            "oak1 guess": mp("oak1", "guess"),
            "oak1 full 12 MP": f"{s['full_oak1_median']:.3f} / {s['full_oak1_p90']:.3f}",
            "bg precompute scan / oak1 (decode + detect)":
                f"{s['bg_per_frame']['scan']:.3f} / {s['bg_per_frame']['oak1']:.3f}",
            "oak2 det": mp("oak2", "det"), "rs det": mp("rs", "det"),
            "GPU peak MB (3 folds loaded)": s["gpu_peak_mb_3_folds_loaded"],
        })
    if not rows:
        return []
    return ["## 5. Runtime (RTX 5090, fp16, median / p90 seconds per frame)", "",
            md_table(pd.DataFrame(rows)), ""]


def main() -> int:
    ms = models()
    lines = ["# L3 comparison: " + ", ".join(ms), ""]
    lines += det_tables(ms) + guess_tables(ms) + timing_table(ms)
    txt = "\n".join(lines)
    (env.OUT / "compare.md").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
