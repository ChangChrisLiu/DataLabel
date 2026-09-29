"""Tables for the L1 report, from ``events_<tag>.csv``.

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l1_localise.analyse --tag v1

Derived here (never in the runner, so nothing is tuned where it is measured):

* **M1** -- M0's top box, withheld when the top blob's area is outside the
  class band by more than a factor ``k``;
* **M4 policies** -- which guess to arm per event, from the M0 / M3 / M4 rows;
* the **held-out choice** of the M3 and M4 variant: picked on desktop 24 only,
  then reported on desktops 13 + 33 and on all three.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l1_localise import base  # noqa: E402

VIEWS = list(base.VIEWS)
BUCKETS = [b[0] for b in base.BUCKETS]
KS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, math.inf]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def pct(x) -> str:
    x = list(x)
    return f"{100.0 * np.mean(x):.1f}" if x else "-"


def med(x) -> str:
    x = [v for v in x if v == v]
    return f"{np.median(x):.2f}" if x else "-"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def load(tag: str) -> pd.DataFrame:
    """One or more runs (``v1,v2``); a method measured twice keeps the first."""
    parts = [pd.read_csv(base.OUT / f"events_{t}.csv") for t in tag.split(",")]
    d = pd.concat(parts, ignore_index=True)
    d["ev"] = (d["desktop"].astype(str) + ":" + d["view"] + ":"
               + d["step"].astype(str))
    return d.drop_duplicates(subset=["ev", "method"], keep="first"
                             ).reset_index(drop=True)


def event_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per event (the columns shared by every method)."""
    keep = ["ev", "desktop", "view", "step", "cls", "label", "group", "bucket",
            "bucketB", "part_width", "partB_width", "loc_instance", "gtB_same",
            "sib_step", "sib_gap", "sib_n", "band_ship", "band_lodo", "roi_area",
            "gtA_dE", "gtB_dE", "instance", "target"]
    keep = [k for k in keep if k in d.columns]
    return d[d["method"] == "M0"][keep].set_index("ev")


def hits(d: pd.DataFrame, method: str, gt: str, col: str = "in1") -> pd.Series:
    s = d[d["method"] == method].set_index("ev")[f"{gt}_{col}"]
    return s


def with_fallback(d: pd.DataFrame, primary: str, fallback: str = "M0") -> pd.DataFrame:
    """``primary``'s rows where it ran, ``fallback``'s elsewhere."""
    p = d[d["method"] == primary].set_index("ev")
    f = d[d["method"] == fallback].set_index("ev")
    out = f.copy()
    out.loc[p.index] = p
    out["method"] = f"{primary}|{fallback}"
    return out.reset_index()


# --------------------------------------------------------------------------- #
# M1: the gate
# --------------------------------------------------------------------------- #
def band_of(row, which: str):
    raw = row.get(f"band_{which}")
    if not isinstance(raw, str) or not raw:
        return None
    return tuple(json.loads(raw))


def m1_outside(row, which: str) -> float:
    """Factor by which M0's top-blob area is outside the class band (1 = in)."""
    area = row.get("p1_area")
    band = band_of(row, which)
    if band is None or not (area == area) or area <= 0:
        return 1.0
    lo, hi = band
    if area < lo:
        return lo / area
    if area > hi:
        return area / hi
    return 1.0


def m1_table(d: pd.DataFrame, gt: str, which: str, views=None,
             good_col: str = "in1", base_m: str = "M0") -> pd.DataFrame:
    """``good_col``: ``in1`` = the point is in the part; ``touch1`` = the box
    contains any part pixel (a looser "not wrong").  ``base_m``: whose top
    blob is gated (B2's ``M0`` or the app's ``M0app``)."""
    m0 = d[d["method"] == base_m].copy()
    if views:
        m0 = m0[m0["view"].isin(views)]
    if gt == "B":
        m0 = m0[m0["loc_instance"].notna() & (m0["loc_instance"] != "")]
    m0["out"] = [m1_outside(r, which) for _, r in m0.iterrows()]
    has = m0["n_props"] > 0
    good = (m0[f"{gt}_{good_col}"] == 1) & has
    wrong = (~good) & has
    rows = []
    for k in KS:
        withhold = has & (m0["out"] > k)
        armed = has & ~withhold
        rows.append({
            "k": "inf" if k == math.inf else f"{k:g}",
            "boxes offered": int(has.sum()),
            "withheld %": pct(withhold[has]),
            "precision of armed %": pct(good[armed]) if armed.any() else "-",
            "good withheld %": pct(withhold[good]) if good.any() else "-",
            "wrong cut %": pct(withhold[wrong]) if wrong.any() else "-",
            "point-in-part (all events) %": pct((good & armed)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# per view x bucket
# --------------------------------------------------------------------------- #
def grid(d: pd.DataFrame, methods: list[str], gt: str, col: str = "in1",
         events=None) -> pd.DataFrame:
    """point-in-part % per view x bucket (n in brackets), one row per method."""
    bcol = "bucket" if gt == "A" else "bucketB"
    rows = []
    for m in methods:
        sel = d[d["method"] == m]
        if events is not None:
            sel = sel[sel["ev"].isin(events)]
        if gt == "B":
            sel = sel[sel[bcol].notna() & (sel[bcol] != "")]
        row = {"method": m}
        for v in VIEWS:
            for b in BUCKETS:
                s = sel[(sel["view"] == v) & (sel[bcol] == b)][f"{gt}_{col}"]
                row[f"{v} {b}"] = f"{pct(s)} ({len(s)})" if len(s) else "-"
        for b in BUCKETS:
            s = sel[sel[bcol] == b][f"{gt}_{col}"]
            row[f"all {b}"] = f"{pct(s)} ({len(s)})" if len(s) else "-"
        s = sel[f"{gt}_{col}"]
        row["all"] = f"{pct(s)} ({len(s)})"
        rows.append(row)
    return pd.DataFrame(rows)


def compact(d: pd.DataFrame, methods: list[str], gt: str, col: str = "in1",
            events=None) -> pd.DataFrame:
    """Views as rows, buckets as columns -- the headline layout."""
    bcol = "bucket" if gt == "A" else "bucketB"
    rows = []
    for v in VIEWS + ["all"]:
        for m in methods:
            sel = d[d["method"] == m]
            if events is not None:
                sel = sel[sel["ev"].isin(events)]
            if gt == "B":
                sel = sel[sel[bcol].notna() & (sel[bcol] != "")]
            if v != "all":
                sel = sel[sel["view"] == v]
            row = {"view": v, "method": m}
            for b in BUCKETS:
                s = sel[sel[bcol] == b][f"{gt}_{col}"]
                row[b] = f"{pct(s)} ({len(s)})" if len(s) else "-"
            s = sel[f"{gt}_{col}"]
            row["all"] = f"{pct(s)} ({len(s)})"
            rows.append(row)
    return pd.DataFrame(rows)


def offsets(d: pd.DataFrame, methods: list[str], gt: str, events=None):
    """Median distance from the armed point to the part, in part widths."""
    bcol = "bucket" if gt == "A" else "bucketB"
    rows = []
    for m in methods:
        sel = d[d["method"] == m]
        if events is not None:
            sel = sel[sel["ev"].isin(events)]
        if gt == "B":
            sel = sel[sel[bcol].notna() & (sel[bcol] != "")]
        row = {"method": m}
        for b in BUCKETS:
            s = sel[sel[bcol] == b]
            row[f"{b} dist/width"] = med(s[f"{gt}_dist1n"].fillna(np.inf))
            row[f"{b} px"] = med(s[f"{gt}_dist1"].fillna(np.inf))
        rows.append(row)
    return pd.DataFrame(rows)


def runtime(d: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for m in methods:
        row = {"method": m}
        for v in VIEWS:
            s = d[(d["method"] == m) & (d["view"] == v)]["secs"]
            row[v] = (f"{np.median(s):.3f} / {np.percentile(s, 90):.3f}"
                      if len(s) else "-")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# policies (M4 combined with M1's gate)
# --------------------------------------------------------------------------- #
def policy(d: pd.DataFrame, appearance: str | None, k: float,
           which: str = "lodo", order: str = "sib_first", name: str = "",
           last: str | None = None, base_m: str = "M0") -> pd.DataFrame:
    """Per event: an appearance guess, M0's box, a last resort, or nothing.

    ``sib_first``: the appearance guess whenever a sibling exists, otherwise
    M0's box if its area is within ``k`` of the band, otherwise ``last``'s
    guess (e.g. M5), otherwise nothing.
    ``band_first``: M0's box if its area fits (within ``k``), otherwise the
    appearance guess if a sibling exists, otherwise ``last``, otherwise
    nothing.  ``appearance=None`` with no ``last`` is plain M1.
    """
    m0 = d[d["method"] == base_m].set_index("ev")
    ap = (d[d["method"] == appearance].set_index("ev") if appearance
          else m0.iloc[0:0])
    lr = d[d["method"] == last].set_index("ev") if last else m0.iloc[0:0]
    out = []
    for ev, r in m0.iterrows():
        fits = r["n_props"] > 0 and m1_outside(r, which) <= k
        use = None
        if order == "sib_first":
            if ev in ap.index:
                use = ap.loc[ev]
            elif fits:
                use = r
        else:
            if fits:
                use = r
            elif ev in ap.index:
                use = ap.loc[ev]
        if use is None and ev in lr.index:
            use = lr.loc[ev]
        if use is None:
            row = r.copy()
            for c in list(row.index):
                if c.startswith(("A_", "B_")):
                    row[c] = 0 if c.endswith(("in1", "in3", "inall", "touch1")) else np.nan
            row["n_props"] = 0
            row["withheld"] = 1
        else:
            row = use.copy()
            row["withheld"] = 0
            row["secs"] = float(r["secs"]) + (float(use["secs"]) if use is not r else 0.0)
        row["method"] = name or f"P[{appearance},{order},k={k:g},{last}]"
        out.append(row)
    return pd.DataFrame(out).reset_index().rename(columns={"index": "ev"})


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args(argv)
    d = load(args.tag)
    E = event_table(d)
    out: list[str] = [f"# L1 tables ({args.tag})", ""]

    # ---- reproduction ------------------------------------------------------
    out += ["## M0 reproduces B2", ""]
    rows = []
    for v in VIEWS:
        s = d[(d["method"] == "M0") & (d["view"] == v)]["A_in1"]
        rows.append({"view": v, "n": len(s), "point-in-part top-1 %": pct(s)})
    out += [md(pd.DataFrame(rows)), ""]

    # ---- GT audit ----------------------------------------------------------
    out += ["## Ground-truth audit", ""]
    rows = []
    for b in BUCKETS:
        e = E[E["bucket"] == b]
        rows.append({
            "bucket (GT-A width)": b, "n": len(e),
            "GT-B = GT-A": int((e["gtB_same"] == 1).sum()),
            "GT-B differs": int((e["gtB_same"] == 0).sum()),
            "GT-B unresolved": int(e["loc_instance"].isna().sum()),
            "median dE in GT-A box": med(e["gtA_dE"]),
            "median dE in GT-B box": med(e["gtB_dE"]),
        })
    out += [md(pd.DataFrame(rows)), ""]
    diff = E[E["gtB_same"] == 0]
    out += [f"Where they differ (n={len(diff)}): median dE inside the GT-A box "
            f"{med(diff['gtA_dE'])}, inside the GT-B box {med(diff['gtB_dE'])}; "
            f"GT-A box dE < 3 on {int((diff['gtA_dE'] < 3).sum())}, "
            f"GT-B box dE < 3 on {int((diff['gtB_dE'] < 3).sum())}.", ""]

    # ---- coverage ------------------------------------------------------------
    out += ["## M3 coverage: a sibling of the target class already drawn", ""]
    rows = []
    for v in VIEWS + ["all"]:
        e = E if v == "all" else E[E["view"] == v]
        row = {"view": v}
        for b in BUCKETS:
            eb = e[e["bucket"] == b]
            if len(eb):
                row[b] = (f"{pct(eb['sib_n'] > 0)} ({len(eb)}); "
                          f"at k: {pct(eb['sib_gap'] == 0)}")
            else:
                row[b] = "-"
        row["all"] = f"{pct(e['sib_n'] > 0)} ({len(e)})"
        rows.append(row)
    out += [md(pd.DataFrame(rows)), ""]

    # ---- variant choice on D24 -------------------------------------------------
    variants = [m for m in d["method"].unique()
                if m.startswith(("M3", "M4", "M5"))]
    d24 = d[d["desktop"] == 24]
    small24 = set(E[(E["desktop"] == 24) & (E["bucket"] == "<40")].index)
    rows = []
    for m in variants:
        s = d24[d24["method"] == m]
        c = s[s["ev"].isin(small24)]
        rows.append({"variant": m, "n": len(s),
                     "A in1 %": pct(s["A_in1"]), "B in1 %": pct(s["B_in1"].fillna(0)),
                     "A in1 <40 %": pct(c["A_in1"]),
                     "B in1 <40 %": pct(c["B_in1"].fillna(0)),
                     "A in3 %": pct(s["A_in3"]),
                     "oak s med": f"{s[s['view'].str.startswith('oak')]['secs'].median():.3f}"})
    ch = pd.DataFrame(rows)
    out += ["## Variant choice on desktop 24 only", "",
            "M3/M4 rows: the sibling-covered D24 events; M5 rows: every D24 "
            "event.  Chosen by GT-B top-1 (GT-A rewards landing on a sibling "
            "that is still there, see the audit), GT-A as the tie-break.", "",
            md(ch), ""]
    ch["score"] = [float(b) + 0.01 * float(a)
                   for a, b in zip(ch["A in1 %"], ch["B in1 %"])]
    pick = {}
    for fam in ("M3", "M4", "M5"):
        # stable: on a tie the variant listed first (the simpler one) wins
        sel = ch[ch["variant"].str.startswith(fam)].sort_values(
            "score", ascending=False, kind="stable")
        pick[fam] = sel.iloc[0]["variant"] if len(sel) else None
    best3, best4, best5 = pick["M3"], pick["M4"], pick["M5"]
    out += [f"Chosen on D24: M3 = `{best3}`, M4 = `{best4}`, M5 = `{best5}`.", ""]

    # ---- M1 ------------------------------------------------------------------
    for which in ("lodo", "ship"):
        for gt in ("A", "B"):
            out += [f"## M1 sweep -- band `{which}`, GT-{gt}, all views", "",
                    md(m1_table(d, gt, which)), ""]
    out += ["## M1 sweep -- band `lodo`, GT-A, good = box touches the part", "",
            md(m1_table(d, "A", "lodo", good_col="touch1")), ""]
    if "M0app" in set(d["method"]):
        for which in ("lodo", "ship"):
            for gt in ("A", "B"):
                out += [f"## M1 sweep on the app's own blob (`M0app`) -- band "
                        f"`{which}`, GT-{gt}", "",
                        md(m1_table(d, gt, which, base_m="M0app")), ""]
        for v in VIEWS:
            out += [f"### M1 on `M0app` -- band `lodo`, GT-B, {v}", "",
                    md(m1_table(d, "B", "lodo", [v], base_m="M0app")), ""]
    for v in VIEWS:
        out += [f"### M1 sweep -- band `lodo`, GT-A, {v}", "",
                md(m1_table(d, "A", "lodo", [v])), ""]

    # ---- policies ------------------------------------------------------------
    pols = [
        policy(d, None, 3.0, "lodo", name="M1 (k=3)"),
        policy(d, None, 5.0, "lodo", name="M1 (k=5)"),
        policy(d, best4, 3.0, "lodo", "sib_first", name="P1: M4 | M1(k=3)"),
        policy(d, best4, 3.0, "lodo", "band_first", name="P2: M1(k=3) | M4"),
        policy(d, best4, math.inf, "lodo", "sib_first", name="P3: M4 | M0"),
    ]
    if best5:
        pols += [
            policy(d, best4, 3.0, "lodo", "sib_first", last=best5,
                   name="P4: M4 | M1(k=3) | M5"),
            policy(d, None, 3.0, "lodo", "band_first", last=best5,
                   name="P5: M1(k=3) | M5"),
        ]
    have_app = "M0app" in set(d["method"])
    if have_app:
        pols += [
            policy(d, None, 3.0, "lodo", name="M1app (k=3)", base_m="M0app"),
            policy(d, best4, 3.0, "lodo", "sib_first", base_m="M0app",
                   name="P1app: M4 | M1app(k=3)"),
            policy(d, best4, math.inf, "lodo", "sib_first", base_m="M0app",
                   name="P3app: M4 | M0app"),
        ]
    extra = [with_fallback(d, best3)]
    if best3 != "M3" and "M3" in set(d["method"]):
        extra.append(with_fallback(d, "M3"))
    allp = pd.concat([d] + pols + extra, ignore_index=True)
    main_methods = (["M0"] + (["M0app"] if have_app else [])
                    + ["split", "M2e"] + (["M2e_app"] if "M2e_app" in set(d["method"])
                                          else [])
                    + [f"{best3}|M0"] + ([best5] if best5 else [])
                    + [p["method"].iloc[0] for p in pols])

    for gt in ("A", "B"):
        out += [f"## Headline, GT-{gt}: point-in-part top-1 %, view x part width "
                f"(n)", "", md(compact(allp, main_methods, gt)), ""]
        ub = (["M0"] + (["M0app"] if have_app else [])
              + ["split", "M2e", f"{best3}|M0", "P3: M4 | M0"]
              + ([best5] if best5 else []))
        out += [f"## Upper bound, GT-{gt}: any of the top 3", "",
                md(compact(allp, ub, gt, "in3")), ""]
        out += [f"## Upper bound, GT-{gt}: any candidate (M0/M2: top 8 blobs; "
                f"M3/M4: top 5, or all re-ranked; M5: top 5)", "",
                md(compact(allp, ub, gt, "inall")), ""]
        out += [f"## Offset of the armed point, GT-{gt} (median; withheld = inf)",
                "", md(offsets(allp, main_methods, gt)), ""]

    # sibling-covered only: M0 vs M3 vs M4 on the same events
    cov = set(E[E["sib_n"] > 0].index)
    fam = ["M0", "M2e", "M3", "M3j", "M3js", best3, "M4_b1", "M4js_b1", best4]
    fam = list(dict.fromkeys(m for m in fam if m in set(d["method"])))
    for gt in ("A", "B"):
        out += [f"## Sibling-covered events only, GT-{gt}", "",
                md(compact(allp, fam, gt, events=cov)), ""]
    # the sequence prior: only where the part removed at k+1 is known
    if "prev_same_cls" in d.columns:
        # only the later runs recorded these; take them from any row
        same = d.groupby("ev")["prev_same_cls"].max()
        isb = d.groupby("ev")["prev_is_gtB"].max()
        pv = set(same.index[same == 1])
        leak = set(isb.index[isb == 1])
        out += [f"## Events with the k+1 part known and of the same class "
                f"(n={len(pv & cov)} sibling-covered); the 'k+1 part' is this "
                f"event's own part (a stale draft) on {len(leak & pv & cov)} of "
                f"them, excluded below", ""]
        ok = (pv & cov) - leak
        pf = [m for m in ["M0", "M3js", "M3js_p", "M4js_b1", "M4js_b1_p",
                          "M4js_b1_p05", "M4js_b1_p_s10", "M4js_rr10_p",
                          "M4js_b1_rr10_p", "M5cs", "M5cs_p"]
              if m in set(d["method"])]
        for gt in ("A", "B"):
            out += [f"### GT-{gt}", "", md(compact(allp, pf, gt, events=ok)), ""]
    # held-out view: D13 + D33 only
    ho = set(E[E["desktop"].isin([13, 33])].index)
    for gt in ("A", "B"):
        out += [f"## Held-out desktops 13 + 33, GT-{gt}", "",
                md(compact(allp, main_methods, gt, events=ho)), ""]

    # ---- M2 ------------------------------------------------------------------
    out += ["## M2 registration", ""]
    rows = []
    for v in VIEWS:
        for m in ("M0c", "M2p", "M2e"):
            s = d[(d["method"] == m) & (d["view"] == v)]
            sh = np.hypot(s.get("reg_dx", pd.Series(dtype=float)).fillna(0),
                          s.get("reg_dy", pd.Series(dtype=float)).fillna(0))
            top_wrong = s[(s["n_props"] > 0)]["A_touch1"]
            rows.append({
                "view": v, "method": m, "n": len(s),
                "median |shift| px": f"{np.median(sh):.2f}" if m != "M0c" else "-",
                "p90 |shift| px": f"{np.percentile(sh, 90):.2f}" if m != "M0c" else "-",
                "rejected": int(s.get("reg_rejected", pd.Series(dtype=float)).fillna(0).sum()),
                "median blobs >= min_area": med(s["n_blobs"]),
                "top box misses part (A) %": pct(1 - top_wrong),
                "in1 A %": pct(s["A_in1"]),
                "in1 A <40 %": pct(s[s["bucket"] == "<40"]["A_in1"]),
            })
    out += [md(pd.DataFrame(rows)), ""]

    # ---- runtime -------------------------------------------------------------
    out += ["## Runtime, seconds per event (median / p90), frames already decoded",
            "", md(runtime(allp, list(dict.fromkeys(
                ["M0", "M0c", "M2p", "M2e", "split", "M3", best3, best4]
                + ([best5] if best5 else []))))), "",
            "Contended numbers (other variants and a test suite shared the "
            "CPU); `timing.py` has the isolated ones.", ""]

    text = "\n".join(out)
    stem = args.tag.replace(",", "+")
    path = base.OUT / f"tables_{stem}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    allp.to_csv(base.OUT / f"events_{stem}_with_policies.csv", index=False)

    # the report's headline: the methods the task names, one table per GT
    rep = [m for m in ["M0", "M0app", "M2e", "M3|M0", f"{best3}|M0",
                       "P3: M4 | M0", "M1app (k=3)", "P4: M4 | M1(k=3) | M5",
                       best5]
           if m and m in set(allp["method"])]
    rep = list(dict.fromkeys(rep))
    htxt = [f"M3 variant `{best3}`, M4 variant `{best4}`, M5 `{best5}` "
            f"(chosen on D24).", ""]
    for gt in ("A", "B"):
        htxt += [f"### GT-{gt}: point-in-part top-1 % (n)", "",
                 md(compact(allp, rep, gt)), ""]
        htxt += [f"### GT-{gt}: any of the top 3 % (n)", "",
                 md(compact(allp, rep, gt, "in3")), ""]
    (base.OUT / "headline.md").write_text("\n".join(htxt), encoding="utf-8")

    # one row per event, the headline methods side by side
    key = [m for m in ["M0", "M0app", "M2e", "M3", best3, best4, "M4js_b1",
                       best5, "M1app (k=3)", "P3: M4 | M0", "P4: M4 | M1(k=3) | M5"]
           if m and m in set(allp["method"])]
    key = list(dict.fromkeys(key))
    wide = E.reset_index()[["ev", "desktop", "view", "step", "target", "label",
                            "bucket", "part_width", "instance", "loc_instance",
                            "bucketB", "partB_width", "sib_n", "sib_gap",
                            "gtA_dE", "gtB_dE"]].set_index("ev")
    for m in key:
        s = allp[allp["method"] == m].set_index("ev")
        for c in ("A_in1", "B_in1", "A_dist1", "B_dist1", "p1_box"):
            if c in s.columns:
                wide[f"{m}:{c}"] = s[c]
        if "withheld" in s.columns:
            wide[f"{m}:withheld"] = s["withheld"]
    wide.to_csv(base.OUT / "events_headline.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
