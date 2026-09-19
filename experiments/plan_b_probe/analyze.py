"""Stage 2: rig epochs (Q1) and within-sequence events (Q2) from cached features.

Everything here works off the .npz caches, so it is cheap to re-run with
different thresholds.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C            # noqa: E402
import estimate as E          # noqa: E402
from extract import Store, ORIG   # noqa: E402

# A move is called when the table landmarks shift by at least this many pixels
# of the ORIGINAL frame.  The static-camera noise floor measured on consecutive
# frames is 0.0 px median, so this is a wide margin.
MOVE_PX = 2.0
CHASSIS_PX = 12.0     # centroid shift that counts as the chassis being moved
BOUNDARY_K = 3        # frames from each side used for a between-desktop estimate


def valid_steps(st: Store):
    return [i for i in range(len(st.steps)) if st.nkp[i] > 0]


def work_wh(st: Store):
    """Working-image (w, h) for this view, for the grid-displacement measure."""
    sh = st.shapes[st.shapes[:, 0] > 0]
    if not len(sh):
        return None
    return int(sh[0][1]), int(sh[0][0])


# ------------------------------------------------------------------ Q2 events

def within_desktop(view: str, desktop: int, chstore=None):
    """Consecutive-step table motion for one desktop."""
    st = Store(view, desktop)
    up = st.upscale
    wh = work_wh(st)
    rows = []
    idx = list(range(len(st.steps)))
    prev = None
    for i in idx:
        kp, des = st.at(i)
        if kp is None:
            prev = None
            continue
        if prev is not None:
            pk, pd, pi = prev
            r = E.estimate(pk, pd, kp, des, up, wh)
            ch = _chassis_delta(chstore, pi, i, up, wh) if chstore is not None else None
            rows.append({
                "desktop": desktop, "view": view,
                "step_from": int(st.steps[pi]), "step_to": int(st.steps[i]),
                "ok": r["ok"], "reason": r["reason"],
                "n_match": r["n_match"], "n_inlier": r["n_inlier"],
                "table_px": r["px"], "rot_deg": r["rot_deg"], "scale": r["scale"],
                "chassis_px": None if ch is None else ch[0],
                "chassis_area_ratio": None if ch is None else ch[1],
                "step_from_idx": pi, "step_to_idx": i,
            })
        prev = (kp, des, i)
    return rows


def _chassis_delta(ch, i, j, up, wh=None):
    """(rigid shift in original px, area ratio) of the chassis itself.

    Measured by matching features *inside* the chassis, not by its outline: a
    teardown removes parts at nearly every step, which moves the blob's centroid
    and changes its area without the chassis itself having been touched.  A
    rigid fit over the parts that are still there is not fooled by that.
    """
    a, b = ch.blob[i], ch.blob[j]
    ar = float(b[2] / a[2]) if (np.isfinite(a).all() and np.isfinite(b).all()
                                and a[2] > 0) else float("nan")
    ka, da = ch.at(i)
    kb, db = ch.at(j)
    if ka is None or kb is None:
        return None, ar
    r = E.estimate(ka, da, kb, db, up, wh)
    return (r["px"] if r["ok"] else None), ar


def classify(table_px, table_ok, chassis_px, area_ratio):
    """camera / chassis / both / none / undetermined."""
    if not table_ok:
        return "undetermined"
    cam = table_px is not None and table_px >= MOVE_PX
    # The blob measure is only trustworthy when the blob kept its size; a hand
    # over the chassis or a big part coming off changes the area a lot.
    blob_ok = (chassis_px is not None and area_ratio is not None
               and np.isfinite(area_ratio) and 0.8 <= area_ratio <= 1.25)
    ch = blob_ok and chassis_px >= CHASSIS_PX
    if cam and ch:
        return "both"
    if cam:
        return "camera"
    if ch:
        return "chassis"
    return "none"


# ------------------------------------------------------- Q1 between desktops

def boundary_estimate(view, d0, d1, k=BOUNDARY_K):
    """Median table motion between the end of d0 and the start of d1.

    Several frames on each side are used because a single boundary frame can be
    a bad one (table buried under removed parts, hand in shot).
    """
    s0, s1 = Store(view, d0), Store(view, d1)
    up = s0.upscale
    wh = work_wh(s0)
    a = valid_steps(s0)[-k:]
    b = valid_steps(s1)[:k]
    res = []
    for i in a:
        ka, da = s0.at(i)
        for j in b:
            kb, db = s1.at(j)
            r = E.estimate(ka, da, kb, db, up, wh)
            if r["ok"]:
                res.append(r)
    if not res:
        return {"ok": False, "px": None, "rot_deg": None, "scale": None,
                "n_pairs": 0, "n_inlier": 0, "reason": "no_reliable_pair"}
    px = float(np.median([r["px"] for r in res]))
    return {
        "ok": True, "px": px,
        "rot_deg": float(np.median([r["rot_deg"] for r in res])),
        "scale": float(np.median([r["scale"] for r in res])),
        "n_pairs": len(res),
        "n_inlier": int(np.median([r["n_inlier"] for r in res])),
        "spread_px": float(np.percentile([r["px"] for r in res], 90) -
                           np.percentile([r["px"] for r in res], 10)),
        "reason": "",
    }


def run_view(view: str, with_chassis=True):
    from extract_chassis import ChStore
    events, boundaries = [], []
    for d in range(1, 67):
        ch = None
        if with_chassis:
            try:
                ch = ChStore(view, d)
            except Exception:
                ch = None
        events += within_desktop(view, d, ch)
    for d in range(1, 66):
        r = boundary_estimate(view, d, d + 1)
        r.update({"view": view, "desktop_from": d, "desktop_to": d + 1})
        boundaries.append(r)
    return events, boundaries


def epochs_from_boundaries(boundaries):
    """Contiguous runs of desktops with no detected between-desktop move."""
    eps, cur, start = [], 1, 1
    for b in boundaries:
        moved = b["ok"] and b["px"] is not None and b["px"] >= MOVE_PX
        if moved:
            eps.append((cur, start, b["desktop_from"]))
            cur += 1
            start = b["desktop_to"]
    eps.append((cur, start, 66))
    return eps


if __name__ == "__main__":
    views = sys.argv[1:] or list(C.VIEWS)
    os.makedirs(C.OUT, exist_ok=True)
    all_ev, all_bd = [], []
    for v in views:
        ev, bd = run_view(v)
        all_ev += ev
        all_bd += bd
        print("== %s: %d step pairs, %d boundaries" % (v, len(ev), len(bd)))
        for b in bd:
            if not b["ok"] or (b["px"] or 0) >= MOVE_PX:
                print("   d%02d->d%02d px=%s rot=%s scale=%s pairs=%d %s" % (
                    b["desktop_from"], b["desktop_to"],
                    None if b["px"] is None else round(b["px"], 2),
                    None if b["rot_deg"] is None else round(b["rot_deg"], 3),
                    None if b["scale"] is None else round(b["scale"], 4),
                    b["n_pairs"], b["reason"]))
    np.save(os.path.join(C.TMP, "events_raw.npy"), np.array(all_ev, dtype=object))
    np.save(os.path.join(C.TMP, "bounds_raw.npy"), np.array(all_bd, dtype=object))
    print("saved raw results")
