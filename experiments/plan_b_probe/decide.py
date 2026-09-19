"""Final decisions for Q1 and Q2, with the tape check as the arbiter.

The ORB stage is only a candidate generator: it is sensitive but it will happily
fit a transform to a bright part that slid across the table.  Every candidate is
therefore re-tested on the tape shape, which cannot be confused with hardware --
warp frame A's tape by the proposed transform and see whether it lines up with
frame B's better than before.  A candidate that does not improve the tape
overlap is reported as "no table motion", whatever the keypoints said.
"""
from __future__ import annotations

import csv
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C        # noqa: E402
import verify as VF       # noqa: E402
import validate as V      # noqa: E402
import analyze as A       # noqa: E402

CAND_PX = 1.0       # ORB magnitude worth testing
IOU_GAIN = 0.02     # tape overlap must actually improve
REPORT_PX = 2.0     # magnitude reported as a move
CHASSIS_PX = 12.0
BIG_PX = 50.0       # beyond this the tape cannot be expected to overlap at all
MIN_IOU_BASE = 0.05  # below this the tape test has no purchase either way

# Verdicts for the handful of pairs the automatic test cannot settle, each read
# off the overlay PNG by eye.  They are listed here rather than buried in prose
# so that the claim and its evidence stay together.
MANUAL = {
    ("step", "oak1", 36, 18, 19): ("camera",
                                   "tape rectangle grossly displaced in the zoom crop"),
    ("step", "oak1", 64, 19, 20): ("undetermined",
                                   "camera is inside the tower; almost no table in frame"),
    ("step", "oak1", 39, 23, 24): ("undetermined",
                                   "only a thin tape sliver visible; cannot judge"),
    ("step", "scan", 63, 31, 32): ("chassis",
                                   "chassis rotated ~90 deg, table sliver stays aligned"),
    ("bound", "oak1", 20, 21): ("camera",
                                "tape grid clearly doubled; magnitude undetermined"),
}


def load(view):
    with open(os.path.join(C.TMP, "raw_%s.pkl" % view), "rb") as f:
        return pickle.load(f)


def load_fallbacks():
    out = {}
    for name in ("bigmove.pkl", "tape.pkl"):
        p = os.path.join(C.TMP, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[name] = pickle.load(f)
    return out


def adjudicate(view, d_a, s_a, d_b, s_b):
    """Run the tape test on one pair.  Returns (verdict, detail)."""
    fs = V.frames(view)
    if (d_a, s_a) not in fs or (d_b, s_b) not in fs:
        return "undetermined", {"reason": "missing_frame"}
    S = VF.transform_for(view, d_a, s_a, s_b, desktop_b=d_b)
    if S is None:
        return "undetermined", {"reason": "no_transform"}
    chk = VF.check(fs[(d_a, s_a)].image_path(), fs[(d_b, s_b)].image_path(), view, S)
    if not chk["usable"]:
        return "undetermined", {"reason": "too_little_tape", **chk}
    gain = chk["iou_after"] - chk["iou_before"]
    chk["iou_gain"] = gain
    # If the tape barely overlaps under either hypothesis there is nothing to
    # compare, and calling that a rejection would be inventing a result.
    if max(chk["iou_before"], chk["iou_after"]) < MIN_IOU_BASE:
        chk["reason"] = "tape_no_overlap"
        return "undetermined", chk
    return ("confirmed" if gain > IOU_GAIN else "rejected"), chk


def decide_events(view):
    raw = load(view)
    out = []
    for e in raw["events"]:
        row = dict(e)
        row["verdict"] = ""
        row["iou_before"] = row["iou_after"] = row["iou_gain"] = None
        if e["ok"] and e["table_px"] is not None and e["table_px"] >= CAND_PX:
            v, det = adjudicate(view, e["desktop"], e["step_from"],
                                e["desktop"], e["step_to"])
            row["verdict"] = v
            for k in ("iou_before", "iou_after", "iou_gain"):
                row[k] = det.get(k)
        elif e["ok"]:
            row["verdict"] = "static"
        else:
            row["verdict"] = "undetermined"
        key = ("step", view, e["desktop"], e["step_from"], e["step_to"])
        if key in MANUAL:
            row["verdict"], row["manual_note"] = MANUAL[key][0], MANUAL[key][1]
            row["verdict"] = {"camera": "confirmed", "chassis": "chassis_manual",
                              "undetermined": "undetermined"}[MANUAL[key][0]]
            row["decided_by"] = "eye"
        else:
            row["decided_by"] = "tape_test" if row["verdict"] in (
                "confirmed", "rejected") else "auto"
            row["manual_note"] = ""
        out.append(row)
    return out


def classify(row):
    """camera / chassis / both / none / undetermined for one step pair."""
    if row["verdict"] == "chassis_manual":
        return "chassis"
    if row["verdict"] == "undetermined":
        return "undetermined"
    cam = (row["verdict"] == "confirmed"
           and row["table_px"] is not None and row["table_px"] >= REPORT_PX)
    ar = row.get("chassis_area_ratio")
    cp = row.get("chassis_px")
    blob_ok = (cp is not None and ar is not None and np.isfinite(ar)
               and 0.8 <= ar <= 1.25)
    ch = bool(blob_ok and cp >= CHASSIS_PX)
    if cam and ch:
        return "both"
    if cam:
        return "camera"
    if ch:
        return "chassis"
    return "none"


def decide_bounds(view, k=A.BOUNDARY_K):
    raw = load(view)
    fb = load_fallbacks()
    out = []
    for b in raw["bounds"]:
        d0, d1 = b["desktop_from"], b["desktop_to"]
        sa, sb = V.valid_steps(view, d0), V.valid_steps(view, d1)
        row = dict(b)
        row["verdict"] = ""
        row["iou_before"] = row["iou_after"] = row["iou_gain"] = None
        row["fallback_px"] = None
        row["n_checked"] = 0
        if not sa or not sb:
            row["verdict"] = "undetermined"
            out.append(row)
            continue
        if b["ok"] and b["px"] is not None and b["px"] >= CAND_PX:
            # The boundary magnitude is a median over the last k frames of d0
            # against the first k of d1, so the tape test has to cover the same
            # pairs -- adjudicating one arbitrary pair was measuring something
            # else and disagreeing with itself.
            gains, befores, afters = [], [], []
            for i in sa[-k:]:
                for j in sb[:k]:
                    v, det = adjudicate(view, d0, i, d1, j)
                    if v in ("confirmed", "rejected"):
                        gains.append(det["iou_gain"])
                        befores.append(det["iou_before"])
                        afters.append(det["iou_after"])
            row["n_checked"] = len(gains)
            if gains:
                row["iou_gain"] = float(np.median(gains))
                row["iou_before"] = float(np.median(befores))
                row["iou_after"] = float(np.median(afters))
                row["verdict"] = ("confirmed" if row["iou_gain"] > IOU_GAIN
                                  else "rejected")
            else:
                row["verdict"] = "undetermined"
        elif b["ok"]:
            row["verdict"] = "static"
        else:
            row["verdict"] = "undetermined"
            r = fb.get("bigmove.pkl", {}).get(("bound", view, d0, d1))
            if r and r["ok"]:
                row["fallback_px"] = r["px"]
                # SIFT recovered the pair ORB could not.  A small value settles
                # it as static; a large one still wants an eye on it, because
                # the fallback has no tape check behind it.
                if r["px"] < REPORT_PX:
                    row["verdict"] = "static"
                    row["decided_by"] = "sift_fallback"
        key = ("bound", view, d0, d1)
        if key in MANUAL:
            row["verdict"] = {"camera": "confirmed",
                              "undetermined": "undetermined"}[MANUAL[key][0]]
            row["manual_note"] = MANUAL[key][1]
            row["decided_by"] = "eye"
        else:
            row["manual_note"] = ""
            row["decided_by"] = "tape_test" if row["verdict"] in (
                "confirmed", "rejected") else "auto"
        out.append(row)
    return out


def is_move(b):
    return b["verdict"] == "confirmed" and (
        b.get("decided_by") == "eye"
        or (b["px"] is not None and b["px"] >= REPORT_PX))


def epochs(bounds, events, view):
    """Spans over which the camera held still, in (desktop, step) coordinates.

    Both kinds of break count: a move between two desktops, and a move partway
    through one desktop's sequence -- the latter splits that desktop, so an
    epoch is a range of (desktop, step), not a range of desktops.

    Unresolved boundaries are not turned into breaks; that would report a count
    driven by where the method failed rather than by where the rig moved.  They
    are listed as `uncertain_breaks` so the count is a stated lower bound.
    """
    breaks = []
    for b in bounds:
        if is_move(b):
            breaks.append((b["desktop_from"], 10 ** 9, "after desktop %d"
                           % b["desktop_from"], b["px"]))
    for e in events:
        if e["kind"] in ("camera", "both"):
            breaks.append((e["desktop"], e["step_from"],
                           "desktop %d step %d->%d" % (e["desktop"],
                                                       e["step_from"], e["step_to"]),
                           e["table_px"]))
    breaks.sort(key=lambda x: (x[0], x[1]))
    pending = [b["desktop_from"] for b in bounds if b["verdict"] == "undetermined"]

    eps = []
    start = (1, 0)
    for i, (d, s, label, px) in enumerate(breaks):
        eps.append({"epoch": i + 1, "view": view,
                    "first_desktop": start[0], "first_step": start[1],
                    "last_desktop": d,
                    "last_step": "end" if s == 10 ** 9 else s,
                    "break_at": label, "break_px": px,
                    "uncertain_breaks": ""})
        start = (d + 1, 0) if s == 10 ** 9 else (d, s + 1)
    eps.append({"epoch": len(breaks) + 1, "view": view,
                "first_desktop": start[0], "first_step": start[1],
                "last_desktop": 66, "last_step": "end",
                "break_at": "", "break_px": None,
                "uncertain_breaks": ";".join(str(x) for x in pending)})
    return eps


def _r(x, n=3):
    return "" if x is None else round(float(x), n)


def write_csvs(events, bounds, eps):
    os.makedirs(C.OUT, exist_ok=True)
    ev_path = os.path.join(C.OUT, "events.csv")
    cols = ["view", "desktop", "step_from", "step_to", "kind", "magnitude_px",
            "rot_deg", "scale", "verdict", "decided_by", "n_inlier", "n_match",
            "iou_before", "iou_after", "iou_gain", "chassis_px",
            "chassis_area_ratio", "reason", "note"]
    with open(ev_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in events:
            if e["kind"] == "none":
                continue
            w.writerow({
                "view": e["view"], "desktop": e["desktop"],
                "step_from": e["step_from"], "step_to": e["step_to"],
                "kind": e["kind"],
                "magnitude_px": _r(e["table_px"], 2) if e["kind"] in
                ("camera", "both") else _r(e.get("chassis_px"), 2),
                "rot_deg": _r(e["rot_deg"], 4), "scale": _r(e["scale"], 5),
                "verdict": e["verdict"], "decided_by": e.get("decided_by", ""),
                "n_inlier": e["n_inlier"], "n_match": e["n_match"],
                "iou_before": _r(e["iou_before"]), "iou_after": _r(e["iou_after"]),
                "iou_gain": _r(e["iou_gain"]),
                "chassis_px": _r(e.get("chassis_px"), 2),
                "chassis_area_ratio": _r(e.get("chassis_area_ratio"), 3),
                "reason": e.get("reason", ""), "note": e.get("manual_note", ""),
            })

    ep_path = os.path.join(C.OUT, "rig_epochs.csv")
    with open(ep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "view", "epoch", "first_desktop", "first_step", "last_desktop",
            "last_step", "n_desktops_spanned", "break_at", "break_px",
            "uncertain_breaks"])
        w.writeheader()
        for e in eps:
            w.writerow({
                "view": e["view"], "epoch": e["epoch"],
                "first_desktop": e["first_desktop"], "first_step": e["first_step"],
                "last_desktop": e["last_desktop"], "last_step": e["last_step"],
                "n_desktops_spanned": e["last_desktop"] - e["first_desktop"] + 1,
                "break_at": e["break_at"], "break_px": _r(e["break_px"], 2),
                "uncertain_breaks": e["uncertain_breaks"],
            })

    bd_path = os.path.join(C.OUT, "desktop_boundaries.csv")
    with open(bd_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "view", "desktop_from", "desktop_to", "verdict", "decided_by",
            "magnitude_px", "rot_deg", "scale", "n_pairs", "n_checked",
            "iou_before", "iou_after", "iou_gain", "fallback_px", "note"])
        w.writeheader()
        for b in bounds:
            w.writerow({
                "view": b["view"], "desktop_from": b["desktop_from"],
                "desktop_to": b["desktop_to"], "verdict": b["verdict"],
                "decided_by": b.get("decided_by", ""),
                "magnitude_px": _r(b["px"], 2), "rot_deg": _r(b["rot_deg"], 4),
                "scale": _r(b["scale"], 5), "n_pairs": b.get("n_pairs", ""),
                "n_checked": b.get("n_checked", ""),
                "iou_before": _r(b["iou_before"]), "iou_after": _r(b["iou_after"]),
                "iou_gain": _r(b["iou_gain"]),
                "fallback_px": _r(b.get("fallback_px"), 2),
                "note": b.get("manual_note", ""),
            })
    print("wrote", ev_path, ep_path, bd_path)


def main():
    views = sys.argv[1:] or list(C.VIEWS)
    all_ev, all_bd, all_ep = [], [], []
    for v in views:
        ev = decide_events(v)
        bd = decide_bounds(v)
        for r in ev:
            r["kind"] = classify(r)
        all_ev += ev
        all_bd += bd
        all_ep += epochs(bd, ev, v)
        nc = sum(1 for r in ev if r["kind"] in ("camera", "both"))
        nch = sum(1 for r in ev if r["kind"] == "chassis")
        nu = sum(1 for r in ev if r["kind"] == "undetermined")
        nb = sum(1 for b in bd if b["verdict"] == "confirmed"
                 and b["px"] is not None and b["px"] >= REPORT_PX)
        nbu = sum(1 for b in bd if b["verdict"] == "undetermined")
        print("%-5s events: camera=%d chassis=%d undetermined=%d | "
              "boundaries: moved=%d undetermined=%d | epochs=%d"
              % (v, nc, nch, nu, nb, nbu, len([e for e in all_ep if e["view"] == v])))
    with open(os.path.join(C.TMP, "decided.pkl"), "wb") as f:
        pickle.dump({"events": all_ev, "bounds": all_bd, "epochs": all_ep}, f)
    write_csvs(all_ev, all_bd, all_ep)


if __name__ == "__main__":
    main()
