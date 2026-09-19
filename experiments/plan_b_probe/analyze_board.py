"""Q3: is the cam1<->cam2 relation constant across desktops, and where does it jump?"""
import csv
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C            # noqa: E402
import board_relation as BR   # noqa: E402


def main():
    with open(os.path.join(C.TMP, "board_raw.pkl"), "rb") as f:
        res = pickle.load(f)
    res.sort(key=lambda r: r["desktop"])
    ref = next((r for r in res if r["H"] is not None), None)
    refrt = next((r for r in res if r["R"] is not None), None)
    print("reference capture: desktop %d (H), desktop %s (R,t)"
          % (ref["desktop"], refrt["desktop"] if refrt else "none"))

    rows = []
    prev = None
    print("\n%-4s %-6s %-6s %-9s %-11s %-11s %-10s %-9s" % (
        "desk", "c1/40", "c2/40", "H_reproj", "dH_vs_ref", "dH_vs_prev",
        "dR_prev", "dt_prev"))
    for r in res:
        dref = BR.h_disagreement(r["H"], ref["H"])
        dprev = BR.h_disagreement(r["H"], prev["H"]) if prev else None
        dR = dt = None
        # Compare the rigid transform against a fixed reference rather than the
        # previous capture: it is board-independent, so it separates "the rig
        # changed" from "the board was put down differently".
        if refrt is not None and r["R"] is not None:
            dR, dt = BR.rt_disagreement(r["R"], r["t"], refrt["R"], refrt["t"])
        print("%-4d %-6d %-6d %-9.2f %-11s %-11s %-10s %-9s" % (
            r["desktop"], r["n_tags_c1"], r["n_tags_c2"], r["H_reproj_px"],
            "%.2f" % dref if dref is not None else "-",
            "%.2f" % dprev if dprev is not None else "-",
            "%.3f deg" % dR if dR is not None else "-",
            "%.1f mm" % dt if dt is not None else "-"))
        rows.append({
            "desktop": r["desktop"],
            "n_tags_cam1_aligned": r["n_tags_c1"],
            "n_tags_cam2_aligned": r["n_tags_c2"],
            "n_tags_cam1_12mp": r["n_tags_c1_12mp"],
            "n_tags_cam2_12mp": r["n_tags_c2_12mp"],
            "n_common": r["n_common"],
            "H_reproj_px": round(r["H_reproj_px"], 3) if r["H_reproj_px"] else "",
            "H_shift_vs_ref_px": round(dref, 2) if dref is not None else "",
            "H_shift_vs_prev_px": round(dprev, 2) if dprev is not None else "",
            "kabsch_rmse_mm": round(r["kabsch_rmse_mm"], 2) if r["kabsch_rmse_mm"] else "",
            "plane_resid_mm_c1": round(r["plane_resid_mm_c1"], 2) if r["plane_resid_mm_c1"] else "",
            "plane_resid_mm_c2": round(r["plane_resid_mm_c2"], 2) if r["plane_resid_mm_c2"] else "",
            "rot_vs_ref_deg": round(dR, 4) if dR is not None else "",
            "transl_vs_ref_mm": round(dt, 2) if dt is not None else "",
            "depth_plane_usable": "no" if r.get("plane_rejected") else
                                  ("yes" if r["R"] is not None else "n/a"),
        })
        prev = r

    d = np.array([x["H_shift_vs_ref_px"] for x in rows if x["H_shift_vs_ref_px"] != ""],
                 dtype=float)
    print("\nH disagreement vs reference over %d captures: "
          "median %.2f px, p90 %.2f px, max %.2f px" %
          (len(d), np.median(d), np.percentile(d, 90), d.max()))
    os.makedirs(C.OUT, exist_ok=True)
    out = os.path.join(C.OUT, "tag_board.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
