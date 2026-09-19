"""Q3 runner: per-desktop board detection + cam1<->cam2 relation -> tag_board.csv"""
import csv
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C            # noqa: E402
import board as B             # noqa: E402
import board_relation as BR   # noqa: E402


def main():
    ds = B.oak_calib_desktops()
    if len(sys.argv) > 1:
        ds = [int(x) for x in sys.argv[1:]]
    res = []
    for d in ds:
        try:
            r = BR.pair(d)
        except Exception as e:                       # noqa: BLE001
            print("d%02d FAILED %s" % (d, e), flush=True)
            continue
        if r is None:
            print("d%02d no usable capture pair" % d, flush=True)
            continue
        res.append(r)
        print("d%02d tags c1=%d/40 c2=%d/40 (12mp %d/%d) common=%d "
              "H_reproj=%.2fpx kabsch=%.2fmm plane_res=%.2f/%.2fmm" % (
                  d, r["n_tags_c1"], r["n_tags_c2"], r["n_tags_c1_12mp"],
                  r["n_tags_c2_12mp"], r["n_common"],
                  r["H_reproj_px"] or float("nan"),
                  r["kabsch_rmse_mm"] or float("nan"),
                  r["plane_resid_mm_c1"] or float("nan"),
                  r["plane_resid_mm_c2"] or float("nan")), flush=True)
    with open(os.path.join(C.TMP, "board_raw.pkl"), "wb") as f:
        pickle.dump(res, f)
    print("saved", len(res), "captures")


if __name__ == "__main__":
    main()
