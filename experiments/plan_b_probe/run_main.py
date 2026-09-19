"""Stage 2 runner: compute every step pair and every desktop boundary, cache raw.

Thresholding and reporting are deliberately left to report.py so they can be
re-tuned after looking at the validation overlays, without recomputing.
"""
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C     # noqa: E402
import analyze as A    # noqa: E402


def main():
    views = sys.argv[1:] or list(C.VIEWS)
    out = {}
    for v in views:
        t0 = time.time()
        ev, bd = A.run_view(v)
        out[v] = {"events": ev, "bounds": bd}
        print("%s: %d step-pairs, %d boundaries in %.1f s"
              % (v, len(ev), len(bd), time.time() - t0), flush=True)
        with open(os.path.join(C.TMP, "raw_%s.pkl" % v), "wb") as f:
            pickle.dump(out[v], f)
    print("done")


if __name__ == "__main__":
    main()
