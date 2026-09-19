"""Render every boundary the estimator could not resolve.

These matter most: a camera move large enough to break the matching looks
exactly like a failure, so each one has to be looked at rather than assumed.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C      # noqa: E402
import validate as V    # noqa: E402


def load(view):
    with open(os.path.join(C.TMP, "raw_%s.pkl" % view), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    os.makedirs(C.OVERLAYS, exist_ok=True)
    n = 0
    for v in C.VIEWS:
        for b in load(v)["bounds"]:
            if not b["ok"]:
                p = V.render_boundary(v, b["desktop_from"], b["desktop_to"],
                                      "undet", None, "undetermined")
                if p:
                    n += 1
                    print(os.path.basename(p))
    print("rendered", n)
