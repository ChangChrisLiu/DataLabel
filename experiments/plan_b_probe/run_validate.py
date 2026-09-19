"""Pick the validation sample and render it."""
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C        # noqa: E402
import validate as V      # noqa: E402

FLAG_PX = 2.0


def load(view):
    with open(os.path.join(C.TMP, "raw_%s.pkl" % view), "rb") as f:
        return pickle.load(f)


# Hand-picked flagged cases: the biggest, the borderline, and the suspicious.
FLAG_STEPS = [
    ("oak1", 2, 5, 6), ("oak1", 4, 6, 7), ("oak1", 24, 8, 9),
    ("oak1", 29, 46, 47), ("oak1", 47, 14, 15), ("oak1", 36, 18, 19),
    ("oak1", 64, 19, 20), ("oak2", 1, 33, 34), ("scan", 11, 46, 47),
    ("scan", 24, 8, 9), ("scan", 36, 9, 10), ("scan", 55, 21, 22),
    ("scan", 61, 1, 2), ("scan", 63, 31, 32), ("rs", 24, 8, 9),
    ("oak1", 23, 8, 9), ("scan", 13, 30, 31),
]
FLAG_BOUNDS = [
    ("oak1", 3, 4), ("oak1", 10, 11), ("oak1", 29, 30), ("oak1", 1, 2),
    ("oak1", 58, 59), ("oak1", 7, 8), ("oak2", 3, 4), ("oak2", 31, 32),
    ("scan", 56, 57), ("scan", 45, 46), ("scan", 1, 2), ("scan", 25, 26),
]
# Borderline, just under the threshold -- these test for missed moves.
NEAR_BOUNDS = [
    ("oak1", 9, 10), ("oak1", 17, 18), ("oak1", 26, 27), ("oak2", 15, 16),
    ("scan", 17, 18), ("scan", 48, 49),
]


def main():
    os.makedirs(C.OVERLAYS, exist_ok=True)
    idx = {v: load(v) for v in C.VIEWS}
    px_step, px_bound = {}, {}
    for v in C.VIEWS:
        for e in idx[v]["events"]:
            px_step[(v, e["desktop"], e["step_from"], e["step_to"])] = \
                (e["table_px"], e["ok"])
        for b in idx[v]["bounds"]:
            px_bound[(v, b["desktop_from"], b["desktop_to"])] = (b["px"], b["ok"])

    n = 0
    for v, d, a, b in FLAG_STEPS:
        px, _ok = px_step.get((v, d, a, b), (None, False))
        if V.render_step_pair(v, d, a, b, "flag", px, "flag"):
            n += 1
    for v, d0, d1 in FLAG_BOUNDS:
        px, _ok = px_bound.get((v, d0, d1), (None, False))
        if V.render_boundary(v, d0, d1, "flag", px, "flag"):
            n += 1
    for v, d0, d1 in NEAR_BOUNDS:
        px, _ok = px_bound.get((v, d0, d1), (None, False))
        if V.render_boundary(v, d0, d1, "near", px, "near"):
            n += 1

    # Non-events: quiet step pairs, sampled across views, biased towards pairs
    # where the chassis changed a lot so the table is the only thing tested.
    rnd = random.Random(7)
    for v in C.VIEWS:
        cands = [e for e in idx[v]["events"]
                 if e["ok"] and e["table_px"] < 0.5 and e["chassis_px"] is not None]
        cands.sort(key=lambda e: -e["chassis_px"])
        pick = cands[:2] + rnd.sample(cands[2:], 2)
        for e in pick:
            if V.render_step_pair(v, e["desktop"], e["step_from"], e["step_to"],
                                  "still", e["table_px"], "non-event"):
                n += 1
    print("rendered", n, "overlays into", C.OVERLAYS)
    big = [(f, os.path.getsize(os.path.join(C.OVERLAYS, f)))
           for f in os.listdir(C.OVERLAYS)]
    big.sort(key=lambda x: -x[1])
    print("largest: ", ["%s %.2fMB" % (f, s / 1e6) for f, s in big[:3]])


if __name__ == "__main__":
    main()
