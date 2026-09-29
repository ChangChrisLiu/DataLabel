"""The annotator's D13/scan case, with the app's ROI instead of the harness's.

The report that started L1: on frame 38 the box sat on screw.motherboard.04
(121 px); on frame 37 it was a tall 6,332 px box on a black cable while the
part to draw was screw.motherboard.03.  The harness's stand-in ROI (the LS
union box padded 10 %) is not the ROI the annotator confirmed
(``pose_segment.roi_json`` = [627, 294, 1385, 1144] for D13/scan), so the rank-1
blob differs a little.  This replays both frames the way the worker does
(``diff_delta_e(max_side=1600)`` then ``diff_blobs`` with its **default**
``min_area=80``), with both ROIs, and prints the top blobs next to the M4 guess.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tda.core.diffmap import diff_blobs, diff_delta_e  # noqa: E402

from experiments.l1_localise import base  # noqa: E402
from experiments.l1_localise import methods as M  # noqa: E402
from experiments.l1_localise.run_l1 import VARIANTS  # noqa: E402
from experiments.l1_localise.siblings import (  # noqa: E402
    PrevParts, known_boxes_at, sibling_pool,
)


def main() -> int:
    events, _shapes, by_frame, con = base.load_events(views=("scan",))
    user_roi = tuple(json.loads(con.execute(
        "SELECT roi_json FROM pose_segment WHERE desktop=13 AND view='scan'"
    ).fetchone()["roi_json"]))
    frames = base.Frames(con, 13, "scan")
    variant = sys.argv[1] if len(sys.argv) > 1 else "M4_b1"
    cfg, _ = VARIANTS[variant]
    prev_parts = PrevParts(con, by_frame)
    for step in (38, 39):
        ev = next(e for e in events if e.desktop == 13 and e.step == step)
        img_j, img_k = frames.get(step - 1), frames.get(step)
        for name, roi in (("harness ROI", ev.roi), ("app ROI", user_roi)):
            # B2's order (j, k) and the app's (k, j): diff_delta_e is not
            # symmetric, and only the app's order gives the annotator's boxes
            for order, (a, b) in (("B2 order (j,k)", (img_j, img_k)),
                                  ("app order (k,j)", (img_k, img_j))):
                delta = diff_delta_e(a, b, roi=roi, max_side=1600)
                top = [(bl.box, bl.area) for bl in diff_blobs(delta)[:3]]
                print(f"frame {step - 1} (k={step}), {name} {roi}, {order}: "
                      f"top blobs {top}")
            _p, d2, org, _i, ck = M.m2(img_j, img_k, roi, 80, "ecc")
            s, pool = sibling_pool(ev, by_frame)
            temps = [M.cut_template(frames.get(s), tuple(p.box), s, p.instance,
                                    pad_frac=cfg.pad_frac) for p in pool]
            prev = prev_parts.get(ev)
            pp = None
            if prev is not None and prev[1] == ev.cls:
                pp = (0.5 * (prev[0][0] + prev[0][2]), 0.5 * (prev[0][1] + prev[0][3]))
            props, _ = M.m3(img_j, img_k, roi, [t for t in temps if t], cfg,
                            known_boxes=known_boxes_at(ev, by_frame), delta=d2,
                            delta_origin=org, k_reg=ck, k_origin=org,
                            prior_point=pp)
            print(f"    {variant} guess: box {props[0].box}, point "
                  f"{tuple(round(v) for v in props[0].point)}; "
                  f"part removed at k+1 at {pp}")
        gt = by_frame[(13, "scan", step - 1)]
        print(f"    drafts at frame {step - 1}: "
              f"{[(k, s.box) for k, s in gt.items() if s.cls == 'screw']}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
