"""Is the B2 ground truth the part that was removed, or just the last ordinal?

``ls:<Label>#<n>`` ordinals are assigned per frame by ascending bbox ``min x``
(``transfer/common.py``), so when one of N same-label drafts disappears between
``k-1`` and ``k`` the *key* that disappears is always ``#N`` -- whichever screw
was physically taken out.  B2's event resolution matches on keys, so for a class
with siblings its "removed part" may be a part that is still there at ``k``.

This probe checks every event by **location** instead: is there a draft of the
same class at ``k`` whose box overlaps the GT draft's box at ``k-1``?  If so the
GT part did not leave.  And which draft at ``k-1`` has *no* same-class partner
at ``k`` -- that is the part that did.

Writes ``gt_check.csv``; the location-resolved GT is used by ``run_l1.py`` as a
second, clearly separated ground truth (``gt=loc``).
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l1_localise import base  # noqa: E402

#: Two boxes of the same class this similar are the same physical part.
SAME_IOU = 0.3


def check(events, by_frame):
    rows = []
    for ev in events:
        before = by_frame.get((ev.desktop, ev.view, ev.j), {})
        after = by_frame.get((ev.desktop, ev.view, ev.step), {})
        same_b = [s for s in before.values() if s.cls == ev.cls]
        same_a = [s for s in after.values() if s.cls == ev.cls]
        gt = before[ev.instance]

        def best(shape, pool):
            ious = [base.box_iou(shape.box, o.box) for o in pool]
            return max(ious) if ious else 0.0

        gt_still = best(gt, same_a)
        orphans = [s for s in same_b if best(s, same_a) < SAME_IOU]
        loc = orphans[0].instance if len(orphans) == 1 else ""
        rows.append({
            "desktop": ev.desktop, "view": ev.view, "step": ev.step,
            "instance": ev.instance, "label": ev.label, "cls": ev.cls,
            "part_width": round(ev.part_width, 1), "bucket": ev.bucket,
            "n_same_before": len(same_b), "n_same_after": len(same_a),
            "gt_iou_with_after": round(gt_still, 3),
            "gt_still_present": int(gt_still >= SAME_IOU),
            "n_orphans": len(orphans),
            "loc_instance": loc,
            "loc_equals_gt": int(loc == ev.instance) if loc else "",
        })
    return rows


def main() -> int:
    events, _shapes, by_frame, con = base.load_events()
    print(f"{len(events)} events, pinned to B2's set")
    rows = check(events, by_frame)
    base.OUT.mkdir(parents=True, exist_ok=True)
    path = base.OUT / "gt_check.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    still = [r for r in rows if r["gt_still_present"]]
    print(f"GT draft still has a same-class box at k (IoU>={SAME_IOU}): "
          f"{len(still)}/{len(rows)}")
    by_b = Counter((r["bucket"], r["gt_still_present"]) for r in rows)
    for b, *_ in base.BUCKETS:
        print(f"  {b:>8}: {by_b[(b, 1)]} of {by_b[(b, 0)] + by_b[(b, 1)]}")
    loc = Counter((r["loc_equals_gt"] if r["loc_instance"] else "unresolved")
                  for r in rows)
    print(f"location-resolved GT: same as B2 {loc[1]}, different {loc[0]}, "
          f"unresolved {loc['unresolved']}")
    print(f"wrote {path}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
