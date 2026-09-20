"""Look at the OAK ROI proposal on real frames (plan B task B3, step 3).

    D:\\Anaconda\\envs\\tda\\python.exe experiments/roi_oak_eval.py \\
        --db D:/DataSet/.cache/tmp/planb_ro.sqlite --out D:/DataSet/.cache/tmp/b3/roi

For every frame asked for it prints what the detector measured -- whether a tape
square was found, the candidates it offered, which one the referee accepted and
what fraction of the frame that box covers -- and writes a downscaled overlay so
the answer can be **looked at**, which is the only way a box is "correct".

The frames default to the first and the last of twelve desktops spread over the
rig epochs in both OAK views, plus the two frames either side of the D36 oak1
camera move.  The database is opened **read-only** and nothing is written to it;
``F:`` is only read.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tda.core.cache import full_frame, suggest_roi  # noqa: E402
from tda.core.cache_roi_detect import (  # noqa: E402
    OAK_MAX_AREA_FRAC,
    OAK_MAX_ASPECT,
    OAK_MIN_AREA_FRAC,
    OAK_MIN_ASPECT,
    OAK_MIN_RECTANGULARITY,
    OAK_WORK_SIDE,
    _downscaled,
    box_plausibility,
    oak_board_mask,
    oak_chassis_candidates,
)

#: Twelve machines spread over the rig epochs: the first Dell batch, the D21
#: re-framing, the HP small-form-factor and mid-tower runs, the D36 camera move,
#: the big-tower tail and the Apple G4.  D13 is a Dell SFF and D34 an HP MT.
DESKTOPS = (1, 13, 21, 24, 29, 34, 36, 45, 52, 58, 63, 64)
#: Extra frames: the D36 oak1 camera moved by ~190 px between these two.
EXTRA = ((36, "oak1", 18), (36, "oak1", 19))
SHOT_WIDTH = 1000


def frames(conn, desktops) -> list[tuple[int, str, int]]:
    out: list[tuple[int, str, int]] = []
    for desktop in desktops:
        for view in ("oak1", "oak2"):
            row = conn.execute(
                "SELECT MIN(step) lo, MAX(step) hi FROM frame WHERE desktop=? AND view=?",
                (desktop, view)).fetchone()
            if row is None or row["lo"] is None:
                continue
            out.append((desktop, view, int(row["lo"])))
            out.append((desktop, view, int(row["hi"])))
    return out


def report_one(img: np.ndarray, view: str) -> dict:
    """What the detector saw, at the resolution it measured on."""
    small, scale = _downscaled(img, OAK_WORK_SIDE)
    height, width = small.shape[:2]
    board = oak_board_mask(small)
    candidates = oak_chassis_candidates(small)
    scored = [
        (box, fill, box_plausibility(
            box, fill, width, height, OAK_MIN_RECTANGULARITY,
            area_band=(OAK_MIN_AREA_FRAC, OAK_MAX_AREA_FRAC),
            aspect_band=(OAK_MIN_ASPECT, OAK_MAX_ASPECT)))
        for box, fill in candidates
    ]
    full = img.shape[1], img.shape[0]
    box = suggest_roi(img, view)
    proposed = box != full_frame(*full)
    return {
        "board": board is not None,
        "candidates": [
            {"frac": round((b[2] - b[0]) * (b[3] - b[1]) / float(width * height), 3),
             "aspect": round((b[2] - b[0]) / float(max(b[3] - b[1], 1)), 2),
             "fill": round(f, 2), "ok": s is not None}
            for b, f, s in scored
        ],
        "box": box,
        "proposed": proposed,
        "frac": round((box[2] - box[0]) * (box[3] - box[1])
                      / float(full[0] * full[1]), 3) if proposed else None,
    }


def overlay(img: np.ndarray, found: dict, label: str, out: Path,
            debug: bool = False) -> None:
    height, width = img.shape[:2]
    factor = SHOT_WIDTH / float(width)
    small = cv2.resize(img, (SHOT_WIDTH, int(round(height * factor))),
                       interpolation=cv2.INTER_AREA)
    if debug:
        work, scale = _downscaled(img, OAK_WORK_SIDE)
        board = oak_board_mask(work)
        if board is not None:
            edge = cv2.resize(board, (small.shape[1], small.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(small, contours, -1, (255, 200, 0), 2)
        for index, (box, fill) in enumerate(oak_chassis_candidates(work)):
            x0, y0, x1, y1 = (int(round(v * scale * factor)) for v in box)
            cv2.rectangle(small, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(small, f"#{index} f{fill:.2f}", (x0 + 4, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    if found["proposed"]:
        x0, y0, x1, y1 = (int(round(v * factor)) for v in found["box"])
        cv2.rectangle(small, (x0, y0), (x1, y1), (0, 0, 255), 3)
        text = f"{label}  {found['frac']:.0%} of the frame"
    else:
        text = f"{label}  NO PROPOSAL (board={found['board']})"
    cv2.rectangle(small, (0, 0), (small.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(small, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="D:/DataSet/.cache/tmp/planb_ro.sqlite")
    ap.add_argument("--out", default="D:/DataSet/.cache/tmp/b3/roi")
    ap.add_argument("--desktops", default=",".join(str(d) for d in DESKTOPS))
    ap.add_argument("--debug", action="store_true",
                    help="draw the board outline and every candidate, not just "
                         "the accepted box")
    ap.add_argument("--grow", type=float, default=None,
                    help="override OAK_BOARD_GROW for a tuning sweep")
    args = ap.parse_args(argv)

    if args.grow is not None:
        from tda.core import cache_roi_detect as detect

        detect.OAK_BOARD_GROW = float(args.grow)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    wanted = frames(conn, [int(d) for d in args.desktops.split(",") if d])
    wanted += list(EXTRA)

    out_dir = Path(args.out)
    print(f"{'frame':22s} {'board':6s} {'proposed':9s} {'frac':6s}  candidates")
    counts = {"oak1": [0, 0], "oak2": [0, 0]}
    for desktop, view, step in wanted:
        row = conn.execute(
            "SELECT path FROM frame WHERE desktop=? AND view=? AND step=?",
            (desktop, view, step)).fetchone()
        img = None if row is None or not row["path"] else cv2.imread(row["path"])
        label = f"D{desktop:02d} {view} s{step:03d}"
        if img is None:
            print(f"{label:22s} -- unreadable")
            continue
        found = report_one(img, view)
        counts[view][1] += 1
        counts[view][0] += int(found["proposed"])
        cands = " ".join(
            f"[{c['frac']:.2f} a{c['aspect']:.2f} f{c['fill']:.2f}"
            f"{'+' if c['ok'] else '-'}]" for c in found["candidates"]
        ) or "(none)"
        frac = "-" if found["frac"] is None else f"{found['frac']:.3f}"
        print(f"{label:22s} {str(found['board']):6s} {str(found['proposed']):9s} "
              f"{frac:6s}  {cands}")
        overlay(img, found, label, out_dir / f"{label.replace(' ', '_')}.jpg",
                debug=args.debug)
    conn.close()
    for view, (made, total) in counts.items():
        print(f"{view}: a proposal on {made}/{total} frames")
    print(f"overlays in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
