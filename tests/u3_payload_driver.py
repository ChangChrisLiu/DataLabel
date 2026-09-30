"""Walk the synthetic scene in a real window and print what every frame arms (tasks U3, U4).

    python tests/u3_payload_driver.py <out.json> <work dir> [--detector] [--views scan,oak1]

Run by ``tests/test_u3_cross_tree.py`` twice: once inside a copy of main
(``git archive``; the detector off there) and once in this tree with
``--detector``: the **real** RF-DETR backend
(:class:`tda.models.detector.RFDetrTileDetector` -- its tiling, merge floor
and process hygiene) over a fake ``rfdetr`` (``tests/fake_rfdetr.py``) that
finds the scene's bright square on **every** frame as a screw at 0.9, plus a
screw at 0.09 and a box of the head's extra slot that must never come out.
So the only thing keeping it off a frame is the rule under test: the card
asks for no screw that comes back on its own (frame 12's are captive, back in
with the fan).

The scene is walked once per view in ``--views`` (U4: scan, oak1 and oak2 --
the same 64 x 64 frames, written as the view's own file type).  After each
walk ``live_check`` revisits frame 12 with the screws' ``parent`` taken off
the card, where the detector must arm the square -- the proof that it was
live.  Per frame it records the box on the canvas, both SAM tools' boxes, the
chip, the cross, the rank, the status line, the ``Shift+C`` alternates, and
the SAM request a click in the middle of the box sends (box, points,
multimask, crop shape and pixel sum, mask input).  Imports only what both
trees have; the tree is the one this file sits in.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TDA_PROMPT_GATE"] = "off"
os.environ["TDA_DETECTOR"] = "off"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def _detector(work: Path):
    """The real RF-DETR backend over the fake rfdetr: no weights, no GPU."""
    import torch

    import fake_rfdetr
    from tda.models.detector import DetectorConfig

    torch.cuda.is_available = lambda: True      # nothing touches it: the model is fake
    if getattr(sys.modules.get("rfdetr"), "OTHERS", None) is None:
        fake_rfdetr.install()                   # once: a second import changes nothing
    weights = work / "fake_rfdetr.pt"
    weights.write_bytes(b"not a model")
    views = ("scan", "oak1", "oak2")
    return DetectorConfig(source=work / "detector.yaml", model=weights,
                          classes=("screw",), views=views, conf=0.10, roi_crop=True,
                          work_scale=tuple((v, 1.0) for v in sorted(views)), tile=640,
                          stride=512, yield_ms=0.0, cache_root=work / "det",
                          backend="rfdetr", merge_floor=0.10)


def walk(view: str, work: Path, use_detector: bool) -> dict:
    import numpy as np
    from PySide6.QtWidgets import QApplication

    import app_scene
    from app_scene import StubSamQueue, close_window, make_paths, make_session
    from tda.ui import app_actions as A
    from tda.ui.app import MainWindow

    # The scene is a scan view; write it as ``view`` (its file type, its rows)
    # and give the "second view" rows to another camera.
    app_scene.VIEW = view
    if not hasattr(app_scene, "_u4_seed_second_view"):
        app_scene._u4_seed_second_view = app_scene.seed_second_view
    original = app_scene._u4_seed_second_view
    other = "scan" if view == "oak1" else "oak1"
    app_scene.seed_second_view = lambda db, v=other, steps=(1, 2): original(db, v, steps)

    app = QApplication.instance() or QApplication([])
    work.mkdir(parents=True, exist_ok=True)
    session = make_session(work)
    assert session.view == view, session.view
    queue = StubSamQueue()
    win = MainWindow(session, make_paths(work), "tester", sam_queue=queue)
    win.resize(900, 700)
    win.show()
    app.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    win.sam_queue = queue
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi() is None:
        # An OAK proposal finds no chassis in 64 px frames: drag the one the
        # scanner's proposal gives (9, 9, 55, 55), as the annotator would.
        if not win.roi_editing:
            win.act_edit_roi()
            win.wait_for_roi_proposal()
        win.on_roi_box((9, 9, 55, 55))
        win.act_commit()
    assert win.roi() == (9, 9, 55, 55), win.roi()

    if use_detector:
        win.enable_detector(_detector(work), cache_root=str(work / "det"))

    def settle() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            app.processEvents()
            done = win.assist.wait(0.02)
            worker = getattr(win, "det_worker", None)
            if done and (worker is None or worker.wait(0.02)):
                break
        for _ in range(3):
            app.processEvents()

    def held() -> dict:
        return {"canvas": win.canvas.prompt_band()[0], "S": win.sam_point.prompt_box,
                "X": win.sam_box.prompt_box, "window": win._prompt_box}

    frames: dict = {}
    for step in sorted(session.steps(), reverse=True):
        session.goto(step)
        settle()
        card = session.task_card()
        screws = [r["instance"] for r in card if r.get("kind") == "add_shape"
                  and not r.get("done") and str(r.get("cls")) == "screw"]
        record = {
            "open_screw_rows": screws,
            "held": held(), "chip": win.canvas.prompt_band()[1],
            "point": win.canvas.prompt_point(), "rank": win.prompt_rank(),
            "line": win.status_message(),
            "withheld": win._withheld_box,
            "alternates": [[float(v) for v in p.box] for p in win.prompt_alternates()],
        }
        rows = [r for r in card if r.get("instance")]
        if rows:
            win.task_card.sigRequestEdit.emit(str(rows[0]["instance"]))
            win.act_tool("sam_point")
            app.processEvents()
            record["edit_line"] = win.status_message()
            record["held_editing"] = held()
            box = win._prompt_box
            x, y = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) if box else (32.0, 26.0)
            queue.requests.clear()
            win.sam_point.on_press(x, y, None)
            if queue.requests:
                req = queue.requests[-1]
                record["request"] = {
                    "box": None if req.box is None else [float(v) for v in req.box],
                    "points": [[float(v) for v in p] for p in req.points],
                    "multimask": bool(req.multimask),
                    "crop_shape": list(req.image_crop.shape),
                    "crop_sum": int(np.asarray(req.image_crop, dtype=np.int64).sum()),
                    "mask_input": (None if req.mask_input is None
                                   else int(np.asarray(req.mask_input).sum())),
                }
            win.act_clear_edit()
            app.processEvents()
        frames[str(step)] = record
    if use_detector:
        # The detector is live: frame 12's screws come back with the fan (a
        # parent) and so ask for nothing; take the parent off and the same
        # frame arms the detector's screw.
        real = session.task_card

        def loose():
            return [{k: v for k, v in row.items()
                     if not (k == "parent" and row.get("cls") == "screw")}
                    for row in real()]

        session.task_card = loose
        session.goto(12)          # from frame 1, where the walk ended: a new visit
        settle()
        answer = win._det_frames.get(12)
        frames["live_check"] = {
            "step": 12, "held": held(), "chip": win.canvas.prompt_band()[1],
            "roi": list(win.roi() or ()), "det_state": win.det_state,
            "dets": None if answer is None else [d.row() for d in answer.dets],
            "alternates": [[float(v) for v in p.box] for p in win.prompt_alternates()],
        }
    close_window(win)
    return frames


def main(argv) -> int:
    out, work = Path(argv[0]), Path(argv[1])
    use_detector = "--detector" in argv
    views = ["scan"]
    if "--views" in argv:
        views = argv[argv.index("--views") + 1].split(",")
    result = {view: walk(view, work / view, use_detector) for view in views}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
