"""Walk the synthetic scene in a real window and print what every frame arms (task U3).

    python tests/u3_payload_driver.py <out.json> <work dir> [--detector]

Run by ``tests/test_u3_cross_tree.py`` twice: once inside a copy of main
``95d3386`` (no detector exists there) and once in this tree with
``--detector`` -- a stub model that finds three screws on **every** frame, so
the only thing keeping it off a frame is the rule under test (the card asks
for no screw).  Per frame it records the box on the canvas, both SAM tools'
boxes, the chip, the cross, the rank, the status line, the ``Shift+C``
alternates, and the SAM request a click in the middle of the box sends
(box, points, multimask, crop shape and pixel sum, mask input).  Imports only
what both trees have; the tree is the one this file sits in.
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


def main(argv) -> int:
    out, work = Path(argv[0]), Path(argv[1])
    use_detector = "--detector" in argv
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from app_scene import StubSamQueue, close_window, make_paths, make_session
    from tda.ui import app_actions as A
    from tda.ui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    work.mkdir(parents=True, exist_ok=True)
    session = make_session(work)
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

    if use_detector:
        from tda.models.detector import Det, DetectorConfig

        boxes = [(26.0, 20.0, 36.0, 32.0), (12.0, 40.0, 20.0, 48.0), (40.0, 40.0, 48.0, 48.0)]

        class Stub:
            names = {0: "screw"}
            identity = "stub-cross-tree"

            def detect(self, img, crop, view, step=None):
                return [Det(b, "screw", 0.9 - 0.1 * i) for i, b in enumerate(boxes)]

        cfg = DetectorConfig(source=work / "detector.yaml", model=work / "stub.pt",
                             classes=("screw",), views=("scan",), conf=0.10,
                             roi_crop=True, work_scale=(("scan", 1.0),), tile=640,
                             stride=512, yield_ms=0.0, cache_root=work / "det")
        win.enable_detector(cfg, factory=lambda _c: Stub(), cache_root=str(work / "det"))

    def settle() -> None:
        deadline = time.monotonic() + 20
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

    frames = {}
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
    close_window(win)
    out.write_text(json.dumps(frames, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
