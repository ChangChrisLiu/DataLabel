"""U3: the small-part detector's box as rank 1 -- the window with a stub model.

The scene is ``app_scene``'s synthetic D13/scan: on frame 12 the card asks for
the CPU cooler and its four screws back (neighbour 13), on frames 10 and 9 for
a drive cage and an SSD -- no screw.  The model is :class:`StubModel`, the dE
a table (``change_fn``) unless a test is about the real one; no GPU.

Frame 12's screws are *captive*: their rows come back with the fan (a
``parent``), which never asks for the detector (round 2).  The tests about the
detector itself therefore stand on a frame 12 whose screws come back on their
own -- :func:`loose_screws` takes the ``parent`` off the screw rows, which is
what D13's motherboard screws on frames 35-40 look like -- and the tests about
the captive case use the scene as it is.

* ranking, the "already drawn" skip, and the fallback to the difference map;
* the rank-1 source switch (the U2h gate does not judge a detector box);
* a late answer arms only while the prompt is untouched;
* ``Shift+C``: the detector's other candidates first, then the difference map;
* the chip, the hover, the log line, refusal and canvas == tools.

The engine, its cache and its file are ``tests/test_detector.py``; the diff
path against main ``95d3386`` is ``tests/test_u3_cross_tree.py``.
"""
from __future__ import annotations

import logging
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app_scene import StubSamQueue, close_window, make_paths, make_session
from test_app_prompt_alt import _blob, _proposal, _start_edit
from test_app_u2h import assert_agree, held, nothing_armed
from tda.models.detector import Det, DetectorConfig, DetectorUnavailable
from tda.ui import app_actions as A
from tda.ui import prompt_gate as PG
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_assist import (
    DET_CHIP,
    DET_CHIP_RANK,
    PROMPT_ARMED,
    PROMPT_ARMED_DET,
    PROMPT_CHIP,
    PROMPT_CHIP_RANK,
    DiffOffer,
)
from tda.ui.app_detect import DetCandidate, already_drawn, detector_asked
from tda.ui.app_guide import HINT_DET_ALT_LABEL, HINT_DET_LABEL, HINT_DIFF_RGB

SCREW_STEP = 12
PLAIN_STEPS = (10, 9)
#: Three screws on frame 12, inside the ROI (9, 9, 55, 55) the fixture stores.
A_BOX = (26, 20, 36, 32)       # where the bright square is on frame 12
B_BOX = (12, 40, 20, 48)
C_BOX = (40, 40, 48, 48)
CHANGE = {A_BOX: 9.0, B_BOX: 5.0, C_BOX: 2.0}


def fbox(box) -> tuple:
    return tuple(float(v) for v in box)


def _far_outside(win) -> tuple[float, float]:
    """A pixel on screen and in the ROI, well clear of the armed box."""
    from tda.ui.canvas.sam_tools import box_holds, prompt_box_margin

    vx0, vy0, vx1, vy1 = win.canvas.viewport_image_rect()
    x0, y0, x1, y1 = win.roi()
    box = win._prompt_box
    margin = prompt_box_margin(win.canvas.zoom_factor()) + 4
    for x, y in ((x0 + 1.5, y1 - 1.5), (x1 - 1.5, y1 - 1.5), (x0 + 1.5, y0 + 1.5),
                 (x1 - 1.5, y0 + 1.5)):
        if not (vx0 <= x < vx1 and vy0 <= y < vy1):
            continue
        if box is None or not box_holds(box, x, y, margin):
            return x, y
    raise AssertionError(f"no pixel clear of {box}")


def det(box, conf: float = 0.8, cls: str = "screw") -> Det:
    return Det(fbox(box), cls, conf)


class StubModel:
    """``detect`` from a table ``{step: [Det]}``; ``hold`` blocks some steps."""

    names = {0: "screw", 1: "connector"}

    def __init__(self, table, hold=(), identity: str = "stub-u3-model") -> None:
        self.table = dict(table)
        self.identity = identity
        self.calls: list = []
        self.hold = set(hold)
        self.release = threading.Event()

    def detect(self, img, crop, view, step=None):
        self.calls.append(step)
        if step in self.hold:
            self.release.wait(20.0)
        return list(self.table.get(step, []))

    def describe(self) -> str:
        return "stub model"


def det_config(tmp_path, views=("scan",), classes=("screw",), conf=0.10) -> DetectorConfig:
    return DetectorConfig(source=tmp_path / "detector.yaml", model=tmp_path / "stub.pt",
                          classes=tuple(classes), views=tuple(views), conf=conf,
                          roi_crop=True, work_scale=tuple((v, 1.0) for v in views),
                          tile=640, stride=512, yield_ms=0.0, cache_root=tmp_path / "det")


def enable(win, tmp_path, table, change=CHANGE, hold=(), views=("scan",),
           factory=None, model=None):
    model = model or StubModel(table, hold=hold)
    change_fn = None if change is None else (
        lambda j, k, box: float(change.get(tuple(int(v) for v in box), 0.5)))
    win.enable_detector(det_config(tmp_path, views=views),
                        factory=factory or (lambda _c: model),
                        cache_root=str(tmp_path / "det"), change_fn=change_fn)
    return model


def pump(win, timeout: float = 20.0) -> None:
    """Let the comparison and the detector finish and deliver everything."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        diff_done = win.assist.wait(0.02)
        det_done = win.det_worker is None or win.det_worker.wait(0.02)
        if diff_done and det_done:
            break
    for _ in range(3):
        QApplication.processEvents()


def go(win, step: int) -> None:
    win.session.goto(step)
    pump(win)


def request_inside(win, box) -> tuple:
    """The SAM request a click in the middle of ``box`` sends, as comparable values."""
    win.sam_queue.requests.clear()
    win.sam_point.reset_prompt()
    win.hand_prompt_box_to_tools()
    win.sam_point.on_press((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, None)
    req = win.sam_queue.requests[-1]
    return (req.box, [tuple(p) for p in req.points], req.multimask,
            req.image_crop.shape, int(np.asarray(req.image_crop).sum()),
            None if req.mask_input is None else int(req.mask_input.sum()))


class Keep(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list = []

    def emit(self, record):
        self.lines.append((record.levelname, record.getMessage()))


@pytest.fixture
def logged():
    keep = Keep()
    loggers = [logging.getLogger(n) for n in ("tda.app", "tda.ui.app_detect")]
    for logger in loggers:
        logger.addHandler(keep)
    yield keep.lines
    for logger in loggers:
        logger.removeHandler(keep)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def loose_screws(win) -> None:
    """Frame 12's screws without their ``parent``: screws that come back on their own.

    The session's card, with the ``parent`` key taken off every screw row --
    the shape of a motherboard screw's row, which is removed by itself.
    """
    real = win.session.task_card

    def card():
        return [{k: v for k, v in row.items()
                 if not (k == "parent" and row.get("cls") == "screw")}
                for row in real()]

    win.session.task_card = card


def _window(tmp_path, loose: bool = True):
    session = make_session(tmp_path)
    queue = StubSamQueue()
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=queue)
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    win.sam_queue = queue
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if loose:
        loose_screws(win)
    return win


@pytest.fixture
def window(qapp, tmp_path):
    win = _window(tmp_path)
    yield win
    close_window(win)


def test_the_scene_is_what_these_tests_assume(window):
    assert window.roi() == (9, 9, 55, 55)
    go(window, SCREW_STEP)
    card = window.session.task_card()
    assert {r["cls"] for r in card if r.get("kind") == api.KIND_ADD_SHAPE} >= {"screw"}
    assert detector_asked(card) == {"cpu_cooler", "screw"}, "loose_screws did not apply"
    assert window.session.task_neighbour() == SCREW_STEP + 1
    for step in PLAIN_STEPS:
        go(window, step)
        classes = {r["cls"] for r in window.session.task_card()
                   if r.get("kind") == api.KIND_ADD_SHAPE}
        assert classes and "screw" not in classes


# --------------------------------------------------------------------------- #
# ranking, skip, fallback
# --------------------------------------------------------------------------- #
def test_rank_one_is_the_screw_that_changes_most(window, tmp_path, logged):
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX, 0.6), det(B_BOX, 0.9),
                                           det(C_BOX, 0.8)]})
    go(window, SCREW_STEP)
    assert window._prompt_box == fbox(A_BOX)
    assert window.canvas.prompt_band() == (fbox(A_BOX), "SAM 提示框（检测器：螺丝）")
    assert DET_CHIP.format(name="螺丝") == "SAM 提示框（检测器：螺丝）"
    assert window.canvas.prompt_point() is None
    assert window.status_message() == PROMPT_ARMED_DET.format(name="螺丝", cls="screw")
    assert " px" not in window.status_message() and "conf" not in window.status_message()
    assert_agree(window, "the detector's arming")
    # the box on screen is the box a click carries
    _start_edit(window)
    window.act_tool("sam_point")
    assert request_inside(window, fbox(A_BOX))[0] is not None
    # one log line: the frame, every candidate (box, conf, dE), the chosen one
    lines = [m for _l, m in logged if m.startswith("prompt box from the detector")]
    assert lines, logged
    line = lines[0]
    assert "D13/scan step 12 (dE against step 13)" in line
    assert "chose (26, 20, 36, 32) conf 0.600 dE 9.00" in line
    assert ("candidates (box, conf, dE) [((26, 20, 36, 32), 0.6, 9.0), "
            "((12, 40, 20, 48), 0.9, 5.0), ((40, 40, 48, 48), 0.8, 2.0)]") in line


def test_only_the_asked_class_above_conf_inside_the_roi_is_a_candidate(window, tmp_path):
    outside = (1, 1, 7, 7)
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX, 0.09), det(B_BOX, 0.95, "connector"),
                                           det(outside, 0.99), det(C_BOX, 0.5)]},
           change={A_BOX: 9.0, B_BOX: 8.0, outside: 9.5, C_BOX: 2.0})
    go(window, SCREW_STEP)
    ranking = window.detector_ranking()
    assert [c.box for c in ranking.chosen] == [fbox(C_BOX)]
    assert window._prompt_box == fbox(C_BOX)


def test_equal_de_keeps_the_detectors_own_order(window, tmp_path):
    enable(window, tmp_path, {SCREW_STEP: [det(B_BOX, 0.9), det(C_BOX, 0.7)]},
           change={B_BOX: 4.0, C_BOX: 4.0})
    go(window, SCREW_STEP)
    assert window._prompt_box == fbox(B_BOX)


def _draw(win, instance: str, box) -> None:
    """Commit ``instance`` on the frame on screen as a rectangle over ``box``."""
    mask = np.zeros(win.overlay.hw, dtype=bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    win.session.begin_edit(instance)
    win.session.set_editing_mask(mask)
    win.session.commit_edit(api.SCOPE_KEYFRAME)
    pump(win)


def test_a_screw_already_drawn_is_skipped(window, tmp_path, logged):
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX, 0.9), det(B_BOX), det(C_BOX)]})
    go(window, SCREW_STEP)
    assert window._prompt_box == fbox(A_BOX)
    # the walk draws a screw where A is (shifted a pixel: a real mask is not a box)
    _draw(window, "screw.cpu_cooler.01", (A_BOX[0] + 1, A_BOX[1], A_BOX[2] + 1, A_BOX[3]))
    ranking = window.detector_ranking()
    assert [c.box for c in ranking.skipped] == [fbox(A_BOX)]
    assert [c.box for c in ranking.chosen] == [fbox(B_BOX), fbox(C_BOX)]
    assert window._prompt_box == fbox(B_BOX), "the commit's re-arm kept the drawn screw"
    line = [m for _l, m in logged if m.startswith("prompt box from the detector")][-1]
    assert "skipped as drawn [((26, 20, 36, 32), 0.9, 9.0)]" in line


def test_a_drawn_instance_of_another_class_is_not_a_reason_to_skip(window, tmp_path):
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX, 0.9), det(B_BOX)]})
    go(window, SCREW_STEP)
    _draw(window, "cpu_cooler.fan.01", (20, 15, 45, 40))       # covers A, not a screw
    assert window._prompt_box == fbox(A_BOX)


def test_already_drawn_both_ways_within_two_pixels_or_at_iou_03():
    drawn = [(21.0, 10.0, 31.0, 20.0)]
    assert not already_drawn((10.0, 10.0, 18.0, 20.0), drawn)       # centre 14: 19 is the edge
    assert already_drawn((12.0, 10.0, 26.0, 20.0), drawn)           # centre 19: within 2 px
    assert already_drawn((0.0, 0.0, 100.0, 100.0), [(40, 40, 50, 50)])   # the drawn centre
    assert already_drawn((40.0, 40.0, 50.0, 50.0), [(0, 0, 100, 100)])   # the candidate's
    # IoU: from 0.3 up it is drawn, just under it is not -- with both centres
    # more than 2 px outside the other box, so only the IoU rule can decide
    a = (0.0, 0.0, 100.0, 100.0)
    assert already_drawn(a, [(53.0, 0.0, 153.0, 100.0)])            # IoU 0.307
    assert not already_drawn(a, [(54.0, 0.0, 154.0, 100.0)])        # IoU 0.299
    assert not already_drawn(a, [])


def test_no_candidate_left_is_todays_box(window, tmp_path):
    enable(window, tmp_path, {SCREW_STEP: []})
    go(window, SCREW_STEP)
    blob = window.assist_result["unexplained"][0]
    assert window._prompt_box == fbox(blob.box)
    assert window.canvas.prompt_band() == (fbox(blob.box), PROMPT_CHIP)
    assert window.status_message() == PROMPT_ARMED
    assert window._prompt_source == "diff"


def test_a_view_the_detector_is_not_for_is_todays_box(window, tmp_path):
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX)]}, views=("oak1",))
    go(window, SCREW_STEP)
    blob = window.assist_result["unexplained"][0]
    assert window._prompt_box == fbox(blob.box)
    assert window.detector_ranking() is None


@pytest.mark.parametrize("step", PLAIN_STEPS)
def test_a_frame_that_asks_for_no_screw_is_todays_box(window, tmp_path, step):
    enable(window, tmp_path, {s: [det(A_BOX), det(B_BOX)] for s in range(1, 15)})
    go(window, step)
    blob = window.assist_result["unexplained"][0]
    assert window._prompt_box == fbox(blob.box)
    assert window.canvas.prompt_band() == (fbox(blob.box), PROMPT_CHIP)
    assert window.detector_ranking() is None


def _state(win) -> dict:
    return {"held": held(win), "band": win.canvas.prompt_band(),
            "point": win.canvas.prompt_point(), "rank": win.prompt_rank(),
            "line": win.status_message(), "withheld": win._withheld_box,
            "alternates": [tuple(p.box) for p in win.prompt_alternates()]}


@pytest.mark.parametrize("step", PLAIN_STEPS)
def test_the_diff_path_is_byte_identical_with_the_detector_on(qapp, tmp_path, step):
    """Detector on (answering screws everywhere) vs off: same arming, same request."""
    seen = {}
    for name in ("off", "on"):
        win = _window(tmp_path / name)
        try:
            if name == "on":
                enable(win, tmp_path / name,
                       {s: [det(A_BOX), det(B_BOX), det(C_BOX)] for s in range(1, 15)})
            go(win, step)
            state = _state(win)
            _start_edit(win)
            win.act_tool("sam_point")
            box = win._prompt_box
            state["request"] = request_inside(win, box)
            state["edit_line"] = win.status_message()
            seen[name] = state
        finally:
            close_window(win)
    assert seen["on"] == seen["off"]


# --------------------------------------------------------------------------- #
# round 2: screws that come back with a parent never ask for the detector
# --------------------------------------------------------------------------- #
def test_only_rows_that_come_back_on_their_own_ask_for_the_detector():
    fan = {"instance": "cpu_cooler.fan.01", "kind": api.KIND_ADD_SHAPE, "done": False,
           "cls": "cpu_cooler"}
    captive = [{"instance": f"screw.cpu_cooler.0{i}", "kind": api.KIND_ADD_SHAPE,
                "done": False, "cls": "screw", "parent": "cpu_cooler.fan.01",
                "attrs": {"role": "cpu_cooler", "captive": True}} for i in (1, 2, 3, 4)]
    assert detector_asked([fan, *captive]) == {"cpu_cooler"}
    board = {"instance": "screw.motherboard.03", "kind": api.KIND_ADD_SHAPE,
             "done": False, "cls": "screw"}
    assert detector_asked([fan, *captive, board]) == {"cpu_cooler", "screw"}
    assert detector_asked([dict(board, done=True)]) == set()
    assert detector_asked([dict(board, kind=api.KIND_SPLIT_KEYFRAME)]) == set()


def test_captive_screws_that_come_back_with_their_fan_keep_the_difference_maps_box(
        qapp, tmp_path):
    """Frame 12 as the scene has it: the fan plus four captive screws with a parent.

    The detector answers screws there, and rank 1 is still the difference
    map's box over the fan -- byte for byte what a window without a detector
    arms, and what a click sends.
    """
    seen = {}
    for name in ("off", "on"):
        win = _window(tmp_path / name, loose=False)
        try:
            model = None
            if name == "on":
                model = enable(win, tmp_path / name,
                               {s: [det(A_BOX), det(B_BOX), det(C_BOX)] for s in range(1, 15)})
            go(win, SCREW_STEP)
            card = win.session.task_card()
            captive = [r for r in card if r.get("cls") == "screw"]
            assert len(captive) == 4 and all(r.get("parent") == "cpu_cooler.fan.01"
                                             for r in captive)
            assert detector_asked(card) == {"cpu_cooler"}
            if model is not None:
                assert SCREW_STEP in win._det_frames, "the detector never answered frame 12"
                assert win.detector_ranking() is None
            state = _state(win)
            _start_edit(win)
            win.act_tool("sam_point")
            state["request"] = request_inside(win, win._prompt_box)
            state["edit_line"] = win.status_message()
            seen[name] = state
        finally:
            close_window(win)
    assert seen["on"] == seen["off"]
    assert seen["on"]["band"][1] == PROMPT_CHIP and seen["on"]["line"] == PROMPT_ARMED


def test_the_gate_does_not_judge_a_detector_box(window, tmp_path):
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX)]})
    window.prompt_gate = PG.PromptGate(["scan"], 3.0, {"scan": {
        c: (10_000.0, 20_000.0) for c in ("screw", "cpu_cooler")}})
    go(window, SCREW_STEP)
    assert window._prompt_box == fbox(A_BOX) and window._withheld_box is None
    # ... while it still withholds the difference map's box where there is no candidate
    window.enable_detector(det_config(tmp_path), factory=lambda _c: StubModel({}),
                           cache_root=str(tmp_path / "det2"))
    window.session.goto(SCREW_STEP + 1)
    go(window, SCREW_STEP)
    nothing_armed(window)
    assert window._withheld_box is not None


# --------------------------------------------------------------------------- #
# the background pass and a late answer
# --------------------------------------------------------------------------- #
def test_the_pass_walks_the_view_from_the_frame_on_screen(window, tmp_path):
    model = enable(window, tmp_path, {})
    pump(window)
    assert window.session.current().step == 14
    assert model.calls == list(range(14, 0, -1))
    assert set(window._det_frames) == set(range(1, 15))
    stats = window.det_last_pass
    assert stats["frames"] == 14 and stats["detected"] == 14


def test_the_frame_on_screen_goes_first_and_the_walk_goes_on_from_it(window, tmp_path):
    model = enable(window, tmp_path, {}, hold={14})
    deadline = time.monotonic() + 5
    while model.calls != [14] and time.monotonic() < deadline:
        time.sleep(0.01)
    window.session.goto(3)
    QApplication.processEvents()
    model.release.set()
    pump(window)
    assert model.calls[:4] == [14, 3, 2, 1]
    assert sorted(model.calls[4:]) == list(range(4, 14))


def test_a_frame_change_does_not_wait_for_the_detector(window, tmp_path):
    model = enable(window, tmp_path, {SCREW_STEP: [det(A_BOX)]}, hold={SCREW_STEP})
    started = time.perf_counter()
    window.session.goto(SCREW_STEP)
    QApplication.processEvents()
    assert time.perf_counter() - started < 1.0
    assert window.assist.wait(10)
    QApplication.processEvents()
    blob = window.assist_result["unexplained"][0]
    assert window._prompt_box == fbox(blob.box), "rank 1 waited for the detector"
    model.release.set()
    pump(window)


def _late(window, tmp_path):
    """On frame 12 with the difference map's box armed and the detector still busy."""
    model = enable(window, tmp_path, {SCREW_STEP: [det(A_BOX), det(B_BOX)]},
                   hold={SCREW_STEP})
    window.session.goto(SCREW_STEP)
    assert window.assist.wait(10)
    QApplication.processEvents()
    blob = window.assist_result["unexplained"][0]
    assert window._prompt_box == fbox(blob.box)
    assert window._det_late == window.session.current()
    return model, fbox(blob.box)


def test_a_late_answer_arms_while_the_prompt_is_untouched(window, tmp_path, logged):
    model, _diff = _late(window, tmp_path)
    model.release.set()
    pump(window)
    assert window._prompt_box == fbox(A_BOX)
    assert window.canvas.prompt_band()[1] == DET_CHIP.format(name="螺丝")
    assert_agree(window, "the late arming")
    assert any("(late answer)" in m for _l, m in logged)


def _click_inside(win, box):
    win.sam_point.on_press((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, None)


TOUCHES = {
    "click": lambda w, box: _click_inside(w, box),
    "canvas press": lambda w, box: w.canvas.sigMousePress.emit(30.0, 30.0, None),
    "refusal": lambda w, box: w.sam_point.on_press(*_far_outside(w), None),
    "shift_c": lambda w, box: w.act_cycle_prompt_box(),
    "edit start": lambda w, box: _start_edit(w),
    "esc": lambda w, box: w.reset_sam_prompt(),
    "tool stroke": lambda w, box: (w.act_tool("brush"),
                                   w.canvas.sigMousePress.emit(20.0, 20.0, None)),
}


@pytest.mark.parametrize("touch", sorted(TOUCHES), ids=sorted(TOUCHES))
def test_a_late_answer_is_dropped_once_the_prompt_was_touched(window, tmp_path, touch,
                                                               logged):
    model, diff_box = _late(window, tmp_path)
    before = window._prompt_box
    TOUCHES[touch](window, diff_box)
    QApplication.processEvents()
    after_touch = window._prompt_box
    model.release.set()
    pump(window)
    assert window._prompt_box == after_touch, f"the late answer moved the box after {touch}"
    assert window._prompt_box != fbox(A_BOX) or before == fbox(A_BOX)
    assert SCREW_STEP in window._det_frames, "the answer itself must be kept"
    assert any("late detector answer for step 12 kept, not armed" in m for _l, m in logged)


def test_a_late_answer_for_a_frame_left_is_kept_for_the_next_visit(window, tmp_path):
    model, _diff = _late(window, tmp_path)
    window.session.goto(PLAIN_STEPS[0])
    pump(window, 2.0)
    blob = window.assist_result["unexplained"][0]
    model.release.set()
    pump(window)
    assert window._prompt_box == fbox(blob.box), "frame 12's answer armed on frame 10"
    go(window, SCREW_STEP)
    assert window._prompt_box == fbox(A_BOX)


def test_storing_the_roi_plans_the_view_and_arms_the_frame_on_screen(qapp, tmp_path):
    """A view with no ROI yet has nothing to detect in; storing one starts the pass."""
    session = make_session(tmp_path)
    queue = StubSamQueue()
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=queue)
    try:
        win.resize(900, 700)
        win.show()
        QApplication.processEvents()
        win.set_mode(A.MODE_ANNOTATE)
        loose_screws(win)
        assert win.roi() is None
        model = enable(win, tmp_path, {SCREW_STEP: [det(A_BOX), det(B_BOX)]})
        pump(win)
        assert model.calls == [], "frames were detected without an ROI to crop to"
        go(win, SCREW_STEP)
        if not win.roi_editing:
            win.start_roi_edit()
        win.wait_for_roi_proposal()
        # the annotator drags the rectangle (a press on the canvas) and stores it
        win.canvas.sigMousePress.emit(10.0, 10.0, None)
        win.act_commit()
        pump(win)
        assert win.roi() == (9, 9, 55, 55)
        assert model.calls[0] == SCREW_STEP, "the frame on screen was not asked for first"
        assert sorted(model.calls) == list(range(1, 15))
        assert win._prompt_box == fbox(A_BOX)
    finally:
        close_window(win)


def test_a_warm_cache_serves_a_new_window_without_the_model(qapp, tmp_path):
    first = _window(tmp_path / "w")
    try:
        enable(first, tmp_path / "w", {SCREW_STEP: [det(A_BOX)]}, change=None)
        pump(first)
    finally:
        close_window(first)
    second = _window(tmp_path / "w2")
    try:
        # the same frames on disk: a second scene at the same paths
        second.session.cache_dir = first.session.cache_dir
        model = StubModel({SCREW_STEP: [det(C_BOX)]})
        second.enable_detector(det_config(tmp_path / "w"), factory=lambda _c: model,
                               cache_root=str(tmp_path / "w" / "det"))
        go(second, SCREW_STEP)
        assert model.calls == [], "a warm cache called the model"
        assert second._prompt_box == fbox(A_BOX)
        # the real dE of the square that moved, from the first window's pass
        assert second._det_frames[SCREW_STEP].change[0] > 20.0
    finally:
        close_window(second)


# --------------------------------------------------------------------------- #
# off
# --------------------------------------------------------------------------- #
def test_a_model_that_cannot_run_is_a_warning_and_the_difference_map(window, tmp_path,
                                                                      logged):
    def broken(_config):
        raise DetectorUnavailable("CUDA is not available")

    enable(window, tmp_path, {}, factory=broken)
    pump(window)
    assert window.det_state == "off"
    assert ("WARNING", "small-part detector OFF, the guess is the difference map's: "
                       "DetectorUnavailable: CUDA is not available") in logged
    go(window, SCREW_STEP)
    blob = window.assist_result["unexplained"][0]
    assert window._prompt_box == fbox(blob.box)


def test_a_model_that_loads_is_one_info_line(window, tmp_path, logged):
    enable(window, tmp_path, {})
    pump(window)
    on = [m for level, m in logged if m.startswith("small-part detector on")]
    assert len(on) == 1 and "['scan']" in on[0] and "['screw']" in on[0]
    assert "stub model" in on[0]


def test_a_broken_detector_file_never_stops_the_window_opening(qapp, tmp_path, monkeypatch):
    bad = tmp_path / "bad.yaml"
    bad.write_text("views: [scan\n", encoding="utf-8")
    monkeypatch.setenv("TDA_DETECTOR", str(bad))
    win = _window(tmp_path / "w")
    try:
        assert win.det_worker is None and win.det_state == "off"
        go(win, SCREW_STEP)
        assert win._prompt_box is not None
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# Shift+C
# --------------------------------------------------------------------------- #
def _with_diff_offers(win):
    """Hang a hand-made comparison on frame 12: its own box and three split pieces."""
    blob = _blob((10, 12, 50, 30), area=400)
    split = [_proposal((14, 14, 22, 22)), _proposal((30, 14, 38, 22)),
             _proposal((12, 40, 20, 48))]          # the last one is screw B again
    win.assist_result = dict(win.assist_result, blobs=[blob], unexplained=[blob],
                             explained=[], proposals=split)
    win.begin_add_shape(blob)
    return blob, split


def test_shift_c_walks_the_detector_first_then_the_difference_map(window, tmp_path):
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX), det(B_BOX), det(C_BOX)]})
    go(window, SCREW_STEP)
    blob, split = _with_diff_offers(window)
    assert window._prompt_box == fbox(A_BOX)
    alts = window.prompt_alternates()
    assert [type(a).__name__ for a in alts] == ["DetCandidate", "DetCandidate", "DiffOffer",
                                                "PartProposal", "PartProposal"]
    assert [tuple(a.box) for a in alts] == [fbox(B_BOX), fbox(C_BOX), fbox(blob.box),
                                           tuple(split[0].box), tuple(split[1].box)]
    total = len(alts) + 1
    seen = []
    for _ in range(total):
        window.act_cycle_prompt_box()
        seen.append((window._prompt_box, window.canvas.prompt_band()[1],
                     window.canvas.prompt_point()))
    assert seen[0] == (fbox(B_BOX), DET_CHIP_RANK.format(rank=2, total=6, name="螺丝"), None)
    assert seen[1] == (fbox(C_BOX), DET_CHIP_RANK.format(rank=3, total=6, name="螺丝"), None)
    assert seen[2] == (fbox(blob.box), PROMPT_CHIP_RANK.format(rank=4, total=6), None)
    assert seen[3][1] == PROMPT_CHIP_RANK.format(rank=5, total=6)
    assert seen[3][2] is not None, "a split piece keeps its click-here cross"
    assert seen[5] == (fbox(A_BOX), DET_CHIP_RANK.format(rank=1, total=6, name="螺丝"), None)
    assert window.prompt_rank() == 1
    assert "检测器找到的那个螺丝" in window.status_message()


def test_shift_c_status_lines_say_who_found_the_box(window, tmp_path):
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX), det(B_BOX)]})
    go(window, SCREW_STEP)
    _with_diff_offers(window)
    window.act_cycle_prompt_box()
    assert window.status_message().startswith("提示框 2/")
    assert "检测器找到的另一个螺丝" in window.status_message()
    window.act_cycle_prompt_box()
    assert "差异图原本给的那块" in window.status_message()


def test_rank_one_comes_back_with_its_detector_chip_after_a_reset(window, tmp_path):
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX), det(B_BOX)]})
    go(window, SCREW_STEP)
    window.act_cycle_prompt_box()
    assert window._prompt_box == fbox(B_BOX)
    window.reset_prompt_rank()
    assert window.canvas.prompt_band() == (fbox(A_BOX), DET_CHIP.format(name="螺丝"))
    assert_agree(window, "the reset")


# --------------------------------------------------------------------------- #
# refusal, canvas == tools, the hover
# --------------------------------------------------------------------------- #
def test_a_refused_detector_box_stays_off_until_the_next_visit(window, tmp_path):
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX), det(B_BOX)]})
    go(window, SCREW_STEP)
    _start_edit(window)
    window.act_tool("sam_point")
    assert window._prompt_box == fbox(A_BOX)
    x, y = _far_outside(window)
    window.sam_point.on_press(x, y, None)
    assert window.sam_queue.requests[-1].box is None
    nothing_armed(window)
    for reset in (window.act_clear_edit, window.reset_prompt_rank,
                  lambda: window.begin_add_shape(None)):
        reset()
        nothing_armed(window)
    window.session.goto(SCREW_STEP - 1)
    go(window, SCREW_STEP)
    assert window._prompt_box == fbox(A_BOX)


@pytest.mark.parametrize("rank", [1, 2])
def test_the_canvas_box_is_the_box_both_sam_tools_hold(window, tmp_path, rank):
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX), det(B_BOX)]})
    go(window, SCREW_STEP)
    _start_edit(window)
    window.act_tool("sam_point")
    for _ in range(rank - 1):
        window.act_cycle_prompt_box()
    assert_agree(window, "arming")
    for step in ("brush", "sam_box", "sam_point"):
        window.act_tool(step)
        assert_agree(window, f"the switch to {step}")
    window.act_flash_compare(True)
    assert_agree(window, "Tab held")
    window.act_flash_compare(False)
    box = window.canvas.prompt_band()[0]
    window.sam_queue.requests.clear()
    _click_inside(window, box)
    assert window.sam_queue.requests[-1].box is not None


def test_the_hover_names_the_detector(window, tmp_path):
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX), det(B_BOX)]})
    go(window, SCREW_STEP)
    screw = next(r["instance"] for r in window.session.task_card() if r["cls"] == "screw")
    hints = [h for h in window.card_hints(screw) if tuple(h[2]) == tuple(HINT_DIFF_RGB)]
    assert [(tuple(h[0]), h[1]) for h in hints] == [
        (fbox(A_BOX), HINT_DET_LABEL.format(name="螺丝"))]
    window.act_cycle_prompt_box()
    hints = [h for h in window.card_hints(screw) if tuple(h[2]) == tuple(HINT_DIFF_RGB)]
    assert hints[0][1] == HINT_DET_ALT_LABEL.format(name="螺丝", rank=2)


def test_the_real_de_ranks_the_screw_that_left_first(window, tmp_path):
    """No dE table: the square that moves between 12 and 13 beats the still board."""
    enable(window, tmp_path, {SCREW_STEP: [det(C_BOX, 0.9), det(A_BOX, 0.5)]}, change=None)
    go(window, SCREW_STEP)
    ranking = window.detector_ranking()
    assert [c.box for c in ranking.chosen] == [fbox(A_BOX), fbox(C_BOX)]
    assert ranking.chosen[0].change > 20.0 > 1.0 > ranking.chosen[1].change
    assert isinstance(ranking.chosen[0], DetCandidate)
    assert not isinstance(ranking.chosen[0], DiffOffer)


# --------------------------------------------------------------------------- #
# round 3 (the review)
# --------------------------------------------------------------------------- #
def _loosen(session) -> None:
    """:func:`loose_screws` on a session, before any window exists."""
    real = session.task_card
    session.task_card = lambda: [
        {k: v for k, v in row.items() if not (k == "parent" and row.get("cls") == "screw")}
        for row in real()]


def _built_with_a_detector(tmp_path, monkeypatch, factory, identity=None,
                           cache_root=None, cache_dir=None, change=CHANGE):
    """A window whose detector is switched on *while it is built* -- the app's own way.

    ``load_detector_config`` answers a stub configuration and the engine is
    given the stub model; the ROI is stored and the session stands on frame
    12 before ``MainWindow`` exists.
    """
    from tda.models.det_engine import DetectionEngine
    from tda.ui import app_detect

    session = make_session(tmp_path)
    session.db.set_pose_segment_roi(13, "scan", 1, [9, 9, 55, 55], annotator="tester",
                                    hw=(64, 64))
    if cache_dir is not None:
        session.cache_dir = str(cache_dir)
    _loosen(session)
    session.goto(SCREW_STEP)
    config = det_config(tmp_path)
    change_fn = None if change is None else (
        lambda j, k, box: float(change.get(tuple(int(v) for v in box), 0.5)))
    root = cache_root or tmp_path / "det"

    def engine(cfg, _root, factory=None, **_kw):
        extra = {} if change_fn is None else {"change_fn": change_fn}
        return DetectionEngine(cfg, root, factory=engine.factory, identity=identity, **extra)

    engine.factory = factory
    monkeypatch.setattr(app_detect, "load_detector_config", lambda: config)
    monkeypatch.setattr(app_detect, "DetectionEngine", engine)
    queue = StubSamQueue()
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=queue)
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.sam_queue = queue
    return win


def test_the_frame_the_window_opens_on_gets_a_late_answer_while_untouched(
        qapp, tmp_path, monkeypatch):
    """Review item 1: the first frame's visit is recorded after the first render."""
    model = StubModel({SCREW_STEP: [det(A_BOX), det(B_BOX)]}, hold={SCREW_STEP})
    win = _built_with_a_detector(tmp_path, monkeypatch, lambda _c: model)
    # the window's own logger setup replaces handlers added before it existed
    keep = Keep()
    logging.getLogger("tda.app").addHandler(keep)
    logged = keep.lines
    try:
        assert win.det_worker is not None and win.session.current().step == SCREW_STEP
        assert win.roi() == (9, 9, 55, 55) and not win.roi_editing
        assert win.assist.wait(10)
        QApplication.processEvents()
        blob = win.assist_result["unexplained"][0]
        assert win._prompt_box == fbox(blob.box), "the difference map armed first"
        assert win._det_late == win.session.current()
        model.release.set()
        pump(win)
        assert win._prompt_box == fbox(A_BOX)
        assert win.canvas.prompt_band()[1] == DET_CHIP.format(name="螺丝")
        assert any("(late answer)" in m for _l, m in logged)
        assert not any("kept, not armed" in m for _l, m in logged)
    finally:
        logging.getLogger("tda.app").removeHandler(keep)
        model.release.set()
        close_window(win)


def test_a_warm_cache_arms_the_first_frame_before_the_model_has_loaded(
        qapp, tmp_path, monkeypatch):
    """Review item 3: the cache is keyed by the model file, not by a loaded model."""
    first = _built_with_a_detector(
        tmp_path / "a", monkeypatch,
        lambda _c: StubModel({SCREW_STEP: [det(A_BOX), det(B_BOX)]}, identity="warm-u3"),
        identity="warm-u3", cache_root=tmp_path / "det")
    try:
        pump(first)
        assert first._prompt_box == fbox(A_BOX)
        scene = first.session.cache_dir
    finally:
        close_window(first)

    gate = threading.Event()
    built = []

    def slow(_config):
        gate.wait(30)
        built.append(True)
        return StubModel({}, identity="warm-u3")

    started = time.perf_counter()
    second = _built_with_a_detector(tmp_path / "b", monkeypatch, slow, identity="warm-u3",
                                    cache_root=tmp_path / "det", cache_dir=scene)
    try:
        assert second.det_worker.served_before_load(10)
        assert second.assist.wait(10)
        for _ in range(5):
            QApplication.processEvents()
        assert second.det_state == "loading" and built == [], "the model had loaded"
        assert SCREW_STEP in second._det_frames
        assert second._prompt_box == fbox(A_BOX)
        assert time.perf_counter() - started < 5.0
    finally:
        gate.set()
        close_window(second)


def test_a_detector_that_raises_on_arrival_leaves_the_difference_map(window, tmp_path,
                                                                     monkeypatch, logged):
    """Review item 4: a detector failure before the comparison is asked for is logged."""
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX)]})
    pump(window)

    def boom(*_a, **_k):
        raise RuntimeError("detector broke")

    monkeypatch.setattr(window, "_det_ready", boom)
    go(window, SCREW_STEP)
    blob = window.assist_result["unexplained"][0]
    assert window._prompt_box == fbox(blob.box)
    assert any(m.startswith("detector: frame") and "planning failed" in m
               for _l, m in logged), logged
    monkeypatch.setattr(window, "detector_ranking", boom)
    window.clear_prompt_box()
    window.begin_add_shape(blob)
    assert window._prompt_box == fbox(blob.box)
    assert any("ranking failed" in m for _l, m in logged)


def test_a_detector_that_raises_while_the_window_is_built_is_off(qapp, tmp_path, monkeypatch,
                                                                logged):
    from tda.ui import app_detect

    def boom():
        raise RuntimeError("config reader broke")

    monkeypatch.setattr(app_detect, "load_detector_config", boom)
    win = _window(tmp_path)
    try:
        assert win.det_worker is None and win.det_state == "off"
        go(win, SCREW_STEP)
        assert win._prompt_box is not None
        assert any("failed while the window was built" in m for _l, m in logged)
    finally:
        close_window(win)


def test_small_boxes_of_one_screw_are_one_shift_c_candidate(window, tmp_path):
    """Review item 5: D13/scan 40 offered the same screw three times (IoU 0.76)."""
    from tda.ui.app_assist import _same_candidate

    assert _same_candidate((898, 800, 919, 820), (898, 799, 922, 822))    # frame 40
    assert not _same_candidate((898, 800, 919, 820), (700, 600, 1000, 900))  # a big box
    assert not _same_candidate((12, 40, 20, 48), (40, 40, 48, 48))
    enable(window, tmp_path, {SCREW_STEP: [det(A_BOX), det(B_BOX)]})
    go(window, SCREW_STEP)
    near_b = _blob((12, 39, 21, 49), area=60)          # IoU 0.71 with B, centre inside it
    other = _proposal((40, 40, 48, 48))
    window.assist_result = dict(window.assist_result, blobs=[near_b], unexplained=[near_b],
                                explained=[], proposals=[other])
    window.begin_add_shape(near_b)
    alts = window.prompt_alternates()
    assert [tuple(a.box) for a in alts] == [fbox(B_BOX), tuple(other.box)]


def test_the_worker_lets_go_of_the_model_cache_when_the_pass_is_over(window, tmp_path):
    """Review item 7: a pass that ends releases the decoded frames and the GPU cache."""
    released = []

    class Releasing:            # (StubModel's own ``release`` is its hold event)
        names = {0: "screw"}
        identity = "stub-releasing"

        def detect(self, img, crop, view, step=None):
            return []

        def release(self):
            released.append(True)

    enable(window, tmp_path, {}, model=Releasing())
    pump(window)
    assert released, "the idle worker kept its GPU cache"
    assert window.det_worker.engine._decoded == {}
