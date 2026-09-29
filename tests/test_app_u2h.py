"""U2h: the prompt box after L1 -- no box when unsure, and the box on screen is the box sent.

* **1** -- on the views ``configs/prompt_gate.yaml`` lists (scan), the
  difference map's rank-1 box is withheld when its blob's area is outside the
  area band of every class the card asks to add back by more than ``k``;
  nothing is armed or drawn and the status line says why.  Unit tests of the
  gate itself are in ``tests/test_prompt_gate.py``.
* **2** -- the margin around the box a first click may land in is eight
  *screen* pixels at least, not two, at the 29 % an OAK frame opens at.
* **3** -- a box turned down by a click outside it stays off until the frame
  is visited again, whatever resets the prompt in between.
* **4** -- the canvas' prompt box is the box both SAM tools hold, on every
  path that resets one or the other.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app_scene import StubSamQueue, close_window, make_paths, make_session
from test_app_prompt_alt import (
    _arm_with_alternates,
    _blob,
    _far_outside,
    _install,
    _proposal,
    _stand_on_a_card_item,
    _start_edit,
)
from tda.ui import app_actions as A
from tda.ui import prompt_gate as PG
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_assist import PROMPT_ARMED, PROMPT_CHIP, PROMPT_WITHHELD, WITHHELD_JOIN
from tda.ui.app_guide import HINT_DIFF_RGB
from tda.ui.canvas.sam_tools import (
    PROMPT_BOX_MARGIN_PX,
    PROMPT_BOX_MARGIN_SCREEN_PX,
    prompt_box_margin,
)
from tda.ui.class_names import class_zh


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
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
    yield win
    close_window(win)


# --------------------------------------------------------------------------- #
# 4: the canvas shows exactly what SAM holds
# --------------------------------------------------------------------------- #
def held(win) -> dict:
    """The prompt box in the four places it lives."""
    return {"canvas": win.canvas.prompt_band()[0], "S": win.sam_point.prompt_box,
            "X": win.sam_box.prompt_box, "window": win._prompt_box}


def assert_agree(win, where: str) -> None:
    seen = held(win)
    assert len(set(seen.values())) == 1, f"after {where}: {seen}"


def _tab(win):
    win.act_flash_compare(True)
    if not win.is_flashing():
        pytest.skip("the scene has no neighbour frame to flash")
    assert_agree(win, "Tab held")
    win.act_flash_compare(False)


def _review(win):
    win.set_mode(A.MODE_REVIEW)
    assert_agree(win, "the switch to Review")
    win.set_mode(A.MODE_ANNOTATE)


def _esc(win):
    win.act_clear_edit()


def _undo(win):
    mask = np.zeros(win.overlay.hw, dtype=bool)
    mask[2:9, 2:9] = True
    win.set_editing_mask(mask, undoable=True)
    assert_agree(win, "a stroke")
    win.act_undo()
    QApplication.processEvents()


def _tool_switch(win):
    win.act_tool("brush")
    assert_agree(win, "the switch to the brush")
    win.act_tool("sam_box")
    assert_agree(win, "the switch to X")
    win.act_tool("sam_point")


def _set_sam_instance(win):
    win.set_sam_instance("somebody-else")


def _reset_prompt_rank(win):
    win.reset_prompt_rank()


PATHS = {"tab": _tab, "review": _review, "esc": _esc, "undo": _undo,
         "tool_switch": _tool_switch, "set_sam_instance": _set_sam_instance,
         "reset_prompt_rank": _reset_prompt_rank}


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("path", sorted(PATHS), ids=sorted(PATHS))
def test_the_canvas_box_is_the_box_both_sam_tools_hold(window, path, rank):
    rank1, _alts = _arm_with_alternates(window)
    _start_edit(window)
    window.act_tool("sam_point")
    window.begin_add_shape(rank1)
    for _ in range(rank - 1):
        window.act_cycle_prompt_box()
    assert window.prompt_rank() == rank
    assert_agree(window, "arming")

    PATHS[path](window)

    assert_agree(window, path)
    # ... and a click is sent with exactly that box, or with none
    box = window.canvas.prompt_band()[0]
    if window.session.editing_instance is None:
        _start_edit(window)
        window.act_tool("sam_point")
        assert_agree(window, "the next edit")
        box = window.canvas.prompt_band()[0]
    if box is None:
        return
    window.sam_queue.requests.clear()
    window.sam_point.on_press((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, None)
    assert window.sam_queue.requests, "the click inside the box sent nothing"
    assert window.sam_queue.requests[-1].box is not None, (
        f"after {path} the canvas showed {box} and the click went out point-only")


def test_the_box_goes_with_the_first_click_on_the_next_part(window):
    """Click on part A, commit, start part B: the box on screen is sent with B."""
    rank1, _alts = _arm_with_alternates(window)
    card = [row for row in window.session.task_card() if row.get("instance")]
    if len(card) < 2:
        pytest.skip("the card has one part")
    first, second = str(card[0]["instance"]), str(card[1]["instance"])
    window.task_card.sigRequestEdit.emit(first)
    window.act_tool("sam_point")
    window.begin_add_shape(rank1)
    box = window._prompt_box
    window.sam_point.on_press((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, None)
    window.sam_queue.flush()
    QApplication.processEvents()
    window.act_clear_edit()                         # (a commit resets the same way)
    window.task_card.sigRequestEdit.emit(second)
    window.act_tool("sam_point")
    window.begin_add_shape(rank1)
    assert_agree(window, "starting the second part")
    window.sam_queue.requests.clear()
    window.sam_point.on_press((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, None)
    assert window.sam_queue.requests[-1].box is not None, (
        "the first click on the second part went out without the box on screen")


# --------------------------------------------------------------------------- #
# 1: no box when unsure
# --------------------------------------------------------------------------- #
def open_add_rows(win) -> list:
    """The card's open ✚ rows: what the frame asks to add back."""
    return [r for r in win.session.task_card() if r.get("kind") == api.KIND_ADD_SHAPE
            and not r.get("done") and r.get("instance")]


def classes_asked(win) -> list:
    return list(dict.fromkeys(str(r["cls"]) for r in open_add_rows(win)))


def gate_on(win, band=(10_000.0, 20_000.0), views=("scan",), **per_class):
    """A gate on ``views`` with ``band`` for every class the card asks for."""
    bands = {cls: per_class.get(cls, band) for cls in classes_asked(win)}
    bands = {cls: b for cls, b in bands.items() if b is not None}
    win.prompt_gate = PG.PromptGate(views, 3.0, {"scan": bands})


def withheld_line(win) -> str:
    names = dict.fromkeys(class_zh(r.get("cls"), r.get("attrs"), str(r["instance"]))
                          for r in open_add_rows(win))
    return PROMPT_WITHHELD.format(name=WITHHELD_JOIN.join(names))


def nothing_armed(win) -> None:
    assert set(held(win).values()) == {None}, held(win)
    assert win.canvas.prompt_band() == (None, "")
    assert win.canvas.prompt_point() is None


def test_a_box_the_size_of_no_part_asked_for_is_withheld(window):
    _stand_on_a_card_item(window)
    assert classes_asked(window), "the scene's card asks for nothing to be added"
    gate_on(window)                              # 40 px blob vs 10,000-20,000
    rank1, alts = _arm_with_alternates(window)

    nothing_armed(window)
    assert window.prompt_rank() == 1
    line = window.status_message()
    assert line == withheld_line(window)
    assert line.startswith("这一步要补的是「") and "不给提示框" in line
    assert line.endswith("no guess this time: click the part") and " px" not in line
    # nothing to hover either: the gate took that guess back
    for row in open_add_rows(window):
        hints = window.card_hints(str(row["instance"]))
        assert not [h for h in hints if tuple(h[2]) == tuple(HINT_DIFF_RGB)], hints
    # a click goes out point-only, as the line says
    _start_edit(window)
    window.act_tool("sam_point")
    window.begin_add_shape(rank1)                # the edit start asks again: still no
    nothing_armed(window)
    window.sam_point.on_press(float(alts[-1].point[0]), float(alts[-1].point[1]), None)
    assert window.sam_queue.requests[-1].box is None


def test_shift_c_and_the_heat_map_stay_when_the_box_is_withheld(window):
    _stand_on_a_card_item(window)
    real = window.assist_result
    assert real is not None and real.get("delta") is not None
    gate_on(window)
    rank1, alts = _arm_with_alternates(window)
    nothing_armed(window)

    window.act_cycle_prompt_box()
    alt = tuple(float(v) for v in alts[0].box)
    assert window.prompt_rank() == 2 and set(held(window).values()) == {alt}
    for _ in range(len(alts)):
        window.act_cycle_prompt_box()
    assert window.prompt_rank() == 1
    nothing_armed(window)                        # rank 1 is still "no box"
    assert "不给框" in window.status_message()

    # the heat map, on the frame's real comparison
    window.assist_result = real
    window.act_toggle_heat()
    assert window.heat_visible and window.heat_item.isVisible()


def test_the_withheld_box_is_not_offered_again_as_an_alternate(window):
    """Behind a withheld rank 1 the alternates are the ones they would have been."""
    _stand_on_a_card_item(window)
    gate_on(window)
    x0, y0 = window.roi()[0], window.roi()[1]
    rank1 = _blob((x0 + 1, y0 + 1, x0 + 41, y0 + 41))
    same = _proposal((x0 + 2, y0 + 2, x0 + 41, y0 + 41))       # IoU > 0.8 with rank 1
    other = _proposal((x0 + 60, y0 + 1, x0 + 80, y0 + 21))
    _install(window, rank1, [same, other])
    nothing_armed(window)
    assert [tuple(p.box) for p in window.prompt_alternates()] == [tuple(other.box)]


def _armed_without_a_gate(win):
    """Rank 1 armed with the gate off: what main does."""
    win.prompt_gate = PG.PromptGate()
    rank1, alts = _arm_with_alternates(win)
    return rank1, alts, _arming(win)


def _arming(win) -> dict:
    band = win.canvas.prompt_band()
    return {"held": held(win), "band": band, "point": win.canvas.prompt_point(),
            "rank": win.prompt_rank(), "line": win.status_message()}


def _request_inside(win, box) -> tuple:
    win.sam_queue.requests.clear()
    win.sam_point.reset_prompt()
    win.hand_prompt_box_to_tools()
    win.sam_point.on_press((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, None)
    req = win.sam_queue.requests[-1]
    return (req.box, [tuple(p) for p in req.points], req.multimask,
            req.image_crop.shape, None if req.mask_input is None else req.mask_input.sum())


@pytest.mark.parametrize("how", ["fits", "another view", "one of two fits",
                                 "a class without a band"])
def test_rank_one_is_byte_identical_whenever_it_is_not_withheld(window, how):
    rank1, _alts, off = _armed_without_a_gate(window)
    box = tuple(float(v) for v in rank1.box)
    assert off["held"]["canvas"] == box and off["line"] == PROMPT_ARMED
    assert off["band"] == (box, PROMPT_CHIP)
    _start_edit(window)
    window.act_tool("sam_point")
    window.begin_add_shape(rank1)
    main = _request_inside(window, box)

    asked = classes_asked(window)
    if how == "fits":
        gate_on(window, band=(30.0, 50.0))
    elif how == "another view":
        gate_on(window, views=("oak1",))
    elif how == "one of two fits":
        if len(asked) < 2:
            pytest.skip("the card asks for one class")
        gate_on(window, **{asked[0]: (30.0, 50.0)})
    else:
        gate_on(window, **{asked[0]: None})
    window.clear_prompt_box()
    window.begin_add_shape(rank1)
    assert _arming(window) == off
    assert _request_inside(window, box) == main


def test_d13_scan_37_the_cable_box_is_withheld_for_a_motherboard_screw(window):
    """The shipped gate on the report's numbers: 6,332 px, screw.motherboard.03."""
    _stand_on_a_card_item(window)
    window.prompt_gate = PG.load_prompt_gate(str(PG.DEFAULT_FILE))
    x0, y0 = window.roi()[0], window.roi()[1]
    screw = {"instance": "screw.motherboard.03", "kind": api.KIND_ADD_SHAPE,
             "done": False, "cls": "screw", "attrs": {"role": "motherboard"}}
    window.begin_add_shape(_blob((x0 + 1, y0 + 1, x0 + 9, y0 + 20), area=6332),
                           rows=[screw])
    nothing_armed(window)
    assert window.status_message() == PROMPT_WITHHELD.format(name="主板螺丝")
    # frame 38's 121 px blob, and the same 6,332 px on a view the gate is off for
    window.begin_add_shape(_blob((x0 + 1, y0 + 1, x0 + 9, y0 + 20), area=121),
                           rows=[screw])
    assert window._prompt_box is not None and window.status_message() == PROMPT_ARMED


def test_two_classes_are_named_in_the_line(window):
    _stand_on_a_card_item(window)
    window.prompt_gate = PG.PromptGate(["scan"], 3.0, {"scan": {
        "screw": (187.0, 413.0), "ram_module": (1665.0, 5937.0)}})
    rows = [{"instance": "screw.motherboard.03", "kind": api.KIND_ADD_SHAPE,
             "done": False, "cls": "screw", "attrs": {"role": "motherboard"}},
            {"instance": "ram_module.01", "kind": api.KIND_ADD_SHAPE,
             "done": False, "cls": "ram_module", "attrs": {}},
            {"instance": "ram_module.02", "kind": api.KIND_ADD_SHAPE,
             "done": True, "cls": "ram_module", "attrs": {}}]
    x0, y0 = window.roi()[0], window.roi()[1]
    window.begin_add_shape(_blob((x0 + 1, y0 + 1, x0 + 9, y0 + 20), area=60_000),
                           rows=rows)
    assert window.status_message() == PROMPT_WITHHELD.format(name="主板螺丝」或「内存条")
    window.begin_add_shape(_blob((x0 + 1, y0 + 1, x0 + 9, y0 + 20), area=6_000),
                           rows=rows)                         # the RAM's band
    assert window._prompt_box is not None


# --------------------------------------------------------------------------- #
# 2: the margin is at least eight screen pixels
# --------------------------------------------------------------------------- #
def test_the_margin_is_eight_image_or_eight_screen_pixels_whichever_is_more():
    assert PROMPT_BOX_MARGIN_PX == PROMPT_BOX_MARGIN_SCREEN_PX == 8.0
    assert prompt_box_margin(0.29) == pytest.approx(8.0 / 0.29)    # 27.6 image px
    assert prompt_box_margin(1.09) == 8.0                          # 7.3 < 8
    assert prompt_box_margin(1.0) == 8.0
    for broken in (None, 0, -1, float("nan"), "x"):
        assert prompt_box_margin(broken) == 8.0


@pytest.mark.parametrize("zoom", [0.29, 1.09])
def test_a_first_click_just_past_the_box_at_the_zoom_on_screen(window, zoom):
    rank1, _alts = _arm_with_alternates(window)
    _start_edit(window)
    window.act_tool("sam_point")
    window.canvas.set_zoom(zoom)
    assert window.canvas.zoom_factor() == pytest.approx(zoom)
    margin = prompt_box_margin(zoom)
    box = tuple(float(v) for v in rank1.box)
    mid = (box[1] + box[3]) / 2.0

    window.begin_add_shape(rank1)
    window.sam_point.on_press(box[2] + margin, mid, None)          # on the edge: in
    assert window._prompt_box == box and window.sam_point.box_refused is False

    window.sam_point.reset_prompt()
    window.begin_add_shape(rank1)
    window.sam_point.on_press(box[2] + margin + 0.5, mid, None)    # past it: out
    assert window._prompt_box is None and window.sam_point.box_refused is True
    if zoom < 1.0:
        # eight image pixels -- what main allowed -- is well inside it here
        assert margin > PROMPT_BOX_MARGIN_PX + 15


# --------------------------------------------------------------------------- #
# 3: a refused box stays off until the next frame visit
# --------------------------------------------------------------------------- #
def _refuse(win, rank: int):
    """Arm, walk to ``rank``, and turn that box down with a click outside it."""
    rank1, alts = _arm_with_alternates(win)
    instance = _start_edit(win)
    win.act_tool("sam_point")
    win.begin_add_shape(rank1)
    for _ in range(rank - 1):
        win.act_cycle_prompt_box()
    assert win.prompt_rank() == rank and win._prompt_box is not None
    x, y = _far_outside(win)
    win.sam_point.on_press(x, y, None)
    assert win._prompt_box is None, "the click did not refuse the box"
    return rank1, instance


def _via_restore(win, instance):
    mask = np.zeros(win.overlay.hw, dtype=bool)
    mask[2:8, 2:8] = True
    win._restore_offer = {"instance": instance, "mask": mask,
                          "key": win.session.current(), "adopted": []}
    win.restore_pending()


def _via_roi(win, _instance):
    win.start_roi_edit()
    win.act_clear_edit()                     # Esc the rectangle: back to the frame


REFUSAL_PATHS = {
    "tab": lambda w, _i: _tab(w),
    "review": lambda w, _i: _review(w),
    "esc": lambda w, _i: w.act_clear_edit(),
    "sidecar_restore": _via_restore,
    "roi_re_edit": _via_roi,
}


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("path", sorted(REFUSAL_PATHS), ids=sorted(REFUSAL_PATHS))
def test_a_refused_box_is_not_armed_again_by_a_reset(window, path, rank):
    rank1, instance = _refuse(window, rank)

    REFUSAL_PATHS[path](window, instance)

    assert window.prompt_rank() == 1
    nothing_armed(window)
    # nor by the comparison, or the next part, arming rank 1 by itself
    window.begin_add_shape(rank1)
    nothing_armed(window)


def test_a_refused_box_comes_back_on_the_next_visit_to_the_frame(window):
    rank1, _instance = _refuse(window, 2)
    window.act_clear_edit()
    step = window.session.current().step
    window.session.goto(step - 1)
    QApplication.processEvents()
    window.session.goto(step)
    QApplication.processEvents()
    window.begin_add_shape(rank1)
    assert window._prompt_box == tuple(float(v) for v in rank1.box)
    assert_agree(window, "the next visit")


def test_the_frame_announced_again_is_not_a_visit(window):
    """A commit re-announces the frame; the refusal outlives it."""
    rank1, _instance = _refuse(window, 1)
    window.act_clear_edit()
    window.session.goto(window.session.current().step, force=True)
    QApplication.processEvents()
    window.begin_add_shape(rank1)
    nothing_armed(window)
