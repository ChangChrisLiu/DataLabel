"""``Shift+C``: alternate box prompts from the split difference map (task B2b).

The measurement behind this is in
``D:\\DataSet\\experiments_out\\plan_b_probe\\diff_eval\\report.md``: replacing
the armed prompt with :func:`tda.core.diff_split.propose_parts` missed the
plan's gate overall, but on parts wider than 100 px the median SAM IoU went
0.320 -> 0.762 -- while nine large parts were over-split and got much worse.
So the split proposals are offered *behind* the existing box rather than
instead of it, and the first thing every test here protects is that an
annotator who never presses ``Shift+C`` sees byte-identical behaviour.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app_scene import (
    DESKTOP,
    LAST_STEP,
    VIEW,
    StubSamQueue,
    close_window,
    make_paths,
    make_session,
)
from tda.core.model import FrameKey
from tda.ui import app_actions as A
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_diff import AssistController


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
# scene helpers
# --------------------------------------------------------------------------- #
def _blob(box, area: int = 40, score: float = 10.0):
    from tda.core.diffmap import DiffBlob

    return DiffBlob(box=tuple(int(v) for v in box), area=int(area),
                    score=float(score))


def _proposal(box, point=None, score: float = 10.0, blob_index: int = 0):
    from tda.core.diff_split import PartProposal

    box = tuple(int(v) for v in box)
    if point is None:
        point = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    area = max(1, ((box[2] - box[0]) * (box[3] - box[1])) // 2)
    return PartProposal(box=box, point=(int(point[0]), int(point[1])),
                        score=float(score), blob_index=int(blob_index), area=area)


def _wait_for_assist(win, timeout: float = 5.0) -> dict:
    assert win.assist.wait(timeout), "the difference map did not finish"
    QApplication.processEvents()
    assert win.assist_result is not None
    return win.assist_result


def _stand_on_a_card_item(win) -> None:
    """Put the window on a frame whose card asks for a shape to be drawn."""
    for step in range(LAST_STEP - 1, 1, -1):
        win.session.goto(step)
        _wait_for_assist(win)
        rows = win.session.task_card()
        index = win.task_card.current_index()
        if 0 <= index < len(rows) and rows[index].get("kind") == api.KIND_ADD_SHAPE:
            return
    pytest.skip("no frame of the scene asks for a shape to be drawn")


def _start_edit(win) -> str:
    card = [row for row in win.session.task_card() if row.get("instance")]
    instance = str(card[0]["instance"])
    win.task_card.sigRequestEdit.emit(instance)
    return instance


def _install(win, rank1, proposals) -> None:
    """Hang a hand-made comparison on the frame and arm rank 1 from it."""
    win.assist_result = {"key": win.session.current(), "blobs": [rank1],
                         "delta": None, "roi": win.roi(), "expected": [],
                         "explained": [], "unexplained": [rank1],
                         "proposals": list(proposals)}
    win._assist_asked = win._assist_subject()
    win.expected_now = lambda: []
    win.act_tool("sam_point")
    win.begin_add_shape(rank1)


def _arm_with_alternates(win, count: int = 3):
    """Stand on a card item, arm rank 1, hang ``count`` alternates on the frame.

    The proposals are hand-made rather than measured: what is under test is the
    cycling, and the synthetic scene's own difference map is not guaranteed to
    split into three pieces on any given frame.
    """
    _stand_on_a_card_item(win)
    roi = win.roi()
    assert roi is not None, "the fixture did not accept the proposed ROI"
    x0, y0, x1, y1 = roi
    w = max(4, (x1 - x0) // 12)
    h = max(4, (y1 - y0) // 12)
    rank1 = _blob((x0 + 1, y0 + 1, x0 + 1 + w, y0 + 1 + h))
    alts = [_proposal((x0 + 1 + (i + 1) * (w + 2), y0 + 1,
                       x0 + 1 + (i + 1) * (w + 2) + w, y0 + 1 + h))
            for i in range(count)]
    _install(win, rank1, alts)
    return rank1, alts


# --------------------------------------------------------------------------- #
# the worker computes them with the blobs
# --------------------------------------------------------------------------- #
def test_the_worker_proposes_split_alternates_next_to_the_blobs():
    """One comparison, one payload: the alternates never run on the GUI thread."""
    from tda.core.diff_split import PartProposal

    controller = AssistController()
    try:
        a = np.full((96, 96, 3), 30, dtype=np.uint8)
        b = a.copy()
        b[20:40, 20:40] = 220          # a part
        b[60:80, 60:80] = 200          # and a second change
        payload = controller.compute(FrameKey(1, 2, VIEW), b, a, (0, 0, 96, 96), [])
        assert "proposals" in payload
        assert payload["proposals"], "two bright squares produced no proposal"
        assert all(isinstance(p, PartProposal) for p in payload["proposals"])
    finally:
        controller.shutdown()


def test_a_comparison_that_raises_in_the_splitter_still_delivers_its_blobs(
        monkeypatch):
    """The alternates are an extra; the blobs are the product."""
    from tda.ui import app_diff

    def boom(*_a, **_k):
        raise RuntimeError("no")

    monkeypatch.setattr(app_diff, "propose_parts", boom)
    controller = AssistController()
    try:
        a = np.full((96, 96, 3), 30, dtype=np.uint8)
        b = a.copy()
        b[20:40, 20:40] = 220
        payload = controller.compute(FrameKey(1, 2, VIEW), b, a, (0, 0, 96, 96), [])
        assert payload["unexplained"], "the blobs were lost with the proposals"
        assert payload["proposals"] == []
    finally:
        controller.shutdown()


# --------------------------------------------------------------------------- #
# rank 1 is exactly today
# --------------------------------------------------------------------------- #
def test_rank_one_is_exactly_todays_box_while_shift_c_is_never_pressed(window):
    """The whole point: an annotator who ignores the key sees no change at all."""
    rank1, alts = _arm_with_alternates(window)
    assert alts, "the fixture built no alternates"
    box = tuple(float(v) for v in rank1.box)
    assert window._prompt_box == box
    assert window.sam_point.prompt_box == box
    assert window.sam_box.prompt_box == box
    assert window.canvas._rubber_band == box
    assert window.prompt_rank() == 1
    assert window.canvas.prompt_point() is None


def test_walking_all_the_way_round_rebuilds_the_same_sam_request(window):
    rank1, alts = _arm_with_alternates(window)
    _start_edit(window)
    window.act_tool("sam_point")
    window.begin_add_shape(rank1)
    point = (float(rank1.box[0] + 2), float(rank1.box[1] + 2))
    window.sam_point.on_press(point[0], point[1], None)
    assert window.sam_queue.requests, "no SAM request was submitted"
    first = window.sam_queue.requests[-1]

    for _ in range(len(alts) + 1):
        window.act_cycle_prompt_box()
    assert window.prompt_rank() == 1
    assert window._prompt_box == tuple(float(v) for v in rank1.box)

    window.sam_point.reset_prompt()
    window.begin_add_shape(rank1)
    window.sam_point.on_press(point[0], point[1], None)
    again = window.sam_queue.requests[-1]
    assert again.box == first.box
    assert again.multimask == first.multimask
    assert [tuple(p) for p in again.points] == [tuple(p) for p in first.points]


# --------------------------------------------------------------------------- #
# cycling
# --------------------------------------------------------------------------- #
def test_shift_c_walks_the_split_proposals_and_wraps(window):
    rank1, alts = _arm_with_alternates(window)
    total = len(alts) + 1
    for index, proposal in enumerate(alts, start=2):
        window.act_cycle_prompt_box()
        assert window.prompt_rank() == index
        box = tuple(float(v) for v in proposal.box)
        assert window._prompt_box == box
        assert window.sam_point.prompt_box == box
        assert window.sam_box.prompt_box == box
        assert window.canvas._rubber_band == box
        assert window.canvas.prompt_point() == tuple(proposal.point)
        message = window.status_message()
        assert f"{index}/{total}" in message and "提示框" in message

    window.act_cycle_prompt_box()               # wrap
    assert window.prompt_rank() == 1
    assert window._prompt_box == tuple(float(v) for v in rank1.box)
    assert window.canvas.prompt_point() is None


def test_an_alternate_is_what_the_next_sam_click_carries(window):
    _rank1, alts = _arm_with_alternates(window)
    _start_edit(window)
    window.act_tool("sam_point")
    window.act_cycle_prompt_box()
    assert window.sam_point.prompt_box == tuple(float(v) for v in alts[0].box)
    window.sam_point.on_press(float(alts[0].point[0]), float(alts[0].point[1]), None)
    assert window.sam_queue.requests[-1].box is not None


def test_shift_c_says_so_when_the_frame_has_no_alternate(window):
    _stand_on_a_card_item(window)
    roi = window.roi()
    rank1 = _blob((roi[0] + 1, roi[1] + 1, roi[0] + 9, roi[1] + 9))
    _install(window, rank1, [])
    before = window._prompt_box

    window.act_cycle_prompt_box()

    assert window._prompt_box == before
    assert window.prompt_rank() == 1
    message = window.status_message()
    assert "没有" in message or "no other" in message


# --------------------------------------------------------------------------- #
# what may never be offered
# --------------------------------------------------------------------------- #
def test_an_alternate_covering_most_of_the_roi_is_never_offered(window):
    """The existing rule: a box over 60 % of the ROI is not a prompt at any rank."""
    _stand_on_a_card_item(window)
    x0, y0, x1, y1 = window.roi()
    rank1 = _blob((x0 + 1, y0 + 1, x0 + 9, y0 + 9))
    huge = _proposal((x0, y0, x1, y1))
    small = _proposal((x1 - 12, y1 - 12, x1 - 2, y1 - 2))
    _install(window, rank1, [huge, small])

    offered = [tuple(p.box) for p in window.prompt_alternates()]
    assert tuple(huge.box) not in offered
    assert tuple(small.box) in offered


def test_an_alternate_that_repeats_rank_one_is_not_offered(window):
    _stand_on_a_card_item(window)
    x0, y0 = window.roi()[0], window.roi()[1]
    rank1 = _blob((x0 + 1, y0 + 1, x0 + 41, y0 + 41))
    same = _proposal((x0 + 2, y0 + 2, x0 + 41, y0 + 41))       # box IoU > 0.8
    other = _proposal((x0 + 60, y0 + 1, x0 + 80, y0 + 21))
    _install(window, rank1, [same, other])

    assert [tuple(p.box) for p in window.prompt_alternates()] == [tuple(other.box)]


def test_at_most_three_alternates_are_offered(window):
    _stand_on_a_card_item(window)
    x0, y0 = window.roi()[0], window.roi()[1]
    rank1 = _blob((x0 + 1, y0 + 1, x0 + 11, y0 + 11))
    many = [_proposal((x0 + 20 * (i + 1), y0 + 1, x0 + 20 * (i + 1) + 10, y0 + 11))
            for i in range(6)]
    _install(window, rank1, many)

    assert len(window.prompt_alternates()) == 3


# --------------------------------------------------------------------------- #
# reset hygiene
# --------------------------------------------------------------------------- #
RESET_FUNNELS = {
    # the editing layer is replaced from outside the tool: commit, Esc, undo
    "reset_sam_prompt": lambda w: w.reset_sam_prompt(),
    # the target changed, so the prompt was about the previous part
    "set_sam_instance": lambda w: w.set_sam_instance("somebody-else"),
    # Tab: the canvas is showing another frame
    "flash_start": lambda w: w._pause_tools(True),
    "clear_prompt_box": lambda w: w.clear_prompt_box(),
}


@pytest.mark.parametrize("name", sorted(RESET_FUNNELS), ids=sorted(RESET_FUNNELS))
def test_the_rank_goes_back_to_one_wherever_the_prompt_is_reset(window, name):
    rank1, _alts = _arm_with_alternates(window)
    window.act_cycle_prompt_box()
    assert window.prompt_rank() == 2

    RESET_FUNNELS[name](window)

    assert window.prompt_rank() == 1
    if name == "clear_prompt_box":
        assert window._prompt_box is None
    else:
        assert window._prompt_box == tuple(float(v) for v in rank1.box)
    assert window.canvas.prompt_point() is None


def test_esc_puts_the_armed_box_back_to_rank_one(window):
    rank1, _alts = _arm_with_alternates(window)
    _start_edit(window)
    window.begin_add_shape(rank1)
    window.act_cycle_prompt_box()
    assert window.prompt_rank() == 2

    window.act_clear_edit()

    assert window.prompt_rank() == 1
    assert window._prompt_box == tuple(float(v) for v in rank1.box)


def test_a_frame_change_takes_the_alternates_with_it(window):
    """A comparison that lands after the frame moved on must arm nothing."""
    _rank1, alts = _arm_with_alternates(window)
    window.act_cycle_prompt_box()
    stale = window.assist_result

    window.session.goto(window.session.current().step - 1)
    QApplication.processEvents()

    assert window.prompt_rank() == 1
    # the superseded payload is still an object somebody could hand back
    window.assist_result = stale
    assert window.prompt_alternates() == []
    assert window._prompt_box != tuple(float(v) for v in alts[0].box)


def test_a_late_comparison_of_the_previous_frame_arms_nothing(window):
    """The worker's own token drops it; the window must not take it either."""
    _rank1, alts = _arm_with_alternates(window)
    stale = dict(window.assist_result)
    window.session.goto(window.session.current().step - 1)
    QApplication.processEvents()
    before = window._prompt_box

    window._on_blobs(stale)          # as if it had crossed the thread late

    assert window.assist_result is None or \
        window.assist_result.get("key") == window.session.current()
    assert window.prompt_alternates() == []
    window.act_cycle_prompt_box()
    assert window.prompt_rank() == 1
    assert window._prompt_box == before
    assert window._prompt_box != tuple(float(v) for v in alts[0].box)


def test_re_editing_the_roi_puts_the_box_back_to_rank_one(window):
    """The ROI is what the difference map was computed inside."""
    rank1, _alts = _arm_with_alternates(window)
    window.act_cycle_prompt_box()
    assert window.prompt_rank() == 2

    window.start_roi_edit()

    assert window.prompt_rank() == 1
    assert window._prompt_box == tuple(float(v) for v in rank1.box)


# --------------------------------------------------------------------------- #
# when the key does nothing
# --------------------------------------------------------------------------- #
def test_shift_c_is_inert_while_the_roi_rectangle_is_being_edited(window):
    _rank1, _alts = _arm_with_alternates(window)
    window.start_roi_edit()
    before = window._prompt_box

    window.act_cycle_prompt_box()

    assert window.prompt_rank() == 1
    assert window._prompt_box == before


def test_shift_c_is_inert_while_a_neighbour_frame_is_flashed(window):
    _rank1, _alts = _arm_with_alternates(window)
    window.act_flash_compare(True)
    if not window.is_flashing():
        pytest.skip("the scene has no neighbour frame to flash")
    before = window._prompt_box

    window.act_cycle_prompt_box()

    assert window.prompt_rank() == 1
    assert window._prompt_box == before
    window.act_flash_compare(False)


def test_shift_c_is_inert_outside_annotate_mode(window):
    _rank1, _alts = _arm_with_alternates(window)
    window.set_mode(A.MODE_REVIEW)
    before = window._prompt_box

    window.act_cycle_prompt_box()

    assert window.prompt_rank() == 1
    assert window._prompt_box == before


def test_shift_c_is_inert_while_a_draft_ghost_owns_the_keys(window):
    _rank1, _alts = _arm_with_alternates(window)
    _start_edit(window)
    window._draft_candidates = [object()]
    window._draft_index = 0
    assert window.showing_draft_ghost()
    before = window._prompt_box

    window.act_cycle_prompt_box()

    assert window.prompt_rank() == 1
    assert window._prompt_box == before


# --------------------------------------------------------------------------- #
# the key map
# --------------------------------------------------------------------------- #
def test_shift_c_is_in_the_actions_table_and_c_still_cycles_masks():
    names = {a.name: a for a in A.ACTIONS}
    assert names["cycle_candidate"].keys == ("C",)
    assert names["cycle_candidate"].slot == "act_cycle_candidate"
    alt = names["cycle_prompt_box"]
    assert alt.keys == ("Shift+C",)
    assert alt.slot == "act_cycle_prompt_box"
    assert alt.modes == (A.MODE_ANNOTATE,)
    assert alt.label_zh in A.shortcut_markdown()
