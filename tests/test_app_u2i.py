"""U2i: the review minors of U2h, window side.

* **4a** -- a box turned down by a click outside it is not shown again as the
  hover's "差异最大处", exactly like a box the gate withheld.
* **4b** -- refusing a ``Shift+C`` alternate starts the walk again: the next
  ``Shift+C`` is rank 2, not rank 3.
* **4c** -- starting an edit on a frame whose box was withheld keeps saying
  there is no box, instead of overwriting the one line that said so.

The gate's loading and its bands are ``tests/test_prompt_gate.py``.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import StubSamQueue, close_window, make_paths, make_session
from test_app_prompt_alt import _arm_with_alternates, _stand_on_a_card_item, _start_edit
from test_app_u2h import _refuse, gate_on, nothing_armed, open_add_rows
from tda.ui import app_actions as A
from tda.ui import prompt_gate as PG
from tda.ui.app import MainWindow
from tda.ui.app_assist import EDITING_NO_BOX
from tda.ui.app_guide import HINT_DIFF_RGB


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


def diff_hints(win) -> list:
    """The orange "差异最大处" outlines a hover on each open ✚ row would draw."""
    out = []
    for row in open_add_rows(win):
        out += [h for h in win.card_hints(str(row["instance"]))
                if tuple(h[2]) == tuple(HINT_DIFF_RGB)]
    return out


# --------------------------------------------------------------------------- #
# 4a
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rank", [1, 2])
def test_a_refused_box_is_not_the_hovers_biggest_change(window, rank):
    rank1, _instance = _refuse(window, rank)
    assert window.assist_result is not None
    assert diff_hints(window) == [], "the refused guess came back on the hover"
    # ... until the annotator asks for a box again
    window.act_cycle_prompt_box()
    shown = [tuple(h[0]) for h in diff_hints(window)]
    assert shown and shown[0] == window._prompt_box


def test_the_hover_still_points_at_a_box_nobody_refused(window):
    rank1, _alts = _arm_with_alternates(window)
    box = tuple(float(v) for v in rank1.box)
    assert [tuple(h[0]) for h in diff_hints(window)][:1] == [box]


# --------------------------------------------------------------------------- #
# 4b
# --------------------------------------------------------------------------- #
def test_refusing_an_alternate_starts_the_walk_again(window):
    _rank1, _instance = _refuse(window, 2)
    assert window.prompt_rank() == 1
    nothing_armed(window)
    alts = window.prompt_alternates()
    window.act_cycle_prompt_box()
    assert window.prompt_rank() == 2, "Shift+C skipped a rank after the refusal"
    assert window._prompt_box == tuple(float(v) for v in alts[0].box)


def test_refusing_rank_one_leaves_the_walk_at_one(window):
    _rank1, _instance = _refuse(window, 1)
    assert window.prompt_rank() == 1
    window.act_cycle_prompt_box()
    assert window.prompt_rank() == 2


# --------------------------------------------------------------------------- #
# 4c
# --------------------------------------------------------------------------- #
def test_starting_an_edit_on_a_withheld_frame_still_says_there_is_no_box(window):
    _stand_on_a_card_item(window)
    gate_on(window)                              # the 40 px blob fits nothing
    _arm_with_alternates(window)
    nothing_armed(window)
    instance = _start_edit(window)
    QApplication.processEvents()
    line = window.status_message()
    assert line.startswith(f"editing {instance} / 正在画 "), line
    assert line.endswith(EDITING_NO_BOX), line
    assert "这一帧没有提示框：直接在零件上点 S" in line


def test_an_edit_on_a_frame_with_a_box_says_nothing_about_boxes(window):
    window.prompt_gate = PG.PromptGate()
    _arm_with_alternates(window)
    assert window._prompt_box is not None
    instance = _start_edit(window)
    QApplication.processEvents()
    line = window.status_message()
    assert line.startswith(f"editing {instance} / 正在画 ") and EDITING_NO_BOX not in line
