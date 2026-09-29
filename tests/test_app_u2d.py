"""U2d: the minors the U2c review left, each against the real window.

1. The ROI reminder bar (editor closed) said "no chassis" for any rectangle
   that would be refused -- a 20 px drag left behind by Esc included.
2. The open editor's bar said "Enter 保存；Esc 先跳过" twice and mentioned a
   Shift+R that only matters after Esc.
3. A good drag after a refused one left the refusal in the status line.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import StubSamQueue, close_window, make_paths, make_session
from tda.ui.app import MainWindow
from tda.ui.app_roi import (
    NO_CHASSIS_CLOSED,
    NO_CHASSIS_DRAG,
    NO_CHASSIS_FOUND,
    ROI_BAR_KEYS,
    ROI_BAR_KEYS_STORED,
    ROI_BOX_READY,
    ROI_TOO_SMALL,
    roi_min_side,
)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def open_window(tmp_path: Path, session=None, show=False, **kwargs) -> MainWindow:
    session = session if session is not None else make_session(tmp_path, **kwargs)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(1400, 900)
    if show:
        win.show()
        QApplication.processEvents()
    return win


def roi_bar_text(win: MainWindow) -> str:
    return win.roi_bar.label.text() if win.roi_bar.isVisibleTo(win) else ""


# --------------------------------------------------------------------------- #
# item 1: the reminder bar names the refusal 确认建议框 would answer with
# --------------------------------------------------------------------------- #
def test_a_too_small_drag_left_behind_by_esc_is_called_too_small(qapp, tmp_path):
    """D13/rs: drag a 20 px box, Esc -- the bar used to say "no chassis"."""
    win = open_window(tmp_path, show=True)
    try:
        win.wait_for_roi_proposal()
        assert win.roi_editing
        win.on_roi_box((10.0, 10.0, 14.0, 14.0))
        win.act_clear_edit()                       # Esc: the editor closes
        assert win.roi_editing is False and win.roi_unanswered()

        too_small = ROI_TOO_SMALL.format(floor=roi_min_side(win.overlay.hw))
        bar = roi_bar_text(win)
        assert too_small in bar, bar
        assert NO_CHASSIS_CLOSED not in bar and "没找到机箱" not in bar
        # ... which is exactly what the button answers
        win.act_accept_roi_proposal()
        assert win.status_message() == too_small
        assert win.roi() is None
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 2: the open editor's bar says each thing once, the keys at the end
# --------------------------------------------------------------------------- #
def says_the_keys_once(bar: str, keys: str = ROI_BAR_KEYS) -> None:
    assert bar.endswith(keys), bar
    assert bar.count("Enter") == keys.count("Enter"), bar
    assert bar.count("Esc") == keys.count("Esc"), bar
    assert "Shift+R" not in bar, bar
    for how in ("拖边或角调整", "框里按住整体移动", "空白处拖动重画", "差异图"):
        assert bar.count(how) == 1, (how, bar)


def test_the_open_editor_bar_says_the_keys_once(qapp, tmp_path):
    win = open_window(tmp_path, show=True)
    try:
        win.wait_for_roi_proposal()
        assert win.roi_editing and win.roi() is None
        says_the_keys_once(roi_bar_text(win))

        win.on_roi_box((10.0, 10.0, 14.0, 14.0))          # too small: a reason
        bar = roi_bar_text(win)
        assert "太小" in bar
        says_the_keys_once(bar)
    finally:
        close_window(win)


def test_a_failed_proposal_is_one_reason_and_one_ending(qapp, tmp_path, monkeypatch):
    """The detector's whole frame: the reason once, no second Enter/Esc."""
    from tda.ui import app_roi_worker

    monkeypatch.setattr(app_roi_worker, "suggest_roi_over",
                        lambda images, view: (0, 0, 64, 64))
    win = open_window(tmp_path, show=True)
    try:
        assert win.wait_for_roi_proposal() is True
        bar = roi_bar_text(win)
        assert bar.count(NO_CHASSIS_DRAG) == 1
        says_the_keys_once(bar)
    finally:
        close_window(win)


def test_the_stored_editor_bar_says_esc_keeps_it(qapp, tmp_path):
    win = open_window(tmp_path, show=True)
    try:
        win.wait_for_roi_proposal()
        win.act_commit()                                  # store the proposal
        assert win.roi() is not None
        win.act_edit_roi()                                # Shift+R: open it again
        assert win.roi_editing
        bar = roi_bar_text(win)
        says_the_keys_once(bar, ROI_BAR_KEYS_STORED)
        assert "先跳过" not in bar

        win.on_roi_box((10.0, 10.0, 14.0, 14.0))
        bar = roi_bar_text(win)
        assert "太小" in bar
        says_the_keys_once(bar, ROI_BAR_KEYS_STORED)
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 3: a storable drag replaces the refusal the last one left
# --------------------------------------------------------------------------- #
def test_a_good_drag_after_refused_ones_says_it_is_ready(qapp, tmp_path, monkeypatch):
    from tda.ui import app_roi_worker

    monkeypatch.setattr(app_roi_worker, "suggest_roi_over",
                        lambda images, view: (0, 0, 64, 64))
    win = open_window(tmp_path, show=True)
    try:
        assert win.wait_for_roi_proposal() is True
        assert NO_CHASSIS_FOUND in win.status_message()

        win.on_roi_box((10.0, 10.0, 14.0, 14.0))
        too_small = ROI_TOO_SMALL.format(floor=roi_min_side(win.overlay.hw))
        assert win.status_message() == too_small

        win.on_roi_box((8.0, 8.0, 56.0, 56.0))
        assert win.status_message() == ROI_BOX_READY
        assert win.roi_refusal(win.roi_draft) == ""
        win.act_commit()                                  # ... and it is: Enter stores it
        assert win.roi() == (8, 8, 56, 56)
    finally:
        close_window(win)
