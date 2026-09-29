"""U2d: the minors the U2c review left, each against the real window.

1. The ROI reminder bar (editor closed) said "no chassis" for any rectangle
   that would be refused -- a 20 px drag left behind by Esc included.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import StubSamQueue, close_window, make_paths, make_session
from tda.ui.app import MainWindow
from tda.ui.app_roi import NO_CHASSIS_CLOSED, ROI_TOO_SMALL, roi_min_side


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
