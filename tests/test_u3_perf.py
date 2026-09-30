"""U3: arriving on a frame with the detector's cache warm stays inside the frame-change budget.

The window of ``test_session_perf``'s 12 MP budgets, on a 4032x3040 scene and
on a 1600x1600 one, with a stored ROI and a stub detector whose pass over the
view has finished -- the state every frame after the first is in.  Measured
exactly like the window's own frame change (``act_step(-1)`` to a repainted
canvas, best and median of five, :data:`BUDGET_WINDOW_FRAME_CHANGE`), on the
frame whose card asks for four screws back -- with their ``parent`` taken off
the card, so that they ask for the detector -- and the detector's rank 1 is
what the comparison's arrival arms.

The arming itself happens later, when the comparison lands on the GUI thread;
its cost with the detector (ranking, the skip, the log line) is measured too
and bounded against the same handler with the detector off, interleaved in the
same run so that the machine's load falls on both alike.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import statistics
import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from session_scene import DESKTOP, VIEW, make_session
from test_session_perf import (
    BEST_OF_12MP,
    BUDGET_WINDOW_FRAME_CHANGE,
    OAK_HW,
    _median_under,
    _open_window,
    _settle,
    _under,
    best_of,
)
from tda.models.detector import Det, DetectorConfig

SCREW_STEP = 12
#: The comparison's arrival may cost this much more with the detector ranking.
BUDGET_DETECTOR_ARMING_EXTRA = 0.02


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


class _Stub:
    names = {0: "screw"}
    identity = "stub-perf"

    def __init__(self, hw):
        h, w = hw
        self.boxes = [(w * 0.30, h * 0.30, w * 0.30 + 24, h * 0.30 + 24),
                      (w * 0.50, h * 0.40, w * 0.50 + 24, h * 0.40 + 24),
                      (w * 0.60, h * 0.60, w * 0.60 + 24, h * 0.60 + 24)]

    def detect(self, img, crop, view, step=None):
        return [Det(tuple(float(v) for v in b), "screw", 0.9 - 0.1 * i)
                for i, b in enumerate(self.boxes)]


def _config(tmp_path: Path) -> DetectorConfig:
    return DetectorConfig(source=tmp_path / "detector.yaml", model=tmp_path / "stub.pt",
                          classes=("screw",), views=(VIEW,), conf=0.10, roi_crop=True,
                          work_scale=((VIEW, 1.0),), tile=640, stride=512, yield_ms=0.0,
                          cache_root=tmp_path / "det")


def _wait(window, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if window.assist.wait(0.05) and (window.det_worker is None
                                         or window.det_worker.wait(0.05)):
            break
    for _ in range(3):
        QApplication.processEvents()


@pytest.mark.slow
@pytest.mark.parametrize("hw", [OAK_HW, (1600, 1600)], ids=["12mp", "scan"])
def test_arrival_with_a_warm_detector_cache_is_inside_the_frame_change_budget(
        qapp, tmp_path, hw):
    from tda.core import masks as masks_mod

    masks_mod.CHECK_ENCODE_WINDOW, check = False, masks_mod.CHECK_ENCODE_WINDOW
    session = make_session(tmp_path, last_step=14, hw=hw)
    h, w = hw
    roi = [int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)]
    session.db.set_pose_segment(DESKTOP, VIEW, 1, 1, 14, 14, None, None)
    session.db.set_pose_segment_roi(DESKTOP, VIEW, 1, roi, annotator="tester", hw=hw)
    session.goto(SCREW_STEP + 1)
    # frame 12's cooler screws are captive (a parent) and never ask for the
    # detector (round 2); measured here as screws that come back on their own,
    # the way D13's motherboard screws do on frames 35-40
    real_card = session.task_card
    session.task_card = lambda: [
        {k: v for k, v in row.items() if not (k == "parent" and row.get("cls") == "screw")}
        for row in real_card()]
    window = _open_window(session, tmp_path)
    try:
        stub = _Stub(hw)
        window.enable_detector(_config(tmp_path), factory=lambda _c: stub,
                               cache_root=str(tmp_path / "det"))
        _wait(window)
        assert set(window._det_frames) == set(range(1, 15)), "the pass did not finish"
        runs = {"on": [], "off": []}
        arming = {"on": [], "off": []}
        armed_by = {}

        def before(attempt: int) -> None:
            session.goto(SCREW_STEP + 1, force=True)
            session.drain_prefetch(timeout=120.0)
            _wait(window)
            _settle(window)

        def change(attempt: int) -> None:
            window.act_step(-1)
            _settle(window)

        def land(mode: str) -> None:
            # the comparison finishes on its worker; its delivery -- and the
            # arming it triggers -- is GUI-thread time, measured on its own
            assert window.assist.wait(60.0)
            started = time.perf_counter()
            QApplication.processEvents()
            arming[mode].append(time.perf_counter() - started)
            armed_by[mode] = window._prompt_source

        for attempt in range(BEST_OF_12MP):
            for mode in ("on", "off"):
                if mode == "off":
                    saved, window.det_worker = window.det_worker, None
                _, one = best_of(change, before, times=1)
                runs[mode] += one
                land(mode)
                assert window.session.current().step == SCREW_STEP
                if mode == "off":
                    window.det_worker = saved
                    window._det_visit = None
        # (the scene's frames differ only in brightness: the difference map
        # finds no blob, so with the detector off nothing is armed at all)
        assert armed_by["on"] == "det:screw" and armed_by["off"] != "det:screw", armed_by
    finally:
        masks_mod.CHECK_ENCODE_WINDOW = check
        window.shutdown()
        window.hide()
        QApplication.processEvents()
    print(f"\n[u3 perf {hw}] arrival on {runs['on']} off {runs['off']}; "
          f"arming on {arming['on']} off {arming['off']}")
    for check_fn in (_under, _median_under):
        check_fn(BUDGET_WINDOW_FRAME_CHANGE,
                 f"arrival with a warm detector cache ({hw[1]}x{hw[0]})", runs["on"])
    extra = statistics.median(arming["on"]) - statistics.median(arming["off"])
    assert extra <= BUDGET_DETECTOR_ARMING_EXTRA, (
        f"arming with the detector costs {extra * 1000:.1f} ms more at the median; "
        f"on {arming['on']}, off {arming['off']}")
