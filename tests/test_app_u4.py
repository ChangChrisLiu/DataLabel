"""U4: the RF-DETR backend in the window, through the detector file -- and its fallbacks.

The scene is ``app_scene``'s synthetic D13/scan with frame 12's screws loose
(``test_app_u3.loose_screws``); the window reads its detector file from
``TDA_DETECTOR`` as the app does, and the backend is the real
:class:`~tda.models.detector.RFDetrTileDetector` over a fake ``rfdetr``
(``tests/fake_rfdetr.py``: no weights, no GPU).  Whatever goes wrong -- the
package missing, no CUDA, weights rfdetr cannot read, a malformed file -- is
one WARNING, the detector is off, the window opens, and frame 12's box is the
difference map's.
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import yaml

import fake_rfdetr as FR
from app_scene import close_window
from test_app_u3 import SCREW_STEP, Keep, _window, fbox, go, pump, qapp  # noqa: F401
from tda.ui.app_assist import DET_CHIP

#: Where the bright square is on frame 12 (x 14 + (12 % 5) * 6, y 20..32).
SQUARE_12 = (26.0, 20.0, 36.0, 32.0)


@pytest.fixture
def logged():
    keep = Keep()
    loggers = [logging.getLogger(n) for n in ("tda.app", "tda.ui.app_detect",
                                              "tda.models.detector")]
    for logger in loggers:
        logger.addHandler(keep)
    yield keep.lines
    for logger in loggers:
        logger.removeHandler(keep)


@pytest.fixture
def restored(monkeypatch):
    """The process as it was, whatever the backend under test left (it should leave nothing)."""
    import warnings

    import PIL.Image
    torch = pytest.importorskip("torch")
    precision = torch.get_float32_matmul_precision()
    pil_max, filters = PIL.Image.MAX_IMAGE_PIXELS, list(warnings.filters)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")          # the backend sets it on purpose
    yield torch
    if torch.get_float32_matmul_precision() != precision:
        torch.set_float32_matmul_precision(precision)
    PIL.Image.MAX_IMAGE_PIXELS = pil_max
    warnings.filters[:] = filters
    vars(__import__("numpy")).pop("complex_", None)
    for name in ("KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK"):
        os.environ.pop(name, None)


def detector_file(tmp_path, monkeypatch, **over):
    weights = tmp_path / "rf.pt"
    weights.write_bytes(b"rf-detr weights, or so the fake says")
    data = {"enabled": True, "backend": "rfdetr", "model": str(weights),
            "classes": ["screw"], "views": ["scan", "oak1", "oak2"], "conf": 0.10,
            "merge_floor": 0.10,
            "tiling": {"roi_crop": True, "work_scale": {"scan": 1.0, "oak1": 1.0, "oak2": 1.0},
                       "tile": 640, "stride": 512},
            "device": 0, "half": True, "yield_ms": 0, "cache_root": str(tmp_path / "det")}
    data.update(over)
    path = tmp_path / "detector.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("TDA_DETECTOR", str(path))
    return path


def _open(tmp_path):
    win = _window(tmp_path / "w")
    pump(win, 30.0)
    return win


def test_the_file_turns_the_rfdetr_backend_on_and_it_arms_the_loose_screw(
        qapp, tmp_path, monkeypatch, restored, logged):
    monkeypatch.setattr(restored.cuda, "is_available", lambda: True)
    FR.install(monkeypatch)
    detector_file(tmp_path, monkeypatch)
    win = _open(tmp_path)
    try:
        assert win.det_state == "on"
        on = [m for level, m in logged if m.startswith("small-part detector on")]
        assert len(on) == 1 and "rfdetr model" in on[0] and "merge floor 0.1" in on[0]
        go(win, SCREW_STEP)
        assert win._prompt_box == SQUARE_12
        assert win.canvas.prompt_band()[1] == DET_CHIP.format(name="螺丝")
        # out of the merge: the square alone (the 0.09 box and the extra slot never were)
        assert [(d.box, d.cls) for d in win._det_frames[SCREW_STEP].dets] == [
            (SQUARE_12, "screw")]
    finally:
        close_window(win)


@pytest.mark.parametrize("what", ["no rfdetr", "no CUDA", "unreadable weights",
                                  "malformed file"])
def test_whatever_stops_rfdetr_the_window_opens_on_the_difference_map(
        qapp, tmp_path, monkeypatch, restored, logged, what):
    import importlib.util

    monkeypatch.setattr(restored.cuda, "is_available", lambda: what != "no CUDA")
    FR.install(monkeypatch, fail_load=(RuntimeError("invalid load key, '\\x0f'")
                                       if what == "unreadable weights" else None))
    if what == "no rfdetr":
        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name, *a: None if name == "rfdetr" else real(name, *a))
    detector_file(tmp_path, monkeypatch,
                  **({"merge_floor": 0.5} if what == "malformed file" else {}))
    win = _open(tmp_path)
    try:
        assert win.det_state == "off"
        warnings = [m for level, m in logged if level == "WARNING"
                    and m.startswith("small-part detector OFF")]
        assert len(warnings) == 1, logged
        assert {"no rfdetr": "rfdetr is not installed",
                "no CUDA": "CUDA is not available",
                "unreadable weights": "rfdetr cannot load",
                "malformed file": "merge_floor = 0.5 is above conf"}[what] in warnings[0]
        go(win, SCREW_STEP)
        blob = win.assist_result["unexplained"][0]
        assert win._prompt_box == fbox(blob.box) and win._prompt_source == "diff"
    finally:
        close_window(win)
