"""Shared pytest fixtures for tda."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every windowed encode checks its own promise for the whole test session.
#
# ``masks.encode_rle(mask, window)`` trusts the caller that the mask is empty
# outside the window, and a window that is wrong would truncate stored geometry
# without saying anything. Turning the check on here makes every call site in
# the codebase -- present and future -- guarded on every run, at four ``any``
# passes per call, while the annotator still pays nothing.
from tda.core import masks as _masks  # noqa: E402

_masks.CHECK_ENCODE_WINDOW = True


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "tda_test.sqlite")


@pytest.fixture
def small_img() -> np.ndarray:
    """64x64 RGB image: dark background with a bright 20x20 square at (20..40)."""
    img = np.full((64, 64, 3), 30, dtype=np.uint8)
    img[20:40, 20:40] = 220
    return img


@pytest.fixture
def paths_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "paths.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(autouse=True)
def _offscreen_qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
