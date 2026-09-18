"""The exports against a lazily-compiled truth table (spec 3.4, 8.1).

Since a commit only compiles the frame in front of the annotator, a frame that
has never been visited carries no ``compiled_mask`` rows at all -- and the COCO
export used to answer ``if not rows: continue``, i.e. drop the whole image
without a word.  An export is the one moment the truth table has to be
complete, so it brings it up to date first: pending re-checks are run, and the
view is refreshed unless only the frozen rows are wanted.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from tda.core import masks
from tda.core.export.coco import export_coco
from tda.core.export.vlm import export_vlm
from tda.core.model import FrameKey
from tda.ui import session_api as api
from tda.ui.session import AnnotationSession
from session_scene import (
    CHASSIS,
    COOLER,
    DESKTOP,
    VIEW,
    cell,
    draw,
    make_session,
    rect,
    seed_shapes,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def session(qapp, tmp_path: Path) -> AnnotationSession:
    made = make_session(tmp_path)
    made.sweeper_enabled = False  # nothing may compile behind the assertions
    made.sweeper.stop()
    yield made
    made.close()


def test_an_unvisited_frame_is_exported_with_its_mask(session, tmp_path):
    session.goto(10)
    draw(session, CHASSIS, cell(1), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(2), api.SCOPE_KEYFRAME)
    # step 6 is covered by both keyframes but was never opened, so it has no rows
    assert session.db.compiled(FrameKey(DESKTOP, 6, VIEW)) == {}

    out = tmp_path / "coco.json"
    export_coco(session.db, session.tax, [DESKTOP], VIEW, str(out), only_verified=False)
    doc = json.loads(out.read_text(encoding="utf-8"))

    steps = {img["extra"]["step"] for img in doc["images"]}
    assert 6 in steps
    image = next(i for i in doc["images"] if i["extra"]["step"] == 6)
    anns = [a for a in doc["annotations"] if a["image_id"] == image["id"]]
    assert anns and any(a.get("segmentation") for a in anns)


def test_an_export_drains_the_pending_rechecks(session, tmp_path):
    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True
    session.goto(10)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)
    assert session.truth.pending_rechecks(DESKTOP, VIEW) == [12]

    export_coco(session.db, session.tax, [DESKTOP], VIEW, str(tmp_path / "c.json"),
                only_verified=False)
    assert session.truth.pending_rechecks(DESKTOP, VIEW) == []


def test_only_verified_does_not_recompile_the_view(session, tmp_path, monkeypatch):
    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True

    calls: list = []
    import tda.core.truth as truth_mod

    original = truth_mod.compile_frame
    monkeypatch.setattr(truth_mod, "compile_frame",
                        lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    export_coco(session.db, session.tax, [DESKTOP], VIEW, str(tmp_path / "v.json"),
                only_verified=True)
    assert calls == []  # the frozen rows are the truth; nothing is derived again


def test_the_vlm_export_also_refreshes_first(session, tmp_path):
    session.goto(10)
    draw(session, CHASSIS, cell(1), api.SCOPE_KEYFRAME)
    assert session.db.compiled(FrameKey(DESKTOP, 5, VIEW)) == {}

    out = tmp_path / "vlm.jsonl"
    export_vlm(session.db, session.tax, [DESKTOP], VIEW, str(out), only_verified=False)
    assert session.db.compiled(FrameKey(DESKTOP, 5, VIEW))


def test_an_export_refuses_to_run_on_an_unchecked_frozen_frame(session, tmp_path,
                                                               monkeypatch):
    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True
    session.goto(10)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)

    # the drain cannot make progress: the export must say so rather than export
    monkeypatch.setattr("tda.core.truth.TruthService.run_pending_rechecks",
                        lambda self, desktop, view: {"conflicts": 0})
    with pytest.raises(RuntimeError, match="re-check"):
        export_coco(session.db, session.tax, [DESKTOP], VIEW, str(tmp_path / "x.json"),
                    only_verified=False)


def test_status_counts_the_pending_rechecks(session):
    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True
    session.goto(10)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)

    assert session.db.count_per_view("rechecks")[(DESKTOP, VIEW)] == 1
