"""Parts added in S1, review round 2 (task U5b).

* a re-import that would take over an added key is **refused** unless
  ``--adopt-extras`` hands it to the sheet -- for a row of the sheet and for
  the implied-instance rule alike -- and the report lists what it kept;
* undoing a deletion recompiles the stored cascade and every undo re-stamps
  ``graph_version``;
* the spec 7.4 replay and the planner's default state start from the
  hand-written initial states, so an ``open`` added clip linked by ``of`` is
  no breach;
* the window drops staged S1 edits on "Yes", asks before a desktop switch and
  before closing, and ``Revert`` clears its unsaved-edit flag;
* ``cli check`` compiles every frame, so two runs report the same problems.
"""
from __future__ import annotations

import csv
import os
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from steps_fixtures import FIXTURES, seeded_db
from tda.core import extra as X
from tda.core import masks
from tda.core.db import Db
from tda.core.graph import edges_from_db, graph_version, validate_sequence
from tda.core.graph_edit import violations
from tda.core.graph_plan import _simulated
from tda.core.model import ShapeKeyframe, ShapePart, is_extra
from tda.core.states import initial_overrides
from tda.core.taxonomy import load_taxonomy
from tda.core.truth_inputs import state_of
from tda.pipeline_logs import REFUSED_EXTRAS, import_logs_into_db
from tda.ui.steps_model import StepTableData

DESKTOP = 13
BOARD = "motherboard.01"
EXTRAS = tuple(f"ram_latch.{i:02d}" for i in (5, 6, 7, 8))
ANNOTATOR = "chang"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def db(tmp_db_path, tax):
    conn = seeded_db(tmp_db_path, tax, desktops=(DESKTOP,))
    yield conn
    conn.close()


def _sheet(tmp_path: Path, *, clips: int = 0, drop_board: bool = False) -> Path:
    """D13's sheet, with ``clips`` more "RAM clip N" rows after clip 4 (and
    every later step renumbered), or without its "Motherboard" row."""
    out = tmp_path / "logs"
    out.mkdir(exist_ok=True)
    shutil.copy(FIXTURES / "desktop_13_meta.csv", out / "desktop_13_meta.csv")
    with open(FIXTURES / "desktop_13.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    width = len(head)
    written = [head]
    for row in body:
        seq = int(row[0]) if row and row[0].strip().isdigit() else None
        name = row[1].strip() if len(row) > 1 else ""
        if drop_board and name == "Motherboard":
            continue
        if seq is not None and seq > 17:
            row = [str(seq + clips), *row[1:]]
        written.append(row)
        if seq == 17:
            for i in range(clips):
                written.append([str(18 + i), f"RAM clip {5 + i}", *[""] * (width - 2)])
    with open(out / "desktop_13.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(written)
    return out


def _add_clips(db: Db, tax, state: str = "open", attrs=None, count: int = 4) -> list[str]:
    data = StepTableData.load(db, DESKTOP, tax, ANNOTATOR)
    keys = data.add_extras("ram_latch", count, state=state, attrs=attrs)
    data.save(db)
    return keys


def _shape(db: Db, key: str) -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[2:6, 2:6] = True
    db.add_keyframe(ShapeKeyframe(id=None, instance=key, desktop=DESKTOP, view="scan",
                                  pose_segment=1, anchor_step=41,
                                  parts=[ShapePart("main", rle=masks.encode_rle(mask))]))


def _ops(db: Db, kind: str) -> list[dict]:
    return [op for op in db.ops(DESKTOP, X.OP_VIEW) if op["kind"] == kind]


def _events(db: Db) -> list[tuple]:
    return sorted((e.step, e.target, e.attr, e.old, e.new, bool(e.auto))
                  for e in db.events(DESKTOP))


# --------------------------------------------------------------------------- #
# 1. a re-import never takes an added key over in silence
# --------------------------------------------------------------------------- #
def test_a_re_import_that_would_take_over_added_keys_is_refused(db, tax, tmp_path):
    _add_clips(db, tax)
    _shape(db, "ram_latch.05")
    before = (db.instances(DESKTOP), _events(db), len(db.steps(DESKTOP)))

    run = import_logs_into_db(db, str(_sheet(tmp_path, clips=4)), tax,
                              desktops={DESKTOP}, force=True)
    one = run.runs[0]
    assert (one.status, one.reason) == ("refused", REFUSED_EXTRAS)
    text = one.issues[0]
    for key in EXTRAS:
        assert f"{key} (the sheet)" in text
    assert "--adopt-extras" in text and "Nothing was written" in text
    # nothing moved: rows, provenance, the hand-set states, the step table
    assert (db.instances(DESKTOP), _events(db), len(db.steps(DESKTOP))) == before
    assert all(is_extra(db.instances(DESKTOP)[key]) for key in EXTRAS)


def test_adopt_extras_hands_the_keys_to_the_sheet_and_the_op_log_can_undo_it(
        db, tax, tmp_path):
    _add_clips(db, tax)
    _shape(db, "ram_latch.05")
    run = import_logs_into_db(db, str(_sheet(tmp_path, clips=4)), tax,
                              desktops={DESKTOP}, force=True, adopt_extras=True)
    one = run.runs[0]
    assert one.status == "imported", one.issues
    assert len(one.adopted) == 4 and "adopted ram_latch.05" in one.adopted[0]
    assert one.extras_kept == []
    stored = db.instances(DESKTOP)
    for i, key in enumerate(EXTRAS, start=5):
        assert not is_extra(stored[key])                      # provenance gone
        assert stored[key].raw_names == [f"RAM clip {i}"]     # the sheet's row now
    assert not [e for e in db.events(DESKTOP) if e.target in EXTRAS and not e.auto]
    assert db.keyframes(DESKTOP, "scan", "ram_latch.05")      # the shape stayed
    # the log's clips start closed and are opened by the log's own steps
    assert state_of(db, tax, DESKTOP, 1)["ram_latch.05"].state == "closed"
    assert state_of(db, tax, DESKTOP, 18)["ram_latch.05"].state == "open"
    op = _ops(db, X.OP_ADOPT)[-1]
    assert op["payload"]["taken_by"] == {key: "log" for key in EXTRAS}
    assert [r["key"] for r in op["inverse"]["instances"]] == list(EXTRAS)
    assert len(op["inverse"]["events"]) == 4

    X.undo_op(db, op, ANNOTATOR, tax)
    back = db.instances(DESKTOP)
    assert all(is_extra(back[key]) for key in EXTRAS)
    assert state_of(db, tax, DESKTOP, 1)["ram_latch.05"].state == "open"


def test_the_implied_rule_is_refused_and_adopted_the_same_way(tmp_path, tax):
    """An added motherboard vs the ``motherboard.01`` the implied rule would make."""
    from tda.ui.steps_delete import delete_instance

    sheet = _sheet(tmp_path, drop_board=True)
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        first = import_logs_into_db(db, str(sheet), tax, desktops={DESKTOP})
        assert first.runs[0].status == "imported"
        assert db.instances(DESKTOP)[BOARD].attrs.get("implied")
        data = StepTableData.load(db, DESKTOP, tax, ANNOTATOR)
        delete_instance(data, db, BOARD)                  # "no" to the implied board
        assert data.add_extras("motherboard", 1) == [BOARD]   # ... then added by hand
        data.save(db)
        db.reset_declined_implied(DESKTOP)                # --reset-declined

        refused = import_logs_into_db(db, str(sheet), tax, desktops={DESKTOP}, force=True)
        assert refused.runs[0].reason == REFUSED_EXTRAS
        assert f"{BOARD} (the implied rule)" in refused.runs[0].issues[0]
        assert is_extra(db.instances(DESKTOP)[BOARD])

        adopted = import_logs_into_db(db, str(sheet), tax, desktops={DESKTOP},
                                      force=True, adopt_extras=True)
        assert adopted.runs[0].status == "imported", adopted.runs[0].issues
        board = db.instances(DESKTOP)[BOARD]
        assert board.attrs.get("implied") and not is_extra(board)
        assert _ops(db, X.OP_ADOPT)[-1]["payload"]["taken_by"] == {BOARD: "implied"}
    finally:
        db.close()


def test_the_report_lists_the_added_instances_it_kept(db, tax, tmp_path):
    from tda.cli import logs_report

    _add_clips(db, tax)
    run = import_logs_into_db(db, str(_sheet(tmp_path)), tax, desktops={DESKTOP},
                              force=True)
    one = run.runs[0]
    assert one.status == "imported" and one.extras_kept == list(EXTRAS)
    report = logs_report(run)
    assert f"kept 4 instance(s) added in S1 (not in the sheet): {', '.join(EXTRAS)}" in report
    assert "instances added in S1 and kept (not in the sheet): 4" in report


def test_the_cli_refuses_then_adopts_with_the_flag(tmp_path, capsys, tax):
    from app_scene import make_db, write_paths_yaml
    from tda.cli import EXIT_ERROR, EXIT_OK, main

    db, _paths, _tax = make_db(tmp_path, last_step=42)
    _add_clips(db, tax)
    db.close()
    paths_yaml = write_paths_yaml(tmp_path)
    sheet = _sheet(tmp_path, clips=4)
    base = ["--paths", paths_yaml, "import-logs", "--force", "--desktops", "13",
            "--dir", str(sheet), "--report", str(tmp_path / "report.md")]

    assert main(base) == EXIT_ERROR
    out = capsys.readouterr().out
    assert "refused: D13 would take over instances an annotator added in S1" in out
    assert main([*base, "--adopt-extras"]) == EXIT_OK
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "adopted ram_latch.05" in report


# --------------------------------------------------------------------------- #
# 2. undo puts the derived rows and the graph stamp back too
# --------------------------------------------------------------------------- #
def test_undoing_a_delete_puts_the_stored_cascade_back(db, tax):
    _add_clips(db, tax)
    before = _events(db)
    data = StepTableData.load(db, DESKTOP, tax, ANNOTATOR)
    data.delete_instance(db, "ram_latch.08")
    assert _events(db) != before
    X.undo_op(db, _ops(db, X.OP_DELETE)[-1], ANNOTATOR, tax)
    assert _events(db) == before     # the cascade at 42 included, from `open`


def test_every_undo_restamps_graph_version(db, tax):
    """An added clip that names its module makes a rule edge: undo takes it along."""
    base = (db.get_desktop(DESKTOP) or {}).get("graph_version")
    _add_clips(db, tax, attrs={"of": "ram_module.01"}, count=1)
    edges = [(e.type, e.target, e.blocker) for e in edges_from_db(db, DESKTOP)]
    assert ("locked_by", "ram_module.01", "ram_latch.05") in edges
    with_clip = graph_version(db, DESKTOP)
    assert db.get_desktop(DESKTOP)["graph_version"] == with_clip != base

    # delete, then undo the delete: the rule edge and the stamp come back
    data = StepTableData.load(db, DESKTOP, tax, ANNOTATOR)
    data.delete_instance(db, "ram_latch.05")
    assert db.get_desktop(DESKTOP).get("graph_version") == graph_version(db, DESKTOP)
    X.undo_op(db, _ops(db, X.OP_DELETE)[-1], ANNOTATOR, tax)
    assert graph_version(db, DESKTOP) == with_clip
    assert db.get_desktop(DESKTOP)["graph_version"] == with_clip

    # undo the addition itself: the edge goes, and the stamp follows it
    X.undo_op(db, _ops(db, X.OP_ADD)[-1], ANNOTATOR, tax)
    assert "ram_latch.05" not in db.instances(DESKTOP)
    assert db.get_desktop(DESKTOP).get("graph_version") == graph_version(db, DESKTOP)
    assert graph_version(db, DESKTOP) != with_clip


def test_an_undo_is_refused_while_another_instance_points_at_the_part(db, tax):
    keys = _add_clips(db, tax, count=1)
    data = StepTableData.load(db, DESKTOP, tax, ANNOTATOR)
    data.apply_instance_edit("cable_clip.01" if "cable_clip.01" in data.instances
                             else "ram_latch.01", "parent", keys[0])
    data.save(db)
    with pytest.raises(X.ExtraError, match="named by"):
        X.undo_op(db, _ops(db, X.OP_ADD)[-1], ANNOTATOR, tax)
    assert keys[0] in db.instances(DESKTOP)


# --------------------------------------------------------------------------- #
# 3. the replay and the planner start from the hand-set state
# --------------------------------------------------------------------------- #
def test_an_open_added_clip_linked_by_of_is_no_breach(db, tax):
    data = StepTableData.load(db, DESKTOP, tax, ANNOTATOR)
    data.add_extras("ram_latch", 1, state="open", attrs={"of": "ram_module.01"})
    # staged: the Relations tab replays the session it is looking at
    assert not [v for v in data.relations.violations() if "ram_latch.05" in v.text]
    data.save(db)

    assert not [v for v in violations(db, DESKTOP, tax) if "ram_latch.05" in v.text]
    instances = db.instances(DESKTOP)
    edges = edges_from_db(db, DESKTOP)
    actions = db.actions(DESKTOP)
    initial = initial_overrides(e for e in db.events(DESKTOP) if not e.auto)
    assert initial == {"ram_latch.05": "open"}
    # the test means something: without the initial state it *is* a breach
    assert [t for t in validate_sequence(instances, edges, actions, tax)
            if "ram_latch.05" in t]
    assert not [t for t in validate_sequence(instances, edges, actions, tax,
                                             initial=initial) if "ram_latch.05" in t]
    assert _simulated(edges, instances, tax, None, initial)["ram_latch.05"] == "open"
    assert _simulated(edges, instances, tax, None)["ram_latch.05"] == "closed"


def test_constraints_validate_reports_no_breach_for_it(db, tax):
    from tda.cli_graph import constraints_into_db

    data = StepTableData.load(db, DESKTOP, tax, ANNOTATOR)
    data.add_extras("ram_latch", 1, state="open", attrs={"of": "ram_module.01"})
    data.save(db)
    run = constraints_into_db(db, tax, {DESKTOP}, validate=True)
    lines = [t for one in run.runs for t in one.violations]
    assert not [t for t in lines if "ram_latch.05" in t]


# --------------------------------------------------------------------------- #
# 4. the window: staged S1 edits are dropped on "Yes", asked about everywhere
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp, tmp_path):
    from app_scene import StubSamQueue, close_window, make_paths, make_session
    from tda.ui import app_actions as A
    from tda.ui.app import MainWindow

    win = MainWindow(make_session(tmp_path, last_step=42), make_paths(tmp_path),
                     "tester", sam_queue=StubSamQueue())
    win.set_mode(A.MODE_STEPS)
    yield win
    close_window(win)


def _answer(monkeypatch, button) -> list[str]:
    asked: list[str] = []

    def question(_parent, _title, text, *_a, **_k):
        asked.append(text)
        return button

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return asked


def test_leaving_steps_with_yes_drops_the_staged_edits(window, monkeypatch):
    from tda.ui import app_actions as A

    panel = window.steps_panel
    panel.add_parts("ram_latch", 4)
    asked = _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    window.set_mode(A.MODE_ANNOTATE)
    assert window.mode == A.MODE_ANNOTATE
    assert "Yes drops them" in asked[0] and "丢掉" in asked[0]
    assert not set(EXTRAS) & set(panel.data.instances) and not panel.data.extras
    assert window._steps_dirty is False
    assert not set(EXTRAS) & set(window.db.instances(DESKTOP))


def test_switching_desktop_asks_first(window, monkeypatch):
    panel = window.steps_panel
    panel.add_parts("ram_latch", 4)
    monkeypatch.setattr(window, "_has_frames", lambda *_a: True)
    asked = _answer(monkeypatch, QMessageBox.StandardButton.No)
    window.act_set_desktop(14)
    assert asked and int(window.session.desktop) == DESKTOP
    assert panel.data.extras and window._steps_dirty          # kept, still unsaved

    moved: list = []
    monkeypatch.setattr(window, "leave_frame", lambda move: moved.append(move) or True)
    _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    window.act_set_desktop(14)
    assert moved and not panel.data.extras and window._steps_dirty is False


def test_closing_the_window_asks_first(window, monkeypatch):
    window.steps_panel.add_parts("ram_latch", 4)
    asked = _answer(monkeypatch, QMessageBox.StandardButton.No)
    event = QCloseEvent()
    window.closeEvent(event)
    assert asked and not event.isAccepted()
    assert window.steps_panel.data.extras                   # nothing was lost


def test_revert_clears_the_windows_unsaved_flag(window):
    window.steps_panel.add_parts("ram_latch", 4)
    assert window._steps_dirty
    window.steps_panel.revert()
    assert window._steps_dirty is False


# --------------------------------------------------------------------------- #
# 5. `check` sees every frame, every time
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("frozen", [False, True])
def test_check_reports_the_same_problems_on_a_second_run(tmp_path, capsys, frozen):
    from app_scene import freeze_mask, make_db, write_paths_yaml
    from tda.cli import main

    db, _paths, _tax = make_db(tmp_path)
    if frozen:
        freeze_mask(db)
    db.close()
    argv = ["--paths", write_paths_yaml(tmp_path), "check", "--desktop", str(DESKTOP),
            "--view", "scan"]
    results = []
    for _ in range(2):
        code = main(argv)
        out = capsys.readouterr().out
        results.append((code, int(out.split("problems: ")[1].split()[0]),
                        int(out.split("open conflicts: ")[1].split()[0])))
    assert results[0] == results[1]
    assert results[0][1] > 0 and results[0][0] == 1
