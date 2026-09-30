"""Parts the picture shows and the log never names (task U5b).

D13 has four RAM slots and eight retention clips. The sheet opened the four
clips of the two filled slots (steps 14-17 -> ``ram_latch.01``-``04``); the four
of the two empty slots are in every frame and were in no row of the database.
Stage S1 now lets the annotator add them, and these tests pin what that means:

* the view model -- keys, defaults, the initial state, staging, ``Apply``,
  ``Revert``, the op log and its inverse, deletion only without shapes;
* the state machine and the task card -- an added clip stands in its initial
  state from step 1 until the board leaves, and comes back with it as a ✚ row
  "跟 motherboard.01 一起装回来的";
* every consumer downstream -- ``infer-relations``, the rule edges and the
  planner, ``import-logs --force``, ``check``, ``status``, the COCO and the VLM
  exports -- treats it as an ordinary visible part and guesses nothing about it;
* the dialog, offscreen.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox

import session_scene as SS
import vlm_scene as VS
from steps_fixtures import FIXTURES, seeded_db
from tda.core import extra as X
from tda.core import masks
from tda.core.db import Db
from tda.core.graph import (
    find_dead_ends,
    find_deadlocks,
    propose_edges,
    remaining_plan,
    validate_sequence,
)
from tda.core.graph_rules import infer_relational_fields
from tda.core.model import FrameKey, InstanceRec, ShapeKeyframe, ShapePart, ZOrderRec
from tda.core.states import (
    INITIAL_STEP,
    diff_states,
    gone_with_parent,
    needs_geom,
    validate_events,
)
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import events_of, state_of
from tda.ui import session_api as api
from tda.ui.session import AnnotationSession
from tda.ui.session_tasks import task_card_for
from tda.ui.steps_model import EditError, StepTableData

DESKTOP = 13
BOARD = "motherboard.01"
BOARD_STEP = 42
LOGGED = tuple(f"ram_latch.{i:02d}" for i in (1, 2, 3, 4))
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


@pytest.fixture
def data(db, tax) -> StepTableData:
    return StepTableData.load(db, DESKTOP, tax, ANNOTATOR)


def _applied(data: StepTableData, db: Db, **kwargs) -> list[str]:
    """Add the four clips of the empty slots and press Apply."""
    keys = data.add_extras("ram_latch", 4, **kwargs)
    data.save(db)
    return keys


def _ops(db: Db, kind: str) -> list[dict]:
    return [op for op in db.ops(DESKTOP, X.OP_VIEW) if op["kind"] == kind]


def _keyframe(instance: str, desktop: int = DESKTOP, view: str = "scan",
              anchor: int = 41, hw=(64, 64)) -> ShapeKeyframe:
    mask = np.zeros(hw, dtype=bool)
    mask[2:6, 2:6] = True
    return ShapeKeyframe(id=None, instance=instance, desktop=desktop, view=view,
                         pose_segment=1, anchor_step=anchor,
                         parts=[ShapePart("main", rle=masks.encode_rle(mask))])


# --------------------------------------------------------------------------- #
# keys and defaults
# --------------------------------------------------------------------------- #
def test_keys_continue_after_the_last_ordinal_of_the_class(data):
    assert data.preview_extras("ram_latch", 4) == list(EXTRAS)
    assert data.add_extras("ram_latch", 4) == list(EXTRAS)
    # a second batch continues the run it left, staged or not
    assert data.add_extras("ram_latch", 1) == ["ram_latch.09"]

    # a class with a discriminator continues its own run, not the class's
    top = max(int(k.rsplit(".", 1)[1]) for k in data.instances
              if k.startswith("screw.motherboard."))
    screws = data.add_extras("screw", 2, attrs={"role": "motherboard"})
    assert screws == [f"screw.motherboard.{top + 1:02d}", f"screw.motherboard.{top + 2:02d}"]


def test_the_defaults_come_from_the_taxonomy(data):
    defaults = data.extra_defaults("ram_latch")
    assert defaults["parent"] == BOARD            # the one instance of its host_class
    assert defaults["parents"][0] == BOARD
    assert defaults["state"] == "closed"          # the class's default_state
    assert defaults["states"] == ["closed", "open"]   # never `removed`

    keys = data.add_extras("ram_latch", 4)
    for key in keys:
        rec = data.instances[key]
        assert rec.cls == "ram_latch"
        assert rec.parent == BOARD and rec.attached   # leaves with the board
        assert rec.raw_names == []                    # no sheet ever named it
        assert rec.attrs[X.ADDED_BY] == ANNOTATOR
        assert rec.attrs[X.REASON_ATTR] == X.REASON == "not in the log"
        assert rec.attrs[X.ADDED_AT]
        assert "of" not in rec.attrs                  # an empty slot locks nothing
        assert rec.socket_host is None and rec.cable is None
        assert X.is_extra(rec)
    assert data.extras[-1].events == []   # the default state needs no event

    # a class that declares no host rides on nothing unless the annotator says so
    assert data.extra_defaults("cable_clip")["parent"] is None
    clip = data.add_extras("cable_clip", 1)[0]
    assert data.instances[clip].parent is None and not data.instances[clip].attached


def test_a_state_other_than_the_default_is_one_manual_event_before_step_1(data):
    keys = data.add_extras("ram_latch", 2, state="open", note="空槽的卡扣")
    events = data.extras[-1].events
    assert [(e.step, e.target, e.attr, e.old, e.new, e.auto) for e in events] == [
        (INITIAL_STEP, key, "state", "closed", "open", False) for key in keys]
    assert data.instances[keys[0]].attrs[X.NOTE_ATTR] == "空槽的卡扣"


@pytest.mark.parametrize("kwargs", [
    {"cls": "chassis"},
    {"cls": "no_such_class"},
    {"cls": "ram_latch", "count": 0},
    {"cls": "ram_latch", "count": 17},
    {"cls": "ram_latch", "state": "removed"},
    {"cls": "ram_latch", "state": "fastened"},
    {"cls": "ram_latch", "parent": "motherboard.09"},
    {"cls": "screw", "attrs": {"role": "mainboard"}},
])
def test_what_the_taxonomy_does_not_allow_is_refused(data, kwargs):
    before = dict(data.instances)
    with pytest.raises(EditError):
        data.add_extras(**kwargs)
    assert data.instances == before and not data.extras


def test_adding_them_raises_no_open_question(data):
    before = list(data.issues)
    data.add_extras("ram_latch", 4)
    assert data.issues == before   # not an orphan: no action is its definition


# --------------------------------------------------------------------------- #
# the state machine
# --------------------------------------------------------------------------- #
def test_an_added_clip_is_in_the_chassis_from_step_1_until_the_board_leaves(data, db, tax):
    _applied(data, db)
    for step in range(1, BOARD_STEP):
        state = state_of(db, tax, DESKTOP, step)
        for key in EXTRAS:
            assert (state[key].state, state[key].placement) == ("closed", "in_chassis"), step
    gone = state_of(db, tax, DESKTOP, BOARD_STEP)
    for key in EXTRAS:
        assert gone[key].state == "removed"
        assert gone_with_parent(gone, key)   # inside the board, not beside it


def test_it_needs_a_mask_wherever_its_state_does(data, db, tax):
    _applied(data, db)
    instances = db.instances(DESKTOP)
    for step in (1, 20, BOARD_STEP - 1):
        needs = needs_geom(instances, state_of(db, tax, DESKTOP, step), tax)
        assert {needs.get(key) for key in EXTRAS} == {"mask"}, step
    needs = needs_geom(instances, state_of(db, tax, DESKTOP, BOARD_STEP), tax)
    assert not set(EXTRAS) & set(needs)


def test_the_difference_between_frames_names_it_only_where_the_board_moves(data, db, tax):
    _applied(data, db)
    rows = diff_states(state_of(db, tax, DESKTOP, BOARD_STEP),
                       state_of(db, tax, DESKTOP, BOARD_STEP - 1))
    for key in EXTRAS:
        assert (key, "state", "removed", "closed") in rows
    early = diff_states(state_of(db, tax, DESKTOP, 1), state_of(db, tax, DESKTOP, 2))
    assert not [row for row in early if row[0] in EXTRAS]


def test_an_initial_state_holds_until_the_cascade_and_the_log_agrees(data, db, tax):
    messages = _applied(data, db, state="open") and data.messages
    assert not [m for m in messages if any(key in m for key in EXTRAS)]
    assert state_of(db, tax, DESKTOP, 1)["ram_latch.05"].state == "open"
    assert state_of(db, tax, DESKTOP, BOARD_STEP - 1)["ram_latch.05"].state == "open"
    # the cascade the derived log records starts from the state the picture showed
    cascade = [e for e in events_of(db, tax, DESKTOP)
               if e.auto and e.target in EXTRAS and e.attr == "state"]
    assert [(e.step, e.old, e.new) for e in cascade] == [(BOARD_STEP, "open", "removed")] * 4
    # and so does the copy Apply stored
    stored = [e for e in db.events(DESKTOP)
              if e.auto and e.target in EXTRAS and e.attr == "state"]
    assert {(e.old, e.new) for e in stored} == {("open", "removed")}
    assert not validate_events(db.instances(DESKTOP), db.events(DESKTOP), tax)


# --------------------------------------------------------------------------- #
# Apply, Revert and the op log
# --------------------------------------------------------------------------- #
def test_revert_drops_a_staged_addition(data, db, tax):
    data.add_extras("ram_latch", 4, state="open")
    reverted = StepTableData.load(db, DESKTOP, tax, ANNOTATOR)   # what Revert does
    assert not set(EXTRAS) & set(reverted.instances)
    assert not set(EXTRAS) & set(db.instances(DESKTOP))
    assert not _ops(db, X.OP_ADD)


def test_apply_writes_the_rows_their_events_and_one_op(data, db):
    data.add_extras("ram_latch", 4, state="open")
    data.save(db)
    assert set(EXTRAS) <= set(db.instances(DESKTOP))
    initial = [e for e in db.events(DESKTOP) if not e.auto and e.target in EXTRAS]
    assert {(e.step, e.new) for e in initial} == {(INITIAL_STEP, "open")}
    ops = _ops(db, X.OP_ADD)
    assert len(ops) == 1                      # one dialog, one op
    op = ops[0]
    assert [rec["key"] for rec in op["payload"]["instances"]] == list(EXTRAS)
    assert op["inverse"]["delete_instances"] == list(EXTRAS)
    assert len(op["inverse"]["delete_events"]) == 4
    assert op["annotator"] == ANNOTATOR
    assert data.extras == []
    data.save(db)                             # a second Apply adds nothing
    assert len(_ops(db, X.OP_ADD)) == 1


def test_apply_queues_the_frozen_frames_for_a_re_check(data, db):
    db.upsert_frame(FrameKey(DESKTOP, 7, "scan"), "x", {}, None,
                    {"review_status": "verified"})
    db.put_compiled(FrameKey(DESKTOP, 7, "scan"), "chassis", None, 0.0, "visible",
                    "in_chassis", "verified", "h")
    _applied(data, db)
    assert db.rechecks(DESKTOP, "scan") == [7]


def test_the_op_logs_inverse_undoes_the_addition(data, db, tax):
    _applied(data, db, state="open")
    op = _ops(db, X.OP_ADD)[0]
    X.undo_op(db, op, ANNOTATOR)
    assert not set(EXTRAS) & set(db.instances(DESKTOP))
    assert not [e for e in db.events(DESKTOP) if e.target in EXTRAS]
    assert not set(EXTRAS) & set(state_of(db, tax, DESKTOP, 1))
    undo = _ops(db, X.OP_UNDO)
    assert len(undo) == 1 and undo[0]["payload"]["undid"] == op["id"]


def test_the_undo_is_refused_once_one_of_them_has_a_shape(data, db):
    _applied(data, db)
    db.add_keyframe(_keyframe("ram_latch.06"))
    with pytest.raises(X.ExtraError, match="ram_latch.06"):
        X.undo_op(db, _ops(db, X.OP_ADD)[0])
    assert set(EXTRAS) <= set(db.instances(DESKTOP))   # nothing half-done


# --------------------------------------------------------------------------- #
# deletion
# --------------------------------------------------------------------------- #
def test_an_added_part_without_shapes_is_deleted_and_the_op_log_can_bring_it_back(
        data, db, tax):
    _applied(data, db, state="open")
    data.delete_instance(db, "ram_latch.08")
    assert "ram_latch.08" not in db.instances(DESKTOP)
    assert not [e for e in db.events(DESKTOP) if e.target == "ram_latch.08" and not e.auto]
    op = _ops(db, X.OP_DELETE)[-1]
    assert op["inverse"]["instances"][0]["key"] == "ram_latch.08"
    assert [e["new"] for e in op["inverse"]["events"]] == ["open"]

    X.undo_op(db, op)
    back = db.instances(DESKTOP)["ram_latch.08"]
    assert back.parent == BOARD and back.attached and X.is_extra(back)
    assert state_of(db, tax, DESKTOP, 1)["ram_latch.08"].state == "open"


def test_an_added_part_with_a_shape_cannot_be_deleted(data, db, tax):
    _applied(data, db)
    db.add_keyframe(_keyframe("ram_latch.05"))
    fresh = StepTableData.load(db, DESKTOP, tax, ANNOTATOR)
    with pytest.raises(EditError, match="shape keyframes"):
        fresh.delete_instance(db, "ram_latch.05")
    assert "ram_latch.05" in db.instances(DESKTOP)


def test_a_staged_part_is_only_unstaged(data, db):
    keys = data.add_extras("ram_latch", 2)
    data.delete_instance(db, keys[1])
    assert keys[1] not in data.instances
    assert data.extras[0].keys == [keys[0]]
    data.save(db)
    assert keys[0] in db.instances(DESKTOP) and keys[1] not in db.instances(DESKTOP)
    assert [r["key"] for r in _ops(db, X.OP_ADD)[0]["payload"]["instances"]] == [keys[0]]
    assert not _ops(db, X.OP_DELETE)


def test_an_instance_the_log_names_is_still_not_deletable(data, db):
    with pytest.raises(EditError, match="target of step"):
        data.delete_instance(db, LOGGED[0])


# --------------------------------------------------------------------------- #
# the task card
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _session(tmp_path: Path, tax, state: str = "closed") -> AnnotationSession:
    """The whole D13 sheet with the four clips added, opened on the scanner."""
    db = Db(str(tmp_path / "tda.sqlite"))
    cache_dir = tmp_path / "cache"
    SS.seed_db(db, tax, cache_dir, last_step=BOARD_STEP)
    batch = X.plan_extras(db.instances(DESKTOP), tax, DESKTOP, "ram_latch", 4,
                          parent=BOARD, state=state, annotator=ANNOTATOR)
    assert batch.keys == list(EXTRAS)
    X.write_batch(db, batch, ANNOTATOR)
    session = AnnotationSession(db, tax, TruthService(db, tax), str(cache_dir), ANNOTATOR)
    session.open(DESKTOP, "scan")
    return session


@pytest.mark.parametrize("state", ["closed", "open"])
def test_they_come_back_with_the_board_on_the_card(qapp, tmp_path, tax, state):
    from tda.ui.panels.taskcard import ADDED_RAW, row_view

    session = _session(tmp_path, tax, state)
    try:
        session.goto(BOARD_STEP - 1)
        items = {item["instance"]: item for item in session.task_card()}
        for key in EXTRAS:
            item = items[key]
            assert item["kind"] == api.KIND_ADD_SHAPE
            assert item["parent"] == BOARD
            assert f"back in with {BOARD}" in item["text"]
            view = row_view(item)
            assert "（跟 motherboard.01 一起装回来的）" in view["sentence"]
            assert view["raw"] == ADDED_RAW       # no log name: says where it came from
            assert item["done"] is False
        assert row_view(items[LOGGED[0]])["raw"].startswith("日志：")
        order = [item["instance"] for item in session.task_card()]
        assert order[0] == BOARD   # the board leads what came back with it
        assert set(EXTRAS) <= set(order[1:])
    finally:
        session.close(force=True)


def test_the_start_cards_list_them_where_they_are_in_the_picture(qapp, tmp_path, tax):
    session = _session(tmp_path, tax)
    try:
        db = session.db
        # reverse walk: the board is out on the last frame, and so are they
        last = {i["instance"] for i in task_card_for(db, tax, DESKTOP, "scan", BOARD_STEP)}
        assert not set(EXTRAS) & last
        # a forward walk starts on the machine not yet taken apart: they are there
        first = {i["instance"]: i for i in task_card_for(db, tax, DESKTOP, "scan", 1)}
        for key in EXTRAS:
            assert first[key]["kind"] == api.KIND_ADD_SHAPE and first[key]["done"] is False
        # and no other frame of the sheet asks about them
        diffed = {i["instance"] for i in task_card_for(db, tax, DESKTOP, "scan", 1, 2)}
        assert not set(EXTRAS) & diffed
    finally:
        session.close(force=True)


# --------------------------------------------------------------------------- #
# downstream: relations, the graph, the importer
# --------------------------------------------------------------------------- #
def test_infer_relations_fills_nothing_on_them_and_invents_no_action(data, db, tax):
    from tda.cli_relations import infer_relations_into_db

    _applied(data, db, state="open")
    data.add_extras("screw", 1, attrs={"role": "motherboard"})
    data.save(db)
    actions = db.actions(DESKTOP)
    run = infer_relations_into_db(db, tax, {DESKTOP})
    one = run.runs[0]
    assert one.status == "applied", one.error
    added = X.extra_keys(db.instances(DESKTOP))
    assert not [line for line in one.fills + one.unresolved
                if any(key in line for key in added)]
    stored = db.instances(DESKTOP)
    for key in EXTRAS:
        assert "of" not in stored[key].attrs and stored[key].parent == BOARD
    assert all(stored[key].fastens is None for key in added if key.startswith("screw."))
    assert db.actions(DESKTOP) == actions
    assert not validate_events(stored, db.events(DESKTOP), tax)


def test_the_heuristic_would_have_paired_them_with_a_module_and_does_not():
    tax = load_taxonomy()
    board = InstanceRec(key=BOARD, desktop=1, cls="motherboard")
    module = InstanceRec(key="ram_module.01", desktop=1, cls="ram_module")
    clip = InstanceRec(key="ram_latch.01", desktop=1, cls="ram_latch",
                       attrs={X.ADDED_BY: ANNOTATOR})
    instances = {r.key: r for r in (board, module, clip)}
    fills = infer_relational_fields(instances, tax)
    assert "of" not in clip.attrs
    assert fills == [f"ram_latch.01.parent = {BOARD}", "ram_latch.01.attached = True"]


def _with_every_kind_of_extra(data: StepTableData) -> list[str]:
    """Extras of every class a template edge could start from."""
    keys = data.add_extras("ram_latch", 4)
    keys += data.add_extras("psu_latch", 1)
    keys += data.add_extras("cpu_socket_lever", 1)
    keys += data.add_extras("connector", 1, parent=None, attrs={"kind": "atx_24pin"})
    keys += data.add_extras("cable_clip", 1)
    keys += data.add_extras("screw", 1, attrs={"role": "motherboard"})
    return keys


def test_no_rule_edge_starts_or_ends_at_an_added_part(data, tax):
    base = propose_edges(dict(data.instances), tax)
    keys = _with_every_kind_of_extra(data)
    edges = propose_edges(data.instances, tax)
    assert not [e for e in edges if e.target in keys or e.blocker in keys]
    triple = lambda es: sorted((e.type, e.target, e.blocker) for e in es)  # noqa: E731
    assert triple(edges) == triple(base)


def test_the_planner_carries_them_as_inert_nodes(data, db, tax):
    base = dict(db.instances(DESKTOP))
    keys = _with_every_kind_of_extra(data)
    data.save(db)
    instances = db.instances(DESKTOP)
    edges = propose_edges(instances, tax)
    state = state_of(db, tax, DESKTOP, BOARD_STEP - 1)
    plan = remaining_plan(instances, edges, state, BOARD, tax)
    assert plan is not None
    assert not [(verb, key) for verb, key in plan if key in keys]
    base_state = {k: v for k, v in state.items() if k in base}
    assert plan == remaining_plan(base, propose_edges(base, tax), base_state, BOARD, tax)
    assert find_deadlocks(edges, instances, tax) == find_deadlocks(
        propose_edges(base, tax), base, tax)
    assert find_dead_ends(edges, instances, tax) == find_dead_ends(
        propose_edges(base, tax), base, tax)
    actions = db.actions(DESKTOP)
    assert validate_sequence(instances, edges, actions, tax) == validate_sequence(
        base, propose_edges(base, tax), actions, tax)


def test_a_forced_re_import_keeps_them_and_their_cascade(data, db, tax, tmp_path):
    from tda.pipeline_logs import import_logs_into_db

    _applied(data, db, state="open")
    run = import_logs_into_db(db, str(FIXTURES), tax, desktops={DESKTOP}, force=True)
    one = run.runs[0]
    assert one.status == "imported", one.issues
    assert not one.dropped and not one.kept_for_review
    assert set(EXTRAS) <= set(db.instances(DESKTOP))
    stored = [e for e in db.events(DESKTOP)
              if e.auto and e.target in EXTRAS and e.attr == "state"]
    assert [(e.step, e.old, e.new) for e in stored] == [(BOARD_STEP, "open", "removed")] * 4


# --------------------------------------------------------------------------- #
# downstream: check and status
# --------------------------------------------------------------------------- #
def test_check_asks_for_their_shapes_and_status_names_them(tmp_path, capsys, tax):
    from app_scene import make_db, write_paths_yaml
    from tda.cli import EXIT_ERROR, EXIT_OK, main

    db, _paths, _tax = make_db(tmp_path, last_step=BOARD_STEP)
    batch = X.plan_extras(db.instances(DESKTOP), tax, DESKTOP, "ram_latch", 4,
                          parent=BOARD, annotator=ANNOTATOR)
    X.write_batch(db, batch, ANNOTATOR)
    db.close()
    paths_yaml = write_paths_yaml(tmp_path)

    code = main(["--paths", paths_yaml, "check", "--desktop", str(DESKTOP),
                 "--view", "scan", "--limit", "100000"])
    out = capsys.readouterr().out
    assert code == EXIT_ERROR          # nothing is drawn yet: missing shapes
    for key in EXTRAS:
        assert f"missing_shape:{key}" in out

    assert main(["--paths", paths_yaml, "status", "--desktop", str(DESKTOP)]) == EXIT_OK
    detail = capsys.readouterr().out
    assert f"added in S1 (not in the log): 4 - {', '.join(EXTRAS)}" in detail


# --------------------------------------------------------------------------- #
# downstream: the exports
# --------------------------------------------------------------------------- #
EXTRA = "ram_latch.02"
EXTRA_RECT = (22, 30, 32, 40)   # on the board, clear of every other part


@pytest.fixture
def exported_scene(tmp_db_path: str):
    """The VLM scene plus one added clip, drawn in both views, then confirmed.

    Confirmed only once the clip is drawn: it sits on the board, so adding it to
    frames already frozen would move the board's visible mask -- a conflict the
    truth service rightly raises, and not what these tests are about.
    """
    db = Db(tmp_db_path)
    tax = VS.build(db, verified_steps=())
    batch = X.plan_extras(db.instances(VS.DESKTOP), tax, VS.DESKTOP, "ram_latch", 1,
                          parent=VS.BOARD, annotator=ANNOTATOR)
    assert batch.keys == [EXTRA]
    X.write_batch(db, batch, ANNOTATOR)
    for view in VS.VIEWS:
        db.add_keyframe(ShapeKeyframe(
            id=None, instance=EXTRA, desktop=VS.DESKTOP, view=view, pose_segment=1,
            anchor_step=VS.LAST_STEP,
            parts=[ShapePart("main", rle=masks.encode_rle(VS.rect(EXTRA_RECT)))]))
        order = list(db.zorder(VS.DESKTOP, view, 1).order)
        db.set_zorder(ZOrderRec(VS.DESKTOP, view, 1, [*order, (EXTRA, "main")]))
    VS.freeze(db, tax)
    yield db, tax
    db.close()


def test_coco_exports_it_as_an_ordinary_mask(exported_scene, tmp_path):
    from tda.core.export.coco import category_ids, export_coco

    db, tax = exported_scene
    doc = export_coco(db, tax, [VS.DESKTOP], VS.VIEW, str(tmp_path / "c.json"))
    mine = [a for a in doc["annotations"] if a["attributes"]["instance_key"] == EXTRA]
    assert len(mine) == len(VS.STEPS)            # every confirmed frame shows it
    for ann in mine:
        assert ann["category_id"] == category_ids(tax)["ram_latch"]
        assert ann["attributes"]["state"] == "closed"
        assert ann["area"] == 100


def test_the_vlm_export_asks_its_state_and_nothing_else_about_it(exported_scene, tmp_path):
    from tda.core.export.vlm import export_vlm
    from vlm_checker import Checker

    db, tax = exported_scene
    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [VS.DESKTOP], VS.VIEW, str(out))
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert Checker(db, tax, VS.DESKTOP).check_all(records) == len(records)

    def about(record) -> bool:
        return EXTRA in json.dumps(record["label"], ensure_ascii=False)

    states = [r for r in records if r["prompt"]["task"] == "V2" and about(r)
              and r["label"]["answer_check"].get("instance") == EXTRA]
    assert states and all(r["label"]["answer"] == {"state": "closed"} for r in states)
    assert [r for r in records if r["prompt"]["task"] == "V1" and about(r)]
    # no history task names it: nothing was ever done to it, and nothing about it
    # is demonstrated or certainly blocked, so V4 never asks about it either
    for task in ("V3", "V4", "V10", "V16"):
        assert not [r for r in records if r["prompt"]["task"] == task and about(r)], task
    # V6's answer is the logged next action, never it; the clip *can* be opened,
    # so it is in the graph-permitted upper bound, exactly like the logged clip
    for record in (r for r in records if r["prompt"]["task"] == "V6"):
        check = record["label"]["answer_check"]
        assert record["label"]["answer"]["target"] != EXTRA
        assert EXTRA not in json.dumps(check.get("blocked_actions"))
        named = [a for a in check.get("permitted_upper_bound") or []
                 if a["target"] == EXTRA]
        assert named in ([], [{"verb": "open", "target": EXTRA}])
    # V5 likewise: never a must, only the permitted upper bound
    for record in (r for r in records if r["prompt"]["task"] == "V5"):
        answer = record["label"]["answer"]
        assert EXTRA not in json.dumps([answer.get("must_include"),
                                        answer.get("must_not_include")])
        named = [a for a in answer.get("permitted_upper_bound") or []
                 if a["target"] == EXTRA]
        assert named in ([], [{"verb": "open", "target": EXTRA}])


# --------------------------------------------------------------------------- #
# the dialog and the panel (offscreen)
# --------------------------------------------------------------------------- #
@pytest.fixture
def panel(qapp, db, tmp_path, tax):
    from tda.ui.panels.steptable import StepTablePanel

    widget = StepTablePanel(db, DESKTOP, taxonomy=tax, cache_dir=tmp_path / "cache",
                            annotator=ANNOTATOR)
    yield widget
    widget.deleteLater()


def _pick(combo, value) -> None:
    index = combo.findData(value)
    assert index >= 0, value
    combo.setCurrentIndex(index)


def test_the_button_sits_above_the_instance_table(panel):
    from tda.ui.panels.add_parts import BUTTON

    assert panel.tabs.widget(1) is panel.instances_page
    layout = panel.instances_page.layout()
    top = layout.itemAt(0).layout()
    assert top.itemAt(0).widget() is panel.add_parts_button
    assert layout.itemAt(1).widget() is panel.instances_view
    assert panel.add_parts_button.text() == BUTTON == "＋ 添加零件 / Add parts"


def test_the_dialog_prefills_from_the_taxonomy_and_previews_the_keys(panel):
    dialog = panel.open_add_parts()
    try:
        _pick(dialog.class_combo, "ram_latch")
        assert "内存卡扣" in dialog.class_combo.currentText()
        dialog.count_spin.setValue(4)
        assert dialog.parent_combo.currentData() == BOARD
        assert [dialog.state_combo.itemData(i) for i in range(dialog.state_combo.count())] \
            == ["closed", "open"]
        assert dialog.state_combo.currentData() == "closed"
        assert dialog.disc_combo.isHidden()
        text = dialog.preview.text()
        assert "ram_latch.05" in text and "ram_latch.08" in text and BOARD in text
        assert dialog.keys() == list(EXTRAS)

        _pick(dialog.class_combo, "screw")         # a class with a role
        assert not dialog.disc_combo.isHidden()
        assert dialog.disc_combo.currentText() == "motherboard"
        assert dialog.keys()[0].startswith("screw.motherboard.")
        assert dialog.parent_combo.currentData() == ""   # screws declare no host
    finally:
        dialog.reject()


def test_accepting_the_dialog_stages_and_apply_writes(panel, db):
    from tda.ui.panels.steptable_models import EXTRA_RAW_NAMES, INSTANCE_COLUMNS

    edited: list[int] = []
    panel.sigEdited.connect(lambda: edited.append(1))
    dialog = panel.open_add_parts()
    _pick(dialog.class_combo, "ram_latch")
    dialog.count_spin.setValue(4)
    _pick(dialog.state_combo, "open")
    dialog.note_edit.setText("空槽")
    dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).click()

    assert edited == [1]
    assert set(EXTRAS) <= set(panel.data.instances)
    assert not set(EXTRAS) & set(db.instances(DESKTOP))    # staged, not written
    assert "Apply" in panel.status.text()
    model = panel.instances_model
    row = model.keys.index("ram_latch.05")
    raw = [c.title for c in INSTANCE_COLUMNS].index("Raw names")
    assert model.index(row, raw).data() == EXTRA_RAW_NAMES
    assert "chang" in model.index(row, 0).data(3)      # Qt.ToolTipRole

    panel.apply()
    assert set(EXTRAS) <= set(db.instances(DESKTOP))
    assert len(_ops(db, X.OP_ADD)) == 1
    assert panel.last_extra_class == "ram_latch"
    assert db.instances(DESKTOP)["ram_latch.05"].attrs[X.NOTE_ATTR] == "空槽"


def test_revert_drops_what_the_dialog_staged(panel, db):
    assert panel.add_parts("ram_latch", 4) == list(EXTRAS)
    panel.revert()
    assert not set(EXTRAS) & set(panel.data.instances)
    assert not set(EXTRAS) & set(db.instances(DESKTOP))


def test_the_context_menu_deletes_an_applied_part_without_shapes(panel, db):
    panel.add_parts("ram_latch", 4)
    panel.apply()
    menu = panel.instances_menu(panel.instances_model.keys.index("ram_latch.08"))
    assert menu.actions()[0].text() == "Delete ram_latch.08"
    menu.actions()[0].trigger()
    menu.deleteLater()
    assert "ram_latch.08" not in db.instances(DESKTOP)
    assert len(_ops(db, X.OP_DELETE)) == 1


def test_a_refused_addition_says_why(panel):
    assert panel.add_parts("ram_latch", 0) == []
    assert panel.status.text().startswith("Rejected")
    assert not panel.data.extras


def test_the_window_names_the_annotator_and_gates_a_staged_part(qapp, tmp_path, monkeypatch):
    """The part is ``added_by`` whoever opened the window, and leaving Steps
    mode with one staged asks first, like every other unsaved S1 edit."""
    from PySide6.QtWidgets import QMessageBox

    from app_scene import StubSamQueue, close_window, make_paths, make_session
    from tda.ui import app_actions as A
    from tda.ui.app import MainWindow

    window = MainWindow(make_session(tmp_path, last_step=BOARD_STEP), make_paths(tmp_path),
                        "tester", sam_queue=StubSamQueue())
    try:
        window.set_mode(A.MODE_STEPS)
        panel = window.steps_panel
        assert panel.add_parts("ram_latch", 4) == list(EXTRAS)
        assert panel.data.instances["ram_latch.05"].attrs[X.ADDED_BY] == "tester"
        assert window._steps_dirty
        monkeypatch.setattr(QMessageBox, "question",
                            staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
        window.set_mode(A.MODE_ANNOTATE)
        assert window.mode == A.MODE_STEPS   # refused: the addition is not applied
    finally:
        close_window(window)
