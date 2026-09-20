"""The full P0 VLM task set of spec 8.2 (task B6).

Every assertion here is made twice: once by naming what the record should say,
and once by :mod:`tests.vlm_checker`, which re-derives the answer from the
database without touching a line of the generator. A question this repository
cannot re-derive is a question the export is not allowed to emit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import vlm_scene as S
from tda.core.db import Db
from tda.core.export.vlm import TASKS, export_vlm
from tda.core.export.vlm_reasoning import OPS, TERMINALS
from tda.core.export.vlm_tasks import PERCEPTION_TASKS, PLANNING_TASKS
from vlm_checker import Checker


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def scene(tmp_db_path: str):
    db = Db(tmp_db_path)
    tax = S.build(db)
    yield db, tax
    db.close()


def _export(db, tax, out: Path, view: str = S.VIEW, **kw) -> dict:
    return export_vlm(db, tax, [S.DESKTOP], view, str(out), **kw)


def _run(db, tax, out: Path, view: str = S.VIEW, **kw) -> list[dict]:
    _export(db, tax, out, view, **kw)
    return _read(out)


def _read(out: Path) -> list[dict]:
    return [json.loads(line) for line in
            out.read_text(encoding="utf-8").splitlines() if line]


def _of(records, task: str) -> list[dict]:
    return [r for r in records if r["task"] == task]


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #
def test_every_p0_task_is_emitted_and_re_derivable(scene, tmp_path: Path):
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    tasks = {r["task"] for r in records}
    assert tasks == set(TASKS), sorted(tasks)

    checker = Checker(db, tax, S.DESKTOP)
    assert checker.check_all(records) == len(records)
    # every check type the format defines was actually exercised
    assert set(checker.by_type) == {"boxes", "exact", "feasibility", "set",
                                    "member_of", "history", "plan"}


def test_every_record_carries_the_contract_fields(scene, tmp_path: Path):
    db, tax = scene
    for record in _run(db, tax, tmp_path / "v.jsonl"):
        assert record["graph_version"] and isinstance(record["graph_version"], str)
        assert record["layer"] in ("L1", "L2", "L3", "L4")
        assert record["tier"] == "gold"
        assert record["model_family"] == "EliteDesk 800 G2 TWR"
        assert record["chassis_type"] == "twr"
        assert record["template_id"].startswith(record["task"] + ".")
        assert record["answer_check"]["type"]
        assert isinstance(record["verified"], bool)


def test_nothing_in_a_record_depends_on_a_split(scene, tmp_path: Path):
    """There is no split yet (spec 8.1: it is cut at freeze, platform-disjoint).

    So a record may not carry one -- and must carry the chassis family, which is
    what the cut will follow.
    """
    db, tax = scene
    db.upsert_desktop(S.DESKTOP, {"brand": "HP", "split": "train",
                                  "model_family": "EliteDesk 800 G2 TWR",
                                  "chassis_type": "twr"})
    for record in _run(db, tax, tmp_path / "v.jsonl"):
        assert "split" not in json.dumps(record)
        assert record["model_family"]


def test_image_paths_are_relative_to_the_dataset_root(scene, tmp_path: Path):
    db, tax = scene
    for record in _run(db, tax, tmp_path / "v.jsonl"):
        assert record["images"]
        for image in record["images"]:
            assert not image.startswith(("F:", "D:", "/"))
            assert image.split("/")[0] in S.VIEWS
            assert image.endswith(".png")


def test_every_rationale_stays_inside_the_closed_operation_set(scene, tmp_path: Path):
    db, tax = scene
    seen = set()
    for record in _run(db, tax, tmp_path / "v.jsonl"):
        steps = record["rationale"]["steps"]
        assert record["rationale"]["depth"] == len(steps)
        assert steps[-1]["op"] in TERMINALS
        for step in steps:
            assert step["op"] in OPS
            seen.add(step["op"])
            if step["op"] == "observe":
                assert step["evidence"]["bbox"] and step["evidence"]["visibility"]
    assert {"observe", "propagate_state", "recall_relation", "check_precondition",
            "compare_frames", "conclude", "abstain"} <= seen


def test_the_two_sources_of_truth_are_kept_apart(scene, tmp_path: Path):
    """Planning survives without a single verified frame; perception does not."""
    db, tax = scene
    summary = _export(db, tax, tmp_path / "all.jsonl")
    assert summary["by_source"]["perception"] > 0
    assert summary["by_source"]["planning"] > 0

    other = Db(str(tmp_path / "bare.sqlite"))
    S.build(other, views=(S.VIEW,), verified_steps=())
    records = _run(other, tax, tmp_path / "bare.jsonl")
    assert {r["task"] for r in records} & PERCEPTION_TASKS == set()
    assert {r["task"] for r in records} == PLANNING_TASKS - {"V15"}
    assert Checker(other, tax, S.DESKTOP).check_all(records) == len(records)
    other.close()


# --------------------------------------------------------------------------- #
# no leakage, no nonsense
# --------------------------------------------------------------------------- #
def test_no_draft_and_no_part_that_left_inside_its_parent_is_asked_about(
    scene, tmp_path: Path
):
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    text = json.dumps(records)
    assert "ls:" not in text

    # the latch is moulded into the board: once the board is out it is inside it
    after = [r for r in records if r["step"] == S.LAST_STEP]
    assert after
    board_gone = {k for k, v in Checker(db, tax, S.DESKTOP).state_at(S.LAST_STEP).items()
                  if v["left_with"]}
    assert S.LATCH not in board_gone  # the board never leaves in this scene
    for record in records:
        for action in record["answer"].get("done", []):
            assert action["target"] in db.instances(S.DESKTOP)


def test_a_latch_that_left_inside_the_board_is_asked_nothing(tmp_path: Path):
    """The ram latch after the board went: no state, no plan, no localisation."""
    db = Db(str(tmp_path / "board.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=())
    # the board comes out at the last step, taking the latch with it
    from tda.core.model import ActionRec

    actions = list(db.actions(S.DESKTOP))
    actions.append(ActionRec(S.DESKTOP, S.LAST_STEP, 1, S.BOARD, "remove", tool="hand"))
    db.replace_steps(S.DESKTOP, db.steps(S.DESKTOP), actions)

    records = _run(db, tax, tmp_path / "v.jsonl")
    # V10 still *remembers* opening the latch -- that happened, and history is
    # what V10 is. Every present-tense question stops mentioning it.
    present = [r for r in records if r["step"] == S.LAST_STEP and r["task"] != "V10"]
    assert present
    for record in present:
        blob = json.dumps(record["answer"]) + json.dumps(record["answer_check"])
        assert S.LATCH not in blob, record["id"]
    assert any(S.LATCH in json.dumps(r["answer"]) for r in records
               if r["task"] == "V10" and r["step"] == S.LAST_STEP)
    db.close()


def test_a_connector_is_never_asked_for_a_state_the_taxonomy_does_not_track(
    scene, tmp_path: Path
):
    """An unplugged connector is untracked (spec 6.2), so "what state?" is a trap."""
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    asked = {r["answer_check"].get("instance") for r in records
             if r["task"] in ("V2", "V14", "V15")}
    assert S.PLUG not in asked
    assert S.CHASSIS not in asked   # one state, "present": the same rule
    assert S.PSU in asked and S.RAM in asked


def test_an_implied_instance_is_marked_as_one(scene, tmp_path: Path):
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    flags = {}
    for record in records:
        for key, attrs in (record["evidence"].get("attributes") or {}).items():
            if "implied" in attrs:
                flags.setdefault(key, set()).add(attrs["implied"])
    assert flags[S.BOARD] == {True}
    assert flags[S.PSU] == {False}


def test_perception_never_asks_about_a_row_it_cannot_point_at(scene, tmp_path: Path):
    """V1 and V8 skip ``occluded_full`` / ``out_of_view``; V14 is what asks."""
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    for record in _of(records, "V1"):
        rows = db.compiled(_key(record))
        for item in record["answer"]["components"]:
            assert rows[item["instance"]]["visibility"] in (
                "visible", "occluded_partial", "visible_tiny")
    grounded = {(r["step"], r["answer"]["instance"]) for r in _of(records, "V8")}
    assert (4, S.RAM) not in grounded      # occluded_full there
    assert (2, S.PLUG) not in grounded     # out_of_view there


# --------------------------------------------------------------------------- #
# V4 -- feasibility and the failed attempt
# --------------------------------------------------------------------------- #
def test_v4_balances_its_positives_and_negatives(scene, tmp_path: Path):
    db, tax = scene
    per_frame: dict[int, list[dict]] = {}
    for record in _of(_run(db, tax, tmp_path / "v.jsonl"), "V4"):
        per_frame.setdefault(record["step"], []).append(record)
    blocked_frames = 0
    for step, group in per_frame.items():
        feasible = [r for r in group if r["answer"]["feasible"]]
        blocked = [r for r in group if not r["answer"]["feasible"]]
        # two of each per frame, plus at most one real failed attempt
        assert len(feasible) <= 2 and len(blocked) <= 3, step
        assert feasible, step
        blocked_frames += bool(blocked)
        for record in blocked:
            assert record["answer"]["blockers"]
            assert record["negative"]
    # the late frames have nothing left to block; most of the teardown does
    assert blocked_frames >= len(per_frame) - 2


def test_v4_explains_the_failed_attempt_with_the_edge_that_caused_it(
    scene, tmp_path: Path
):
    db, tax = scene
    records = _of(_run(db, tax, tmp_path / "v.jsonl"), "V4")
    failed = [r for r in records if r.get("negative") == "failed_attempt"]
    assert len(failed) == 1
    record = failed[0]
    assert record["step"] == 5           # the frame before the attempt at step 6
    assert record["answer"] == {"feasible": False, "blockers": [S.LATCH]}
    assert [e["type"] for e in record["answer_check"]["edges"]] == ["locked_by"]
    ops = [s["op"] for s in record["rationale"]["steps"]]
    assert ops.count("recall_relation") == 1 and ops.count("check_precondition") == 1


def test_a_failed_attempt_the_graph_cannot_explain_is_not_a_ground_truth(
    tmp_path: Path
):
    """The three unexplained failures of the constraint report stay unexported."""
    db = Db(str(tmp_path / "unexplained.sqlite"))
    tax = S.build(db)
    db.add_relation(S.DESKTOP, "locked_by", S.RAM, S.LATCH, status="rejected")
    with db.conn:  # drop the rule edge that explains it, leaving nothing
        db.conn.execute("DELETE FROM relation WHERE desktop=? AND type='locked_by'",
                        (S.DESKTOP,))
    records = _of(_run(db, tax, tmp_path / "v.jsonl"), "V4")
    assert not [r for r in records if r.get("negative") == "failed_attempt"]
    db.close()


# --------------------------------------------------------------------------- #
# V5 / V6 -- legality is the graph's
# --------------------------------------------------------------------------- #
def test_v5_is_the_graphs_whole_legal_set_and_holds_the_logged_next_action(
    scene, tmp_path: Path
):
    db, tax = scene
    checker = Checker(db, tax, S.DESKTOP)
    for record in _of(_run(db, tax, tmp_path / "v.jsonl"), "V5"):
        answer = {(a["verb"], a["target"]) for a in record["answer"]["actions"]}
        assert answer == checker.legal(record["step"])
        logged = record["answer_check"]["logged_next"]
        if logged is not None:
            assert tuple(logged) in answer


def test_v6_answers_with_the_log_and_is_graded_on_membership(scene, tmp_path: Path):
    db, tax = scene
    checker = Checker(db, tax, S.DESKTOP)
    records = _of(_run(db, tax, tmp_path / "v.jsonl"), "V6")
    assert records
    with_negative = 0
    for record in records:
        legal = checker.legal(record["step"])
        assert (record["answer"]["verb"], record["answer"]["target"]) in legal
        options = record["answer_check"]["options"]
        assert len(options) >= 2
        with_negative += any(not o["legal"] for o in options)
        for option in options:
            assert ((option["verb"], option["target"]) in legal) is option["legal"]
    # the distractor the spec asks for: an action the graph forbids right now.
    # Late in a teardown nothing is blocked any more, and inventing one would be
    # inventing a constraint.
    assert with_negative >= len(records) - 2


def test_a_step_the_log_and_the_graph_disagree_about_is_never_ground_truth(
    tmp_path: Path
):
    """One of the 18 violations, in miniature: the PSU comes out still screwed in."""
    db = Db(str(tmp_path / "gap.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=(), skip_actions=(2,))
    summary = _export(db, tax, tmp_path / "v.jsonl")
    records = _read(tmp_path / "v.jsonl")

    assert summary["illegal_steps"] == {
        S.DESKTOP: {5: [f"fastened_by({S.PSU}, {S.SCREW})"]}
    }
    # step 4 is the frame whose "next action" is the illegal one
    assert 4 not in {r["step"] for r in records if r["task"] in ("V5", "V6")}
    assert 3 in {r["step"] for r in records if r["task"] == "V5"}
    # ... and no plan may be built across it
    assert not [r for r in records if r["task"] == "V16" and r["step"] < 5]
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)
    db.close()


# --------------------------------------------------------------------------- #
# V10, V12
# --------------------------------------------------------------------------- #
def test_v10_reports_the_history_the_progress_and_the_remainder(scene, tmp_path: Path):
    db, tax = scene
    records = {r["step"]: r for r in _of(_run(db, tax, tmp_path / "v.jsonl"), "V10")}
    first, last = records[1], records[S.LAST_STEP]
    assert first["answer"]["done"] == [] and first["answer"]["progress_bin"] == "0-25"
    assert last["answer"]["remaining_actions"] == 0
    assert last["answer"]["progress_bin"] == "75-100"
    assert [a["verb"] for a in last["answer"]["done"]] == [
        "unscrew", "disconnect", "remove", "open", "remove"
    ]


def test_v12_takes_its_no_change_from_dupli_and_leaves_failed_alone(
    scene, tmp_path: Path
):
    db, tax = scene
    records = {r["step"]: r for r in _of(_run(db, tax, tmp_path / "v.jsonl"), "V12")}
    assert records[3]["answer"] == {"changed": False, "events": []}
    assert records[3]["negative"] == "dupli_no_change"
    assert records[2]["answer"]["changed"] is True
    assert records[2]["answer"]["events"] == [
        {"target": S.SCREW, "old": "fastened", "new": "removed"}
    ]
    assert 6 not in records  # the failed attempt: nothing changed, but a hand moved


# --------------------------------------------------------------------------- #
# V14 / V15 -- the metacognitive pair
# --------------------------------------------------------------------------- #
def test_v14_abstains_exactly_where_the_view_cannot_decide(scene, tmp_path: Path):
    db, tax = scene
    records = _of(_run(db, tax, tmp_path / "v.jsonl"), "V14")
    abstain = {(r["step"], r["answer_check"]["instance"]) for r in records
               if not r["answer"]["answerable"]}
    assert (4, S.RAM) in abstain and (7, S.LATCH) in abstain
    for record in records:
        instance = record["answer_check"]["instance"]
        row = db.compiled(_key(record))[instance]
        seen = row["visibility"] in ("visible", "occluded_partial", "visible_tiny")
        assert record["answer"]["answerable"] is seen
        assert (record["answer"]["state"] is None) is not seen
        terminal = record["rationale"]["steps"][-1]["op"]
        assert terminal == ("conclude" if seen else "abstain")
    per_frame: dict[int, list[bool]] = {}
    for record in records:
        per_frame.setdefault(record["step"], []).append(record["answer"]["answerable"])
    for step, flags in per_frame.items():
        assert sum(flags) <= 2 and len(flags) - sum(flags) <= 2, step


def test_v15_uses_no_geometry_and_writes_each_view_pair_once(scene, tmp_path: Path):
    db, tax = scene
    here = _of(_run(db, tax, tmp_path / "scan.jsonl", view=S.VIEW), "V15")
    there = _of(_run(db, tax, tmp_path / "oak1.jsonl", view=S.OTHER), "V15")
    assert here and not there                      # scan comes first in VIEWS
    assert {tuple(r["views"]) for r in here} == {(S.VIEW, S.OTHER)}

    cross = [r for r in here if r["answer_check"]["derive"] == "cross_view"]
    moments = [r for r in here if r["answer_check"]["derive"] == "same_moment"]
    assert {(r["step"], r["answer_check"]["instance"]) for r in cross} == {
        (4, S.RAM), (7, S.LATCH)
    }
    assert all(r["answer"]["best_view"] == S.OTHER for r in cross)
    assert {r["answer"]["same_moment"] for r in moments} == {True, False}
    # the question is built from visibility alone: no homography, no transform
    for record in here:
        blob = json.dumps(record)
        for banned in ("homography", "transform", "projected", "T_k"):
            assert banned not in blob


# --------------------------------------------------------------------------- #
# V16 -- plan verification
# --------------------------------------------------------------------------- #
def test_v16_pairs_a_true_plan_with_a_swap_the_graph_catches(scene, tmp_path: Path):
    db, tax = scene
    records = _of(_run(db, tax, tmp_path / "v.jsonl"), "V16")
    true = [r for r in records if r["answer"]["valid"]]
    swapped = [r for r in records if not r["answer"]["valid"]]
    assert true and swapped

    for record in true:
        assert record["answer"] == {"valid": True, "first_error_index": None,
                                    "violated_edge": None}
        assert "negative" not in record
    for record in swapped:
        assert record["answer"]["violated_edge"], record["id"]
        assert record["answer"]["first_error_index"] is not None
        assert record["negative"] == "swapped"
        plan = record["answer_check"]["plan"]
        assert 0 <= record["answer"]["first_error_index"] < len(plan)

    # at step 1 the plan is unscrew / disconnect / remove PSU / open latch, so
    # whichever adjacent swap the seed picks, what it breaks is a precondition
    # of taking the PSU out
    first = next(r for r in swapped if r["step"] == 1)
    assert first["answer"]["violated_edge"] in (
        f"fastened_by({S.PSU}, {S.SCREW})", f"connected_to({S.PSU}, {S.PLUG})",
    )


def test_v16_never_calls_a_harmless_reordering_an_error(scene, tmp_path: Path):
    """Legality, not uniqueness: a swap that violates nothing is another legal plan."""
    db, tax = scene
    checker = Checker(db, tax, S.DESKTOP)
    for record in _of(_run(db, tax, tmp_path / "v.jsonl"), "V16"):
        checker.check(record)  # re-derives validity from the graph, not from us
        if not record["answer"]["valid"]:
            assert record["answer"]["violated_edge"] is not None


# --------------------------------------------------------------------------- #
# determinism, filtering, refusals
# --------------------------------------------------------------------------- #
def test_the_same_database_gives_a_byte_identical_file(scene, tmp_path: Path):
    db, tax = scene
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _export(db, tax, a)
    _export(db, tax, b)
    assert a.read_bytes() == b.read_bytes()


def test_the_order_of_the_records_is_frame_then_task(scene, tmp_path: Path):
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    steps = [r["step"] for r in records]
    assert steps == sorted(steps)
    for step in set(steps):
        order = [TASKS.index(r["task"]) for r in records if r["step"] == step]
        assert order == sorted(order)


def test_the_tasks_argument_filters(scene, tmp_path: Path):
    db, tax = scene
    summary = _export(db, tax, tmp_path / "v.jsonl", tasks=("V4", "V5"))
    assert set(summary["by_task"]) == {"V4", "V5"}
    assert summary["by_source"] == {"perception": 0,
                                    "planning": summary["records"]}


def test_only_verified_keeps_the_confirmed_records(scene, tmp_path: Path):
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl", only_verified=True)
    assert records and {r["verified"] for r in records} == {True}
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)


def test_the_export_refuses_a_view_with_a_standing_conflict(scene, tmp_path: Path):
    db, tax = scene
    from tda.core.masks import encode_rle

    db.add_conflict(_frame_key(1), S.PSU, encode_rle(S.rect(S.RECTS[S.PSU])),
                    encode_rle(S.rect(S.RECTS[S.RAM])), 400)
    with pytest.raises(RuntimeError, match="allow_conflicts"):
        _export(db, tax, tmp_path / "v.jsonl")
    summary = _export(db, tax, tmp_path / "v.jsonl", allow_conflicts=True)
    assert summary["open_conflicts"] == 1
    at_one = [r for r in _read(tmp_path / "v.jsonl") if r["step"] == 1]
    assert at_one and not [r for r in at_one if r["verified"]]


def test_an_answer_check_no_checker_can_execute_is_refused(scene, tmp_path: Path):
    db, tax = scene
    from tda.core.export import vlm_tasks

    ctx = vlm_tasks.TaskCtx(
        ctx=None, view=S.VIEW, tier="gold", graph_version=None, meta={},
        edges=[], frames={}, steps=[],
    )
    with pytest.raises(ValueError, match="not one a checker can execute"):
        vlm_tasks.record(ctx, "V1", "x", 1, [], "?", 0, {}, {"type": "vibes"},
                         {}, {"depth": 0, "steps": []}, False)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _frame_key(step: int, view: str = S.VIEW):
    from tda.core.model import FrameKey

    return FrameKey(S.DESKTOP, step, view)


def _key(record: dict):
    return _frame_key(record["step"], record["view"])
