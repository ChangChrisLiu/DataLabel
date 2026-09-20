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
from tda.core.export.vlm import TASKS, export_vlm, manifest_path
from tda.core.export.vlm_reasoning import OPS, TERMINALS
from tda.core.export.vlm_tasks import PERCEPTION_TASKS, PLANNING_TASKS
from vlm_checker import CheckFailure, Checker, scan_prompt_leaks


# --------------------------------------------------------------------------- #
# fixtures and helpers
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


def flat(record: dict) -> dict:
    """The record as one mapping, for assertions that do not care about the split."""
    return {"id": record["id"], **record["prompt"], **record["label"]}


def _of(records, task: str) -> list[dict]:
    return [flat(r) for r in records if r["prompt"]["task"] == task]


def _frame_key(step: int, view: str = S.VIEW):
    from tda.core.model import FrameKey

    return FrameKey(S.DESKTOP, step, view)


def _key(record: dict):
    return _frame_key(record["step"], record["view"])


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #
def test_every_p0_task_is_emitted_and_re_derivable(scene, tmp_path: Path):
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    tasks = {r["prompt"]["task"] for r in records}
    assert tasks == set(TASKS), sorted(tasks)

    checker = Checker(db, tax, S.DESKTOP)
    assert checker.check_all(records) == len(records)
    assert set(checker.by_type) == {"boxes", "exact", "feasibility", "bounded_set",
                                    "next_action", "history", "plan"}


def test_every_record_carries_the_contract_fields(scene, tmp_path: Path):
    db, tax = scene
    for record in _run(db, tax, tmp_path / "v.jsonl"):
        assert set(record) == {"id", "prompt", "label"}
        label = record["label"]
        assert label["graph_version"] and isinstance(label["graph_version"], str)
        assert label["layer"] in ("L1", "L2", "L3", "L4")
        assert label["tier"] == "gold"
        assert label["model_family"] == "EliteDesk 800 G2 TWR"
        assert label["chassis_type"] == "twr"
        assert label["template_id"].startswith(record["prompt"]["task"] + ".")
        assert label["answer_check"]["type"]
        assert isinstance(label["verified"], bool)


def test_nothing_in_a_record_depends_on_a_split(scene, tmp_path: Path):
    """There is no split yet (spec 8.1: it is cut at freeze, platform-disjoint)."""
    db, tax = scene
    db.upsert_desktop(S.DESKTOP, {"brand": "HP", "split": "train",
                                  "model_family": "EliteDesk 800 G2 TWR",
                                  "chassis_type": "twr"})
    for record in _run(db, tax, tmp_path / "v.jsonl"):
        assert "split" not in json.dumps(record)
        assert record["label"]["model_family"]


def test_every_rationale_stays_inside_the_closed_operation_set(scene, tmp_path: Path):
    db, tax = scene
    seen = set()
    for record in _run(db, tax, tmp_path / "v.jsonl"):
        steps = record["label"]["rationale"]["steps"]
        assert record["label"]["rationale"]["depth"] == len(steps)
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
    tasks = {r["prompt"]["task"] for r in records}
    assert tasks & PERCEPTION_TASKS == set()
    assert tasks == PLANNING_TASKS - {"V15"}
    assert Checker(other, tax, S.DESKTOP).check_all(records) == len(records)
    other.close()


# --------------------------------------------------------------------------- #
# prompt side vs label side
# --------------------------------------------------------------------------- #
def test_the_prompt_side_shows_the_model_nothing_it_must_infer(scene, tmp_path: Path):
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    assert not scan_prompt_leaks(records)
    for record in records:
        assert set(record["prompt"]) <= {"task", "images", "question", "options"}
        for image in record["prompt"]["images"]:
            assert image.startswith("img_") and len(image) == 20
            assert "scan" not in image and "D07" not in image


def test_the_record_id_is_opaque_and_the_readable_one_is_label_side(
    scene, tmp_path: Path
):
    """Every harness logs record ids, and the readable one spells the teardown."""
    from tda.core.export.vlm_tasks import opaque_record_id

    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    assert len({r["id"] for r in records}) == len(records)
    for record in records:
        readable = record["label"]["readable_id"]
        assert record["id"] == opaque_record_id(readable)
        assert record["id"].startswith("rec_") and len(record["id"]) == 20
        for banned in ("D07", "scan", "s001", S.PSU):
            assert banned not in record["id"]
        # V15's two halves are `V15-...` and `V15m-...`, so the task is a prefix
        assert readable.startswith(record["prompt"]["task"])


def test_a_planted_leak_is_caught(scene, tmp_path: Path):
    """The scan has to be able to fail, or it is decoration."""
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    records[0]["prompt"]["question"] += " (frame s003 of this teardown)"
    records[1]["prompt"]["question"] += f" -- is {S.PSU} feasible?"
    records[2]["prompt"]["question"] += " look at the oak1 view"
    records[3]["id"] = records[3]["label"]["readable_id"]  # the un-hashed id
    problems = scan_prompt_leaks(records)
    assert len(problems) >= 4
    assert any("step/desktop/view" in p for p in problems)
    assert any("label word" in p for p in problems)
    assert any(p.startswith(records[3]["label"]["readable_id"]) for p in problems)


def test_the_manifest_maps_every_opaque_id_back(scene, tmp_path: Path):
    db, tax = scene
    out = tmp_path / "v.jsonl"
    summary = _export(db, tax, out)
    lines = [json.loads(line) for line in
             manifest_path(str(out)).read_text(encoding="utf-8").splitlines() if line]
    header, entries = lines[0], lines[1:]
    assert header["type"] == "header" and header["salt"]
    assert summary["manifest"] == str(manifest_path(str(out)))
    assert len(entries) == summary["images"]

    by_id = {e["image"]: e for e in entries}
    for record in _read(out):
        for image in record["prompt"]["images"]:
            entry = by_id[image]
            assert entry["desktop"] == S.DESKTOP
            assert entry["path"] == f"{entry['view']}/D07/s{entry['step']:03d}.png"
    # the ids really are a function of the frame, not of the record order
    from tda.core.export.vlm_tasks import opaque_image_id

    assert by_id[opaque_image_id(S.DESKTOP, S.VIEW, 1)]["step"] == 1


# --------------------------------------------------------------------------- #
# no leakage, no nonsense
# --------------------------------------------------------------------------- #
def test_no_draft_and_no_part_that_left_inside_its_parent_is_asked_about(
    scene, tmp_path: Path
):
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    assert "ls:" not in json.dumps(records)


def test_a_latch_that_left_inside_the_board_is_asked_nothing(tmp_path: Path):
    """The ram latch after the board went: no state, no plan, no localisation."""
    db = Db(str(tmp_path / "board.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=())
    from tda.core.model import ActionRec

    actions = list(db.actions(S.DESKTOP))
    actions.append(ActionRec(S.DESKTOP, S.LAST_STEP, 1, S.BOARD, "remove", tool="hand"))
    db.replace_steps(S.DESKTOP, db.steps(S.DESKTOP), actions)

    records = [flat(r) for r in _run(db, tax, tmp_path / "v.jsonl")]
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
    asked = {r["label"]["answer_check"].get("instance") for r in records
             if r["prompt"]["task"] in ("V2", "V14", "V15")}
    assert S.PLUG not in asked
    assert S.CHASSIS not in asked   # one state, "present": the same rule
    assert S.PSU in asked and S.RAM in asked


def test_the_chassis_is_not_a_free_point_in_v1_or_v8(scene, tmp_path: Path):
    """It is in every frame and fills it: scoring on it measures nothing."""
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    for record in _of(records, "V1"):
        assert S.CHASSIS not in {c["instance"] for c in record["answer"]["components"]}
        assert record["answer_check"]["exclude_classes"] == ["chassis"]
    assert S.CHASSIS not in {r["answer"]["instance"] for r in _of(records, "V8")}
    # ... and it is still in the truth table, for the COCO export
    assert S.CHASSIS in db.compiled(_frame_key(1))


def test_an_implied_instance_is_marked_as_one(scene, tmp_path: Path):
    db, tax = scene
    flags = {}
    for record in _run(db, tax, tmp_path / "v.jsonl"):
        for key, attrs in (record["label"]["evidence"].get("attributes") or {}).items():
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


def test_v2_counts_what_the_frame_cannot_see(tmp_path: Path):
    """The guarantee the rationale format exists for: observe, then propagate.

    A second PSU screw is drawn three pixels wide, so the compiler calls it
    ``too_small`` (spec 6.2: keep the centre point only). It is a real screw --
    the count is 2 -- and one this view cannot be asked to point at.
    """
    db = Db(str(tmp_path / "tiny.sqlite"))
    tax = S.build(db, views=(S.VIEW,), tiny_screw=True)
    records = _of(_run(db, tax, tmp_path / "v.jsonl"), "V2")
    counts = [r for r in records if "count" in r["answer"] and r["step"] == 1]
    assert len(counts) == 1
    assert counts[0]["answer"] == {"count": 2}
    assert [s["op"] for s in counts[0]["rationale"]["steps"]] == [
        "observe", "propagate_state", "conclude"
    ]
    assert counts[0]["rationale"]["steps"][0]["target"] == S.SCREW
    assert counts[0]["rationale"]["steps"][1]["target"] == S.TINY_SCREW
    # and the ordinal that would give the count away is kept out of V8
    for record in _of(_read(tmp_path / "v.jsonl"), "V8"):
        if record["step"] == 1 and record["answer"]["instance"].startswith("screw."):
            assert record["answer_check"]["expression"] != "label"
    db.close()


# --------------------------------------------------------------------------- #
# the governing principle: demonstrated, blocked, permitted
# --------------------------------------------------------------------------- #
def test_v4_says_yes_only_to_what_the_log_demonstrated(scene, tmp_path: Path):
    db, tax = scene
    checker = Checker(db, tax, S.DESKTOP)
    records = _of(_run(db, tax, tmp_path / "v.jsonl"), "V4")
    yes = [r for r in records if r["answer"]["feasible"]]
    no = [r for r in records if not r["answer"]["feasible"]]
    assert yes and no
    for record in yes:
        assert record["truth_source"] == "demonstrated"
        shown = checker.demonstrated(record["step"])
        assert (shown[1].verb, shown[1].target) == (
            record["answer_check"]["verb"], record["answer_check"]["target"])
    assert {r["truth_source"] for r in no} <= {"graph_blocked", "failed_attempt"}
    # ... and the two classes are balanced, so the majority baseline is ~50 %
    assert abs(len(yes) - len(no)) <= 2


def test_v5_is_a_bounded_set_and_never_claims_the_upper_bound_is_the_truth(
    scene, tmp_path: Path
):
    db, tax = scene
    checker = Checker(db, tax, S.DESKTOP)
    records = _of(_run(db, tax, tmp_path / "v.jsonl"), "V5")
    assert records
    for record in records:
        answer = record["answer"]
        assert set(answer) == {"must_include", "must_not_include",
                               "permitted_upper_bound"}
        must = {(a["verb"], a["target"]) for a in answer["must_include"]}
        must_not = {(a["verb"], a["target"]) for a in answer["must_not_include"]}
        upper = {(a["verb"], a["target"]) for a in answer["permitted_upper_bound"]}
        assert must == {(checker.demonstrated(record["step"])[1].verb,
                         checker.demonstrated(record["step"])[1].target)}
        assert must_not == checker.blocked(record["step"])
        assert upper == checker.permitted(record["step"])
        assert not must & must_not and must <= upper
        assert record["answer_check"]["type"] == "bounded_set"
        # the question may not promise a complete set of physical possibilities
        for banned in ("every", "all possible", "physically possible", "exhaustive"):
            assert banned not in record["question"].lower()


def test_v6_grades_on_the_demonstrated_action_and_the_blocked_rate(
    scene, tmp_path: Path
):
    db, tax = scene
    checker = Checker(db, tax, S.DESKTOP)
    raw = [r for r in _run(db, tax, tmp_path / "v.jsonl")
           if r["prompt"]["task"] == "V6"]
    assert raw
    with_options = 0
    for entry in raw:
        record = flat(entry)
        shown = checker.demonstrated(record["step"])
        assert (record["answer"]["verb"], record["answer"]["target"]) == (
            shown[1].verb, shown[1].target)
        assert record["answer_check"]["metric"] == "match_rate+blocked_rate"
        assert {(a["verb"], a["target"])
                for a in record["answer_check"]["blocked_actions"]} == \
            checker.blocked(record["step"])
        options = entry["label"].get("options")
        if options is None:
            # late in a teardown nothing is certainly blocked, so the question
            # is open-ended rather than a choice between unknowns -- and says so
            assert "options" not in entry["prompt"]
            assert record["answer_check"]["options_kind"] == "open"
            continue
        assert record["answer_check"]["options_kind"] == "listed"
        with_options += 1
        kinds = [o["kind"] for o in options]
        assert kinds.count("demonstrated") == 1
        assert kinds.count("graph_blocked") >= 2
        # the prompt lists them without saying which is which
        shown_options = entry["prompt"]["options"]
        assert len(shown_options) == len(options)
        assert all(set(o) == {"verb", "target_label"} for o in shown_options)
    assert with_options


def test_v16_valid_plans_are_logged_suffixes_and_errors_can_land_anywhere(
    scene, tmp_path: Path
):
    db, tax = scene
    checker = Checker(db, tax, S.DESKTOP)
    records = _of(_run(db, tax, tmp_path / "v.jsonl"), "V16")
    valid = [r for r in records if r["answer"]["valid"]]
    invalid = [r for r in records if not r["answer"]["valid"]]
    assert valid and invalid
    assert abs(len(valid) - len(invalid)) <= 1     # ~1:1

    for record in valid:
        assert record["truth_source"] == "demonstrated"
        wanted = [(i["verb"], i["target"]) for i in record["answer_check"]["plan"]]
        logged = [(a.verb, a.target) for s in checker.log_steps()
                  if s > record["step"] for a in checker.named(s)]
        assert logged[:len(wanted)] == wanted
    for record in invalid:
        assert record["answer"]["violated_edge"]
        assert record["truth_source"] == "graph_blocked"
    indices = {r["answer"]["first_error_index"] for r in invalid}
    assert indices and indices <= {0, 1, 2, 3}


def test_v16_never_calls_a_harmless_reordering_an_error(scene, tmp_path: Path):
    """Legality, not uniqueness: a rearrangement that violates nothing is unknown."""
    db, tax = scene
    checker = Checker(db, tax, S.DESKTOP)
    for record in _run(db, tax, tmp_path / "v.jsonl"):
        if record["prompt"]["task"] != "V16":
            continue
        checker.check(record)  # re-derives validity from the graph, not from us
        if not record["label"]["answer"]["valid"]:
            assert record["label"]["answer"]["violated_edge"] is not None


def test_v4_explains_the_failed_attempt_with_the_edge_that_caused_it(
    scene, tmp_path: Path
):
    db, tax = scene
    records = _of(_run(db, tax, tmp_path / "v.jsonl"), "V4")
    failed = [r for r in records if r.get("truth_source") == "failed_attempt"]
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
    tax = S.build(db, views=(S.VIEW,), verified_steps=())
    with db.conn:  # drop the rule edge that explains it, leaving nothing
        db.conn.execute("DELETE FROM relation WHERE desktop=? AND type='locked_by'",
                        (S.DESKTOP,))
    records = _of(_run(db, tax, tmp_path / "v.jsonl"), "V4")
    assert not [r for r in records if r.get("truth_source") == "failed_attempt"]
    db.close()


def test_a_step_the_log_and_the_graph_disagree_about_is_never_ground_truth(
    tmp_path: Path
):
    """One of the 18 violations, in miniature: the PSU comes out still screwed in."""
    db = Db(str(tmp_path / "gap.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=(), skip_actions=(2,))
    summary = _export(db, tax, tmp_path / "v.jsonl")
    records = [flat(r) for r in _read(tmp_path / "v.jsonl")]

    assert summary["illegal_steps"] == {
        S.DESKTOP: {5: [f"fastened_by({S.PSU}, {S.SCREW})"]}
    }
    graph_tasks = ("V4", "V5", "V6", "V16")
    assert 4 not in {r["step"] for r in records if r["task"] in graph_tasks}
    assert 3 in {r["step"] for r in records if r["task"] == "V5"}
    assert Checker(db, tax, S.DESKTOP).check_all(_read(tmp_path / "v.jsonl")) \
        == len(records)
    db.close()


# --------------------------------------------------------------------------- #
# desktops and logs a task cannot be asked of
# --------------------------------------------------------------------------- #
def test_a_desktop_with_no_graph_answers_no_affordance_question(tmp_path: Path):
    db = Db(str(tmp_path / "nograph.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=())
    with db.conn:
        db.conn.execute("DELETE FROM relation WHERE desktop=?", (S.DESKTOP,))
    summary = _export(db, tax, tmp_path / "v.jsonl")
    assert summary["excluded_desktops"] == {S.DESKTOP: "no_graph_version"}
    assert set(summary["by_task"]) == {"V3", "V10"}
    db.close()


def test_a_desktop_that_is_mostly_drafts_is_not_excluded_for_that(tmp_path: Path):
    """``ls:*`` rows are reference geometry in a parallel table, not missing parts.

    Five real desktops (D13, D18, D19, D24, D33) carry more drafts than resolved
    instances and 41-49 hard edges each. Counting drafts says nothing about
    whether the identities are settled, and `DesktopCtx` has already dropped
    them by the time any question is asked.
    """
    db = Db(str(tmp_path / "drafts.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=())
    from tda.core.model import InstanceRec

    for n in range(20):  # 8 real instances, 20 drafts
        db.upsert_instance(InstanceRec(key=f"ls:Screw#{n}", desktop=S.DESKTOP,
                                       cls="screw", attrs={"role": "psu"}))
    summary = _export(db, tax, tmp_path / "v.jsonl")
    assert summary["excluded_desktops"] == {}
    assert {"V4", "V5", "V6", "V16"} <= set(summary["by_task"])
    assert "ls:" not in (tmp_path / "v.jsonl").read_text(encoding="utf-8")
    db.close()


def test_a_desktop_whose_edges_are_all_rejected_answers_no_affordance_question(
    tmp_path: Path
):
    db = Db(str(tmp_path / "rejected.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=())
    with db.conn:
        db.conn.execute("UPDATE relation SET status='rejected' WHERE desktop=?",
                        (S.DESKTOP,))
    summary = _export(db, tax, tmp_path / "v.jsonl")
    assert summary["excluded_desktops"] == {S.DESKTOP: "no_constraint_edges"}
    assert not set(summary["by_task"]) & {"V4", "V5", "V6", "V16"}
    db.close()


def test_a_log_with_an_unresolved_target_answers_no_history_question(tmp_path: Path):
    db = Db(str(tmp_path / "unresolved.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=())
    from tda.core.model import ActionRec

    actions = list(db.actions(S.DESKTOP))
    actions.append(ActionRec(S.DESKTOP, 7, 1, "?", "remove", tool="hand"))
    db.replace_steps(S.DESKTOP, db.steps(S.DESKTOP), actions)
    summary = _export(db, tax, tmp_path / "v.jsonl")
    assert summary["v10_excluded"] == {S.DESKTOP: ["?"]}
    assert "V10" not in summary["by_task"]
    assert "V5" in summary["by_task"]  # the graph tasks are unaffected
    db.close()


def test_an_unknown_tool_is_never_a_graded_field(tmp_path: Path):
    db = Db(str(tmp_path / "tool.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=())
    actions = list(db.actions(S.DESKTOP))
    for action in actions:
        if action.step == 5:
            action.tool = "unknown"
    db.replace_steps(S.DESKTOP, db.steps(S.DESKTOP), actions)
    records = [flat(r) for r in _run(db, tax, tmp_path / "v.jsonl")]

    v3 = next(r for r in records if r["task"] == "V3" and r["step"] == 5)
    assert v3["answer"]["tool"] is None
    assert "tool" not in v3["answer_check"]["fields"]
    v6 = next(r for r in records if r["task"] == "V6" and r["answer_check"][
        "reference_step"] == 5)
    assert v6["answer"]["tool"] is None and "tool" not in v6["answer_check"]["fields"]
    assert Checker(db, tax, S.DESKTOP).check_all(_read(tmp_path / "v.jsonl"))
    db.close()


# --------------------------------------------------------------------------- #
# the graph B5 left behind: modes, decisions, deadlocks
# --------------------------------------------------------------------------- #
def _bounded(records, step: int) -> tuple[set, set]:
    """``(must_not_include, permitted_upper_bound)`` of the V5 record at ``step``."""
    record = next(r for r in _of(records, "V5") if r["step"] == step)
    return ({(a["verb"], a["target"]) for a in record["answer"]["must_not_include"]},
            {(a["verb"], a["target"])
             for a in record["answer"]["permitted_upper_bound"]})


@pytest.mark.parametrize("mode,open_blocked,displace_blocked", [
    ("cable_tension", False, False),   # it only stops the part leaving
    ("physical_path", False, True),    # in the way: no moving it either
    ("tool_access", True, True),       # cannot be reached at all
    (None, True, True),                # an unstated mode gates everything
])
def test_a_blocked_by_edge_gates_by_its_mode(tmp_path: Path, mode, open_blocked,
                                             displace_blocked):
    """B5 made `blocked_by` read its mode; the checker learned the same table.

    Two edges, one target each, so each verb is decided by the mode alone: the
    latch has nothing else gating ``open``, and the board's only other edge
    (`connected_to`) gates ``remove`` and nothing more.
    """
    db = Db(str(tmp_path / f"mode_{mode}.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=(), edges=[
        {"type": "blocked_by", "target": S.LATCH, "blocker": S.PSU, "mode": mode},
        {"type": "blocked_by", "target": S.BOARD, "blocker": S.RAM, "mode": mode},
    ])
    records = _run(db, tax, tmp_path / "v.jsonl")
    must_not, permitted = _bounded(records, 1)

    assert (("open", S.LATCH) in must_not) is open_blocked
    assert (("open", S.LATCH) in permitted) is not open_blocked
    assert (("displace", S.BOARD) in must_not) is displace_blocked
    assert (("remove", S.BOARD) in must_not)     # every mode stops removal
    # and the independent checker, which transcribed the table, agrees
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)
    db.close()


@pytest.mark.parametrize("status", ["rejected", "accepted_orphan", "rejected_orphan"])
def test_an_inactive_decision_gates_nothing(tmp_path: Path, status):
    """A rejected edge, and a decision about a rule nobody derives any more."""
    db = Db(str(tmp_path / f"decided_{status}.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=())
    before = _run(db, tax, tmp_path / "before.jsonl")
    assert ("remove", S.RAM) in _bounded(before, 1)[0]     # locked by the latch
    assert any(S.LATCH in r["answer"]["blockers"] for r in _of(before, "V4"))

    S.set_status(db, "locked_by", S.RAM, S.LATCH, status)
    records = _run(db, tax, tmp_path / "after.jsonl")
    must_not, permitted = _bounded(records, 1)
    assert ("remove", S.RAM) not in must_not
    assert ("remove", S.RAM) in permitted
    assert not [r for r in _of(records, "V4") if S.LATCH in r["answer"]["blockers"]]
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)
    db.close()


def test_a_manual_edge_on_a_cable_node_catches_a_reordered_plan(tmp_path: Path):
    """The loom is released at step 4; a plan that removes the PSU first cannot work.

    The edge names a virtual ``cable:*`` node, which has no instance row and no
    mask -- the case where V16's own replay used to disagree with V4 about
    whether a blocker missing from the snapshot counts as satisfied. It goes
    through the shared `unmet` now, so it cannot.
    """
    db = Db(str(tmp_path / "cable.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=(), cable=True, edges=[
        {"type": "blocked_by", "target": S.PSU, "blocker": S.CABLE,
         "mode": "physical_path", "source": "manual"},
    ])
    records = _run(db, tax, tmp_path / "v.jsonl")
    plans = _of(records, "V16")
    assert plans

    # the true plan keeps the release where the log put it
    true_at_3 = next(r for r in plans if r["step"] == 3 and r["answer"]["valid"])
    listed = [(i["verb"], i["target"]) for i in true_at_3["answer_check"]["plan"]]
    assert ("release", S.CABLE) in listed
    assert listed.index(("release", S.CABLE)) < listed.index(("remove", S.PSU))

    caught = [r for r in plans
              if r["answer"].get("violated_edge") == f"blocked_by({S.PSU}, {S.CABLE})"]
    assert caught, [r["answer"].get("violated_edge") for r in plans]
    for record in caught:
        corrupted = [(i["verb"], i["target"]) for i in record["answer_check"]["plan"]]
        bad = record["answer"]["first_error_index"]
        assert corrupted[bad] == ("remove", S.PSU)
        assert corrupted.index(("release", S.CABLE)) > bad
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)
    db.close()


def test_a_deadlocked_graph_answers_no_affordance_question(tmp_path: Path):
    """Two parts each waiting for the other: nothing in the ring can ever be done."""
    db = Db(str(tmp_path / "deadlock.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=(), edges=[
        {"type": "blocked_by", "target": S.PSU, "blocker": S.RAM,
         "mode": "tool_access"},
        {"type": "blocked_by", "target": S.RAM, "blocker": S.PSU,
         "mode": "tool_access"},
    ])
    summary = _export(db, tax, tmp_path / "v.jsonl")
    assert summary["excluded_desktops"] == {S.DESKTOP: "deadlock"}
    assert not set(summary["by_task"]) & {"V4", "V5", "V6", "V16"}
    assert {"V3", "V10"} <= set(summary["by_task"])
    db.close()


def test_a_recommended_edge_never_makes_anything_impossible(tmp_path: Path):
    """A preference is not a physical law (spec 7.1), so it blocks nothing."""
    db = Db(str(tmp_path / "recommended.sqlite"))
    tax = S.build(db, views=(S.VIEW,), verified_steps=(), edges=[
        {"type": "blocked_by", "target": S.LATCH, "blocker": S.PSU,
         "mode": "tool_access", "necessity": "recommended"},
    ])
    records = _run(db, tax, tmp_path / "v.jsonl")
    must_not, permitted = _bounded(records, 1)
    assert ("open", S.LATCH) not in must_not
    assert ("open", S.LATCH) in permitted
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
    assert [(a["verb"], a["target"]) for a in last["answer"]["done"]] == sorted([
        ("unscrew", S.SCREW), ("disconnect", S.PLUG), ("remove", S.PSU),
        ("open", S.LATCH), ("remove", S.RAM),
    ])
    assert last["answer_check"]["done_actions"] == 5


def test_v12_takes_its_no_change_from_dupli_and_balances_against_it(
    scene, tmp_path: Path
):
    """One ``dupli`` row in this scene, so exactly one change rides with it.

    The negatives can only come from ``dupli`` (spec 8.2) and there are sixteen
    of those in the whole dataset: publishing every change would make "yes"
    right 99 % of the time, which is a benchmark that measures nothing.
    """
    db, tax = scene
    records = {r["step"]: r for r in _of(_run(db, tax, tmp_path / "v.jsonl"), "V12")}
    assert len(records) == 2
    assert sum(r["answer"]["changed"] for r in records.values()) == 1
    assert records[3]["answer"] == {"changed": False, "events": []}
    assert records[3]["negative"] == "dupli_no_change"
    positive = next(r for r in records.values() if r["answer"]["changed"])
    assert positive["answer"]["events"]
    assert 6 not in records  # the failed attempt: nothing changed, but a hand moved


def test_a_step_flagged_dupli_counts_even_when_its_type_does_not_say_so(
    scene, tmp_path: Path
):
    db, tax = scene
    steps = db.steps(S.DESKTOP)
    for rec in steps:
        if rec.step == 3:
            rec.step_type, rec.dupli = "normal", True
    db.replace_steps(S.DESKTOP, steps, db.actions(S.DESKTOP))
    records = {r["step"]: r for r in _of(_run(db, tax, tmp_path / "v.jsonl"), "V12")}
    assert records[3]["answer"] == {"changed": False, "events": []}


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
        assert record["rationale"]["steps"][-1]["op"] == (
            "conclude" if seen else "abstain")
    # exactly balanced by construction, per frame and therefore overall
    per_frame: dict[int, list[bool]] = {}
    for record in records:
        per_frame.setdefault(record["step"], []).append(record["answer"]["answerable"])
    for step, flags in per_frame.items():
        assert sum(flags) == len(flags) - sum(flags), step
    assert sum(r["answer"]["answerable"] for r in records) * 2 == len(records)


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
    for record in here:
        blob = json.dumps(record)
        for banned in ("homography", "transform", "projected", "T_k"):
            assert banned not in blob


def test_a_different_moment_negative_really_is_a_different_state(
    scene, tmp_path: Path
):
    """Two steps apart is not enough: a dupli pair leaves the machine unchanged."""
    db, tax = scene
    checker = Checker(db, tax, S.DESKTOP)
    negatives = [r for r in _of(_run(db, tax, tmp_path / "v.jsonl"), "V15")
                 if r["answer_check"]["derive"] == "same_moment"
                 and not r["answer"]["same_moment"]]
    assert negatives
    for record in negatives:
        other = record["answer_check"]["other_step"]
        assert checker.bare_state(record["step"]) != checker.bare_state(other)


# --------------------------------------------------------------------------- #
# the checker has to be able to fail
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task,field,mutate", [
    ("V12", "events", lambda a: a.__setitem__("events", [{"target": "x", "old": "a",
                                                          "new": "b"}])),
    ("V12", "changed", lambda a: a.__setitem__("changed", not a["changed"])),
    ("V15", "same_moment", lambda a: a.__setitem__("same_moment",
                                                   not a["same_moment"])),
    ("V15", "best_view", lambda a: a.__setitem__("best_view", "rs")),
    ("V4", "feasible", lambda a: a.__setitem__("feasible", not a["feasible"])),
    ("V4", "blockers", lambda a: a.__setitem__("blockers", ["psu.01"])),
    ("V5", "must_include", lambda a: a.__setitem__("must_include", [])),
    ("V5", "must_not_include", lambda a: a["must_not_include"].pop()),
    ("V6", "target", lambda a: a.__setitem__("target", "chassis.01")),
    ("V10", "remaining_actions", lambda a: a.__setitem__("remaining_actions", 99)),
    ("V10", "done", lambda a: a["done"].pop()),
    ("V16", "valid", lambda a: a.__setitem__("valid", not a["valid"])),
    ("V16", "first_error_index", lambda a: a.__setitem__("first_error_index", 9)),
    ("V1", "components", lambda a: a["components"].pop()),
    ("V2", "state", lambda a: a.__setitem__("state", "nonsense")),
    ("V2", "count", lambda a: a.__setitem__("count", 99)),
    ("V8", "bbox", lambda a: a.__setitem__("bbox", [0, 0, 1, 1])),
    ("V14", "answerable", lambda a: a.__setitem__("answerable",
                                                  not a["answerable"])),
])
def test_a_corrupted_answer_fails_the_checker(scene, tmp_path: Path, task, field,
                                              mutate):
    """The checker has to be able to fail, or it is decoration.

    Every derivation in :mod:`vlm_checker` is re-computed from the database, so
    changing a published answer must break it. A field the checker merely copied
    out of the record would pass this test only by accident.
    """
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    checker = Checker(db, tax, S.DESKTOP)
    mine = [r for r in records if r["prompt"]["task"] == task
            and field in r["label"]["answer"]]
    assert mine, f"no {task} record carries {field!r}"
    hurt = 0
    for record in mine:
        answer = json.loads(json.dumps(record["label"]["answer"]))
        try:
            mutate(answer)
        except (IndexError, KeyError):
            continue
        if answer == record["label"]["answer"]:
            continue
        record["label"]["answer"] = answer
        with pytest.raises(CheckFailure):
            checker.check(record)
        hurt += 1
    assert hurt, f"no {task} record could be corrupted on {field!r}"


# --------------------------------------------------------------------------- #
# determinism, filtering, refusals
# --------------------------------------------------------------------------- #
def test_the_same_database_gives_a_byte_identical_file(scene, tmp_path: Path):
    db, tax = scene
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _export(db, tax, a)
    _export(db, tax, b)
    assert a.read_bytes() == b.read_bytes()
    assert manifest_path(str(a)).read_bytes() == manifest_path(str(b)).read_bytes()


def test_the_order_of_the_records_is_frame_then_task(scene, tmp_path: Path):
    db, tax = scene
    records = [flat(r) for r in _run(db, tax, tmp_path / "v.jsonl")]
    steps = [r["step"] for r in records]
    assert steps == sorted(steps)
    for step in set(steps):
        order = [TASKS.index(r["task"]) for r in records if r["step"] == step]
        assert order == sorted(order)


def test_the_tasks_argument_filters(scene, tmp_path: Path):
    db, tax = scene
    summary = _export(db, tax, tmp_path / "v.jsonl", tasks=("V4", "V5"))
    assert set(summary["by_task"]) == {"V4", "V5"}
    assert summary["by_source"] == {"perception": 0, "planning": summary["records"]}


def test_only_verified_keeps_the_confirmed_records(scene, tmp_path: Path):
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl", only_verified=True)
    assert records and {r["label"]["verified"] for r in records} == {True}
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
    at_one = [r for r in _read(tmp_path / "v.jsonl") if r["label"]["step"] == 1]
    assert at_one and not [r for r in at_one if r["label"]["verified"]]


def test_a_conflict_in_another_view_does_not_refuse_this_one(scene, tmp_path: Path):
    """The refusal belongs to the view being published, not to its neighbours."""
    db, tax = scene
    from tda.core.masks import encode_rle

    db.add_conflict(_frame_key(4, S.OTHER), S.RAM,
                    encode_rle(S.rect(S.RECTS[S.RAM])),
                    encode_rle(S.rect(S.RECTS[S.PSU])), 400)
    records = _run(db, tax, tmp_path / "v.jsonl")
    assert records
    cross = [r for r in _of(records, "V15")
             if r["answer_check"]["derive"] == "cross_view"]
    assert (4, S.RAM) not in {(r["step"], r["answer_check"]["instance"])
                              for r in cross}
    assert (7, S.LATCH) in {(r["step"], r["answer_check"]["instance"]) for r in cross}


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
