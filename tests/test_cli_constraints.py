"""``python -m tda.cli constraints``: the constraint graph reaches the database.

Spec 7 has been implemented for a while and nothing ever called it.
``propose_edges``, ``find_cycles`` and ``validate_sequence`` had no caller
outside the tests, so the only ``relation`` rows the real database held came
from the Label Studio import, and ``export/vlm.py`` wrote ``"graph_version":
None`` because there was nothing to write.

This command is the missing caller: one transaction per desktop, rule edges
replaced and manual ones left alone, cycles and (with ``--validate``) the spec
7.4 replay checked, a Markdown report for the annotator, and a content hash
stamped in the desktop meta so an export can say which graph it shipped.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from graph_scenes import DESKTOP, bench_instances, good_sequence

from tda.cli import EXIT_ERROR, EXIT_OK, main
from tda.core.db import Db
from tda.core.graph import Edge, edges_to_db, graph_version
from tda.core.model import ActionRec, InstanceRec


@pytest.fixture
def env(tmp_path: Path) -> dict:
    """paths.yaml plus a database holding the bench desktop and a clean teardown."""
    db_path = tmp_path / "annotations" / "tda.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "cache_dir": str(tmp_path / "cache"),
        "db_path": str(db_path),
        "backup_dir": str(tmp_path / "backups"),
        "raw_logs_dir": str(tmp_path / "raw_logs"),
    }
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    db = Db(str(db_path))
    try:
        db.upsert_desktop(DESKTOP, {"brand": "Bench"})
        for rec in bench_instances().values():
            db.upsert_instance(rec)
        db.replace_steps(DESKTOP, [], good_sequence())
    finally:
        db.close()
    return {"paths": str(tmp_path / "paths.yaml"), "cfg": cfg, "tmp": tmp_path}


def run(env: dict, *argv: str) -> int:
    return main(["--paths", env["paths"], *argv])


def open_db(env: dict) -> Db:
    return Db(env["cfg"]["db_path"])


def relations(env: dict) -> list[dict]:
    db = open_db(env)
    try:
        return db.relations(DESKTOP)
    finally:
        db.close()


def triples(rows) -> set[tuple[str, str, str]]:
    return {(r["type"], r["target"], r["blocker"]) for r in rows}


# --------------------------------------------------------------------------- #
# the edges land
# --------------------------------------------------------------------------- #
def test_constraints_stores_the_proposed_edges_as_rule_edges(env):
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_OK
    rows = relations(env)
    assert rows
    assert {r["source"] for r in rows} == {"rule"}
    assert ("covered_by", "cpu.01", "cpu_cooler.fan.01") in triples(rows)


def test_running_it_twice_changes_nothing(env):
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_OK
    first = relations(env)
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_OK
    second = relations(env)
    assert triples(first) == triples(second)
    assert len(first) == len(second)
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_a_manual_edge_is_never_touched(env):
    db = open_db(env)
    try:
        edges_to_db(db, DESKTOP, [
            Edge("blocked_by", "psu.01", "cable:psu", mode="cable_tension",
                 reason="the annotator's hand does not fit", source="manual",
                 status="accepted"),
        ])
    finally:
        db.close()
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_OK
    manual = [r for r in relations(env) if r["source"] == "manual"]
    assert len(manual) == 1
    assert manual[0]["reason"] == "the annotator's hand does not fit"
    assert manual[0]["status"] == "accepted"


def test_a_manual_edge_on_a_proposed_triple_keeps_its_provenance(env):
    """The rule would derive this one too; the human's row still wins."""
    db = open_db(env)
    try:
        edges_to_db(db, DESKTOP, [
            Edge("covered_by", "cpu.01", "cpu_cooler.fan.01", reason="checked by hand",
                 source="manual", status="accepted"),
        ])
    finally:
        db.close()
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_OK
    row = [r for r in relations(env)
           if (r["type"], r["target"], r["blocker"])
           == ("covered_by", "cpu.01", "cpu_cooler.fan.01")]
    assert len(row) == 1
    assert row[0]["source"] == "manual"
    assert row[0]["reason"] == "checked by hand"


def test_a_stale_rule_edge_is_removed_on_the_next_run(env):
    db = open_db(env)
    try:
        edges_to_db(db, DESKTOP, [
            Edge("blocked_by", "psu.01", "motherboard.01", reason="stale", source="rule"),
        ])
    finally:
        db.close()
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_OK
    assert ("blocked_by", "psu.01", "motherboard.01") not in triples(relations(env))


def test_a_label_studio_edge_is_left_alone(env):
    db = open_db(env)
    try:
        edges_to_db(db, DESKTOP, [
            Edge("blocked_by", "psu.01", "chassis", reason="from LS",
                 source="labelstudio"),
        ])
    finally:
        db.close()
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_OK
    kept = [r for r in relations(env) if r["source"] == "labelstudio"]
    assert len(kept) == 1


def test_a_non_constraint_label_studio_row_is_not_part_of_the_graph(env):
    """The import files ``partner_of`` rows in the same table; they gate nothing.

    Feeding them to ``find_cycles`` invented eight cycles on the real database
    out of rows nobody ever claimed were constraints.
    """
    db = open_db(env)
    try:
        edges_to_db(db, DESKTOP, [
            Edge("partner_of", "ls:Screw#1", "ls:Screw#2", reason="labelstudio:scan",
                 source="labelstudio"),
            Edge("partner_of", "ls:Screw#2", "ls:Screw#1", reason="labelstudio:scan",
                 source="labelstudio"),
        ])
    finally:
        db.close()
    out = Path(env["tmp"]) / "partners.md"
    code = run(env, "constraints", "--desktops", str(DESKTOP), "--validate",
               "--report", str(out))
    assert code == EXIT_OK
    assert "- none (the graph is acyclic" in out.read_text(encoding="utf-8")
    kept = [r for r in relations(env) if r["type"] == "partner_of"]
    assert len(kept) == 2


def test_provisional_label_studio_instances_derive_nothing(env):
    db = open_db(env)
    try:
        db.upsert_instance(InstanceRec(key="ls:Screw#1", desktop=DESKTOP, cls="screw",
                                       attrs={"role": "motherboard"},
                                       fastens="motherboard.01"))
    finally:
        db.close()
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_OK
    assert not any("ls:" in b for _t, _g, b in triples(relations(env)))


# --------------------------------------------------------------------------- #
# the stamp
# --------------------------------------------------------------------------- #
def test_the_graph_version_is_stamped_and_readable(env):
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_OK
    db = open_db(env)
    try:
        stamped = (db.get_desktop(DESKTOP) or {}).get("graph_version")
        assert stamped
        assert graph_version(db, DESKTOP) == stamped
    finally:
        db.close()


def test_the_graph_version_follows_the_edge_set(env):
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_OK
    db = open_db(env)
    try:
        before = graph_version(db, DESKTOP)
        edges_to_db(db, DESKTOP, [
            Edge("blocked_by", "psu.01", "chassis", source="manual"),
        ])
        assert graph_version(db, DESKTOP) != before
    finally:
        db.close()


def test_a_desktop_without_edges_has_no_graph_version(env):
    db = open_db(env)
    try:
        db.upsert_desktop(404, {})
        assert graph_version(db, 404) is None
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def test_the_report_lists_the_edges_by_type(env):
    out = Path(env["tmp"]) / "constraints.md"
    assert run(env, "constraints", "--desktops", str(DESKTOP),
               "--report", str(out)) == EXIT_OK
    text = out.read_text(encoding="utf-8")
    assert f"D{DESKTOP:02d}" in text
    for etype in ("fastened_by", "connected_to", "locked_by", "covered_by"):
        assert etype in text
    assert "cycles" in text.lower()


def test_the_report_names_the_violations_with_step_and_reason(env):
    """A teardown that pulls the board out before its screws is a spec 7.4 breach."""
    db = open_db(env)
    try:
        db.replace_steps(DESKTOP, [], [
            ActionRec(desktop=DESKTOP, step=1, idx=0, target="motherboard.01",
                      verb="remove", result="success"),
        ])
    finally:
        db.close()
    out = Path(env["tmp"]) / "violations.md"
    code = run(env, "constraints", "--desktops", str(DESKTOP), "--validate",
               "--report", str(out))
    text = out.read_text(encoding="utf-8")
    assert "step 1" in text and "violates" in text
    assert "fastened_by" in text
    assert code == EXIT_ERROR


def test_a_failed_attempt_with_nothing_in_its_way_is_a_missing_edge_hint(env):
    db = open_db(env)
    try:
        db.replace_steps(DESKTOP, [], [
            ActionRec(desktop=DESKTOP, step=1, idx=0, target="cover.01", verb="open",
                      result="failed"),
        ])
    finally:
        db.close()
    out = Path(env["tmp"]) / "hints.md"
    run(env, "constraints", "--desktops", str(DESKTOP), "--validate", "--report", str(out))
    assert "missing edge?" in out.read_text(encoding="utf-8")


def test_an_unresolved_fan_owner_reaches_the_report(env):
    db = open_db(env)
    try:
        db.delete_instance(DESKTOP, "cpu_cooler.fan.01")
        db.upsert_instance(InstanceRec(key="cpu_cooler.heatsink.01", desktop=DESKTOP,
                                       cls="cpu_cooler", attrs={"kind": "heatsink"}))
        db.upsert_instance(InstanceRec(key="connector.fan.01", desktop=DESKTOP,
                                       cls="connector", attrs={"kind": "fan"},
                                       socket_host="motherboard.01",
                                       cable="cable:cpu_fan"))
    finally:
        db.close()
    out = Path(env["tmp"]) / "fan.md"
    run(env, "constraints", "--desktops", str(DESKTOP), "--report", str(out))
    assert "unresolved fan owner" in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# writing, and not writing
# --------------------------------------------------------------------------- #
def test_dry_run_writes_nothing(env):
    assert run(env, "constraints", "--desktops", str(DESKTOP), "--dry-run") == EXIT_OK
    assert relations(env) == []
    db = open_db(env)
    try:
        assert (db.get_desktop(DESKTOP) or {}).get("graph_version") is None
    finally:
        db.close()
    assert not list(Path(env["cfg"]["backup_dir"]).glob("*.sqlite")) \
        if Path(env["cfg"]["backup_dir"]).exists() else True
    assert not (Path(env["cfg"]["cache_dir"]) / "constraints_report.md").exists()


def test_dry_run_still_writes_a_report_it_was_asked_for(env):
    out = Path(env["tmp"]) / "dry.md"
    assert run(env, "constraints", "--desktops", str(DESKTOP), "--dry-run",
               "--report", str(out)) == EXIT_OK
    assert out.exists()
    assert relations(env) == []


def test_a_real_run_takes_a_safety_backup(env, capsys):
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_OK
    assert "backed the database up first" in capsys.readouterr().out
    assert list(Path(env["cfg"]["backup_dir"]).glob("tda_*.sqlite"))


def test_one_desktop_is_one_transaction(env, monkeypatch):
    """A desktop that raises half way through must leave no edge behind."""
    from tda.core import graph as graph_module

    def boom(db, desktop, edges):
        edges_to_db(db, desktop, edges[:2])
        raise RuntimeError("the disk went away")

    monkeypatch.setattr("tda.cli_graph.edges_to_db", boom)
    assert run(env, "constraints", "--desktops", str(DESKTOP)) == EXIT_ERROR
    assert relations(env) == []
    assert graph_module.edges_to_db is not boom  # the real one is untouched


def test_a_desktop_that_is_not_in_the_database_is_reported_not_fatal(env, capsys):
    assert run(env, "constraints", "--desktops", "404") == EXIT_OK
    assert "404" in capsys.readouterr().out
