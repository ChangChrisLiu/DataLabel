"""A template edge that is dropped must not be dropped quietly.

``save_template`` and ``apply_template`` take an optional ``report`` list and
explain every edge they skip into it. Two of their three callers pass ``None``,
and for those the explanation went nowhere -- a family template that silently
instantiates four edges instead of nine is the kind of thing that is only
noticed months later, in the constraint panel of the twentieth sibling.
"""
from __future__ import annotations

import pytest
from test_graph_plan import _inst, _sibling, bench, tax  # noqa: F401 (re-used fixtures)

from tda.core.graph_rules import Edge
from tda.core.graph_templates import apply_template, save_template

LOGGER = "tda.core.graph_templates"


def test_save_template_logs_what_it_drops_without_a_report(bench, tmp_path, caplog):
    caplog.set_level("WARNING", logger=LOGGER)
    edge = Edge("blocked_by", "psu.01", "ghost.01")
    assert save_template(tmp_path / "t.yaml", bench, [edge]) == 0
    assert any("ghost.01" in r.message for r in caplog.records)


def test_apply_template_logs_a_missing_slot_without_a_report(bench, tmp_path, caplog):
    edge = Edge("locked_by", "psu.01", "psu_latch.01")
    save_template(tmp_path / "t.yaml", bench, [edge])
    sibling = {k: v for k, v in _sibling(bench).items() if v.cls != "psu_latch"}
    caplog.set_level("WARNING", logger=LOGGER)
    assert apply_template(tmp_path / "t.yaml", sibling) == []
    assert any("psu_latch" in r.message for r in caplog.records)


def test_a_report_list_collects_and_the_log_still_sees_it(bench, tmp_path, caplog):
    caplog.set_level("WARNING", logger=LOGGER)
    notes: list[str] = []
    save_template(tmp_path / "t.yaml", bench, [Edge("blocked_by", "psu.01", "ghost.01")],
                  notes)
    assert notes and "ghost.01" in notes[0]
    assert any("ghost.01" in r.message for r in caplog.records)


def test_one_missing_slot_is_one_line_however_many_edges_it_costs(bench, tmp_path, caplog):
    """``apply_template`` reports per *slot*: that is the one thing to fix."""
    edges = [Edge("blocked_by", "psu.01", "psu_latch.01"),
             Edge("locked_by", "psu.01", "psu_latch.01")]
    save_template(tmp_path / "t.yaml", bench, edges)
    sibling = {k: v for k, v in _sibling(bench).items() if v.cls != "psu_latch"}
    caplog.set_level("WARNING", logger=LOGGER)
    apply_template(tmp_path / "t.yaml", sibling)
    assert len([r for r in caplog.records if "psu_latch" in r.message]) == 1


def test_nothing_dropped_logs_nothing(bench, tmp_path, caplog):
    caplog.set_level("WARNING", logger=LOGGER)
    edge = Edge("locked_by", "psu.01", "psu_latch.01")
    assert save_template(tmp_path / "t.yaml", bench, [edge]) == 1
    assert [r for r in caplog.records if r.name == LOGGER] == []
