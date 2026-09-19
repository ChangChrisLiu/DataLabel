"""A fan lead may only be owned by a cooler that actually has a fan.

``graph_rules.connector_owner`` resolved ``cable:cpu_fan`` onto "the one
``cpu_cooler`` of this desktop" and stopped there.  On twelve machines that one
cooler instance is a bare ``kind=heatsink`` -- the fan itself was never
instantiated, a gap in the sheet -- so the rule bound the fan lead to a part
that has no fan, and told the annotator the heatsink could not come out until a
connector that is not on it was unplugged.

The ruling: no owner edge at all, and an "unresolved fan owner" line so stage S1
can ask for the missing instance.
"""
from __future__ import annotations

import pytest
from graph_scenes import bench_instances
from graph_scenes import inst as _inst
from graph_scenes import triples as _triples

from tda.core.graph import connector_owner, propose_edges, unresolved_fan_owners
from tda.core.model import InstanceRec
from tda.core.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def heatsink_only_desktop() -> dict[str, InstanceRec]:
    """The bench with its cooler replaced by a bare heatsink, plus a fan lead."""
    bench = bench_instances()
    del bench["cpu_cooler.fan.01"]
    for key in [k for k, r in bench.items() if r.fastens == "cpu_cooler.fan.01"]:
        bench[key].fastens = "cpu_cooler.heatsink.01"
        bench[key].parent = "cpu_cooler.heatsink.01"
    bench["cpu_cooler.heatsink.01"] = _inst(
        "cpu_cooler.heatsink.01", "cpu_cooler", attrs={"kind": "heatsink"},
        mounted_on="motherboard.01", slot_id="cooler",
    )
    bench["connector.fan.01"] = _inst(
        "connector.fan.01", "connector", attrs={"kind": "fan"},
        socket_host="motherboard.01", cable="cable:cpu_fan",
    )
    return bench


def fan_capable_desktop(kind: str) -> dict[str, InstanceRec]:
    """The bench whose single cooler carries a fan of this ``kind``."""
    bench = bench_instances()
    del bench["cpu_cooler.fan.01"]
    key = f"cpu_cooler.{kind}.01"
    for screw in [k for k, r in bench.items() if r.fastens == "cpu_cooler.fan.01"]:
        bench[screw].fastens = key
        bench[screw].parent = key
    bench[key] = _inst(key, "cpu_cooler", attrs={"kind": kind},
                       mounted_on="motherboard.01", slot_id="cooler")
    bench["connector.fan.01"] = _inst(
        "connector.fan.01", "connector", attrs={"kind": "fan"},
        socket_host="motherboard.01", cable="cable:cpu_fan",
    )
    return bench


# --------------------------------------------------------------------------- #
# the owner
# --------------------------------------------------------------------------- #
def test_a_bare_heatsink_does_not_own_the_fan_lead():
    bench = heatsink_only_desktop()
    assert connector_owner(bench["connector.fan.01"], bench) is None


@pytest.mark.parametrize("kind", ["fan", "heatsink_fan"])
def test_a_cooler_that_can_carry_a_fan_still_owns_its_lead(kind):
    bench = fan_capable_desktop(kind)
    assert connector_owner(bench["connector.fan.01"], bench) == f"cpu_cooler.{kind}.01"


def test_no_edge_binds_the_fan_lead_to_a_bare_heatsink(tax):
    bench = heatsink_only_desktop()
    edges = _triples(propose_edges(bench, tax))
    assert ("connected_to", "cpu_cooler.heatsink.01", "connector.fan.01") not in edges
    # the plug still gates the board it sits in: only the *owner* edge goes
    assert ("connected_to", "motherboard.01", "connector.fan.01") in edges


@pytest.mark.parametrize("kind", ["fan", "heatsink_fan"])
def test_the_edge_survives_where_the_cooler_really_has_a_fan(tax, kind):
    bench = fan_capable_desktop(kind)
    edges = _triples(propose_edges(bench, tax))
    assert ("connected_to", f"cpu_cooler.{kind}.01", "connector.fan.01") in edges


# --------------------------------------------------------------------------- #
# what S1 is told instead
# --------------------------------------------------------------------------- #
def test_the_dropped_edge_is_reported_as_an_unresolved_fan_owner():
    bench = heatsink_only_desktop()
    lines = unresolved_fan_owners(bench)
    assert len(lines) == 1
    assert "connector.fan.01" in lines[0]
    assert "cpu_cooler.heatsink.01" in lines[0]
    assert "unresolved fan owner" in lines[0]


@pytest.mark.parametrize("kind", ["fan", "heatsink_fan"])
def test_a_resolved_fan_owner_is_not_reported(kind):
    assert unresolved_fan_owners(fan_capable_desktop(kind)) == []


def test_a_desktop_without_a_fan_lead_reports_nothing():
    assert unresolved_fan_owners(bench_instances()) == []


def test_a_psu_power_lead_is_never_mistaken_for_a_fan_lead():
    """The kind decides first: a power lead belongs to the PSU, not the cooler."""
    bench = heatsink_only_desktop()
    bench["connector.cpu_power.01"] = _inst(
        "connector.cpu_power.01", "connector", attrs={"kind": "cpu_power"},
        socket_host="motherboard.01", cable="cable:cpu_fan",
    )
    assert connector_owner(bench["connector.cpu_power.01"], bench) == "psu.01"
    assert len(unresolved_fan_owners(bench)) == 1
