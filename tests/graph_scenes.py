"""Builders shared by the constraint-graph tests (not a test module itself).

One small hand-made desktop plus the helpers that read it. Imported by
``test_graph.py`` (the edge rules, preconditions and legal actions) and
``test_graph_plan.py`` (validation, cycles, planning, persistence, templates
and the real-log smoke tests).
"""
from __future__ import annotations

from pathlib import Path

from tda.core.graph import Edge
from tda.core.model import ActionRec, InstanceRec
from tda.core.states import events_from_actions, state_at

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
DESKTOP = 900


def inst(key: str, cls: str, **kw) -> InstanceRec:
    return InstanceRec(key=key, desktop=DESKTOP, cls=cls, **kw)


def bench_instances() -> dict[str, InstanceRec]:
    """One motherboard, one cooler, RAM, a PSU, a SATA drive and a screw cover.

    Slot ids are the family-template slots; every relational field the rules
    read (``fastens``, ``socket_host``, ``cable``, ``of``, and the
    ``parent``/``attached`` pair of everything that rides out of the chassis
    inside something else) is filled in, so ``propose_edges`` can run without
    the heuristics and :func:`infer_relational_fields` finds nothing left to do.
    """
    recs = [
        inst("chassis", "chassis", slot_id="chassis"),
        inst("motherboard.01", "motherboard", mounted_on="chassis", slot_id="mb"),
        inst("cpu.01", "cpu", mounted_on="motherboard.01", slot_id="cpu"),
        inst("cpu_cooler.fan.01", "cpu_cooler", attrs={"kind": "fan"},
             mounted_on="motherboard.01", slot_id="cooler"),
        # the two board-mounted latch classes (taxonomy `host_class`): they
        # leave the chassis inside the board, exactly like a captive screw
        inst("cpu_socket_lever.01", "cpu_socket_lever", parent="motherboard.01",
             attached=True, slot_id="lever"),
        inst("ram_module.01", "ram_module", mounted_on="motherboard.01", slot_id="ram1"),
        inst("psu.01", "psu", mounted_on="chassis", slot_id="psu"),
        inst("psu_latch.01", "psu_latch", slot_id="psu_latch"),
        inst("storage_drive.hdd.01", "storage_drive", attrs={"kind": "hdd"}, slot_id="hdd"),
        inst("cover.01", "cover", attrs={"of": "motherboard_screws"}, slot_id="mb_cover"),
    ]
    for i in (1, 2, 3):
        recs.append(inst(f"screw.motherboard.{i:02d}", "screw",
                         attrs={"role": "motherboard", "captive": False, "head": "PH2"},
                         fastens="motherboard.01", slot_id=f"mb_screw{i}"))
    for i in (1, 2):
        recs.append(inst(f"screw.cpu_cooler.{i:02d}", "screw",
                         attrs={"role": "cpu_cooler", "captive": True, "head": "PH2"},
                         fastens="cpu_cooler.fan.01", parent="cpu_cooler.fan.01",
                         attached=True, slot_id=f"cooler_screw{i}"))
        recs.append(inst(f"ram_latch.{i:02d}", "ram_latch", attrs={"of": "ram_module.01"},
                         parent="motherboard.01", attached=True,
                         slot_id=f"ram1_latch{i}"))
    # PSU 24-pin: socket on the motherboard, cable owned by the PSU.
    recs.append(inst("connector.atx_24pin.01", "connector", attrs={"kind": "atx_24pin"},
                     socket_host="motherboard.01", cable="cable:psu", slot_id="atx"))
    # SATA data: two independent ends, owned by nobody (spec 7.2).
    recs.append(inst("connector.sata_data.01", "connector", attrs={"kind": "sata_data"},
                     socket_host="storage_drive.hdd.01", slot_id="sata_drive_end"))
    recs.append(inst("connector.sata_data.02", "connector", attrs={"kind": "sata_data"},
                     socket_host="motherboard.01", slot_id="sata_mb_end"))
    return {r.key: r for r in recs}


#: Every edge ``propose_edges`` must derive from :func:`bench_instances`.
EXPECTED = {
    ("fastened_by", "motherboard.01", "screw.motherboard.01"),
    ("fastened_by", "motherboard.01", "screw.motherboard.02"),
    ("fastened_by", "motherboard.01", "screw.motherboard.03"),
    ("fastened_by", "cpu_cooler.fan.01", "screw.cpu_cooler.01"),
    ("fastened_by", "cpu_cooler.fan.01", "screw.cpu_cooler.02"),
    ("connected_to", "motherboard.01", "connector.atx_24pin.01"),
    ("connected_to", "psu.01", "connector.atx_24pin.01"),
    ("connected_to", "storage_drive.hdd.01", "connector.sata_data.01"),
    ("connected_to", "motherboard.01", "connector.sata_data.02"),
    ("locked_by", "ram_module.01", "ram_latch.01"),
    ("locked_by", "ram_module.01", "ram_latch.02"),
    ("locked_by", "cpu.01", "cpu_socket_lever.01"),
    ("locked_by", "psu.01", "psu_latch.01"),
    ("covered_by", "screw.motherboard.01", "cover.01"),
    ("covered_by", "screw.motherboard.02", "cover.01"),
    ("covered_by", "screw.motherboard.03", "cover.01"),
    ("covered_by", "cpu.01", "cpu_cooler.fan.01"),
}

#: Bench classes that nothing can remove: a latch, a lever and the chassis
#: itself are not in ``remove.applies_to`` (spec 6.3).
NOT_REMOVABLE = frozenset({"chassis", "cpu_socket_lever", "psu_latch", "ram_latch"})

#: Every class the bench desktop is built from, as a guard on class sweeps.
BENCH_CLASSES = frozenset({
    "chassis", "motherboard", "cpu", "cpu_cooler", "cpu_socket_lever", "ram_module",
    "ram_latch", "psu", "psu_latch", "storage_drive", "cover", "screw", "connector",
})


def triples(edges: list[Edge]) -> set[tuple[str, str, str]]:
    return {(e.type, e.target, e.blocker) for e in edges}


def act(step: int, target: str, verb: str, result: str = "success", idx: int = 0) -> ActionRec:
    return ActionRec(desktop=DESKTOP, step=step, idx=idx, target=target, verb=verb,
                     result=result)


def state_after(instances, actions, tax, step: int = 10**6):
    """The frame state once ``actions`` have been replayed onto the initial one."""
    return state_at(instances, events_from_actions(instances, actions, tax), step, tax)


def good_sequence() -> list[ActionRec]:
    """A correct teardown of the bench motherboard, in order."""
    return [
        act(1, "cover.01", "open"),
        act(2, "connector.atx_24pin.01", "disconnect"),
        act(3, "connector.sata_data.02", "disconnect"),
        act(4, "screw.motherboard.01", "unscrew"),
        act(5, "screw.motherboard.02", "unscrew"),
        act(6, "screw.motherboard.03", "unscrew"),
        act(7, "motherboard.01", "remove"),
    ]
