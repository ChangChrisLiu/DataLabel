"""Hard-constraint edges and the attribute rules that derive them (spec 7.1-7.3).

The :class:`Edge` type and the semantic tables live here so that
:mod:`tda.core.graph` (the reasoning layer) and :mod:`tda.core.graph_templates`
(the family templates) can both import them without a cycle;
:mod:`tda.core.graph` re-exports everything a caller needs.

Two functions matter:

* :func:`propose_edges`          -- the spec 7.3 attribute rules, ~90% coverage.
* :func:`infer_relational_fields`-- fill the relational fields the log importer
  leaves empty, so the rules have something to chew on.

Everything is pure except :func:`infer_relational_fields`, which fills blanks on
the :class:`~tda.core.model.InstanceRec` objects it is given (it never
overwrites a value that is already there) and reports what it filled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tda.core.model import InstanceRec
from tda.core.taxonomy import Taxonomy

__all__ = [
    "CABLE_PREFIX",
    "Edge",
    "HARD_TYPES",
    "REQUIRED_STATES",
    "cable_owner",
    "infer_relational_fields",
    "propose_edges",
]

#: The five hard-constraint edge types of spec 7.2, most-specific first.
HARD_TYPES = ("fastened_by", "connected_to", "locked_by", "covered_by", "blocked_by")

#: Spec 7.2: the blocker must be in one of these states before the target of an
#: edge of this type may be acted on.
REQUIRED_STATES: dict[str, frozenset[str]] = {
    "fastened_by": frozenset({"loosened", "removed"}),
    "connected_to": frozenset({"unplugged", "removed"}),
    "locked_by": frozenset({"open"}),
    "covered_by": frozenset({"open", "removed", "displaced"}),
    "blocked_by": frozenset({"removed", "displaced", "released"}),
}

#: ``blocked_by`` is the only type that carries a mode (spec 7.1).
BLOCKED_MODES = ("physical_path", "tool_access", "cable_tension")

CABLE_PREFIX = "cable:"
REMOVED = "removed"


@dataclass(frozen=True)
class Edge:
    """One hard constraint: ``blocker`` gates every action on ``target``.

    ``necessity`` is ``required`` or ``recommended``; ``mode`` is only set on
    ``blocked_by``; ``source`` is ``rule`` / ``template`` / ``manual``;
    ``status`` is ``proposed`` / ``accepted`` / ``rejected`` (spec 7.3).
    """

    type: str
    target: str
    blocker: str
    necessity: str = "required"
    mode: Optional[str] = None
    reason: str = ""
    source: str = "rule"
    evidence_step: Optional[int] = None
    status: str = "proposed"

    def label(self) -> str:
        """``fastened_by(motherboard.01, screw.motherboard.03)``."""
        return f"{self.type}({self.target}, {self.blocker})"


# --------------------------------------------------------------------------- #
# cables
# --------------------------------------------------------------------------- #
#: ``cable:<owner>`` tokens the logs use that are not taxonomy class names.
CABLE_OWNER_ALIASES = {
    "cpu_fan": "cpu_cooler",
    "cooler_fan": "cpu_cooler",
    "hdd": "storage_drive",
    "ssd": "storage_drive",
    "drive": "storage_drive",
    "optical": "optical_drive",
}

#: Classes whose cables are *not* captive to them, however ``logs.py`` groups
#: them. A drive's SATA data and power leads unplug at both ends, so spec 7.2
#: says each plug gates only the part it sits on -- unlike a PSU harness, a fan
#: lead or a front-panel loom, which are moulded into their part and therefore
#: gate it wherever they plug in.
DETACHABLE_CABLE_OWNERS = frozenset({"storage_drive", "optical_drive"})


def cable_owner(
    cable_id: str,
    instances: Optional[dict[str, InstanceRec]] = None,
) -> Optional[str]:
    """The part a virtual cable node belongs to (spec 3.1 ``cable.owner``).

    Without ``instances`` this answers the owner *class* (``"cable:psu"`` ->
    ``"psu"``); with them it resolves that class to its instance key, but only
    when the desktop has exactly one instance of it -- an ambiguous or absent
    owner yields ``None`` rather than a guess. A key that is not a cable node
    yields ``None`` either way.

    Note this is the *cable's* owner, in the spec-7.1 sense of "the cable is
    permanently attached to this part", which is narrower than the grouping key
    ``logs.py`` writes: see :data:`DETACHABLE_CABLE_OWNERS`.
    """
    if not cable_id.startswith(CABLE_PREFIX):
        return None
    owner = cable_id[len(CABLE_PREFIX):].strip()
    if not owner:
        return None
    owner = CABLE_OWNER_ALIASES.get(owner, owner)
    if owner in DETACHABLE_CABLE_OWNERS:
        return None
    if instances is None:
        return owner
    return _unique_of_class(instances, owner)


def _unique_of_class(instances: dict[str, InstanceRec], cls: str) -> Optional[str]:
    """The key of the one instance of ``cls``, or ``None`` when it is not unique."""
    found = [key for key, rec in sorted(instances.items()) if rec.cls == cls]
    return found[0] if len(found) == 1 else None


def _resolve(instances: dict[str, InstanceRec], ref: Optional[str]) -> Optional[str]:
    """An instance key from a field that may hold a key or a bare class name.

    ``logs._resolve_socket_hosts`` writes the literal ``"motherboard"`` when the
    raw step name says so, which is a class, not a key; the same happens for a
    hand-edited ``fastens`` or ``of``. A class resolves only when it is unique.
    """
    if not ref:
        return None
    if ref in instances:
        return ref
    return _unique_of_class(instances, ref)


# --------------------------------------------------------------------------- #
# propose_edges (spec 7.3 item 1)
# --------------------------------------------------------------------------- #
#: Latch classes whose ``of`` attribute names what they lock.
LATCH_OF_CLASSES = ("ram_latch", "drive_latch", "card_latch", "cooler_latch")

#: Latch classes that lock a fixed part class (the spec-7.3 template edges).
LATCH_TEMPLATE_HOSTS = {"cpu_socket_lever": "cpu", "psu_latch": "psu"}

#: ``cover.of`` -> what the cover hides. ``("cls", ...)`` matches instance
#: classes, ``("role", ...)`` matches ``screw.role``. ``front_bezel`` and
#: ``other`` name no part, so they are left to the manual panel.
COVER_TARGETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cpu_cooler": ("cls", ("cpu_cooler",)),
    "motherboard_screws": ("role", ("motherboard",)),
    "ram": ("cls", ("ram_module",)),
    "expansion_slot": ("cls", ("expansion_card",)),
    "drive": ("cls", ("storage_drive", "optical_drive")),
}

#: Fixed template edges of spec 7.3: the cooler always sits on the CPU.
FIXED_COVERED_BY = (("cpu", "cpu_cooler"),)


def propose_edges(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """Derive every hard edge the recorded attributes imply (spec 7.3 item 1).

    Rules, in order: ``screw.fastens`` -> ``fastened_by``; the connector rule of
    spec 7.2 -> ``connected_to``; ``latch.of`` plus the fixed CPU-lever and
    PSU-latch edges -> ``locked_by``; ``cover.of`` plus the fixed cooler-over-CPU
    edge -> ``covered_by``. ``cable_clip.holds`` yields no edge on its own -- a
    clipped cable only blocks a target once a human says which one, so those
    ``blocked_by`` edges stay manual (spec 7.3 item 3).

    Every edge comes back ``necessity="required"``, ``source="rule"``,
    ``status="proposed"``; duplicates are collapsed, keeping the first reason.
    Instances whose relational fields are still empty simply derive nothing --
    run :func:`infer_relational_fields` first to fill the obvious ones.
    """
    out: list[Edge] = []
    for rule in (_fastened_by, _connected_to, _locked_by, _covered_by):
        out.extend(rule(instances, tax))
    return _dedupe(out)


def _dedupe(edges: list[Edge]) -> list[Edge]:
    """Drop repeated ``(type, target, blocker)`` triples, keeping the first."""
    seen: set[tuple[str, str, str]] = set()
    out: list[Edge] = []
    for edge in edges:
        key = (edge.type, edge.target, edge.blocker)
        if key in seen or edge.target == edge.blocker:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _edge(etype: str, target: str, blocker: str, reason: str) -> Edge:
    return Edge(type=etype, target=target, blocker=blocker, reason=reason, source="rule")


def _fastened_by(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """``screw.fastens = X`` -> ``fastened_by(X, screw)`` (spec 7.1)."""
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "screw":
            continue
        target = _resolve(instances, rec.fastens)
        if target:
            out.append(_edge("fastened_by", target, key, f"{key} fastens {target}"))
    return out


def _connected_to(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """The connector rule of spec 7.2.

    For a part X, every connector whose ``socket_host`` is X *or* whose cable is
    owned by X must be unplugged before X may be removed. The two ends of a SATA
    data cable carry no owner, so each gates only the part it plugs into; every
    plug of the PSU harness gates the PSU as well as its own socket host.
    """
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "connector":
            continue
        host = _resolve(instances, rec.socket_host)
        if host:
            out.append(_edge("connected_to", host, key, f"{key} plugs into {host}"))
        owner = cable_owner(rec.cable or "", instances)
        if owner and owner != host:
            out.append(
                _edge("connected_to", owner, key, f"{key} is on the {owner} cable")
            )
    return out


def _locked_by(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """``latch.of`` -> ``locked_by``, plus the fixed CPU-lever / PSU-latch edges."""
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        target = None
        if rec.cls in LATCH_OF_CLASSES:
            target = _resolve(instances, rec.attrs.get("of"))
        elif rec.cls in LATCH_TEMPLATE_HOSTS:
            target = _unique_of_class(instances, LATCH_TEMPLATE_HOSTS[rec.cls])
        if target:
            out.append(_edge("locked_by", target, key, f"{key} locks {target}"))
    return out


def _covered_by(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """``cover.of`` -> ``covered_by``, plus the fixed cooler-over-CPU edge."""
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "cover":
            continue
        for target in _cover_targets(instances, rec):
            out.append(_edge("covered_by", target, key, f"{key} covers {target}"))
    for under_cls, over_cls in FIXED_COVERED_BY:
        under = _unique_of_class(instances, under_cls)
        over = _unique_of_class(instances, over_cls)
        if under and over:
            out.append(_edge("covered_by", under, over, f"{over} sits on {under}"))
    return out


def _cover_targets(instances: dict[str, InstanceRec], cover: InstanceRec) -> list[str]:
    """Which instances this cover hides, from its ``of`` attribute."""
    of = str(cover.attrs.get("of") or "")
    if not of:
        return []
    if of in instances:  # `of` already names one instance
        return [of]
    rule = COVER_TARGETS.get(of)
    if rule is None:
        return []
    kind, wanted = rule

    def hit(rec: InstanceRec) -> bool:
        if kind == "cls":
            return rec.cls in wanted
        return rec.cls == "screw" and rec.attrs.get("role") in wanted

    return [key for key, rec in sorted(instances.items()) if hit(rec)]


# --------------------------------------------------------------------------- #
# infer_relational_fields
# --------------------------------------------------------------------------- #
#: ``screw.role`` -> the classes the screw may fasten, best candidate first.
SCREW_ROLE_CLASSES: dict[str, tuple[str, ...]] = {
    "motherboard": ("motherboard",),
    "cpu_cooler": ("cpu_cooler",),
    "cooler_bracket": ("cooler_bracket",),
    "drive": ("storage_drive", "drive_cage"),
    "optical_drive": ("optical_drive",),
    "card": ("expansion_card",),
    "psu": ("psu",),
}


def infer_relational_fields(
    instances: dict[str, InstanceRec],
    tax: Taxonomy,
) -> list[str]:
    """Fill the relational fields the log importer leaves empty (spec 2.3).

    ``logs.py`` never sets ``fastens``, ``parent``/``attached`` or a latch's
    ``of``, and it may write a bare class name into ``socket_host``. The rules
    of spec 7.3 need those, so this fills the obvious defaults:

    * ``screw.fastens`` from ``screw.role``, whenever the desktop has exactly
      one instance of the class that role names;
    * a captive screw's ``parent`` = what it fastens, with ``attached=True``
      (spec 7.1: a captive cooler screw leaves with the cooler);
    * ``ram_latch.of`` by nearest ordinal -- the latches are split evenly over
      the modules, so two latches per module pair up with module 1, 2, ...;
    * ``socket_host`` given as a class name -> that class's unique instance.

    Nothing already filled in is ever overwritten, so running this twice is a
    no-op and a human correction survives a re-import. Returns one
    ``"<key>.<field> = <value>"`` line per field it filled; every one of them is
    a guess with source ``"heuristic"``, to be confirmed in the S1 UI.
    """
    filled: list[str] = []

    def put(rec: InstanceRec, field: str, value: object) -> None:
        setattr(rec, field, value)
        filled.append(f"{rec.key}.{field} = {value}")

    for key, rec in sorted(instances.items()):
        if rec.cls == "screw":
            _infer_screw(instances, rec, put, filled)
        elif rec.cls == "connector":
            host = _resolve(instances, rec.socket_host)
            if host and host != rec.socket_host:
                put(rec, "socket_host", host)
    _infer_ram_latches(instances, filled)
    return filled


def _infer_screw(instances, rec: InstanceRec, put, filled: list[str]) -> None:
    """``fastens`` from the role, then ``parent``/``attached`` when captive."""
    if rec.fastens is None:
        for cls in SCREW_ROLE_CLASSES.get(str(rec.attrs.get("role") or ""), ()):
            target = _unique_of_class(instances, cls)
            if target:
                put(rec, "fastens", target)
                break
    target = _resolve(instances, rec.fastens)
    if target and rec.attrs.get("captive"):
        if rec.parent is None:
            put(rec, "parent", target)
        if not rec.attached:
            rec.attached = True
            filled.append(f"{rec.key}.attached = True")


def _infer_ram_latches(instances: dict[str, InstanceRec], filled: list[str]) -> None:
    """Pair RAM latches with modules by ordinal: N latches spread over M modules."""
    latches = [
        rec for _, rec in sorted(instances.items())
        if rec.cls == "ram_latch" and not rec.attrs.get("of")
    ]
    modules = [key for key, rec in sorted(instances.items()) if rec.cls == "ram_module"]
    if not latches or not modules:
        return
    per = max(1, len(latches) // len(modules))
    for i, rec in enumerate(latches):
        module = modules[min(i // per, len(modules) - 1)]
        rec.attrs["of"] = module
        filled.append(f"{rec.key}.attrs.of = {module}")
