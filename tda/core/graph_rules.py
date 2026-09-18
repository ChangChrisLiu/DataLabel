"""The constraint graph's vocabulary, and the rules that derive its edges.

This is the bottom of the graph stack: the :class:`Edge` type, the spec 7.2
semantics, the verb-applicability tables, and the spec 7.3 attribute rules.
:mod:`tda.core.graph` (the reasoning layer), :mod:`tda.core.graph_plan` (the
planner) and :mod:`tda.core.graph_templates` (the family templates) all import
from here, which is what keeps the four modules acyclic;
:mod:`tda.core.graph` re-exports everything a caller needs.

Three functions matter:

* :func:`propose_edges`          -- the spec 7.3 attribute rules, ~90% coverage.
* :func:`infer_relational_fields`-- fill the relational fields the log importer
  leaves empty, so the rules have something to chew on.
* :func:`unresolved_relations`   -- what it deliberately did *not* guess.

Everything is pure except :func:`infer_relational_fields`, which fills blanks on
the :class:`~tda.core.model.InstanceRec` objects it is given (it never
overwrites a value that is already there) and reports what it filled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from tda.core.model import InstanceRec
from tda.core.states import FrameState
from tda.core.taxonomy import Taxonomy

__all__ = [
    "CABLE_DEFAULT_STATE",
    "CABLE_PREFIX",
    "LS_PREFIX",
    "Edge",
    "HARD_TYPES",
    "REQUIRED_STATES",
    "VerbTarget",
    "active_edges",
    "blocker_state",
    "cable_nodes",
    "cable_owner",
    "connector_owner",
    "infer_relational_fields",
    "is_provisional",
    "propose_edges",
    "unresolved_relations",
    "verb_applies",
    "verb_effect",
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
#: Prefix of the provisional keys :mod:`tda.core.ls_import` writes
#: (``ls:Motherboard#1``). Those rows are *drafts*: they carry a real taxonomy
#: class, so a desktop whose sheet named one motherboard can easily hold three
#: more of them once the Label Studio export is in. They therefore take no part
#: in "the one instance of class X" -- neither as the candidate nor as a
#: competitor -- and the heuristics never write onto them (spec 3.2: S1 turns a
#: draft into a real instance, and only then does it carry relations).
LS_PREFIX = "ls:"
REMOVED = "removed"
REJECTED = "rejected"

#: The default state of a ``cable:*`` node, mirroring ``taxonomy.yaml``. A
#: cable enters the frame state only once an event names it (see
#: :func:`tda.core.states.state_at`), so an edge may well point at one that is
#: not in the snapshot yet; it is still routed until something releases it.
CABLE_DEFAULT_STATE = "routed"

#: Which states a verb may be applied *from*, where the taxonomy's effect table
#: alone is too permissive. ``remove`` is handled by :data:`REMOVE_FROM_STATES`.
VERB_FROM_STATES: dict[str, frozenset[str]] = {
    "unscrew": frozenset({"fastened"}),
    "disconnect": frozenset({"plugged"}),
    "open": frozenset({"closed"}),
    "release": frozenset({"closed", "routed"}),
    "displace": frozenset({"installed"}),
}

#: Classes that need an intermediate step before ``remove``: a screw has to come
#: loose first. Every other class can be taken straight out of whatever state it
#: is in -- including a ``connector``, which spec 6.3 lets you remove while it
#: is still plugged, meaning the whole cable was pulled out in one go.
REMOVE_FROM_STATES: dict[str, frozenset[str]] = {
    "screw": frozenset({"loosened"}),
}

#: One action as the graph talks about it: ``(verb, target)``.
VerbTarget = tuple[str, str]


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
# shared helpers: edge status, verb applicability, cable nodes
# --------------------------------------------------------------------------- #
def active_edges(edges: Iterable["Edge"]) -> list["Edge"]:
    """Edges a human has not rejected (spec 7.3: proposed / accepted / rejected)."""
    return [e for e in edges if e.status != REJECTED]


def verb_effect(tax: Taxonomy, cls: str, attrs: dict, verb: str) -> Optional[str]:
    """The state this verb would put the class in, or ``None`` for no effect."""
    effect = tax.apply_verb(cls, attrs, verb)
    if effect is None or effect[0] != "state":
        return None
    return effect[1]


def verb_applies(tax: Taxonomy, cls: str, attrs: dict, verb: str, current: str) -> bool:
    """Can this verb be performed on an instance of ``cls`` in state ``current``?

    Independent of the constraint graph: it only asks whether the verb belongs
    to the class (spec 6.3), does something, and starts from a sane state.
    """
    new = verb_effect(tax, cls, attrs, verb)
    if new is None or current == REMOVED or new == current:
        return False
    allowed = REMOVE_FROM_STATES.get(cls) if verb == "remove" else VERB_FROM_STATES.get(verb)
    return allowed is None or current in allowed


def blocker_state(state: FrameState, blocker: str) -> Optional[str]:
    """A node's state, or ``None`` when it is not a node of this desktop."""
    inst = state.get(blocker)
    if inst is not None:
        return inst.state
    if blocker.startswith(CABLE_PREFIX):
        return CABLE_DEFAULT_STATE
    return None


def cable_nodes(edges: list["Edge"], state: FrameState) -> dict[str, str]:
    """Every virtual cable node this graph mentions, with its current state."""
    keys = {e.blocker for e in edges if e.blocker.startswith(CABLE_PREFIX)}
    keys |= {k for k in state if k.startswith(CABLE_PREFIX)}
    return {k: blocker_state(state, k) or CABLE_DEFAULT_STATE for k in keys}


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
#: them. A drive's SATA leads unplug at both ends, so spec 7.2 says each plug
#: gates only the part it sits on -- unlike a fan lead or a front-panel loom,
#: which are moulded into their part and gate it wherever they plug in.
DETACHABLE_CABLE_OWNERS = frozenset({"storage_drive", "optical_drive"})

#: Connector kinds that are power leads: whatever ``taxonomy_map`` tagged them
#: with, the far end of one is moulded into the PSU, so the PSU harness owns it
#: (spec 7.2: every plug of the harness must be out before the PSU can go).
POWER_LEAD_KINDS = frozenset({"atx_24pin", "cpu_power", "sata_power", "molex"})

#: Connector kinds with a detachable plug at *both* ends, owned by nobody.
TWO_ENDED_KINDS = frozenset({"sata_data"})


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


def connector_owner(
    rec: InstanceRec,
    instances: dict[str, InstanceRec],
) -> Optional[str]:
    """The part one connector's cable is captive to, or ``None``.

    The connector's ``kind`` decides first, because it is the physical fact and
    ``logs.py``'s ``cable_owner`` is only a grouping key:

    * a power lead (:data:`POWER_LEAD_KINDS`) belongs to the PSU harness, even
      when the sheet filed it under the drive it feeds -- so the PSU cannot come
      out until every one of its plugs is pulled;
    * a ``sata_data`` lead has a detachable plug at both ends and belongs to
      nobody, so each end gates only its own socket host (spec 7.2);
    * anything else falls back to the tagged cable owner, which still drops the
      drive classes of :data:`DETACHABLE_CABLE_OWNERS` -- an untyped connector
      filed under a drive is one of its SATA leads.
    """
    kind = str(rec.attrs.get("kind") or "")
    if kind in POWER_LEAD_KINDS:
        return _unique_of_class(instances, "psu")
    if kind in TWO_ENDED_KINDS:
        return None
    return cable_owner(rec.cable or "", instances)


def is_provisional(key: str) -> bool:
    """Is this a Label Studio draft key rather than a settled instance?

    See :data:`LS_PREFIX`. Drafts are invisible to every class-uniqueness
    question here, so ``motherboard.01`` stays "the motherboard" of a desktop
    whose export also drew ``ls:Motherboard#1``.
    """
    return key.startswith(LS_PREFIX)


def _real_instances(instances: dict[str, InstanceRec], cls: str) -> list[str]:
    """Keys of the settled (non-draft) instances of ``cls``, sorted."""
    return [
        key for key, rec in sorted(instances.items())
        if rec.cls == cls and not is_provisional(key)
    ]


def _unique_of_class(instances: dict[str, InstanceRec], cls: str) -> Optional[str]:
    """The key of the one instance of ``cls``, or ``None`` when it is not unique."""
    found = _real_instances(instances, cls)
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
    plug of the PSU harness gates the PSU as well as its own socket host. See
    :func:`connector_owner` for how the owner is decided.
    """
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "connector":
            continue
        host = _resolve(instances, rec.socket_host)
        if host:
            out.append(_edge("connected_to", host, key, f"{key} plugs into {host}"))
        owner = connector_owner(rec, instances)
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
    no-op and a human correction survives a re-import. Provisional ``ls:*``
    drafts are left alone entirely and never count towards "the one instance of
    class X" (:data:`LS_PREFIX`), and an ambiguous reference is *reported* by
    :func:`unresolved_relations` rather than guessed at. Returns one
    ``"<key>.<field> = <value>"`` line per field it filled; every one of them is
    a guess with source ``"heuristic"``, to be confirmed in the S1 UI.
    """
    filled: list[str] = []

    def put(rec: InstanceRec, field: str, value: object) -> None:
        setattr(rec, field, value)
        filled.append(f"{rec.key}.{field} = {value}")

    for key, rec in sorted(instances.items()):
        if is_provisional(key):
            continue
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
        rec for key, rec in sorted(instances.items())
        if rec.cls == "ram_latch" and not rec.attrs.get("of") and not is_provisional(key)
    ]
    modules = _real_instances(instances, "ram_module")
    if not latches or not modules:
        return
    per = max(1, len(latches) // len(modules))
    for i, rec in enumerate(latches):
        module = modules[min(i // per, len(modules) - 1)]
        rec.attrs["of"] = module
        filled.append(f"{rec.key}.attrs.of = {module}")


# --------------------------------------------------------------------------- #
# unresolved_relations
# --------------------------------------------------------------------------- #
#: The four relational columns of spec 7.1 a class name may legitimately sit in
#: until stage S1 narrows it down (mirrors ``steps_values.RELATION_FIELDS``).
RELATION_FIELDS = ("parent", "mounted_on", "fastens", "socket_host")

#: Prefix of every line this reports, so a caller can print them as they are.
UNRESOLVED = "unresolved:"


def unresolved_relations(
    instances: dict[str, InstanceRec],
    tax: Taxonomy,
) -> list[str]:
    """What :func:`infer_relational_fields` refused to guess, one line each.

    Three questions are left to the annotator rather than answered by a coin
    toss, and every one of them is reported here so it does not disappear:

    * a relational field holding a taxonomy *class* the desktop has zero or two
      or more real instances of -- the class name the importer wrote is all the
      sheet said, and picking one of two motherboards is not a heuristic's job;
    * a screw whose ``role`` names no unique part, so ``fastens`` stays empty;
    * a *captive* screw that consequently has no ``parent``, which is what
      makes it stay behind when its part leaves the chassis (spec 3.3).

    Lines start with :data:`UNRESOLVED` and are sorted by instance key. A
    reference to something that is neither an instance nor a class is *not*
    reported here -- that is a dangling pointer, which the S1 step table
    already asks about (:mod:`tda.ui.steps_issues`). Provisional ``ls:*``
    drafts are skipped: they carry no relations yet by construction.
    """
    out: list[str] = []
    for key, rec in sorted(instances.items()):
        if is_provisional(key):
            continue
        for name in RELATION_FIELDS:
            value = getattr(rec, name)
            if value and value in tax.classes and _resolve(instances, value) is None:
                n = len(_real_instances(instances, value))
                out.append(
                    f"{UNRESOLVED} {key}.{name} = {value!r} names a class the desktop "
                    f"has {n} instances of - pick one in S1"
                )
        if rec.cls != "screw":
            continue
        if not rec.fastens:
            role = str(rec.attrs.get("role") or "")
            out.append(
                f"{UNRESOLVED} {key}.fastens is empty - role {role or 'unset'!r} names "
                f"no unique part of this desktop"
            )
        if rec.attrs.get("captive") and not rec.parent:
            out.append(
                f"{UNRESOLVED} {key} is captive but has no parent - it will not leave "
                f"the chassis with the part it is screwed into"
            )
    return out
