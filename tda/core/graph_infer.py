"""Instance lookup and the relational-field heuristics (spec 2.3 / 7.1, 7.3).

The bottom of the graph stack, below :mod:`tda.core.graph_rules`: first "which
instance does this name mean", then the two functions that use it to fill in
what the log importer could not.

* :func:`unique_of_class` / :func:`resolve_ref` -- a class name or a key ->
  one instance key, or ``None`` when the desktop has no unique answer. Label
  Studio drafts (:data:`LS_PREFIX`) never count.
* :func:`infer_relational_fields` -- fills the empty relational fields
  (``screw.fastens``, a captive screw's ``parent``/``attached``, a RAM latch's
  ``of``, a class-level ``socket_host``). Only ever writes into a blank, so it
  is idempotent and a human correction survives a re-import.
* :func:`unresolved_relations` -- what it deliberately did **not** guess, split
  into :data:`NO_CANDIDATE` ("this desktop has no such part at all") and
  :data:`AMBIGUOUS` ("it has several and nothing says which").

Two heuristics decide which part a screw fastens. The desktop usually has
exactly one instance of the role's class and that is the end of it. When it has
several -- the eight machines whose sheet lists a ``cpu_cooler.fan`` *and* a
``cpu_cooler.heatsink`` -- the tie is broken by **physical necessity in time**:
you take a screw out in order to take something out, so a screw fastens the
candidate that is removed *soonest after* the screw's own last action. The raw
sheet name is used only as a veto: when it names a different candidate than the
clock does, the two heuristics disagree and the field is left for the human.
"""
from __future__ import annotations

from typing import Iterable, Optional

from tda.core.model import ActionRec, InstanceRec
from tda.core.taxonomy import Taxonomy

__all__ = [
    "AMBIGUOUS",
    "LS_PREFIX",
    "NO_CANDIDATE",
    "RELATION_FIELDS",
    "SCREW_ROLE_CLASSES",
    "UNRESOLVED",
    "infer_relational_fields",
    "is_provisional",
    "real_instances",
    "resolve_ref",
    "unique_of_class",
    "unresolved_kind",
    "unresolved_relations",
]

#: Prefix of the provisional keys :mod:`tda.core.ls_import` writes
#: (``ls:Motherboard#1``). Those rows are *drafts*: they carry a real taxonomy
#: class, so a desktop whose sheet named one motherboard can easily hold three
#: more of them once the Label Studio export is in. They therefore take no part
#: in "the one instance of class X" -- neither as the candidate nor as a
#: competitor -- and the heuristics never write onto them (spec 3.2: S1 turns a
#: draft into a real instance, and only then does it carry relations).
LS_PREFIX = "ls:"


def is_provisional(key: str) -> bool:
    """Is this a Label Studio draft key rather than a settled instance?

    See :data:`LS_PREFIX`. Drafts are invisible to every class-uniqueness
    question here, so ``motherboard.01`` stays "the motherboard" of a desktop
    whose export also drew ``ls:Motherboard#1``.
    """
    return key.startswith(LS_PREFIX)


def real_instances(instances: dict[str, InstanceRec], cls: str) -> list[str]:
    """Keys of the settled (non-draft) instances of ``cls``, sorted."""
    return [
        key for key, rec in sorted(instances.items())
        if rec.cls == cls and not is_provisional(key)
    ]


def unique_of_class(instances: dict[str, InstanceRec], cls: str) -> Optional[str]:
    """The key of the one instance of ``cls``, or ``None`` when it is not unique."""
    found = real_instances(instances, cls)
    return found[0] if len(found) == 1 else None


def blank(value: object) -> bool:
    """Is this relational field empty? ``None``, ``""`` and whitespace all are.

    Clearing a cell in the S1 table leaves an empty string behind, and a
    hand-edited sheet can leave a stray space. Both mean "not answered yet",
    so they must reach the heuristics as a blank rather than as a value that
    resolves to nothing. Takes ``object`` rather than ``str`` because it is also
    asked about ``attrs`` entries, which come back out of JSON as whatever was
    put in.
    """
    return not str(value or "").strip()


def resolve_ref(instances: dict[str, InstanceRec], ref: Optional[str]) -> Optional[str]:
    """An instance key from a field that may hold a key or a bare class name.

    ``logs._resolve_socket_hosts`` writes the literal ``"motherboard"`` when the
    raw step name says so, which is a class, not a key; the same happens for a
    hand-edited ``fastens`` or ``of``. A class resolves only when it is unique.
    """
    if blank(ref):
        return None
    ref = str(ref).strip()
    if ref in instances:
        return ref
    return unique_of_class(instances, ref)


# --------------------------------------------------------------------------- #
# what a screw fastens
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

#: Words a raw sheet name may use to single one candidate out, and the
#: spellings that count as that word. Checked longest-first so that
#: ``"heat sink"`` is not read as two unrelated tokens.
NAME_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("heatsink", ("heatsink", "heat sink", "heat-sink")),
    ("fan", ("fan",)),
)

#: Verbs that count as "the operator worked on this screw".
SCREW_VERBS = frozenset({"unscrew", "remove"})
_SUCCESS = "success"


class _Clock:
    """When each instance was removed, and when each screw was last touched.

    Built once per desktop from the recorded actions (spec 6.3). ``removed_at``
    only counts a *successful* ``remove`` -- a failed attempt took nothing out
    -- while ``worked_at`` counts every ``unscrew``/``remove`` aimed at a screw,
    successful or not: the question it answers is when the operator was at that
    screw, not what came of it. A *later failed retry* therefore pushes
    ``worked_at`` forward, which can only shrink the set of candidates removed
    at or after it: the heuristic then gives up rather than guessing, which is
    the direction to fail in.
    """

    def __init__(self, actions: Iterable[ActionRec]):
        self.removed_at: dict[str, int] = {}
        self.worked_at: dict[str, int] = {}
        for action in actions:
            if action.verb == "remove" and action.result == _SUCCESS:
                first = self.removed_at.get(action.target)
                if first is None or action.step < first:
                    self.removed_at[action.target] = action.step
            if action.verb in SCREW_VERBS:
                self.worked_at[action.target] = max(
                    self.worked_at.get(action.target, 0), action.step
                )


def _named_candidate(rec: InstanceRec, candidates: list[str]) -> Optional[str]:
    """The candidate the screw's raw sheet name points at, if it points at one.

    ``"Heatsink screw 1"`` names the heatsink, ``"CPU fan screw 1"`` the fan. A
    name carrying both words (or neither), or one whose word fits several
    candidates, singles nobody out and yields ``None``.
    """
    text = " ".join(rec.raw_names).lower()
    seen = [word for word, spellings in NAME_WORDS if any(s in text for s in spellings)]
    if len(seen) != 1:
        return None
    hits = [key for key in candidates if seen[0] in key.lower()]
    return hits[0] if len(hits) == 1 else None


def _by_the_clock(
    rec: InstanceRec, candidates: list[str], clock: _Clock
) -> tuple[Optional[str], str]:
    """Physical necessity: the candidate removed soonest after this screw.

    You undo a fastener in order to take something out, so of the parts that
    could carry this screw the right one is the first to leave *at or after* the
    last time anybody touched the screw. The "at" matters: a compound row that
    reads "unscrew the last screw and lift the fan off" puts both on one step,
    and a strict ``>`` would skip the fan and write the *next* part in -- the
    one wrong answer that would never look wrong. Two candidates leaving on that
    same step, or none leaving at all, is not a tie-break and stays unresolved.
    """
    since = clock.worked_at.get(rec.key)
    if since is None:
        return None, "no action ever names this screw, so there is nothing to time it by"
    after = sorted(
        (step, key) for step, key in
        ((clock.removed_at.get(key), key) for key in candidates)
        if step is not None and step >= since
    )
    if not after:
        return None, (
            f"none of {', '.join(candidates)} is removed at or after the screw's own "
            f"last action at step {since}"
        )
    if len(after) > 1 and after[0][0] == after[1][0]:
        return None, (
            f"{after[0][1]} and {after[1][1]} are both removed at step {after[0][0]}"
        )
    step, pick = after[0]
    named = _named_candidate(rec, candidates)
    if named is not None and named != pick:
        return None, (
            f"the sheet name {' / '.join(rec.raw_names)!r} says {named} but {pick} is "
            f"the first removed (step {step}) after the screw's last action at step "
            f"{since} - the two heuristics disagree"
        )
    return pick, ""


def screw_target(
    instances: dict[str, InstanceRec],
    rec: InstanceRec,
    clock: Optional[_Clock] = None,
) -> tuple[Optional[str], str]:
    """The part this screw fastens, as ``(key, why-not)``.

    The role's classes are tried in order and a class with exactly one real
    instance wins outright. Only when none of them is unique does the first
    class with *several* go to :func:`_by_the_clock`, and only when a ``clock``
    was built at all -- without the desktop's actions the heuristic has nothing
    to reason with and the field simply stays empty.
    """
    role = str(rec.attrs.get("role") or "")
    ambiguous: list[str] = []
    for cls in SCREW_ROLE_CLASSES.get(role, ()):
        found = real_instances(instances, cls)
        if len(found) == 1:
            return found[0], ""
        if len(found) > 1 and not ambiguous:
            ambiguous = found
    if not ambiguous:
        return None, f"role {role or 'unset'!r} names no part of this desktop"
    if clock is None:
        return None, f"role {role!r} names {len(ambiguous)} parts and nothing ranks them"
    return _by_the_clock(rec, ambiguous, clock)


# --------------------------------------------------------------------------- #
# infer_relational_fields
# --------------------------------------------------------------------------- #
def infer_relational_fields(
    instances: dict[str, InstanceRec],
    tax: Taxonomy,
    actions: Optional[Iterable[ActionRec]] = None,
) -> list[str]:
    """Fill the relational fields the log importer leaves empty (spec 2.3).

    ``logs.py`` never sets ``fastens``, ``parent``/``attached`` or a latch's
    ``of``, and it may write a bare class name into ``socket_host``. The rules
    of spec 7.3 need those, so this fills the obvious defaults:

    * ``screw.fastens`` from ``screw.role`` -- see :func:`screw_target`, which
      takes the class's one instance, or, given ``actions``, the candidate the
      step order says the screw was undone for;
    * a captive screw's ``parent`` = what it fastens, with ``attached=True``
      (spec 7.1: a captive cooler screw leaves with the cooler);
    * ``ram_latch.of`` by nearest ordinal -- the latches are split evenly over
      the modules, so two latches per module pair up with module 1, 2, ...;
    * ``socket_host`` given as a class name -> that class's unique instance.

    ``actions`` is the desktop's recorded action list; leaving it out only
    costs the temporal tie-break, nothing else. Nothing already filled in is
    ever overwritten, so running this twice is a no-op and a human correction
    survives a re-import. Provisional ``ls:*`` drafts are left alone entirely
    and never count towards "the one instance of class X"
    (:data:`LS_PREFIX`), and an ambiguous reference is *reported* by
    :func:`unresolved_relations` rather than guessed at. Returns one
    ``"<key>.<field> = <value>"`` line per field it filled; every one of them is
    a guess with source ``"heuristic"``, to be confirmed in the S1 UI.
    """
    filled: list[str] = []
    clock = None if actions is None else _Clock(actions)

    def put(rec: InstanceRec, field: str, value: object) -> None:
        setattr(rec, field, value)
        filled.append(f"{rec.key}.{field} = {value}")

    for key, rec in sorted(instances.items()):
        if is_provisional(key):
            continue
        if rec.cls == "screw":
            _infer_screw(instances, rec, clock, put, filled)
        elif rec.cls == "connector":
            host = resolve_ref(instances, rec.socket_host)
            if host and host != rec.socket_host:
                put(rec, "socket_host", host)
    _infer_ram_latches(instances, filled)
    return filled


def _infer_screw(instances, rec: InstanceRec, clock, put, filled: list[str]) -> None:
    """``fastens`` from the role, then ``parent``/``attached`` when captive.

    ``attached`` is set **only** in the same pass that fills ``parent``. It is a
    bool, so a stored ``False`` cannot say whether it is the dataclass default
    or an annotator who deliberately unticked it in S1; an instance that already
    carries a ``parent`` has been looked at, and re-ticking the flag under the
    annotator on the next run would be exactly the overwrite this heuristic
    promises never to do.
    """
    if blank(rec.fastens):
        target, _why = screw_target(instances, rec, clock)
        if target:
            put(rec, "fastens", target)
    target = resolve_ref(instances, rec.fastens)
    if target and rec.attrs.get("captive") and blank(rec.parent):
        put(rec, "parent", target)
        if not rec.attached:
            rec.attached = True
            filled.append(f"{rec.key}.attached = True")


def _infer_ram_latches(instances: dict[str, InstanceRec], filled: list[str]) -> None:
    """Pair RAM latches with modules by ordinal: N latches spread over M modules.

    ``attrs["of"]`` is read through :func:`blank`, exactly like the relational
    *columns*: clearing that cell in the S1 table leaves an empty string, and a
    hand-edited sheet can leave a stray space. Both mean "not answered yet", and
    a latch pointing at ``" "`` is a latch rule 7.1 can never fire on.
    """
    latches = [
        rec for key, rec in sorted(instances.items())
        if rec.cls == "ram_latch" and blank(rec.attrs.get("of")) and not is_provisional(key)
    ]
    modules = real_instances(instances, "ram_module")
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
UNRESOLVED = "unresolved"
#: The desktop has no instance of the class at all -- S1 has to create one (or
#: decide the part is simply not tracked, user decision C7).
NO_CANDIDATE = "no candidate"
#: The desktop has several and nothing here ranks them.
AMBIGUOUS = "ambiguous"


def _line(kind: str, text: str) -> str:
    return f"{UNRESOLVED} ({kind}): {text}"


def unresolved_kind(line: str) -> str:
    """``no candidate`` or ``ambiguous`` for one line of :func:`unresolved_relations`."""
    head, _, _rest = line.partition(":")
    return head[len(UNRESOLVED):].strip(" ()") or AMBIGUOUS


def unresolved_relations(
    instances: dict[str, InstanceRec],
    tax: Taxonomy,
    actions: Optional[Iterable[ActionRec]] = None,
) -> list[str]:
    """What :func:`infer_relational_fields` refused to guess, one line each.

    Three questions are left to the annotator rather than answered by a coin
    toss, and every one of them is reported here so it does not disappear:

    * a relational field holding a taxonomy *class* the desktop has zero or two
      or more real instances of;
    * a screw whose role names no part, several parts the clock cannot rank, or
      one the clock and the sheet name disagree about (:func:`screw_target`
      supplies the exact wording);
    * a *captive* screw that consequently has no ``parent``, which is what
      makes it leave the chassis with its part (spec 3.3).

    Every line starts with :data:`UNRESOLVED` and carries its kind in brackets
    -- :data:`NO_CANDIDATE` or :data:`AMBIGUOUS` -- because the two want
    different work: one needs an instance created, the other needs one picked.
    :func:`unresolved_kind` reads it back. A reference to something that is
    neither an instance nor a class is *not* reported here -- that is a dangling
    pointer, which the S1 step table already asks about
    (:mod:`tda.ui.steps_issues`). Provisional ``ls:*`` drafts are skipped: they
    carry no relations yet by construction.
    """
    out: list[str] = []
    clock = None if actions is None else _Clock(actions)
    for key, rec in sorted(instances.items()):
        if is_provisional(key):
            continue
        for name in RELATION_FIELDS:
            value = (getattr(rec, name) or "").strip()
            if value and value in tax.classes and resolve_ref(instances, value) is None:
                out.append(_class_line(instances, key, name, value))
        if rec.cls != "screw":
            continue
        kind = AMBIGUOUS
        if blank(rec.fastens):
            target, why = screw_target(instances, rec, clock)
            kind = NO_CANDIDATE if target is None and "names no part" in why else AMBIGUOUS
            out.append(_line(kind, f"{key}.fastens is empty - {why}"))
        if rec.attrs.get("captive") and blank(rec.parent):
            out.append(_line(kind, (
                f"{key} is captive but has no parent - it will not leave the chassis "
                f"with the part it is screwed into"
            )))
    return out


def _class_line(instances, key: str, name: str, value: str) -> str:
    """One unresolved line for a relational field still holding a class name."""
    found = real_instances(instances, value)
    if not found:
        return _line(NO_CANDIDATE, (
            f"{key}.{name} = {value!r} - class {value!r} has no instance on this "
            f"desktop; add the instance in the Instances tab or leave it unresolved"
        ))
    return _line(AMBIGUOUS, (
        f"{key}.{name} = {value!r} names a class the desktop has {len(found)} "
        f"instances of - pick one in S1"
    ))
