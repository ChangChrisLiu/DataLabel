"""Which physical part a log row means (spec 3.2), and when two rows share one.

:mod:`tda.core.logs` reads the sheets; this module answers the one question that
decides how many parts a desktop has -- *is this row a part I have already seen?*
-- and owns the bookkeeping that answer needs.  It is pure: no CSV, no taxonomy
file, no database.

The default answer is **no**.  The annotators restarted their numbering ("CPU
fan screw 1..5", then "Heatsink screw 1..4") and repeated unnumbered names
("Case - motherboard connector", seven separate connectors), so a rule that
merged on name alone once merged 419 actions onto the wrong instance.  Two doors
are open instead:

* :meth:`InstanceLedger.numbered_matches` -- the strict test of spec 3.2: an
  explicit instance number, a matching :func:`identity`, and a verb the instance
  has not had yet;
* :func:`physical_reuse` -- the narrow one below, for the unnumbered
  ``displace``-then-``remove`` pair that is one part in the world and was two in
  the draft.

Both report; nothing here ever merges silently.
"""
from __future__ import annotations

from typing import Callable, Optional

UNRESOLVED = "?"
CHASSIS_KEY = "chassis"

#: Attributes the importer derives itself; they never take part in the
#: instance-identity comparison.
DERIVED_ATTRS = frozenset({"instance_nos", "sheet_no", "captive", "captive_source"})

#: Brands whose CPU-cooler screws are captive (they stay in the bracket).
CAPTIVE_BRANDS = ("dell", "optiplex")
CAPTIVE_ROLE = "cpu_cooler"

#: Verbs that leave the part where it is, so the row after them can still be
#: about the same object. ``loosen`` is not a taxonomy verb today; it is listed
#: because the ruling names it and a sheet may yet use it.
NON_TERMINAL_VERBS = frozenset(
    {"displace", "open", "release", "loosen", "unscrew", "disconnect"}
)

#: The verb the narrow rule applies to, and the state it needs the candidate
#: *not* to be in already.
REMOVE = "remove"
REMOVED = "removed"

#: One entry of an instance's history: ``(step, verb, result)``.
Op = tuple[int, str, str]
#: What a later row must match exactly before it may reuse an instance.
Identity = tuple


def instance_key(cls: str, attrs: dict, ordinal: int) -> str:
    """Build an instance key: class, discriminator, 2-digit ordinal.

    The discriminator is ``attrs["role"]`` (screws) or ``attrs["kind"]``
    (connectors, drives, coolers, cards); ``chassis`` is unique per desktop and
    therefore carries no ordinal.
    """
    if cls == CHASSIS_KEY:
        return CHASSIS_KEY
    parts = [cls]
    disc = attrs.get("role") or attrs.get("kind") or ""
    if disc:
        parts.append(str(disc))
    parts.append(f"{ordinal:02d}")
    return ".".join(parts)


def placeholder_key(cls: str, attrs: dict) -> str:
    """The key of a target a human still has to resolve."""
    if not cls:
        return UNRESOLVED
    disc = attrs.get("role") or attrs.get("kind") or ""
    return ".".join([cls] + ([str(disc)] if disc else []) + [UNRESOLVED])


def identity(cls: str, disc: str, number: Optional[int], attrs: dict) -> Identity:
    """What a later row must match exactly before it may reuse an instance.

    Spec 3.2's class + role + number, plus every attribute that names *which*
    part the row means: the cable owner, the annotator's bracketed qualifier
    and ``of`` (what a cover, cage or latch belongs to).  Two "Connector 1"
    rows whose cables run to different devices are different connectors, and a
    RAM cover is never the heatsink cover.

    ``number`` is ``None`` for an unnumbered row, which is its own identity
    rather than a wildcard: an unnumbered candidate is never matched by a
    numbered row, nor the other way round.
    """
    return (
        cls,
        disc,
        number,
        str(attrs.get("cable_owner") or ""),
        str(attrs.get("qualifier") or ""),
        str(attrs.get("of") or ""),
    )


def attr_conflicts(stored: dict, new: dict) -> list[tuple[str, object, object]]:
    """Attributes the new row states differently from the stored instance."""
    return [
        (name, stored[name], value)
        for name, value in new.items()
        if name not in DERIVED_ATTRS and name in stored and stored[name] != value
    ]


def merge_attrs(stored: dict, new: dict) -> None:
    """Add the new row's extra attributes; conflicts are handled by the caller."""
    for name, value in new.items():
        if name not in DERIVED_ATTRS:
            stored.setdefault(name, value)


def is_captive(cls: str, attrs: dict, meta: dict) -> Optional[bool]:
    """Is this screw captive? ``None`` for anything that is not a screw."""
    if cls != "screw":
        return None
    brand = str(meta.get("brand_model_raw") or "").lower()
    dell = any(token in brand for token in CAPTIVE_BRANDS)
    return bool(dell and attrs.get("role") == CAPTIVE_ROLE)


class InstanceLedger:
    """The instances one desktop has so far, indexed by how a row could find them.

    Three books: the next ordinal per ``(class, discriminator)`` group, the keys
    reachable by :func:`identity` (numbered and unnumbered kept apart, because
    only the unnumbered ones can be reached by the narrow rule), and the verbs
    each instance has already been given.
    """

    def __init__(self) -> None:
        self._ordinals: dict[tuple[str, str], int] = {}
        self._numbered: dict[Identity, list[str]] = {}
        self._unnumbered: dict[Identity, list[str]] = {}
        self._ops: dict[str, list[Op]] = {}

    # -- creation --------------------------------------------------------- #
    def next_ordinal(self, cls: str, disc: str) -> int:
        """The ordinal a new instance of this group takes, counting it in."""
        ordinal = self._ordinals.get((cls, disc), 0) + 1
        self._ordinals[(cls, disc)] = ordinal
        return ordinal

    def remember(self, key: str, cls: str, disc: str, number: Optional[int],
                 attrs: dict) -> None:
        """File a freshly created instance under the identity of its row."""
        book = self._numbered if number is not None else self._unnumbered
        book.setdefault(identity(cls, disc, number, attrs), []).append(key)

    # -- history ---------------------------------------------------------- #
    def record(self, key: str, step: int, verb: str, result: str) -> None:
        """Note that this instance was given a verb at this step."""
        self._ops.setdefault(key, []).append((step, verb, result))

    def ops(self) -> dict[str, list[Op]]:
        """Every instance's history, for the one-verb-per-instance invariant."""
        return self._ops

    def verbs(self, key: str) -> list[str]:
        """Every verb this instance has been given, successful or not."""
        return [verb for _step, verb, _result in self._ops.get(key, [])]

    def last_successful(self, key: str) -> Optional[str]:
        """The verb this instance was last *successfully* given, or ``None``."""
        done = [v for _s, v, result in self._ops.get(key, []) if result == "success"]
        return done[-1] if done else None

    # -- lookup ----------------------------------------------------------- #
    def numbered_matches(self, cls: str, disc: str, number: int,
                         attrs: dict) -> list[str]:
        """Instances a numbered row could be operating again, in creation order."""
        return list(self._numbered.get(identity(cls, disc, number, attrs), []))

    def unnumbered_matches(self, cls: str, disc: str, attrs: dict) -> list[str]:
        """Instances created by an earlier *unnumbered* row of the same identity."""
        return list(self._unnumbered.get(identity(cls, disc, None, attrs), []))


def physical_reuse(
    ledger: InstanceLedger,
    cls: str,
    disc: str,
    attrs: dict,
    state_of: Callable[[str], Optional[str]],
) -> Optional[tuple[str, str]]:
    """The one instance an unnumbered successful ``remove`` may operate again.

    ``(key, the verb it was last successfully given)``, or ``None`` when the row
    is a part of its own.  Every condition has to hold:

    * an earlier **unnumbered** row created it with the same class and the same
      :func:`identity` discriminators -- a numbered candidate belongs to the
      strict test, and a different ``of`` or cable owner is a different part;
    * it is not removed already, so the row is not lifting out something that
      has gone (``state_of`` folds the verbs that succeeded);
    * its latest successful verb is a non-terminal one, i.e. the sheet left it
      mid-disassembly: swung aside, opened, unscrewed.  An instance nothing has
      successfully happened to yet is *not* a candidate -- a failed attempt at
      step 10 says the part is untouched, not that step 20 continues it;
    * it is the only candidate on the desktop.  Two displaced power supplies and
      one ``Power module`` row is a question for a human, not a coin toss.

    This is what turns D22's ``Open Power Module`` (step 12) and ``Power module``
    (step 20) into one ``psu.01`` instead of a ``psu.01`` the annotator masks for
    eight steps and a ``psu.02`` they mask for the rest.
    """
    found = [
        (key, last)
        for key in ledger.unnumbered_matches(cls, disc, attrs)
        for last in [ledger.last_successful(key)]
        if last in NON_TERMINAL_VERBS and state_of(key) != REMOVED
    ]
    return found[0] if len(found) == 1 else None
