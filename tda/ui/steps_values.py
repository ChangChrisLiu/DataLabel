"""Vocabularies, cell-value coercion and note handling for the S1 step table.

Everything :mod:`tda.ui.steps_model` needs that is *about one value* rather than
about the edit session: the enums spec 3.1/6.x fixes (failure reasons, results,
group orders, screw heads, step types, the difficulty range), the error a
refused edit raises, the cached-thumbnail lookup, and the split between the
operator's own step notes and the ``LS:`` lines :mod:`tda.core.ls_import`
writes into the same column.

Qt-free, like the rest of the view-model layer. :mod:`tda.ui.steps_model`
re-exports these names, so callers need only that one import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from tda.core.model import StepType

__all__ = [
    "DIFFICULTY_MAX",
    "DIFFICULTY_MIN",
    "FAILURE_REASONS",
    "GROUP_ORDERS",
    "LS_NOTE_PREFIX",
    "RESULTS",
    "SCREW_HEADS",
    "STEP_TYPES",
    "EditError",
    "as_bool",
    "checked_difficulty",
    "human_notes",
    "ls_notes",
    "merge_notes",
    "text_of",
    "thumb_path",
]

#: ``Action.failure_reason`` enum of spec 3.1.
FAILURE_REASONS = ("blocked_by_cable", "blocked_by_part", "fastener_stuck", "wrong_tool", "other")
RESULTS = ("success", "failed")
#: ``Instance.group_order`` enum of spec 3.1.
GROUP_ORDERS = ("unordered", "sequential", "opposite_pairs")
#: ``screw.head`` vocabulary of spec 6.1.
SCREW_HEADS = ("PH1", "PH2", "PH3", "T15", "T20", "unknown")
STEP_TYPES = tuple(t.value for t in StepType)
DIFFICULTY_MIN, DIFFICULTY_MAX = 1, 5

#: Prefix of the note lines :mod:`tda.core.ls_import` writes into ``Step.notes``.
#: Mirrored here so the head-less UI layer does not import the Label Studio
#: reader; ``tests/test_steps_model.py`` asserts the two stay equal.
LS_NOTE_PREFIX = "LS: "

_TRUE_WORDS = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_WORDS = frozenset({"", "0", "false", "no", "n", "off", "none"})


class EditError(ValueError):
    """An edit the taxonomy or the instance table refuses; the text is UI-ready."""


# --------------------------------------------------------------------------- #
# thumbnails
# --------------------------------------------------------------------------- #
def thumb_path(
    cache_dir: str | Path, desktop: int, step: int, view: str = "scan"
) -> Optional[Path]:
    """Path of one cached frame, or ``None`` when it is not on disk.

    The local cache (spec 2.4) is laid out as
    ``<cache_dir>/<view>/D<nn>/s<kkk>.png``. Step ``0`` -- the "before" cell of
    the very first row -- has no frame, and a cache root that does not exist
    yet is not an error: the panel simply shows an empty cell.
    """
    if step < 1:
        return None
    path = Path(cache_dir) / view / f"D{desktop:02d}" / f"s{step:03d}.png"
    try:
        return path if path.is_file() else None
    except OSError:  # unreachable drive, bad path -- treat as "no thumbnail"
        return None


# --------------------------------------------------------------------------- #
# step notes
# --------------------------------------------------------------------------- #
def human_notes(notes: str) -> str:
    """The operator's own notes, without the imported ``LS:`` lines."""
    kept = [ln for ln in (notes or "").splitlines() if not ln.startswith(LS_NOTE_PREFIX)]
    return "\n".join(kept).strip()


def ls_notes(notes: str) -> list[str]:
    """The imported ``LS:`` lines of a step's notes, in their original order."""
    return [ln for ln in (notes or "").splitlines() if ln.startswith(LS_NOTE_PREFIX)]


def merge_notes(stored: str, edited: str) -> str:
    """Put the edited human text back in front of the preserved ``LS:`` lines.

    The step-table editor only ever shows :func:`human_notes`, so writing the
    cell back must not drop what :mod:`tda.core.ls_import` recorded there.
    """
    lines = [line for line in [text_of(edited)] if line] + ls_notes(stored)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# value coercion
# --------------------------------------------------------------------------- #
def text_of(value: Any) -> str:
    """A cell value as trimmed text; ``None`` becomes the empty string."""
    return "" if value is None else str(value).strip()


def as_bool(value: Any) -> bool:
    """Read a check-box / combo / text value as a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = text_of(value).lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    raise EditError(f"{value!r} is not a yes/no value")


def checked_difficulty(value: Any) -> Optional[int]:
    """Validate ``Action.difficulty``: 1-5, or empty for "not recorded"."""
    if value is None or text_of(value) == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise EditError(f"difficulty {value!r} is not a whole number") from None
    if not DIFFICULTY_MIN <= number <= DIFFICULTY_MAX:
        raise EditError(f"difficulty {number} is outside {DIFFICULTY_MIN}-{DIFFICULTY_MAX}")
    return number
