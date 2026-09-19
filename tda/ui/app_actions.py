"""The one keyboard map of the annotator (spec 4.6), and what is generated from it.

:data:`ACTIONS` is the single source of truth for three things that used to
drift apart: what a key does, what the in-app cheat sheet (``?`` / ``F12``)
shows, and the ``快捷键`` table of ``docs/annotation_guide.md``.  Adding a
binding here adds it everywhere; there is nowhere else to add one.

**The 1-4 vs 1-7 collision.**  The plan gave the four views the number keys and
the seven visibility values the same ones.  The ruling implemented here (and in
the brief) is: the views are ``F1``-``F4``, and ``1``-``7`` set the visibility
of the selected instance.  Visibility is a per-instance judgement made dozens of
times per frame, so it gets the home row; switching view is rare.

**Modes.**  Every action declares the modes it is live in, so the step table
keeps ``Enter`` for its cell editors and the review panel keeps ``R`` for
"rework" while annotate mode uses it for the bench box.  A key with no action in
the current mode is never swallowed -- it goes on to the focused widget.

**Text fields.**  :func:`blocks_shortcuts` is consulted before any lookup, so a
``B`` typed into a note or a combo box is a letter, not a brush.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QKeySequenceEdit,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)

from tda.ui import session_api as api

__all__ = [
    "ACTIONS",
    "GROUPS",
    "MODES",
    "MODE_ANNOTATE",
    "MODE_REVIEW",
    "MODE_STEPS",
    "Action",
    "action_for",
    "blocks_shortcuts",
    "cheat_sheet_html",
    "combos_of",
    "navigates_a_list",
    "shortcut_markdown",
]

MODE_STEPS = "steps"
MODE_ANNOTATE = "annotate"
MODE_REVIEW = "review"
MODES: tuple[str, ...] = (MODE_STEPS, MODE_ANNOTATE, MODE_REVIEW)

#: Modes an action is live in.  ``modes[0]`` is the mode the action belongs to
#: first, which is what the cheat sheet and the tests use.
_ANN = (MODE_ANNOTATE,)
_ANN_REV = (MODE_ANNOTATE, MODE_REVIEW)
_ALL = (MODE_ANNOTATE, MODE_REVIEW, MODE_STEPS)

#: Section keys and their headings in the guide and the cheat sheet.
GROUPS: tuple[tuple[str, str], ...] = (
    ("view", "视图与模式"),
    ("nav", "帧导航"),
    ("tool", "工具"),
    ("edit", "编辑与提交"),
    ("display", "显示"),
    ("session", "会话"),
)


@dataclass(frozen=True)
class Action:
    """One binding: the keys, the window slot they call, and how to name it.

    Attributes:
        name: unique identifier, used by the tests and the cheat sheet anchors.
        keys: portable ``QKeySequence`` strings; all of them trigger the action
            (``Enter``/``Return`` are two spellings of one key).
        slot: method on the main window, called with ``args``.
        args: positional arguments handed to ``slot``.
        group: one of :data:`GROUPS`.
        label / label_zh: English and Simplified Chinese descriptions.
        modes: modes the binding is live in.
        hold: the action is held rather than pressed -- the slot is called with
            ``True`` on press and ``False`` on release (``Tab``, flash compare).
        repeat: holding the key fires the action again and again.  It is the
            difference between "bigger brush" (hold it until the brush is the
            size you want) and "commit" (once, however long the finger stays
            down).  Auto-repeat used to be swallowed for *every* binding, which
            made ``]``, ``.`` and ``PgDn`` do nothing at all when held.
    """

    name: str
    keys: tuple[str, ...]
    slot: str
    label: str
    label_zh: str
    group: str = "edit"
    args: tuple = field(default_factory=tuple)
    modes: tuple[str, ...] = _ANN
    hold: bool = False
    repeat: bool = False


def _view(view: str, key: str, zh: str) -> Action:
    return Action(f"view_{view}", (key,), "act_set_view", f"View: {view}", zh,
                  "view", (view,), _ANN_REV)


def _visibility(index: int) -> Action:
    value = api.VISIBILITY_VALUES[index]
    return Action(f"visibility_{index + 1}", (str(index + 1),),
                  "act_set_visibility", f"Visibility: {value}",
                  f"可见性设为 {value}", "view", (value,), _ANN)


def _tool(name: str, key: str, label: str, zh: str) -> Action:
    return Action(f"tool_{name}", (key,), "act_tool", label, zh, "tool", (name,), _ANN)


ACTIONS: tuple[Action, ...] = (
    # -- views and modes ----------------------------------------------------
    _view("scan", "F1", "切换到扫描仪视图"),
    _view("oak1", "F2", "切换到 OAK 正面视图"),
    _view("oak2", "F3", "切换到 OAK 侧面视图"),
    _view("rs", "F4", "切换到 RealSense 视图"),
    *[_visibility(i) for i in range(7)],
    # -- frame navigation ---------------------------------------------------
    Action("step_back", ("PgDown",), "act_step", "Previous step (k-1)",
           "上一步 k-1（倒序标注的“前进”）", "nav", (-1,), _ANN_REV, repeat=True),
    Action("step_forward", ("PgUp",), "act_step", "Next step (k+1)",
           "下一步 k+1", "nav", (+1,), _ANN_REV, repeat=True),
    Action("step_first", ("Home",), "act_step_edge", "First step",
           "跳到第一步", "nav", ("first",), _ANN_REV),
    Action("step_last", ("End",), "act_step_edge", "Last step",
           "跳到最后一步", "nav", ("last",), _ANN_REV),
    Action("flash_compare", ("Tab",), "act_flash_compare",
           "Hold: flash the frame the task card is written against",
           "按住：闪回任务卡对照的那一帧（你来的方向）", "nav", (True,), _ANN, hold=True),
    Action("flash_other", ("Shift+Tab",), "act_flash_other",
           "Hold: flash the frame on the other side",
           "按住：闪回另一侧的那一帧", "nav", (True,), _ANN, hold=True),
    # -- tools --------------------------------------------------------------
    _tool("brush", "B", "Brush", "画笔（加像素）"),
    _tool("eraser", "E", "Eraser", "橡皮擦（减像素）"),
    _tool("sam_point", "S", "SAM point prompt", "SAM 点提示"),
    _tool("sam_box", "X", "SAM box prompt", "SAM 框提示"),
    _tool("occluder", "O", "Occluder brush", "遮挡层画笔（手/工具）"),
    _tool("bench_box", "R", "Bench box", "台面框（拖框标注已拆下的零件）"),
    Action("radius_down", ("[",), "act_radius", "Smaller brush",
           "笔刷变小", "tool", (-1,), _ANN, repeat=True),
    Action("radius_up", ("]",), "act_radius", "Bigger brush",
           "笔刷变大", "tool", (+1,), _ANN, repeat=True),
    Action("cycle_candidate", ("C",), "act_cycle_candidate", "Next SAM candidate",
           "切换 SAM 候选掩码", "tool", (), _ANN),
    Action("fill_holes", ("Shift+F",), "act_fill_holes", "Fill holes",
           "填补掩码内部空洞", "tool", (), _ANN),
    Action("despeckle", ("Shift+D",), "act_despeckle", "Remove specks",
           "删除小碎块（默认 <16 像素）", "tool", (), _ANN),
    # -- editing ------------------------------------------------------------
    Action("commit", ("Return", "Enter"), "act_commit", "Commit edit",
           "提交编辑（按建议范围）", "edit", (), _ANN),
    Action("commit_override", ("Alt+Return", "Alt+Enter"), "act_commit_override",
           "Commit for this frame only", "只对当前帧生效（帧覆盖）", "edit", (), _ANN),
    Action("commit_split", ("Ctrl+K",), "act_commit_split", "Split keyframe here",
           "从这一帧起拆分关键帧", "edit", (), _ANN),
    Action("clear_edit", ("Esc",), "act_clear_edit", "Discard the edit",
           "放弃当前编辑", "edit", (), _ANN),
    Action("confirm", ("Space",), "act_confirm", "Confirm the frame",
           "确认当前帧并后退一帧", "edit", (), _ANN),
    # Review mode: the canvas is read-only, so its two keys are the queue's.
    Action("review_accept", ("Return", "Enter", "Space"), "act_confirm",
           "Accept the frame the queue points at", "接受队列选中的这一帧",
           "edit", (), (MODE_REVIEW,)),
    Action("review_rework", ("R",), "act_rework_selected",
           "Rework: open it in Annotate mode", "返工：在标注模式下打开这一帧",
           "edit", (), (MODE_REVIEW,)),
    Action("review_keep_old", ("K",), "act_resolve", "Conflict: keep the frozen shape",
           "冲突：保留已冻结的形状", "edit", (api.RESOLVE_KEEP_OLD,), (MODE_REVIEW,)),
    Action("review_accept_new", ("N",), "act_resolve", "Conflict: take the edit",
           "冲突：接受新的编辑", "edit", (api.RESOLVE_ACCEPT_NEW,), (MODE_REVIEW,)),
    Action("undo", ("Ctrl+Z",), "act_undo", "Undo", "撤销", "edit", (), _ANN),
    Action("redo", ("Ctrl+Y",), "act_redo", "Redo", "重做", "edit", (), _ANN),
    Action("toggle_hidden", ("H",), "act_toggle_hidden", "Hide/show instance",
           "隐藏或显示选中实例", "edit", (), _ANN),
    Action("cycle_visibility", ("V",), "act_cycle_visibility", "Cycle visibility",
           "循环切换可见性取值", "edit", (), _ANN),
    Action("zorder_up", ("Ctrl+Up",), "act_move_instance", "One layer up",
           "选中实例上移一层", "edit", (-1,), _ANN, repeat=True),
    Action("zorder_down", ("Ctrl+Down",), "act_move_instance", "One layer down",
           "选中实例下移一层", "edit", (+1,), _ANN, repeat=True),
    Action("edit_roi", ("Shift+R",), "act_edit_roi", "Re-edit the ROI",
           "重新框定 ROI（机箱范围）", "edit", (), _ANN),
    # -- display ------------------------------------------------------------
    Action("toggle_overlays", ("A",), "act_toggle_overlays", "All overlays on/off",
           "开关全部标注图层", "display", (), _ANN_REV),
    Action("toggle_outline", ("Q",), "act_toggle_outline", "Outline / filled",
           "轮廓线与半透明填充切换", "display", (), _ANN_REV),
    Action("opacity_down", (",",), "act_opacity", "Less opaque",
           "图层更透明", "display", (-1,), _ANN_REV, repeat=True),
    Action("opacity_up", (".",), "act_opacity", "More opaque",
           "图层更不透明", "display", (+1,), _ANN_REV, repeat=True),
    Action("toggle_heat", ("D",), "act_toggle_heat", "Difference heat map",
           "开关帧间差异热力图", "display", (), _ANN_REV),
    Action("fit_roi", ("F",), "act_fit_roi", "Fit the ROI",
           "缩放到 ROI", "display", (), _ANN_REV),
    Action("fit_image", ("Shift+0",), "act_fit_image", "Fit the whole frame",
           "缩放到整幅图像", "display", (), _ANN_REV),
    Action("toggle_grid", ("G",), "act_toggle_grid", "Pixel grid",
           "开关像素网格", "display", (), _ANN_REV),
    # -- session ------------------------------------------------------------
    Action("save", ("Ctrl+S",), "act_save", "Save", "保存（清除未保存标记）",
           "session", (), _ALL),
    Action("refresh_all", ("F5",), "act_refresh_all", "Recompile this frame",
           "重新编译当前帧（整视图请用命令行 cli check）", "session", (), _ALL),
    Action("cheat_sheet", ("?", "F12"), "act_cheat_sheet", "Shortcut cheat sheet",
           "快捷键速查表", "session", (), _ALL),
)


# --------------------------------------------------------------------------- #
# key lookup
# --------------------------------------------------------------------------- #
def _as_int(value) -> int:
    """``int`` of a Qt enum, a flag combination or a plain number.

    PySide6 hands a *combination* of ``KeyboardModifier`` flags back as an
    object ``int()`` refuses, while a single flag converts fine -- so both
    spellings are tried rather than assuming either.
    """
    try:
        return int(value)
    except TypeError:
        return int(getattr(value, "value", 0))


#: Modifiers that never take part in a binding: the numeric keypad reports its
#: own flag, and the group switch belongs to the keyboard layout.
_IGNORED_MODIFIERS = (_as_int(Qt.KeyboardModifier.KeypadModifier)
                      | _as_int(Qt.KeyboardModifier.GroupSwitchModifier))
_SHIFT = _as_int(Qt.KeyboardModifier.ShiftModifier)
#: Keys a US layout produces with Shift where the binding names the unshifted
#: character, so both spellings have to resolve to the same action.
_SHIFTED = {_as_int(Qt.Key.Key_ParenRight): _as_int(Qt.Key.Key_0)}


def _combined(spec: str) -> Optional[int]:
    """The ``Qt.Key | modifiers`` integer of a portable key string."""
    seq = QKeySequence.fromString(spec)
    if seq.count() < 1:
        return None
    return int(seq[0].toCombined())


def combos_of(action: Action) -> tuple[int, ...]:
    """Every key combination that triggers ``action``."""
    return tuple(c for c in (_combined(spec) for spec in action.keys) if c is not None)


def _build_index() -> dict[str, dict[int, Action]]:
    index: dict[str, dict[int, Action]] = {mode: {} for mode in MODES}
    for action in ACTIONS:
        for combo in combos_of(action):
            for mode in action.modes:
                index[mode][combo] = action
    return index


_INDEX = _build_index()


def _candidates(key, modifiers) -> list[int]:
    """The combination the event means, plus the layout aliases of the brief."""
    code = _as_int(key)
    mods = _as_int(modifiers) & ~_IGNORED_MODIFIERS
    out = [code | mods]
    if mods & _SHIFT:
        alias = _SHIFTED.get(code)
        if alias is not None:
            out.append(alias | mods)
        # "?" is Shift+/ on a US layout but the binding is written "?"
        if 0x21 <= code <= 0x3F and not (0x30 <= code <= 0x39):
            out.append(code | (mods & ~_SHIFT))
    return out


def action_for(key, modifiers, mode: str = MODE_ANNOTATE) -> Optional[Action]:
    """The action bound to ``key`` + ``modifiers`` in ``mode``, or ``None``."""
    table = _INDEX.get(mode)
    if table is None:
        return None
    for combo in _candidates(key, modifiers):
        found = table.get(combo)
        if found is not None:
            return found
    return None


#: Widget classes that own every keystroke while they have the focus.
_TEXT_WIDGETS = (
    QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox,
    QKeySequenceEdit,
)


def blocks_shortcuts(widget: Optional[QWidget]) -> bool:
    """``True`` when ``widget`` is a text entry that must keep its keystrokes.

    An item view's open editor is a real ``QLineEdit`` child, so a cell being
    edited in the step table is covered by the same check.
    """
    return widget is not None and isinstance(widget, _TEXT_WIDGETS)


#: Keys a focused list or table must keep for moving its own selection.  Page
#: up/down are deliberately **not** here: they are frame navigation everywhere
#: in this window, and a list that scrolled instead would be a second, silent
#: meaning for the annotator's most used pair of keys.
_LIST_KEYS = frozenset({
    int(Qt.Key.Key_Up), int(Qt.Key.Key_Down),
    int(Qt.Key.Key_Left), int(Qt.Key.Key_Right),
})


def navigates_a_list(widget: Optional[QWidget], key) -> bool:
    """Is this an arrow key inside a list or table, i.e. its own navigation?"""
    if widget is None or _as_int(key) not in _LIST_KEYS:
        return False
    return isinstance(widget, QAbstractItemView)


# --------------------------------------------------------------------------- #
# generated documentation
# --------------------------------------------------------------------------- #
def _keys_text(action: Action) -> str:
    return " / ".join(action.keys)


#: What the generated tables add after a binding that answers auto-repeat.
REPEAT_NOTE = "（可长按）"
HOLD_NOTE = "（按住）"


def _how_text(action: Action) -> str:
    """"Hold it down" is part of what a key does, so the tables say so."""
    if action.repeat:
        return REPEAT_NOTE
    return HOLD_NOTE if action.hold else ""


def _by_group() -> list[tuple[str, str, list[Action]]]:
    return [
        (key, title, [a for a in ACTIONS if a.group == key])
        for key, title in GROUPS
    ]


def shortcut_markdown() -> str:
    """The ``快捷键`` table of ``docs/annotation_guide.md``, one row per binding."""
    lines = ["| 快捷键 | 作用 | 分组 |", "|---|---|---|"]
    for _key, title, actions in _by_group():
        for action in actions:
            lines.append(f"| `{_keys_text(action)}` | {action.label_zh}"
                         f"{_how_text(action)} | {title} |")
    return "\n".join(lines)


def cheat_sheet_html() -> str:
    """The same table as HTML, for the in-app cheat sheet (``?`` / ``F12``)."""
    parts = ["<h3>快捷键 / Shortcuts</h3>"]
    for _key, title, actions in _by_group():
        parts.append(f"<h4>{title}</h4><table cellspacing='4'>")
        for action in actions:
            parts.append(
                f"<tr><td><b>{_keys_text(action)}</b></td>"
                f"<td>{action.label_zh}{_how_text(action)}</td>"
                f"<td><i>{action.label}</i></td></tr>"
            )
        parts.append("</table>")
    return "".join(parts)


#: Markers delimiting the generated block inside ``docs/annotation_guide.md``.
GUIDE_BEGIN = "<!-- shortcuts:begin (generated from tda/ui/app_actions.py) -->"
GUIDE_END = "<!-- shortcuts:end -->"


def write_guide_table(path: str) -> bool:
    """Replace the generated block of the annotation guide; ``True`` when it changed.

    Run as ``python -m tda.ui.app_actions`` after touching :data:`ACTIONS`, so
    the guide can never disagree with the keyboard (a test asserts they match).
    """
    from pathlib import Path

    target = Path(path)
    body = target.read_text(encoding="utf-8")
    head, _, rest = body.partition(GUIDE_BEGIN)
    _, _, tail = rest.partition(GUIDE_END)
    if not rest or not tail:
        raise ValueError(f"{target} has no {GUIDE_BEGIN} ... {GUIDE_END} block")
    rebuilt = f"{head}{GUIDE_BEGIN}\n\n{shortcut_markdown()}\n\n{GUIDE_END}{tail}"
    if rebuilt == body:
        return False
    target.write_text(rebuilt, encoding="utf-8")
    return True


def _main(argv: Optional[list[str]] = None) -> int:  # pragma: no cover - a tool
    import argparse
    from pathlib import Path

    default = Path(__file__).resolve().parents[2] / "docs" / "annotation_guide.md"
    ap = argparse.ArgumentParser(description="Regenerate the guide's shortcut table.")
    ap.add_argument("--guide", default=str(default))
    args = ap.parse_args(argv)
    changed = write_guide_table(args.guide)
    print(f"{'rewrote' if changed else 'already up to date:'} {args.guide}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(_main())
