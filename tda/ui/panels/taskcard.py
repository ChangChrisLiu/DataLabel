"""Task card dock: the work that belongs to the frame on screen (spec 4.2).

The session diffs this frame against the neighbour the annotator came from and
turns the result into instructions -- draw this part back into the chassis, box
that one in the staging area, split this keyframe, only flip that state -- and
this panel is their checklist: done items are struck through and the first open
one is highlighted.

The panel decides nothing.  Activating an item only emits
:attr:`TaskCardPanel.sigRequestEdit`; the main window is what calls
``begin_edit``, because it is the one that can refuse -- switching instance
while pixels are uncommitted has to be answerable.  It has **no action buttons**
of its own any more (task U2b round 1): the four commit/confirm buttons it
carried duplicated the tool palette's, and two places for one thing is the
confusion the second trial reported.  When ``confirm_frame`` refuses, the
problems that came with ``sigProblems`` are shown instead of any local check.
On arriving at a frame the pane lists only what the rows do not already say
(task U2c): a missing shape *is* its row, and the card is the to-do list.
What stops ``Space`` comes first; the notes the confirmation accepts follow
under their own greyed heading, 提示（不挡 Space） (task U2d).

**A card that explains itself** (task U2b).  The second trial's annotator stood
on the start frame, read "Draw cover.02 (cover) on this frame" four times and
did not know what any of the four were.  So the card now opens with a sentence
saying what *this* frame's list is, and every row says the part's Chinese name,
its key, the name the step sheet gave it (``日志：CPU Fan Cover``), one plain
sentence of what to do with it and whether it is 待画 / 正在画 / 已完成.  A
**single** click starts the edit (the double-click that "did nothing" had in
fact started one -- nothing on the card said so), and resting the pointer on a
row reports it through :attr:`TaskCardPanel.sigHover`, so the window can
outline where the part probably is.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from tda.core.truth_refusals import (
    FROZEN_DISAGREEMENT,
    INPUTS_CHANGED,
    OPEN_CONFLICT,
    conflict_of,
    is_blocking,
)
from tda.ui import session_api as api
from tda.ui.class_names import instance_label, state_zh, visibility_zh
from tda.ui.panels import session_is_open

__all__ = ["CHIP_DONE", "CHIP_EDITING", "CHIP_TODO", "KIND_ICONS", "KIND_SENTENCES",
           "TaskCardPanel", "card_header", "row_view", "unlisted_problems"]

INSTANCE_ROLE = int(Qt.ItemDataRole.UserRole)
#: The painted view of a row (:func:`row_view`), for the delegate.
VIEW_ROLE = INSTANCE_ROLE + 1
KIND_ROLE = INSTANCE_ROLE + 2
#: The compiler code a row of the problems pane stands for (``""``: none).
CODE_ROLE = INSTANCE_ROLE + 3
#: ``True`` on the pane's one line that is not a problem: the heading of the
#: notes the confirmation accepts (U2d).
NOTES_HEADING_ROLE = INSTANCE_ROLE + 4

#: One glyph per task kind (spec 4.2), so the list can be skimmed vertically.
KIND_ICONS: dict[str, str] = {
    api.KIND_ADD_SHAPE: "✚",  # heavy greek cross: draw a new shape
    api.KIND_SPLIT_KEYFRAME: "✂",  # scissors: split the keyframe
    api.KIND_STATE_ONLY: "≡",  # identical to: shape unchanged
    api.KIND_ADD_BENCH_BOX: "▭",  # rectangle: box it in the staging area
    api.KIND_REMOVE_BENCH_BOX: "⌫",  # erase: the bench box ends here
    api.KIND_CONFIRM: "✔",  # check: nothing to draw
}

#: Rows whose work is already done.
DONE_COLOR = QColor(128, 128, 132)

#: One plain sentence per kind: what to do with the row (ruling U2b-3).
KIND_SENTENCES: dict[str, str] = {
    api.KIND_ADD_SHAPE: "在这一帧画出它的完整形状",
    api.KIND_SPLIT_KEYFRAME: ("它的形状从这一帧起变了（例如盖子关上了），画出这一帧的样子，"
                              "用 Ctrl+K 提交"),
    api.KIND_STATE_ONLY: "只是状态变了，不用画",
    api.KIND_ADD_BENCH_BOX: "它放在台面上，用台面框 R 框一下",
    api.KIND_REMOVE_BENCH_BOX: "它回到机箱里了：台面框到这一帧为止，不用画",
    api.KIND_CONFIRM: "这一帧不用画，直接 Space",
}
#: Kinds a click starts work on; the others only explain themselves.  A ⌫
#: row (the bench box ends here) is not a mask to draw (U2b round 2).
EDITABLE_KINDS = (api.KIND_ADD_SHAPE, api.KIND_SPLIT_KEYFRAME, api.KIND_ADD_BENCH_BOX)

CHIP_TODO = "待画"
CHIP_EDITING = "✎ 正在画"
CHIP_BOXING = "✎ 正在框"
CHIP_DONE = "✔ 已完成"
CHIP_NOTHING = "不用画"
CHIP_CONFIRM = "待确认"

#: What the icons mean, for the header's tooltip.
ICON_LEGEND = (
    "✚ 要画：这一帧多出来的零件，画出它的完整形状\n"
    "✂ 形状变了：画这一帧的新样子，Ctrl+K 提交（从这帧起新版本）\n"
    "≡ 只是状态变了（例如螺丝拧紧），不用画\n"
    "▭ 零件放在台面上：用台面框 R 拖一个框\n"
    "⌫ 零件回到机箱：台面框到这里结束\n"
    "✔ 这一帧没有要画的：Space 确认（任务卡下面列着挡住 Space 的问题时，先处理它们）\n"
    "单击一条就开始画它；鼠标停在一条上，画面上会用虚线框标出它大概在哪。"
)


def _now() -> float:
    import time

    return time.monotonic()


def _double_click_s() -> float:
    """The platform's double-click interval, in seconds."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    interval = app.doubleClickInterval() if app is not None else 400
    return max(0.1, interval / 1000.0) + 0.05


#: What stands where the card would say "直接 Space" while its own pane lists
#: problems that make ``Space`` refuse (U2d round 2).  ``n`` is
#: :meth:`TaskCardPanel.problem_count` -- the number the guide reads too -- so
#: the header, the ✔ row and the guide cannot disagree about one frame.
BLOCKED_SPACE = "但下面还有 {n} 个问题挡住 Space（见下方）"


def space_or_blocked(blockers: int, space: str = "直接 Space") -> str:
    """``space`` when nothing in the pane stops Space, else what does."""
    return BLOCKED_SPACE.format(n=int(blockers)) if blockers else space


#: What a card that would say "直接 Space" says on a frame already confirmed,
#: with nothing in the pane: what the guide says there (U2e round 2).
CONFIRMED_TAIL = "已经确认 ✓（PgDn 下一帧；改了之后再按 Space 重新确认）"
CONFIRMED_TEXT = "这一帧" + CONFIRMED_TAIL


#: How each kind is counted in a mixed header (U2b round 2).
_KIND_COUNTS: tuple[tuple[str, str], ...] = (
    (api.KIND_ADD_SHAPE, "{n} 个要画回去"),
    (api.KIND_SPLIT_KEYFRAME, "{n} 个形状变了"),
    (api.KIND_STATE_ONLY, "{n} 个只是状态变了"),
    (api.KIND_ADD_BENCH_BOX, "{n} 个要用台面框框出来"),
    (api.KIND_REMOVE_BENCH_BOX, "{n} 个台面框到这里结束"),
)


def card_header(step: Optional[int], neighbour: Optional[int],
                kinds: Optional[list[str]] = None, first: Optional[int] = None,
                last: Optional[int] = None, done: Optional[list[bool]] = None,
                blockers: int = 0, confirmed: bool = False) -> str:
    """The sentence above the list: what *this* frame's list is (ruling U2b-3).

    Built from the rows' kinds (round 2): "多了…要画回去" over rows that only
    change a state was an instruction the rows contradicted.  ``first`` /
    ``last`` are the view's first and last step, which is what tells the start
    frame of the reverse walk (the machine taken apart) from the start of a
    forward one (not taken apart yet).  ``done`` runs alongside ``kinds``: the
    start card lists the drawn parts too (round 3), and its header counts what
    is left and what is done rather than calling every row work.
    ``blockers`` is how many problems in the card's pane stop ``Space``: with
    any, no header says "直接 Space" (U2d round 2, :func:`space_or_blocked`).
    ``confirmed`` is a frame already confirmed: with nothing blocking, where
    the header would send the annotator to Space it says :data:`CONFIRMED_TEXT`,
    as the guide does (U2e round 2).
    """
    if step is None:
        return ""
    kinds = [str(k) for k in (kinds if kinds is not None else [api.KIND_ADD_SHAPE])]
    flags = [bool(d) for d in (done if done is not None else [False] * len(kinds))]
    flags += [False] * (len(kinds) - len(flags))
    work = [k for k in kinds if k != api.KIND_CONFIRM]
    if neighbour is None:
        if last is not None and int(step) == int(last):
            where = "起点，已经拆完的样子"
        elif first is not None and int(step) == int(first):
            where = "起点，还没开始拆的样子"
        else:
            where = "起点"
        left = [k for k, d in zip(kinds, flags) if k != api.KIND_CONFIRM and not d]
        drawn = len(work) - len(left)
        if not left and confirmed and not blockers:
            return f"第 {step} 帧（{where}）：{CONFIRMED_TEXT}"
        if not left:
            return (f"第 {step} 帧（{where}）：这一帧的零件都画好了，"
                    f"{space_or_blocked(blockers, '直接 Space 确认')}")
        # Once, here, rather than on every row of a start card that can list
        # sixty parts (round 1b).
        text = (f"第 {step} 帧（{where}）：把这一帧里还看得到的零件都画出来，"
                f"每个都画完整形状{START_NOTE}")
        if drawn:
            text += f"。还剩 {len(left)} 个，{drawn} 个画好了（✔，排在最后）"
        return text
    present = set(work)
    if present <= {api.KIND_STATE_ONLY} and confirmed and not blockers:
        return f"第 {step} 帧：{CONFIRMED_TEXT}"
    if not present:
        return f"第 {step} 帧：这一帧不用画，{space_or_blocked(blockers)}"
    if present == {api.KIND_ADD_SHAPE}:
        if int(neighbour) > int(step):
            return (f"第 {step} 帧：比第 {neighbour} 帧多了下面这些零件"
                    f"（刚被拆掉的，要把它画回去）")
        return f"第 {step} 帧：和第 {neighbour} 帧比，下面这些零件多出来了，要画出来"
    if present == {api.KIND_STATE_ONLY}:
        return (f"第 {step} 帧：这一帧只有状态变化（拧紧/插上…），不用画，"
                f"{space_or_blocked(blockers)}")
    if present == {api.KIND_SPLIT_KEYFRAME}:
        return (f"第 {step} 帧：下面这些零件的形状从这一帧起变了"
                f"（画这一帧的新样子，Ctrl+K 提交）")
    if present == {api.KIND_ADD_BENCH_BOX}:
        return f"第 {step} 帧：下面这些零件放在台面上，用台面框 R 框出来"
    if present == {api.KIND_REMOVE_BENCH_BOX}:
        return f"第 {step} 帧：下面这些零件回到机箱里了，台面框到这里结束，不用画"
    counts = [template.format(n=work.count(kind)) for kind, template in _KIND_COUNTS
              if kind in present]
    return f"第 {step} 帧（对照第 {neighbour} 帧）：" + "、".join(counts)


#: What "完整形状" means, said once -- in the start frame's header -- where
#: every part is drawn for the first time (round 1 item 6, round 1b).
START_NOTE = "（被别的零件挡住的部分也算它的，层级程序会处理）"


def row_view(row: dict, editing: Optional[str] = None, bench: bool = False,
             start: bool = False, blockers: int = 0, confirmed: bool = False) -> dict:
    """What one row shows: title, the log's name, the sentence and the chip.

    ``start`` is the start frame, where nothing "comes back in with" a parent:
    that clause is left off there (what a complete shape is, is said once in
    :func:`card_header`).  ``blockers`` is the header's: a ✔ row does not say
    "直接 Space" while the pane below lists what stops it (U2d round 2).  Pure,
    so the tests read exactly what the delegate paints.
    """
    kind = str(row.get("kind", api.KIND_CONFIRM))
    instance = str(row.get("instance", ""))
    done = bool(row.get("done", False))
    if kind == api.KIND_CONFIRM:
        title = f"这一帧（{instance}）" if instance else "这一帧"
    else:
        title = instance_label(instance, row.get("cls"), row.get("attrs"))
    raw = " / ".join(dict.fromkeys(str(n) for n in (row.get("raw_names") or []) if n))
    sentence = KIND_SENTENCES.get(kind, str(row.get("text", "")))
    if kind == api.KIND_CONFIRM and blockers:
        sentence = f"这一帧不用画，{space_or_blocked(blockers)}"
    elif kind == api.KIND_CONFIRM and confirmed:
        sentence = f"这一帧不用画，{CONFIRMED_TAIL}"
    transition = row.get("transition")
    if kind == api.KIND_SPLIT_KEYFRAME and transition:
        sentence += f"（{state_zh(transition[0])} → {state_zh(transition[1])}）"
    if kind == api.KIND_ADD_SHAPE and row.get("parent") and not start:
        sentence += f"（跟 {row['parent']} 一起装回来的）"
    span = row.get("span")
    if span:
        sentence += f"（这一条跨了第 {span[0]}–{span[-1]} 步：中间的帧没有图像）"
    invisible = row.get("invisible")
    if invisible:
        # A shape exists and a label hides it on this frame (round 3): the trial
        # DB's chassis vanished from step 42 that way, after a stray "4".
        sentence += (f" ⚠ 已设为不可见（{visibility_zh(invisible)}）：这一帧不显示它的"
                     f"形状；选中后按 1 改回可见，或 Ctrl+Z 撤销")
    if editing is not None and instance == editing:
        chip = CHIP_BOXING if bench else CHIP_EDITING
    elif kind in (api.KIND_STATE_ONLY, api.KIND_REMOVE_BENCH_BOX):
        # Neither is drawing: a state that changed, a bench box that ends.
        chip = CHIP_NOTHING
    elif done:
        chip = CHIP_DONE
    elif kind == api.KIND_CONFIRM:
        chip = CHIP_CONFIRM
    else:
        chip = CHIP_TODO
    return {"icon": KIND_ICONS.get(kind, "•"), "title": title,
            "raw": f"日志：{raw}" if raw else "", "sentence": sentence, "chip": chip,
            "done": done, "editing": chip in (CHIP_EDITING, CHIP_BOXING)}


def row_tooltip(row: dict, view: dict, blockers: int = 0, confirmed: bool = False) -> str:
    """The row's tooltip: the session's own words, which name the state
    transition and the program's instruction in full.

    Except on a ✔ row, where the session's English ends "… - confirm the
    frame": Chinese first there, the row's own sentence, and the English
    saying the same thing -- not "confirm" over a pane that blocks it (U2e).
    """
    kind = str(row.get("kind", api.KIND_CONFIRM))
    if kind != api.KIND_CONFIRM:
        return f"{row.get('text', '')}\n[{kind}]"
    if blockers:
        english = f"Nothing to draw here, but {blockers} problem(s) below stop Space"
    elif confirmed:
        english = ("Nothing to draw here, and the frame is confirmed: PgDn for the next "
                   "one; after a change, Space confirms it again")
    else:
        english = str(row.get("text") or "Nothing to draw here - confirm the frame")
    return f"{view['sentence']}\n{english}"


def plain_text(view: dict) -> str:
    """The row as one string: the list item's text, and what a test reads."""
    head = f"{view['icon']}  {view['title']}"
    if view["raw"]:
        head += f" — {view['raw']}"
    return f"{head}  [{view['chip']}]\n{view['sentence']}"


class _RowDelegate(QStyledItemDelegate):
    """Paints a row as title + chip, the log's name, and the sentence.

    Only the rows on screen are painted, and a row's height is a few font
    metrics, so a start frame with sixty items costs no more to build than the
    plain list did.
    """

    PAD = 6
    CHIPS = {
        CHIP_TODO: ((255, 226, 190), (150, 70, 0)),
        CHIP_EDITING: ((47, 50, 56), (255, 232, 64)),
        CHIP_BOXING: ((47, 50, 56), (255, 232, 64)),
        CHIP_DONE: ((214, 240, 218), (30, 110, 50)),
        CHIP_CONFIRM: ((214, 228, 250), (30, 70, 150)),
    }

    @staticmethod
    def _fonts(option) -> tuple[QFont, QFont]:
        bold = QFont(option.font)
        bold.setBold(True)
        small = QFont(option.font)
        small.setPointSizeF(max(7.0, option.font.pointSizeF() - 1.0))
        return bold, small

    def _width(self, option) -> int:
        view = self.parent()
        width = view.viewport().width() if isinstance(view, QListView) else 0
        return max(120, int(width or option.rect.width() or 240))

    def _geometry(self, option, view: dict, width: int) -> dict:
        bold, small = self._fonts(option)
        bm, sm, nm = QFontMetrics(bold), QFontMetrics(small), QFontMetrics(option.font)
        wrap = int(Qt.TextFlag.TextWordWrap)
        chip_w = sm.horizontalAdvance(view["chip"]) + 12
        chip_h = sm.height() + 4
        icon_w = nm.horizontalAdvance(view["icon"]) + 6
        title_room = max(40, width - 2 * self.PAD - chip_w - icon_w - 4)
        title_h = bm.boundingRect(QRect(0, 0, title_room, 1000), wrap,
                                  view["title"]).height()
        text_room = max(40, width - 2 * self.PAD - icon_w)
        raw_h = (sm.boundingRect(QRect(0, 0, text_room, 1000), wrap, view["raw"]).height()
                 if view["raw"] else 0)
        sentence_h = nm.boundingRect(QRect(0, 0, text_room, 1000), wrap,
                                     view["sentence"]).height()
        height = (self.PAD + max(title_h, chip_h) + 2 + raw_h + (2 if raw_h else 0)
                  + sentence_h + self.PAD)
        return {"chip_w": chip_w, "chip_h": chip_h, "icon_w": icon_w,
                "title_room": title_room, "title_h": title_h, "text_room": text_room,
                "raw_h": raw_h, "sentence_h": sentence_h, "height": height}

    def sizeHint(self, option, index) -> QSize:  # noqa: D102
        view = index.data(VIEW_ROLE)
        if not isinstance(view, dict):
            return super().sizeHint(option, index)
        width = self._width(option)
        return QSize(width, self._geometry(option, view, width)["height"])

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: D102
        view = index.data(VIEW_ROLE)
        if not isinstance(view, dict):
            super().paint(painter, option, index)
            return
        rect = option.rect
        geo = self._geometry(option, view, rect.width())
        bold, small = self._fonts(option)
        wrap_left = int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if view["editing"]:
            painter.fillRect(rect, QColor(255, 244, 194))
            painter.fillRect(QRect(rect.left(), rect.top(), 4, rect.height()),
                             QColor(224, 168, 0))
        elif option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(rect, QColor(220, 232, 250))
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(rect, QColor(240, 244, 250))
        grey = view["done"] and not view["editing"]
        fg = QColor(128, 128, 132) if grey else QColor(20, 20, 24)
        x, y = rect.left() + self.PAD, rect.top() + self.PAD
        painter.setFont(option.font)
        painter.setPen(fg)
        painter.drawText(QRect(x, y, geo["icon_w"], geo["title_h"]),
                         int(Qt.AlignmentFlag.AlignLeft), view["icon"])
        tx = x + geo["icon_w"]
        title_font = QFont(bold)
        title_font.setStrikeOut(grey)
        painter.setFont(title_font)
        painter.drawText(QRect(tx, y, geo["title_room"], geo["title_h"]), wrap_left,
                         view["title"])
        painter.setFont(small)
        bg, ink = self.CHIPS.get(view["chip"], ((232, 232, 236), (90, 90, 96)))
        chip = QRectF(rect.right() - self.PAD - geo["chip_w"], y, geo["chip_w"],
                      geo["chip_h"])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*bg))
        painter.drawRoundedRect(chip, geo["chip_h"] / 2.0, geo["chip_h"] / 2.0)
        painter.setPen(QColor(*ink))
        painter.drawText(chip, int(Qt.AlignmentFlag.AlignCenter), view["chip"])
        y += max(geo["title_h"], geo["chip_h"]) + 2
        if view["raw"]:
            painter.setPen(QColor(110, 110, 118))
            painter.drawText(QRect(tx, y, geo["text_room"], geo["raw_h"]), wrap_left,
                             view["raw"])
            y += geo["raw_h"] + 2
        painter.setFont(option.font)
        painter.setPen(fg)
        painter.drawText(QRect(tx, y, geo["text_room"], geo["sentence_h"]), wrap_left,
                         view["sentence"])
        painter.setPen(QPen(QColor(222, 224, 228), 1))
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())
        painter.restore()


#: A compiler problem code looks like ``missing_shape:cpu_cooler.01`` -- one
#: token, a colon, an instance key.  A "how to fix it" sentence has spaces.
_CODE_CHARS = set("abcdefghijklmnopqrstuvwxyz_")


def _is_code(problem: str) -> bool:
    """Is this the compiler's own name for a problem rather than a sentence?"""
    head, sep, rest = str(problem).partition(":")
    return bool(sep) and bool(head) and set(head) <= _CODE_CHARS and " " not in head


#: The one problem a click answers by starting to draw (U2d).
MISSING_SHAPE = "missing_shape:"

#: The session writes a "how to fix it" sentence for exactly these codes
#: (``session_review._how_to_fix``), in the order they appear, so they are the
#: only ones a sentence may be paired with.  Pairing them with *every* code put
#: "Draw c.01 on this frame" next to ``bench_missing:b.01`` on any frame that
#: had both, and left the real missing shape showing its raw code.
EXPLAINED_CODES = ("missing_shape:",)

#: Codes an open card row already *is*: a missing shape is the ✚ / ✂ row of
#: that instance, a missing bench box its ▭ row.  The card is the to-do list,
#: so these are not said a second time anywhere (task U2c).
ROW_CODES = ("missing_shape:", "bench_missing:")

#: The pane's title when it lists what a frame has on arrival: the problems
#: the card's rows do not already ask for (task U2c).
ARRIVAL_TITLE = "清单以外的问题 / Problems not on the list above"
REFUSAL_TITLE = "Problems"
#: The line between what stops ``Space`` and what does not (U2d): the notes
#: follow it, greyed.  Which codes are which is ``is_blocking``'s answer.
NOTES_HEADING = "提示（不挡 Space）/ notes -- Space still works"
NOTE_COLOR = QColor(118, 118, 124)
#: An open conflict about the frame on screen, and how to settle it (U2e).
CONFLICT_SENTENCE = "这一帧有未处理的冲突：点顶部 Review → Conflicts，K 保留旧的 / N 采用新的"
#: The input race: an edit landed while Space was confirming.  Transient.
RACE_SENTENCE = "输入刚变了：再按一次 Space"

#: What every other code means, in one place.  ``{what}`` is the part of the
#: code after the colon -- an instance key, sometimes with a part or a second
#: key after it -- which is what the annotator has to go and look at.  Each
#: sentence ends with what to do about it (U2d); a click on the line selects
#: the part, which is where the keys named here act.
PROBLEM_SENTENCES: tuple[tuple[str, str], ...] = (
    ("bench_missing:", "{what}：已拆到台面上但还没有台面框（按 R 拖一个框）"),
    ("shape_size_mismatch:", "{what}：形状和这一帧的画面尺寸对不上，重画一次"),
    ("zorder_cycle:", "{what}：上下层级前后说反了（互相矛盾）→ Ctrl+Z 撤销刚才改的层级"),
    ("zorder_missing:", "{what}：不在层级顺序里，会被画在最上面 → 选中它用 Ctrl+↑/↓ 放到对的层"),
    ("empty_visible:", "{what}：这一帧里它被完全挡住了 → 选中它按 3（完全遮挡）"),
    ("pose_segment_ambiguous:", "{what}：跨了不止一个位姿段 → 先定好位姿分段"
                                "（接受/拒绝断点提示，或 Ctrl+Shift+B）"),
    ("missing_shape:", "{what}：这一帧还缺形状，把它画出来"),
    # The refusals that are not the compiler's (U2e,
    # tda.core.truth_refusals.blocking_reasons).  A conflict line is clicked
    # to Review with that conflict selected; {what} is "#id part".
    (OPEN_CONFLICT, CONFLICT_SENTENCE + "（{what}）"),
    # Said after a refused Space too, which has just queued it: true either way.
    (FROZEN_DISAGREEMENT, "{what}：确认过的形状后来在别的帧被改了，和这一帧对不上 → 到 Review → "
                          "Conflicts：K 保留旧的 / N 采用新的（冲突队列里还没有它就先按 F5）"),
    (INPUTS_CHANGED, RACE_SENTENCE),
)


def explain_code(code: str) -> str:
    """The human sentence for a problem code, or the code itself when unknown."""
    found = conflict_of(code)
    for prefix, template in PROBLEM_SENTENCES:
        if code.startswith(prefix):
            what = f"#{found[0]} {found[1]}".strip() if found else code[len(prefix):]
            return template.format(what=what)
    return code


def instance_of(code: str) -> str:
    """The instance key a code is about: before any ``/`` part or ``,`` partner.

    A conflict line is about the conflict, not a part to select: ``""``.
    """
    if conflict_of(code) is not None:
        return ""
    tail = code.split(":", 1)[1] if ":" in code else ""
    return tail.split("/", 1)[0].split(",", 1)[0].strip()


def _clickable(row: dict) -> bool:
    """Does a click on this pane line go somewhere (a part, or a conflict)?"""
    return bool(row["instance"]) or conflict_of(row["code"]) is not None


def pair_problems(problems: list[str]) -> list[dict]:
    """``{"text", "code", "instance"}`` per problem, each one listed once.

    ``confirm_frame`` emits the compiler's codes **and** one sentence per code
    it knows how to explain (:data:`EXPLAINED_CODES`), in the same order.  Only
    those codes consume a sentence; every other one is explained from
    :data:`PROBLEM_SENTENCES` here, and a code nobody has written a sentence for
    is shown as it is rather than wearing somebody else's.

    A refusal **opens** with the session's own sentence ("frame 13/scan/step 14
    cannot be verified: ...", "conflict(s) 3, 5 are still open"), which explains
    no code -- it restates them.  Popping sentences off one flat list handed it
    to the first ``missing_shape:`` and pushed every real sentence one code
    along, leaving the last one orphaned under nobody's instance.  So only what
    comes *after* the first code can be paired; anything before it is its own
    row.
    """
    first_code = next((i for i, p in enumerate(problems) if _is_code(p)),
                      len(problems))
    lead, rest = problems[:first_code], problems[first_code:]
    codes = [p for p in rest if _is_code(p)]
    sentences = [p for p in rest if not _is_code(p)]
    rows: list[dict] = [{"text": text, "code": "", "instance": ""}
                        for text in lead]
    for code in codes:
        if code.startswith(EXPLAINED_CODES) and sentences:
            text = sentences.pop(0)
        else:
            text = explain_code(code)
        rows.append({"text": text, "code": code, "instance": instance_of(code)})
    rows.extend({"text": text, "code": "", "instance": ""} for text in sentences)
    return rows


def _is_note(row: dict) -> bool:
    """A compiler code the confirmation accepts: shown, but it stops nothing."""
    return bool(row["code"]) and not is_blocking(row["code"])


def _problem_item(row: dict) -> QListWidgetItem:
    """One line of the pane: the sentence, the code in the tooltip."""
    item = QListWidgetItem(row["text"])
    item.setToolTip(row["code"] or row["text"])
    item.setData(INSTANCE_ROLE, row["instance"])
    item.setData(CODE_ROLE, row["code"])
    return item


def unlisted_problems(problems: list[str], rows: list[dict]) -> list[str]:
    """The compiler codes of a frame that its card's open rows do not ask for.

    Arriving on D13/scan step 34 said "13 problem(s) — 见任务卡" over a card
    with one ✂ row and no problems pane at all (task U2c): the status line
    pointed at nothing.  One of the thirteen *was* that row -- its missing
    shape -- and is left out here, because the row already says it; the other
    twelve (parts no row mentions) are what the pane is for.  Anything that is
    not a compiler code -- a conflict that would not resolve, a re-check that
    failed on another step -- is not about this frame's list and is dropped.
    """
    open_rows = {str(r.get("instance", "")) for r in rows if not r.get("done", False)}
    return [str(p) for p in problems if _is_code(p)
            and not (str(p).startswith(ROW_CODES) and instance_of(str(p)) in open_rows)]


class TaskCardPanel(QWidget):
    """The per-frame instruction list, and the problems its rows do not cover."""

    #: An item was activated: the canvas should start editing this instance.
    sigRequestEdit = Signal(str)
    #: The pointer rests on the row of this instance; ``""`` when it left.
    sigHover = Signal(str)
    #: A row that is not work was clicked; the payload is what it means.
    sigExplain = Signal(str)
    #: A frame's problems arrived, and this many of the pane's lines stop
    #: ``Space`` (:meth:`problem_count`; ``0``: none do -- the pane is hidden or
    #: holds only notes).  The status line says "见任务卡" from this and from
    #: nothing else, so it never points at a pane that is not there, nor at
    #: notes nothing has to be done about (U2d).
    sigProblemsShown = Signal(int)
    #: A problem in the pane was clicked: select this part where the keys
    #: that fix it act -- the instance table (U2d).
    sigPickInstance = Signal(str)
    #: A conflict line in the pane was clicked: open Review on this conflict (U2e).
    sigOpenConflict = Signal(int)

    def __init__(self, session: Optional[api.SessionLike] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session: Optional[api.SessionLike] = None
        self._problems: list[str] = []
        #: Inside :meth:`confirm`: a refusal's problems are kept for it to show.
        self._confirming = False
        self._rows: list[dict] = []
        #: The card rows as the session last gave them (``rows()``).
        self._card: list[dict] = []
        self._editing: Optional[str] = None
        self._editing_bench = False
        #: Is the frame on screen the start frame (no neighbour)?
        self._start = False
        #: What the header was built from at the last refresh, so that a change
        #: in the pane can re-say it without asking the session again.
        self._header_args: dict = {}
        #: :meth:`problem_count` as the header and the rows last said it.
        self._blockers = 0
        #: Has somebody confirmed the frame on screen (U2e round 2)?
        self._confirmed = False
        self._hovered = ""
        #: ``(instance, monotonic time)`` of the last click on a row.
        self._last_click: tuple[str, float] = ("", 0.0)

        #: What this frame's list *is*, in one sentence (ruling U2b-3).
        self.header = QLabel("")
        self.header.setWordWrap(True)
        self.header.setToolTip(ICON_LEGEND)
        self.header.setStyleSheet("QLabel { font-weight: bold; padding: 3px 4px; "
                                  "background: #eef2f8; border-radius: 3px; }")

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._list.setResizeMode(QListView.ResizeMode.Adjust)
        self._list.setUniformItemSizes(False)
        self._list.setMinimumWidth(160)
        self._list.setItemDelegate(_RowDelegate(self._list))
        # A click on a row starts work on the canvas; the keyboard stays there
        # (the next B is a brush, not a list keystroke).
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setMouseTracking(True)
        self._list.itemActivated.connect(self._on_item_activated)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemEntered.connect(self._on_item_entered)
        self._list.viewportEntered.connect(lambda: self._set_hover(""))
        self._list.viewport().installEventFilter(self)

        self._problems_label = QLabel(REFUSAL_TITLE)
        self._problems_list = QListWidget()
        self._problems_list.setMaximumHeight(90)     # it scrolls; 60 of them fit
        self._problems_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._problems_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._problems_list.itemClicked.connect(self._on_problem_clicked)
        self._problems_list.itemActivated.connect(self._on_problem_activated)
        self._problems_label.setVisible(False)
        self._problems_list.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        title = QLabel("任务卡 / Task card")
        title.setToolTip(ICON_LEGEND)
        layout.addWidget(title)
        layout.addWidget(self.header)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._problems_label)
        layout.addWidget(self._problems_list)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if session is not None:
            self.set_session(session)

    # -- wiring -------------------------------------------------------------
    def set_session(self, session: Optional[api.SessionLike]) -> None:
        """Attach a session (or ``None``) and rebuild the card."""
        if self._session is not None:
            self._session.sigFrameChanged.disconnect(self._on_frame_changed)
            self._session.sigProblems.disconnect(self._on_problems)
        self._session = session
        if session is not None:
            session.sigFrameChanged.connect(self._on_frame_changed)
            session.sigProblems.connect(self._on_problems)
        self._hide_problems()
        self.refresh()
        # The session announced the frame it opened on before this panel was
        # listening (the window is built after ``open``), so ask once.
        finder = getattr(session, "current_problems", None)
        if session_is_open(session) and callable(finder):
            self._show_arrival([str(p) for p in finder()])

    def list_widget(self) -> QListWidget:
        """The underlying list, for the main window's layout and for tests."""
        return self._list

    # -- content ------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the card from ``session.task_card()``.

        The only call to the session: the rows are kept (:meth:`rows`) so the
        window's guide can read them without asking the session to diff the
        frame again.
        """
        self._list.clear()
        opened = session_is_open(self._session)
        rows = self._session.task_card() if opened else []
        self._card = [dict(r) for r in rows]
        step, neighbour = self._frame_pair() if opened else (None, None)
        self._start = opened and neighbour is None
        steps = list(self._session.steps()) if opened else []
        kinds = [str(r.get("kind", api.KIND_CONFIRM)) for r in rows]
        done = [bool(r.get("done", False)) for r in rows]
        # confirmed already: the guide's "已经确认 ✓", and now the header's (U2e)
        self._confirmed = bool(opened and step is not None and self._session.frame_status(
            step) in api.CONFIRMED_STATUSES)
        self._header_args = (dict(step=step, neighbour=neighbour, kinds=kinds,
                                  first=min(steps) if steps else None,
                                  last=max(steps) if steps else None, done=done,
                                  confirmed=self._confirmed)
                             if opened else {})
        self._blockers = self.problem_count()
        self._paint_header()
        first_open = -1
        for i, row in enumerate(rows):
            kind = str(row.get("kind", api.KIND_CONFIRM))
            instance = str(row.get("instance", ""))
            view = row_view(row, self._editing, self._editing_bench, self._start,
                            self._blockers, self._confirmed)
            item = QListWidgetItem(plain_text(view))
            item.setData(INSTANCE_ROLE, instance)
            item.setData(KIND_ROLE, kind)
            item.setData(VIEW_ROLE, view)
            item.setToolTip(row_tooltip(row, view, self._blockers, self._confirmed))
            font = item.font()
            done = bool(row.get("done", False))
            if done:
                font.setStrikeOut(True)
                item.setForeground(QBrush(DONE_COLOR))
            elif first_open < 0:
                first_open = i
                font.setBold(True)
            item.setFont(font)
            self._list.addItem(item)
        self._list.setCurrentRow(first_open)

    def _frame_pair(self) -> tuple[int, Optional[int]]:
        """``(step on screen, the step the card is diffed against or None)``."""
        key = self._session.current()
        finder = getattr(self._session, "task_neighbour", None)
        try:
            neighbour = finder() if callable(finder) else None
        except Exception:  # noqa: BLE001 - a header is not worth a failure
            neighbour = None
        return int(key.step), neighbour

    def header_text(self) -> str:
        """The sentence above the list."""
        return self.header.text()

    def _paint_header(self) -> None:
        text = (card_header(**self._header_args, blockers=self._blockers)
                if self._header_args else "")
        self.header.setText(text)
        self.header.setVisible(bool(text))

    def _repaint_rows(self) -> None:
        """Rewrite the rows whose painted view changed; the list is not rebuilt.

        The items are the card's rows in order (:meth:`refresh` adds one per
        row), so each is re-derived from the row at its own index.
        """
        for index in range(min(self._list.count(), len(self._card))):
            item = self._list.item(index)
            old = item.data(VIEW_ROLE)
            if not isinstance(old, dict):
                continue
            view = row_view(self._card[index], self._editing, self._editing_bench,
                            self._start, self._blockers, self._confirmed)
            if view != old:
                item.setData(VIEW_ROLE, view)
                item.setText(plain_text(view))
                item.setToolTip(row_tooltip(self._card[index], view, self._blockers,
                                            self._confirmed))

    def _sync_blockers(self) -> None:
        """The pane changed: the header and a ✔ row say Space only if nothing blocks it.

        The same :meth:`problem_count` the guide and the status line read
        (U2d round 2), so all three agree on the frame on screen.
        """
        count = self.problem_count()
        if count == self._blockers:
            return
        self._blockers = count
        self._paint_header()
        self._repaint_rows()

    def follow_status(self, status: str, problems: Optional[list] = None) -> None:
        """The frame on screen now has ``status``: re-say "已经确认 ✓" only if it holds.

        :meth:`refresh` reads whether the frame is confirmed once per frame, and
        the frame can stop being confirmed while the annotator stands on it --
        the sweeper's re-check demotes it, or turns it into a conflict. The
        window hands the new status over when a queue moves (task U2f); the
        header and the rows are re-derived from what the card already holds,
        without asking the session for the card again.

        ``problems`` is what the frame would be refused for now
        (``session.current_problems()``), read with the status: a re-check
        that queued a conflict about this frame is a reason Space refuses, and
        without it in the pane the header would drop "已经确认 ✓" for "直接
        Space" -- a promise Space would break.
        """
        if problems is not None and not self._confirming:
            self._problems = [str(p) for p in problems]
            self._show_arrival(self._problems)
        confirmed = bool(self._header_args) and str(status) in api.CONFIRMED_STATUSES
        if confirmed == self._confirmed:
            return
        self._confirmed = confirmed
        self._header_args["confirmed"] = confirmed
        self._paint_header()
        self._repaint_rows()

    def rows(self) -> list[dict]:
        """The rows on the card, as the session gave them at the last refresh."""
        return [dict(r) for r in self._card]

    def set_editing(self, instance: Optional[str], bench: bool = False) -> None:
        """Mark the row being worked on "✎ 正在画" (or "✎ 正在框" for a bench box).

        Only the rows whose chip changes are rewritten; the list is not rebuilt.
        """
        instance = str(instance) if instance else None
        if instance == self._editing and bool(bench) == self._editing_bench:
            return
        self._editing, self._editing_bench = instance, bool(bench)
        self._repaint_rows()

    def editing_instance(self) -> Optional[str]:
        """The row marked as being worked on, or ``None``."""
        return self._editing

    def row_texts(self) -> list[str]:
        """Every row as the plain text of what is painted."""
        return [self._list.item(i).text() for i in range(self._list.count())]

    def current_index(self) -> int:
        """Row of the first item that is not done, or ``-1`` when all are."""
        for row in range(self._list.count()):
            if not self._list.item(row).font().strikeOut():
                return row
        return -1

    def current_instance(self) -> Optional[str]:
        """Instance of the highlighted row, or ``None``."""
        item = self._list.currentItem()
        return None if item is None else str(item.data(INSTANCE_ROLE))

    def select_instance(self, instance: str) -> bool:
        """Highlight the row of ``instance``; ``False`` when the card has none.

        The window uses it to put the highlight back after refusing an
        activation: Qt has already moved it by then, and a card pointing at an
        instance that is not the one being edited is the wrong instruction.
        """
        for row in range(self._list.count()):
            if str(self._list.item(row).data(INSTANCE_ROLE)) == instance:
                self._list.setCurrentRow(row)
                return True
        return False

    # -- actions ------------------------------------------------------------
    # ``commit(scope)`` is gone: it called ``session.commit_edit`` straight, so
    # the window never cleared the layer, never dropped the crash sidecar and
    # never showed the scope bar.  ``Enter`` and the palette's 提交 both reach
    # ``MainWindow.act_commit`` now; ``confirm`` below is what ``Space`` calls.

    def confirm(self) -> bool:
        """Confirm the frame; on refusal show the problems the session sent.

        Only the problems that arrived during *this* call are shown: a refusal
        always comes with a fresh ``sigProblems`` (see :class:`api.SessionLike`),
        and showing an older list would attribute another frame's problems to
        this one.  A refusal shows all of them, rows included: that list is
        the answer to "why not?".  A confirmation steps to the next frame, and
        what arrived during the call is *that* frame's, shown as on any
        arrival (task U2c).
        """
        if self._session is None:
            return False
        self._problems = []
        self._confirming = True
        try:
            ok = self._session.confirm_frame()
        finally:
            self._confirming = False
        if ok:
            self._show_arrival(self._problems)
        else:
            self._show_problems(self._problems, REFUSAL_TITLE)
        return ok

    def problem_count(self) -> int:
        """How many things in the pane stop ``Space`` -- not how many lines are shown.

        The refusal's opening line restates the codes listed under it, so
        counting it as well told the annotator "3 problem(s)" for two missing
        shapes.  When no code under it blocks -- an open conflict, which the
        compiler cannot name -- it *is* the problem, and counts.  A code the
        confirmation accepts (``empty_visible``, ``zorder_missing``, ...) is a
        note and stops nothing, so it is not counted (U2d): which codes block
        is :func:`tda.core.truth_verify.is_blocking`, the test
        ``confirm_frame`` itself applies.
        """
        if not self._problems_list.isVisibleTo(self):
            return 0
        blocking = [row for row in self._rows if row["code"] and is_blocking(row["code"])]
        return len(blocking) or len([row for row in self._rows if not row["code"]])

    def blocker_hint(self) -> str:
        """What to say instead of "单击一条去处理" when no blocking line is clickable.

        ``""`` when a click on one of them goes somewhere -- a part, or a
        conflict in Review.  Otherwise the first one's own instruction: after
        an input race that is "输入刚变了：再按一次 Space", and clicking it
        would do nothing (U2e).
        """
        if not self._problems_list.isVisibleTo(self):
            return ""
        blocking = [row for row in self._rows if row["code"] and is_blocking(row["code"])]
        if any(_clickable(row) for row in blocking):
            return ""
        lines = blocking or [row for row in self._rows if not row["code"]]
        return lines[0]["text"] if lines else ""

    def problems(self) -> list[str]:
        """The problems currently on display (empty when none are shown)."""
        if not self._problems_list.isVisibleTo(self):
            return []
        return [item.text() for item in self._problem_items()]

    def _problem_items(self) -> list[QListWidgetItem]:
        """The pane's lines that are problems: the notes' heading is not one."""
        items = (self._problems_list.item(i) for i in range(self._problems_list.count()))
        return [item for item in items if not item.data(NOTES_HEADING_ROLE)]

    def problems_visible(self) -> bool:
        """Whether the problem list is on display."""
        return self._problems_list.isVisibleTo(self)

    # -- keys ---------------------------------------------------------------
    # There is exactly one key map, and it is not here: the main window's
    # ``tda.ui.app_actions.ACTIONS`` table owns every binding, per mode, and
    # calls the plain methods above.  A second table in the panel is how ``Ctrl+K``
    # on this list came to raise out of ``keyPressEvent`` in Steps mode and how
    # ``Enter`` in Review committed with keyframe scope past ``act_commit``.


    # -- slots --------------------------------------------------------------
    def _on_item_activated(self, item: QListWidgetItem) -> None:
        """Report the request; the window decides whether the edit may start.

        The panel used to call ``begin_edit`` itself and then emit, which made
        it impossible to refuse: by the time the window heard about it the
        previous instance's uncommitted pixels were already gone.
        """
        instance = str(item.data(INSTANCE_ROLE))
        if not instance:
            return
        # A double-click is a click first, and the click has already asked
        # for this row: one gesture, one request.
        last, at = self._last_click
        if last == instance and _now() - at <= _double_click_s():
            return
        kind = item.data(KIND_ROLE)
        if kind is not None and str(kind) not in EDITABLE_KINDS:
            self._explain(item, instance)
            return
        self.sigRequestEdit.emit(instance)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """One click starts the work a row asks for (ruling U2b-3).

        A row that is not work -- a state change, a frame with nothing to draw
        -- says what it means instead of starting an edit nobody asked for.
        """
        instance = str(item.data(INSTANCE_ROLE) or "")
        kind = str(item.data(KIND_ROLE) or "")
        if not instance:
            return
        self._last_click = (instance, _now())
        if kind in EDITABLE_KINDS:
            self.sigRequestEdit.emit(instance)
            return
        self._explain(item, instance)

    def _explain(self, item: QListWidgetItem, instance: str) -> None:
        """Say what a row that is not work means (a click, or a double-click)."""
        view = item.data(VIEW_ROLE)
        sentence = view.get("sentence", "") if isinstance(view, dict) else ""
        self.sigExplain.emit(f"{instance}：{sentence}" if sentence else instance)

    # -- hover --------------------------------------------------------------
    def _on_item_entered(self, item: QListWidgetItem) -> None:
        self._set_hover(str(item.data(INSTANCE_ROLE) or ""))

    def _set_hover(self, instance: str) -> None:
        if instance != self._hovered:
            self._hovered = instance
            self.sigHover.emit(instance)

    def hovered(self) -> str:
        """The instance whose row the pointer rests on, ``""`` for none."""
        return self._hovered

    def eventFilter(self, obj, event) -> bool:  # noqa: D102 - the list's viewport
        if obj is self._list.viewport() and event.type() == QEvent.Type.Leave:
            self._set_hover("")
        return super().eventFilter(obj, event)

    def _on_frame_changed(self, _key: object) -> None:
        self._problems = []
        self._hide_problems()
        self.refresh()
        # The rows under the pointer are new ones: whatever it was resting on
        # has gone, and so has the outline the window drew for it.
        self._set_hover("")

    def _on_problems(self, problems: list) -> None:
        self._problems = [str(p) for p in problems]
        if not self._confirming:
            self._show_arrival(self._problems)

    def _show_arrival(self, problems: list[str]) -> None:
        """Put what a frame arrived with in the pane, minus the card's own rows.

        Only a frame's own compiler codes decide the pane: a list with none of
        them in it is a conflict or a failed re-check the window reports in
        its own words, and it must not wipe the problems of the frame that is
        on screen.
        """
        if problems and not any(_is_code(p) for p in problems):
            return
        shown = unlisted_problems(problems, self._card)
        if shown:
            self._show_problems(shown, ARRIVAL_TITLE)
        else:
            self._hide_problems()
        self.announce()

    def announce(self) -> None:
        """Say again what the pane holds that stops Space (:attr:`sigProblemsShown`).

        For a listener that attached after the pane was filled: the window's
        status line, whose card read the first frame's problems while the
        window was still being built (U2d).
        """
        self.sigProblemsShown.emit(self.problem_count())

    def _show_problems(self, problems: list[str], title: str = REFUSAL_TITLE) -> None:
        """One row per problem: the sentence, with the code in the tooltip.

        The session emits the compiler's codes *and* the "how to fix it"
        sentences in one list, so a start frame with 60 missing shapes listed
        every part twice -- once as ``missing_shape:cover.01`` and once as
        "Draw cover.01 on this frame".  They are paired back up here: the
        sentence is what the annotator reads, the code is what they quote in a
        bug report, and the instance is what a click jumps to.
        """
        self._problems_list.clear()
        # What stops Space first, then -- under their own heading, greyed --
        # the notes the confirmation accepts (U2d).  An arrival that listed
        # ``empty_visible`` above a missing shape read as two equal chores.
        rows = pair_problems(problems)
        firsts = [row for row in rows if not _is_note(row)]
        notes = [row for row in rows if _is_note(row)]
        self._rows = firsts + notes
        for row in firsts:
            self._problems_list.addItem(_problem_item(row))
        if notes:
            heading = QListWidgetItem(NOTES_HEADING)
            heading.setFlags(Qt.ItemFlag.NoItemFlags)
            heading.setData(NOTES_HEADING_ROLE, True)
            heading.setForeground(QBrush(NOTE_COLOR))
            font = heading.font()
            font.setItalic(True)
            heading.setFont(font)
            self._problems_list.addItem(heading)
        for row in notes:
            item = _problem_item(row)
            item.setForeground(QBrush(NOTE_COLOR))
            self._problems_list.addItem(item)
        visible = bool(problems)
        self._problems_label.setText(title)
        self._problems_label.setVisible(visible)
        self._problems_list.setVisible(visible)
        self._sync_blockers()

    def problem_rows(self) -> list[dict]:
        """``{"text", "code", "instance", "note"}`` per problem shown, in order.

        ``note`` is a code the confirmation accepts (U2d): those come last,
        under :data:`NOTES_HEADING`, which is not a row.
        """
        return [{"text": item.text(), "code": item.toolTip(),
                 "instance": str(item.data(INSTANCE_ROLE) or ""),
                 "note": _is_note({"code": str(item.data(CODE_ROLE) or "")})}
                for item in self._problem_items()]

    def notes_heading_shown(self) -> bool:
        """Is the "提示（不挡 Space）" line between the problems and the notes?"""
        return any(self._problems_list.item(i).data(NOTES_HEADING_ROLE)
                   for i in range(self._problems_list.count()))

    def activate_problem(self, instance: str) -> None:
        """Jump to the card item a problem is about (a click in the pane)."""
        if instance:
            self.select_instance(instance)

    def _on_problem_clicked(self, item: QListWidgetItem) -> None:
        """One click on a problem goes to deal with it (U2d).

        The guide says "单击一条去处理" when only this pane stands between the
        annotator and ``Space``, and a click used to do nothing at all for a
        part the card has no row for -- which is every problem the pane shows
        on arrival.  A missing shape starts drawing that part, exactly as a
        card row does; any other problem selects its part (on the card and in
        the instance table, where the visibility keys, ``Ctrl+↑/↓`` and ``R``
        act) and says its sentence in the status line.
        """
        instance = str(item.data(INSTANCE_ROLE) or "")
        conflict = conflict_of(str(item.data(CODE_ROLE) or ""))
        if conflict is not None:
            # Settled in Review, where K / N act on the selected conflict (U2e).
            self.sigOpenConflict.emit(conflict[0])
            return
        if not instance:
            return
        self.activate_problem(instance)
        if str(item.data(CODE_ROLE) or "").startswith(MISSING_SHAPE):
            self.sigRequestEdit.emit(instance)
            return
        self.sigPickInstance.emit(instance)
        self.sigExplain.emit(item.text())

    def _on_problem_activated(self, item: QListWidgetItem) -> None:
        """Double click / Enter on a problem: start editing that instance.

        The window decides whether the edit may begin, as for every other way
        of asking.
        """
        instance = str(item.data(INSTANCE_ROLE) or "")
        if instance:
            self.select_instance(instance)
            self.sigRequestEdit.emit(instance)

    def _hide_problems(self) -> None:
        self._problems_list.clear()
        self._rows = []
        self._problems_label.setVisible(False)
        self._problems_list.setVisible(False)
        self._sync_blockers()
