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

from tda.ui import session_api as api
from tda.ui.class_names import instance_label, state_zh
from tda.ui.panels import session_is_open

__all__ = ["CHIP_DONE", "CHIP_EDITING", "CHIP_TODO", "KIND_ICONS", "KIND_SENTENCES",
           "TaskCardPanel", "card_header", "row_view"]

INSTANCE_ROLE = int(Qt.ItemDataRole.UserRole)
#: The painted view of a row (:func:`row_view`), for the delegate.
VIEW_ROLE = INSTANCE_ROLE + 1
KIND_ROLE = INSTANCE_ROLE + 2

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
    "✔ 这一帧没有要画的：直接 Space 确认\n"
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
                last: Optional[int] = None) -> str:
    """The sentence above the list: what *this* frame's list is (ruling U2b-3).

    Built from the rows' kinds (round 2): "多了…要画回去" over rows that only
    change a state was an instruction the rows contradicted.  ``first`` /
    ``last`` are the view's first and last step, which is what tells the start
    frame of the reverse walk (the machine taken apart) from the start of a
    forward one (not taken apart yet).
    """
    if step is None:
        return ""
    kinds = [str(k) for k in (kinds if kinds is not None else [api.KIND_ADD_SHAPE])]
    work = [k for k in kinds if k != api.KIND_CONFIRM]
    if neighbour is None:
        if last is not None and int(step) == int(last):
            where = "起点，已经拆完的样子"
        elif first is not None and int(step) == int(first):
            where = "起点，还没开始拆的样子"
        else:
            where = "起点"
        if not work:
            return f"第 {step} 帧（{where}）：这一帧的零件都画好了，直接 Space 确认"
        # Once, here, rather than on every row of a start card that can list
        # sixty parts (round 1b).
        return (f"第 {step} 帧（{where}）：把这一帧里还看得到的零件都画出来，"
                f"每个都画完整形状{START_NOTE}")
    present = set(work)
    if not present:
        return f"第 {step} 帧：这一帧不用画，直接 Space"
    if present == {api.KIND_ADD_SHAPE}:
        if int(neighbour) > int(step):
            return (f"第 {step} 帧：比第 {neighbour} 帧多了下面这些零件"
                    f"（刚被拆掉的，要把它画回去）")
        return f"第 {step} 帧：和第 {neighbour} 帧比，下面这些零件多出来了，要画出来"
    if present == {api.KIND_STATE_ONLY}:
        return f"第 {step} 帧：这一帧只有状态变化（拧紧/插上…），不用画，直接 Space"
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
             start: bool = False) -> dict:
    """What one row shows: title, the log's name, the sentence and the chip.

    ``start`` is the start frame, where nothing "comes back in with" a parent:
    that clause is left off there (what a complete shape is, is said once in
    :func:`card_header`).  Pure, so the tests read exactly what the delegate
    paints.
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
    transition = row.get("transition")
    if kind == api.KIND_SPLIT_KEYFRAME and transition:
        sentence += f"（{state_zh(transition[0])} → {state_zh(transition[1])}）"
    if kind == api.KIND_ADD_SHAPE and row.get("parent") and not start:
        sentence += f"（跟 {row['parent']} 一起装回来的）"
    span = row.get("span")
    if span:
        sentence += f"（这一条跨了第 {span[0]}–{span[-1]} 步：中间的帧没有图像）"
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


#: The session writes a "how to fix it" sentence for exactly these codes
#: (``session_review._how_to_fix``), in the order they appear, so they are the
#: only ones a sentence may be paired with.  Pairing them with *every* code put
#: "Draw c.01 on this frame" next to ``bench_missing:b.01`` on any frame that
#: had both, and left the real missing shape showing its raw code.
EXPLAINED_CODES = ("missing_shape:",)

#: What every other code means, in one place.  ``{what}`` is the part of the
#: code after the colon -- an instance key, sometimes with a part or a second
#: key after it -- which is what the annotator has to go and look at.
PROBLEM_SENTENCES: tuple[tuple[str, str], ...] = (
    ("bench_missing:", "{what}：已拆到台面上但还没有台面框（按 R 拖一个框）"),
    ("shape_size_mismatch:", "{what}：形状和这一帧的画面尺寸对不上，重画一次"),
    ("zorder_cycle:", "{what}：层级关系互相矛盾，改掉其中一条"),
    ("zorder_missing:", "{what}：不在层级顺序里，会被画在最上面"),
    ("empty_visible:", "{what}：可见部分是空的，可能被完全遮挡或画到了框外"),
    ("pose_segment_ambiguous:", "{what}：跨了不止一个位姿段，先确认位姿分段"),
    ("missing_shape:", "{what}：这一帧还缺形状，把它画出来"),
)


def explain_code(code: str) -> str:
    """The human sentence for a problem code, or the code itself when unknown."""
    for prefix, template in PROBLEM_SENTENCES:
        if code.startswith(prefix):
            return template.format(what=code[len(prefix):])
    return code


def instance_of(code: str) -> str:
    """The instance key a code is about: before any ``/`` part or ``,`` partner."""
    tail = code.split(":", 1)[1] if ":" in code else ""
    return tail.split("/", 1)[0].split(",", 1)[0].strip()


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


class TaskCardPanel(QWidget):
    """The per-frame instruction list and the problems of a refused confirm."""

    #: An item was activated: the canvas should start editing this instance.
    sigRequestEdit = Signal(str)
    #: The pointer rests on the row of this instance; ``""`` when it left.
    sigHover = Signal(str)
    #: A row that is not work was clicked; the payload is what it means.
    sigExplain = Signal(str)

    def __init__(self, session: Optional[api.SessionLike] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session: Optional[api.SessionLike] = None
        self._problems: list[str] = []
        self._rows: list[dict] = []
        #: The card rows as the session last gave them (``rows()``).
        self._card: list[dict] = []
        self._editing: Optional[str] = None
        self._editing_bench = False
        #: Is the frame on screen the start frame (no neighbour)?
        self._start = False
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

        self._problems_label = QLabel("Problems")
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
        self.header.setText(
            card_header(step, neighbour, kinds, first=min(steps) if steps else None,
                        last=max(steps) if steps else None) if opened else "")
        self.header.setVisible(bool(self.header.text()))
        first_open = -1
        for i, row in enumerate(rows):
            kind = str(row.get("kind", api.KIND_CONFIRM))
            instance = str(row.get("instance", ""))
            view = row_view(row, self._editing, self._editing_bench, self._start)
            item = QListWidgetItem(plain_text(view))
            item.setData(INSTANCE_ROLE, instance)
            item.setData(KIND_ROLE, kind)
            item.setData(VIEW_ROLE, view)
            # The session's own words stay reachable: they name the state
            # transition and the program's instruction in full.
            item.setToolTip(f"{row.get('text', '')}\n[{kind}]")
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
        for index in range(self._list.count()):
            item = self._list.item(index)
            old = item.data(VIEW_ROLE)
            row = next((r for r in self._card
                        if str(r.get("instance", "")) == str(item.data(INSTANCE_ROLE))),
                       None)
            if row is None or not isinstance(old, dict):
                continue
            view = row_view(row, self._editing, self._editing_bench, self._start)
            if view != old:
                item.setData(VIEW_ROLE, view)
                item.setText(plain_text(view))

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
        this one.
        """
        if self._session is None:
            return False
        self._problems = []
        ok = self._session.confirm_frame()
        if ok:
            self._hide_problems()
        else:
            self._show_problems(self._problems)
        return ok

    def problem_count(self) -> int:
        """How many things there are to fix -- not how many lines are shown.

        The refusal's opening line restates the codes listed under it, so
        counting it as well told the annotator "3 problem(s)" for two missing
        shapes.  When it is all there is -- an open conflict, which the
        compiler cannot name -- it *is* the problem, and counts.
        """
        if not self._problems_list.isVisibleTo(self):
            return 0
        return len([row for row in self._rows if row["code"]]) or len(self._rows)

    def problems(self) -> list[str]:
        """The problems currently on display (empty when none are shown)."""
        if not self._problems_list.isVisibleTo(self):
            return []
        return [
            self._problems_list.item(i).text()
            for i in range(self._problems_list.count())
        ]

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

    def _show_problems(self, problems: list[str]) -> None:
        """One row per problem: the sentence, with the code in the tooltip.

        The session emits the compiler's codes *and* the "how to fix it"
        sentences in one list, so a start frame with 60 missing shapes listed
        every part twice -- once as ``missing_shape:cover.01`` and once as
        "Draw cover.01 on this frame".  They are paired back up here: the
        sentence is what the annotator reads, the code is what they quote in a
        bug report, and the instance is what a click jumps to.
        """
        self._problems_list.clear()
        self._rows = pair_problems(problems)
        for row in self._rows:
            item = QListWidgetItem(row["text"])
            item.setToolTip(row["code"] or row["text"])
            item.setData(INSTANCE_ROLE, row["instance"])
            self._problems_list.addItem(item)
        visible = bool(problems)
        self._problems_label.setVisible(visible)
        self._problems_list.setVisible(visible)

    def problem_rows(self) -> list[dict]:
        """``{"text", "code", "instance"}`` per row currently shown."""
        out = []
        for i in range(self._problems_list.count()):
            item = self._problems_list.item(i)
            out.append({"text": item.text(), "code": item.toolTip(),
                        "instance": str(item.data(INSTANCE_ROLE) or "")})
        return out

    def activate_problem(self, instance: str) -> None:
        """Jump to the card item a problem is about (a click in the pane)."""
        if instance:
            self.select_instance(instance)

    def _on_problem_clicked(self, item: QListWidgetItem) -> None:
        self.activate_problem(str(item.data(INSTANCE_ROLE) or ""))

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
