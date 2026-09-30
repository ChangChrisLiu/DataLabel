"""What the annotator should do next, as data: the "现在做什么 / What now" plan.

The second trial (task U2b) ended with an annotator who did not know what the
task card's other four items were, and a double-click that looked like it did
nothing -- it had started an edit, and one word in the status bar said so.
The tool's gestures were all there; nothing on the main window said *which*
one was next.

This module is that sentence and nothing else.  :func:`plan_for` reads a small
snapshot of the window (:class:`GuideFacts`) and answers with a
:class:`GuidePlan`: the phase, the one "现在：…" line that names the exact key
or button, the numbered list of the frame's five steps with the current one
marked, and the palette action worth highlighting.  No Qt, no session: the
window gathers the facts from state it already holds, so asking costs a few
dozen attribute reads, and the plan is a frozen value the panel compares with
the previous one before it repaints anything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from tda.ui import session_api as api

__all__ = [
    "CardItem", "DONE", "GuideFacts", "GuidePlan", "NOW", "TODO", "WARN",
    "PHASES", "plan_for",
]

#: Marker states of one line of the numbered list.
DONE, NOW, TODO, WARN = "done", "now", "todo", "warn"

PHASE_STEPS = "steps"
PHASE_REVIEW = "review"
PHASE_CLOSED = "closed"
PHASE_RAW_MISSING = "raw_missing"
PHASE_NO_IMAGE = "no_image"
PHASE_FLASH = "flash"
#: A ``P`` polygon or a ``Y`` circle half drawn (task U5a): ``Enter`` and
#: ``Esc`` are the shape's until it is filled or dropped.
PHASE_SHAPE = "shape"
PHASE_GHOST = "ghost"
PHASE_WARNING = "warning"
PHASE_SCOPE = "scope"
PHASE_ROI = "roi"
PHASE_BENCH = "bench"
PHASE_DRAW_EMPTY = "draw_empty"
PHASE_DRAW_PIXELS = "draw_pixels"
#: A stored shape loaded and not touched: ``Enter`` has nothing to commit.
PHASE_LOADED = "loaded"
PHASE_PICK = "pick"
#: Every card row is done and the card's pane still lists problems that make
#: ``Space`` refuse (U2d): the guide must not send anybody to that key.
PHASE_BLOCKED = "blocked"
PHASE_CONFIRM = "confirm"
PHASE_CONFIRMED = "confirmed"
PHASES: tuple[str, ...] = (
    PHASE_STEPS, PHASE_REVIEW, PHASE_CLOSED, PHASE_RAW_MISSING, PHASE_NO_IMAGE,
    PHASE_FLASH, PHASE_SHAPE, PHASE_GHOST, PHASE_WARNING, PHASE_SCOPE, PHASE_ROI,
    PHASE_BENCH,
    PHASE_DRAW_EMPTY, PHASE_DRAW_PIXELS, PHASE_LOADED, PHASE_PICK, PHASE_BLOCKED,
    PHASE_CONFIRM, PHASE_CONFIRMED,
)

#: The key and the palette action of each commit scope, for the lines that
#: have to name the *right* one (a ✂ row is committed with ``Ctrl+K``).
_SCOPE_KEYS: dict[str, tuple[str, str]] = {
    api.SCOPE_SPLIT: ("Ctrl+K", "commit_split"),
    api.SCOPE_FRAME_OVERRIDE: ("Alt+Enter", "commit_override"),
}

#: Card kinds that are work to *do* on the canvas.  ``state_only`` and
#: ``confirm`` items are never "the next part to click".
_WORK_KINDS = (api.KIND_ADD_SHAPE, api.KIND_SPLIT_KEYFRAME, api.KIND_ADD_BENCH_BOX)


@dataclass(frozen=True)
class CardItem:
    """One task-card row as the guide needs it."""

    instance: str
    kind: str
    done: bool
    #: What the annotator reads: ``导风罩 cover.01``.
    label: str = ""


@dataclass(frozen=True)
class GuideFacts:
    """Everything the plan depends on; all of it already held by the window."""

    mode: str = "annotate"
    is_open: bool = True
    has_image: bool = True
    #: Why no image could be read at all (the raw drive is missing), or ``""``.
    raw_missing: str = ""
    view: str = "scan"
    step: Optional[int] = None
    #: The frame the card is diffed against; ``None`` on the start frame.
    neighbour: Optional[int] = None
    #: The start frame is the *first* step (browsing forward): the machine
    #: whole, not taken apart (U2b round 2, I3).
    forward_start: bool = False
    frame_confirmed: bool = False
    roi_stored: bool = False
    roi_editing: bool = False
    roi_unanswered: bool = False
    #: Label of the instance being edited, or ``None``.
    editing: Optional[str] = None
    #: The task-card kind of the row being edited (``""`` when it is not on
    #: the card): a ✂ row is committed with ``Ctrl+K``, never ``Enter``.
    editing_kind: str = ""
    layer_pixels: bool = False
    layer_dirty: bool = False
    warning: bool = False
    #: The scope the area warning was raised for (its second press repeats it).
    warning_scope: str = ""
    #: The scope the non-modal bar is suggesting, or ``""``.
    scope: str = ""
    #: That scope in the annotator's words (``它在 机箱 chassis 上面（层级）``).
    scope_text: str = ""
    ghost: bool = False
    #: Label of the part the bench box is armed for, or ``None``.
    bench: Optional[str] = None
    flashing: bool = False
    sam_ready: bool = True
    items: tuple[CardItem, ...] = ()
    #: How many problems in the task card's pane stop ``Space``: the codes
    #: ``confirm_frame`` refuses on (``tda.core.truth_verify.is_blocking``),
    #: never the notes it accepts.  The window reads it off the card (U2d).
    blockers: int = 0
    #: When no blocking line in the pane can be clicked, the first one's own
    #: instruction ("输入刚变了：再按一次 Space"); ``""`` when one can (U2e).
    blocker_hint: str = ""
    #: The filled shape half drawn (task U5a): ``"polygon"``, ``"circle"`` or
    #: ``""``, and how many vertices the polygon has so far.
    shape: str = ""
    shape_vertices: int = 0


@dataclass(frozen=True)
class GuidePlan:
    """The answer: what to say, and which palette button deserves a border."""

    phase: str
    title: str
    #: The one sentence naming the exact next key or button.
    now: str
    #: ``(marker state, text)`` per line of the numbered list; empty outside
    #: the per-frame flow (Review, Steps, no image).
    steps: tuple[tuple[str, str], ...] = ()
    #: :data:`tda.ui.app_actions.ACTIONS` name of the suggested button, or ``""``.
    action: str = ""
    extra: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# the five steps of a frame
# --------------------------------------------------------------------------- #
ROI_NOW = "回答机箱范围（ROI）：拖紫红框的白色小方块调整，Enter 保存 / Esc 先跳过"
#: A stored ROI opened to check it: Esc keeps it, it skips nothing (round 2).
ROI_EDIT_NOW = "机箱范围（ROI）：拖白色小方块调整，Enter 保存 / Esc 不改"
ROI_DONE = "机箱范围（ROI）✓"
ROI_NONE = "机箱范围（ROI）：这一段不用 ✓"
ROI_WARN = ("机箱范围（ROI）还没确认：画布下方点「确认建议框」，或 Shift+R 重画"
            "（不挡你继续）")
PICK = "在右边任务卡上单击一个零件"
DRAW = ("画：S 在零件上点一下 / X 拖框；B 画笔补、E 橡皮修边；P 连点填一块；"
        "Y 圆形：拖一下画一颗螺丝")
#: The filled shapes half drawn (task U5a): the "now" line and step ③.
POLYGON_NOW = ("现在：多边形已点 {n} 个点 — 沿边接着点；Enter / 双击 / 点回第一个点 填上，"
               "Backspace 删一个点，Esc 取消")
POLYGON_DRAW = "多边形 {n} 个点：Enter / 双击 / 点回起点 填上"
CIRCLE_NOW = "现在：拖到零件边缘松手，圆就填进去；Esc 取消"
CIRCLE_DRAW = "圆形：拖到边缘松手"
DRAW_BENCH = "在台面上拖一个框框住它（台面框 R），松手就存好"
COMMIT = "Enter 提交（形状从这帧起变了用 Ctrl+K；只这一帧特殊用 Alt+Enter）"
COMMIT_SPLIT = "Ctrl+K 提交（从这帧起新版本）— 这一条不要按 Enter"
CONFIRM = "全部画完：Space 确认整帧，自动退到上一帧"
#: The same line while the card's pane lists what stops Space (U2d).
CONFIRM_BLOCKED = "先处理任务卡下面的 {n} 个问题（它们挡住 Space）"
NOTHING_TO_DRAW = "这一帧不用画"


def _label(item: CardItem) -> str:
    return item.label or item.instance


_BILINGUAL_SPLIT = re.compile(r"\s*/ (?=[A-Za-z])")
_CJK = re.compile(r"[一-鿿]")


def _chinese_half(text: str) -> str:
    """``"中文 / English — path"`` -> ``"中文"``; anything else unchanged."""
    head = _BILINGUAL_SPLIT.split(text, 1)[0].strip()
    return head if _CJK.search(head) else text


def _title(facts: GuideFacts) -> str:
    step = facts.step
    if step is None:
        return "现在做什么"
    if facts.neighbour is None:
        look = "还没开始拆的样子" if facts.forward_start else "已经拆完的样子"
        return f"第 {step} 帧（起点：{look}）"
    return f"第 {step} 帧（对照第 {facts.neighbour} 帧）"


def _roi_line(facts: GuideFacts) -> tuple[str, str]:
    if facts.roi_editing:
        return (NOW, ROI_EDIT_NOW if facts.roi_stored else ROI_NOW)
    if facts.roi_stored:
        return (DONE, ROI_DONE)
    if facts.roi_unanswered:
        return (WARN, ROI_WARN)
    return (DONE, ROI_NONE)


def _open_work(facts: GuideFacts) -> list[CardItem]:
    return [item for item in facts.items if not item.done and item.kind in _WORK_KINDS]


def _steps(facts: GuideFacts, at: int, *, pick_text: str = "", draw_text: str = "",
           commit_text: str = "", nothing: bool = False) -> tuple:
    """The five lines, with ``at`` the one being done now (``-1``: none).

    The last one says Space only while nothing in the card's pane stops it,
    in every phase: "全部画完：Space 确认" over a pane of blockers was a promise
    the confirmation would break (U2d round 2).
    """
    texts = [
        None,
        pick_text or PICK,
        draw_text or DRAW,
        commit_text or COMMIT,
        CONFIRM_BLOCKED.format(n=facts.blockers) if facts.blockers else CONFIRM,
    ]
    if nothing:
        texts[1] = texts[2] = texts[3] = NOTHING_TO_DRAW
    out = [_roi_line(facts)]
    for index in range(1, 5):
        if at < 0 or index < at:
            state = DONE
        elif index == at:
            state = NOW
        else:
            state = TODO
        out.append((state, texts[index]))
    return tuple(out)


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def plan_for(facts: GuideFacts) -> GuidePlan:
    """The phase the annotator is in and the next thing to press."""
    if facts.mode == "steps":
        return GuidePlan(PHASE_STEPS, "Steps 步骤核对",
                         "Steps：核对步骤表和实例表，改完点 Apply；然后点顶部 "
                         "Annotate 回到画布")
    if not facts.is_open:
        return GuidePlan(PHASE_CLOSED, "现在做什么",
                         "这台机器的这个视角没有帧：在顶部换一台机器或换视角（F1–F4）")
    if not facts.has_image and facts.raw_missing:
        # The resolver's own sentence already says what to plug in and that
        # F5 looks again; saying it twice was noise (round 2).  It is
        # bilingual ("中文 / English — path"): the guide speaks Chinese, and the
        # raw-drive bar above the canvas keeps the whole sentence and the path.
        reason = _chinese_half(facts.raw_missing)
        advice = "" if "F5" in reason else " — 接上数据盘后按 F5"
        return GuidePlan(PHASE_RAW_MISSING, _title(facts),
                         f"读不到原始图像：{reason}{advice}")
    if not facts.has_image:
        return GuidePlan(PHASE_NO_IMAGE, _title(facts),
                         f"这一帧在 {facts.view} 视角没有图像 — 按 PgDn / PgUp "
                         f"翻到有图像的帧")
    if facts.mode == "review":
        return GuidePlan(PHASE_REVIEW, "复查 Review",
                         "复查模式，画布只读：在右侧队列选一条 → Enter 接受这一帧，"
                         "R 回标注模式返工；冲突：K 保留旧的，N 采用新的")
    if facts.flashing:
        return GuidePlan(PHASE_FLASH, _title(facts),
                         "正在对照另一帧（画布上不是这一帧）— 松开 Tab 回来")
    return _annotate_plan(facts)


def _annotate_plan(facts: GuideFacts) -> GuidePlan:
    title = _title(facts)
    editing = facts.editing
    if facts.shape and editing:
        # First, as it is first in the Enter / Esc chains: the gesture in the
        # annotator's hand owns both keys until it is filled or dropped (U5a).
        picked = f"已选中：{editing}"
        if facts.shape == "polygon":
            n = int(facts.shape_vertices)
            return GuidePlan(PHASE_SHAPE, title, POLYGON_NOW.format(n=n),
                             _steps(facts, 2, pick_text=picked,
                                    draw_text=POLYGON_DRAW.format(n=n)), "")
        return GuidePlan(PHASE_SHAPE, title, CIRCLE_NOW,
                         _steps(facts, 2, pick_text=picked, draw_text=CIRCLE_DRAW), "")
    if facts.ghost:
        return GuidePlan(PHASE_GHOST, title,
                         "现在：淡蓝色是旧草稿的预览 — Enter 采纳进编辑层再修，"
                         "Esc 取消，Shift+A 换下一个",
                         _steps(facts, 2, pick_text=f"已选中：{editing}" if editing else "",
                                draw_text="旧草稿预览：Enter 采纳 / Esc 取消"),
                         "commit")
    if facts.warning:
        key, action = _SCOPE_KEYS.get(facts.warning_scope, ("Enter", "commit"))
        return GuidePlan(PHASE_WARNING, title,
                         f"现在：面积和这类零件差得多 — 确认没画错就再按一次 {key}，"
                         f"否则 Esc 回去改",
                         _steps(facts, 3, pick_text=f"已选中：{editing}" if editing else "",
                                commit_text=f"面积提示：再按一次 {key} 仍然提交 / Esc 回去改"),
                         action)
    if facts.scope:
        said = facts.scope_text or facts.scope
        return GuidePlan(PHASE_SCOPE, title,
                         f"现在：看画布下方的提示条 — Enter 接受建议（{said}），"
                         f"Alt+Enter 只改这一帧，Ctrl+K 从这帧起新版本",
                         _steps(facts, 3, pick_text=f"已选中：{editing}" if editing else "",
                                commit_text="提交范围建议：Enter / Alt+Enter / Ctrl+K"),
                         "commit")
    if facts.roi_editing:
        # Opening a stored ROI to check it is not a proposal to put off: Esc
        # keeps what is stored (U2b round 2 -- D13/scan already has one).
        now = ("现在：拖边 / 角调整已存的机箱范围（框里按住整体挪），按 Enter 保存；"
               "不改按 Esc（保持原来的）" if facts.roi_stored else
               "现在：调好紫红框（拖白色小方块），按 Enter 保存；暂时不想管就按 Esc")
        return GuidePlan(PHASE_ROI, title, now, _steps(facts, 0), "commit")
    if facts.bench:
        return GuidePlan(PHASE_BENCH, title,
                         f"现在：在台面上拖一个框框住「{facts.bench}」— 松手就存好",
                         _steps(facts, 2, pick_text=f"已选中：{facts.bench}",
                                draw_text=DRAW_BENCH, commit_text="台面框拖完自动保存"),
                         "tool_bench_box")
    if editing:
        picked = f"已选中：{editing}"
        split = facts.editing_kind == api.KIND_SPLIT_KEYFRAME
        if facts.layer_dirty and split:
            # Enter would rewrite the version in force -- the neighbour's shape
            # too -- which is exactly what a ✂ row must not do (U2b round 2, I1).
            return GuidePlan(PHASE_DRAW_PIXELS, title,
                             f"现在：「{editing}」的形状从这一帧起变了 — 画好了按 Ctrl+K"
                             f"（从这帧起新版本）提交；Enter 会连前后帧里的旧样子一起改掉",
                             _steps(facts, 3, pick_text=picked, commit_text=COMMIT_SPLIT),
                             "commit_split")
        if facts.layer_dirty:
            return GuidePlan(PHASE_DRAW_PIXELS, title,
                             f"现在：「{editing}」的形状对了就按 Enter 提交；边缘不对用 "
                             f"B 补 / E 擦，漏掉的大块用 P 连点补，不想要就 Esc",
                             _steps(facts, 3, pick_text=picked), "commit")
        if facts.layer_pixels:
            # A stored shape, loaded and untouched: Enter has nothing to write
            # and says so; the next thing is to change it or leave it.
            nxt = "画好了按 Ctrl+K（从这帧起新版本）" if split else "改完 Enter"
            # An open ✂ row loads the *old* version: it needs drawing, so no
            # button is the next one to press -- not Esc (round 3, item 6).
            open_split = split and not any(
                item.done for item in facts.items
                if editing in (item.label, item.instance)
                and item.kind == api.KIND_SPLIT_KEYFRAME)
            return GuidePlan(PHASE_LOADED, title,
                             f"现在：这是已存的形状：要改就画，{nxt}；不改按 Esc"
                             f"（或直接点下一条）",
                             _steps(facts, 2, pick_text=picked,
                                    draw_text=f"这是已存的形状：要改就画（{nxt}），"
                                              f"不改按 Esc"),
                             "" if open_split else "clear_edit")
        if facts.sam_ready:
            now = (f"现在：按 S，在「{editing}」上点一下（或按 X 拖一个框），"
                   f"SAM 会给出形状")
            action = "tool_sam_point"
        else:
            now = f"现在：SAM 还没准备好 — 按 B 用画笔涂出「{editing}」"
            action = "tool_brush"
        return GuidePlan(PHASE_DRAW_EMPTY, title, now,
                         _steps(facts, 2, pick_text=picked), action)
    work = _open_work(facts)
    if work:
        first = _label(work[0])
        return GuidePlan(PHASE_PICK, title,
                         f"现在：在右边任务卡上单击「{first}」开始画（还剩 {len(work)} 个）",
                         _steps(facts, 1, pick_text=f"{PICK}：{first}"), "")
    nothing = not any(item.kind in _WORK_KINDS for item in facts.items)
    if facts.blockers:
        # The rows are done, and Space would still say no: the pane under the
        # card is the list, and no button is the next one (U2d).  "单击一条"
        # only while a click on one goes somewhere; otherwise the line's own
        # instruction (U2e).
        now = (f"现在：{facts.blocker_hint}" if facts.blocker_hint else
               f"现在：任务卡做完了，但下面还有 {facts.blockers} 个问题挡住 "
               f"Space：单击一条去处理")
        return GuidePlan(PHASE_BLOCKED, title, now, _steps(facts, 4, nothing=nothing), "")
    if facts.frame_confirmed:
        return GuidePlan(PHASE_CONFIRMED, title,
                         "这一帧已经确认 ✓ — 按 PgDn 去上一帧继续；要改哪个零件就在"
                         "任务卡上单击它",
                         _steps(facts, -1, nothing=nothing), "")
    return GuidePlan(PHASE_CONFIRM, title,
                     "现在：任务卡都做完了 — 按 Space 确认整帧，自动退到上一帧",
                     _steps(facts, 4, nothing=nothing), "confirm")
