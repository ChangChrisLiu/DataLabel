"""The window half of task U2b: palette, "What now", the banner, the card's hover.

Mixed into :class:`tda.ui.app.MainWindow`.  The second trial asked for three
things, and none of them is a new *ability* -- every gesture already had a key:

* **buttons** for the keys (:class:`~tda.ui.panels.palette.ToolPalette`).  A
  button runs the very ``act_*`` slot its key runs, through
  :meth:`GuideMixin.run_palette_action`, which is :meth:`KeysMixin.handle_key`
  minus the keyboard: the same mode check, the same end of a stuck ``Tab``
  flash, the same :meth:`dispatch`.  Every other gate lives in the slots and is
  therefore shared for free.
* **the guide on the main window** (:class:`~tda.ui.panels.whatnow.WhatNowPanel`),
  fed by :func:`tda.ui.guide.plan_for` from state the window already holds.
* **visible feedback** when an edit starts: a banner on the canvas and the row
  on the card marked "✎ 正在画", and for an empty shape the tool switched to
  SAM, which is how a new shape is started.

**Cost.**  :meth:`refresh_guidance` runs at the end of ``update_status`` and at
the few places that change what the panel says without it (a bar shown or
hidden, a ghost, ``Esc``).  It reads booleans, the task card's *cached* rows and
:attr:`_edit_facts` -- the two facts about the editing layer that cost a pass
over a 12 MP array are computed only when the layer changes
(:meth:`_note_edit_facts`), never on a mouse move and never on a frame change
with nothing being edited.  Each widget compares before it repaints.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer

from tda.core import masks as _masks
from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.app_roi import BENCH_NEEDS_ITEM, ON_BENCH
from tda.ui.class_names import instance_label
from tda.ui.panels.palette import RADIUS_MAX, RADIUS_MIN

__all__ = ["BANNER_EMPTY", "BANNER_PIXELS", "GuideMixin", "HINT_DWELL_MS", "VIS_NOTE_MS"]

#: The banner's two endings (ruling U2b-4).
BANNER_EMPTY = "用 S 在零件上点一下，或 X 拖框"
BANNER_EMPTY_NO_SAM = "用 B 画笔涂出它（SAM 还没准备好）"
BANNER_PIXELS = "Enter 提交 / Esc 放弃"
#: A ✂ row: the new version is committed with Ctrl+K (U2b round 2, I1).
BANNER_SPLIT = "Ctrl+K 从这帧起新版本 / Esc 放弃"
#: A stored shape, loaded and untouched: Enter has nothing to write (item 2).
BANNER_LOADED = "这是已存的形状：要改就画，改完 Enter；不改按 Esc"
BANNER_LOADED_SPLIT = "这是前一版的形状：画出这一帧的样子，Ctrl+K 提交；不改按 Esc"
#: How long the pointer rests on a card row before its outline is looked up.
#: Passing over rows on the way to another one should not cost a query each.
HINT_DWELL_MS = 120
#: How long a "可见性 → …（Ctrl+Z 撤销）" note stays across the canvas.
VIS_NOTE_MS = 4000
#: ``Alt+Enter`` / ``Ctrl+K``: refused on an untouched layer with the greyed
#: button's own reason (round 3, item 6).
_SCOPE_KEYS = frozenset({"commit_override", "commit_split"})
#: Keys :meth:`GuideMixin.key_refusal` looks at: the drawing and committing
#: ones.  Undo, redo and Esc are never refused by it.
_KEY_GUARD_NAMES = frozenset(A.PALETTE_TOOLS) | _SCOPE_KEYS | frozenset(
    {"commit", "confirm", "toggle_heat", "flash_compare", "cycle_candidate"})
#: At most this many old Label Studio drafts are outlined for one hover: a
#: thirty-screw frame must not become a mess (round 1, item 4).
HINT_MAX_DRAFTS = 5
#: The hover's step window: unbounded, i.e. the pose segment ``drafts_for``
#: clamps it to.  ``Shift+A`` keeps its own ±2 (``ls_adopt.NEAR_STEPS``).
HINT_DRAFT_STEPS = 100_000
HINT_DRAFT_RGB = (96, 208, 255)
HINT_SHAPE_RGB = (80, 220, 120)
HINT_DIFF_RGB = (255, 150, 40)

#: Two outlines this similar are one place, said once with both names.
HINT_SAME_PLACE_IOU = 0.5


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _merge_hints(hints: list) -> list:
    """One outline per place: a draft on the difference map's box is one hint.

    Two labels drawn on top of each other are two labels nobody can read.
    """
    out: list = []
    for box, label, rgb in hints:
        for index, (kept, kept_label, kept_rgb) in enumerate(out):
            if _iou(box, kept) >= HINT_SAME_PLACE_IOU:
                if label not in kept_label.split(" + "):
                    out[index] = (kept, f"{kept_label} + {label}", kept_rgb)
                break
        else:
            out.append((box, label, rgb))
    return out


_SAM_TOOLS = ("sam_point", "sam_box")
_NO_EDIT = (None, False, False)
_VERIFIED = api.CONFIRMED_STATUSES


class GuideMixin:
    """Palette, guide panel, canvas banner and task-card hover, for the window."""

    # ------------------------------------------------------------------ setup
    def _init_guide(self) -> None:
        """Wire the widgets :class:`ShellMixin` built; called once, last."""
        #: ``(instance, layer has pixels, layer differs from what was loaded)``.
        self._edit_facts: tuple = _NO_EDIT
        #: ``(instance, label, raw names)`` of the instance being edited.
        self._edit_label: tuple = (None, "", "")
        #: The SAM tool a new shape starts with: the last one chosen, S until
        #: then (ruling U2b-4, round 2).
        self._last_sam_tool = "sam_point"
        self._hint_for = ""
        self._hint_cache: dict = {}
        self._hint_timer = QTimer(self)
        self._hint_timer.setSingleShot(True)
        self._hint_timer.setInterval(HINT_DWELL_MS)
        self._hint_timer.timeout.connect(self._show_card_hint)
        self._cheat_cache: dict[str, str] = {}
        #: ``(text, frame)``: what a visibility key just did, on the canvas for
        #: :data:`VIS_NOTE_MS` (round 3, item 5).
        self._vis_note: tuple = ("", None)
        self._vis_note_timer = QTimer(self)
        self._vis_note_timer.setSingleShot(True)
        self._vis_note_timer.setInterval(VIS_NOTE_MS)
        self._vis_note_timer.timeout.connect(self.expire_visibility_note)
        self.palette.sigAction.connect(self.run_palette_action)
        self.palette.sigRadius.connect(self.set_brush_radius)
        self.palette.sigRadiusTyped.connect(self._on_radius_typed)
        self.palette.sigRefused.connect(self._on_palette_refused)
        # The minimap lives at the foot of the palette now, off the canvas
        # (round 2, item 5): nothing is drawn over the image any more.
        self.palette.set_minimap(self.canvas.minimap())
        # K / N are live only with a conflict selected: the queue's selection
        # is state the palette's greying depends on.
        for queue in api.QUEUE_NAMES:
            self.review.list_for(queue).currentItemChanged.connect(
                lambda *_a: self.refresh_guidance())
        self.review.tabs().currentChanged.connect(lambda *_a: self.refresh_guidance())
        # 台面框 R is live only with a part on the bench selected (round 3).
        self.instances.table().itemSelectionChanged.connect(self.refresh_guidance)
        self.task_card.list_widget().currentItemChanged.connect(
            lambda *_a: self.refresh_guidance())
        self.task_card.sigHover.connect(self.on_card_hover)
        self.task_card.sigExplain.connect(self.report)
        # A problem clicked in the card's pane selects its part where the
        # keys that fix it act (U2d: the guide says "单击一条去处理").
        self.task_card.sigPickInstance.connect(self.instances.select_instance)
        # ... and a conflict line opens Review on that conflict (U2e).
        self.task_card.sigOpenConflict.connect(self.open_conflict_in_review)
        for bar in (self.warn_bar, self.scope_bar, self.restore_bar, self.roi_bar,
                    getattr(self, "raw_bar", None)):
            if bar is not None:
                bar.sigVisible.connect(self._on_bar_visible)
        self._guidance_ready = True

    @S.guard
    def _on_palette_refused(self, name: str, reason: str) -> None:
        """A click on a greyed button: say why in the status bar (round 2).

        The tooltip said it already, but only to somebody who hovered; a click
        that does nothing and says nothing is a click repeated, harder.
        """
        head = A.short_parts(A.action_named(name))[0]
        self.report(f"{head}：{reason}" if reason else f"{head}：现在不能用")

    def on_canvas_focus(self) -> None:
        """The canvas has the keyboard again: a "快捷键没生效" line is now stale."""
        from tda.ui.app_keys import KEY_SWALLOWED

        if self.status_message() == KEY_SWALLOWED:
            self.report("")

    def flash_visibility_note(self, text: str) -> None:
        """Show ``text`` across the canvas for a few seconds (round 3, item 5)."""
        self._vis_note = (str(text), self.session.current())
        self._vis_note_timer.start()
        self.refresh_guidance()

    def expire_visibility_note(self) -> None:
        """Take the visibility note off the canvas; the banner is the guide's again."""
        self._vis_note = ("", None)
        self._vis_note_timer.stop()
        self.refresh_guidance()

    def forget_card_hints(self) -> None:
        """Drop the hover outlines looked up for this frame: a commit or an
        undo changed the shapes they were read from (round 2)."""
        self._hint_cache.clear()

    # -------------------------------------------------------- the palette
    @S.guard
    def run_palette_action(self, name: str, pressed: bool = True) -> None:
        """A palette button: exactly what its key does (see the module doc)."""
        action = A.action_named(name)
        if self.mode not in action.modes:
            self.report(f"{A.short_parts(action)[0]}：只在标注模式可用 / Annotate mode only")
            return
        if action.hold:
            self.dispatch(action, bool(pressed))
        elif pressed:
            self.end_flash()
            self.dispatch(action)
        # A click on the palette is not a reason for the keyboard to leave the
        # canvas; the buttons never take it, and this puts back a focus some
        # earlier click on a dock may have taken.
        self.focus_canvas()
        self.refresh_guidance()

    @S.guard
    def _on_radius_typed(self, radius: int) -> None:
        """``Enter`` / ``Esc`` in the brush-size box: the keyboard goes home.

        Said in the status line as well, over the "快捷键没生效" the digits
        may have raised on their way into the box -- they were numbers, not
        shortcuts, and the answer to typing one is the size it set.
        """
        self.focus_canvas()
        self.report(f"笔刷大小 r={int(radius)} / brush radius {int(radius)} px")

    @S.guard
    def set_brush_radius(self, radius: int) -> None:
        """One radius for brush, eraser and occluder; the slider, ``[``/``]`` and
        the badge all end here."""
        value = min(max(int(radius), RADIUS_MIN), RADIUS_MAX)
        for tool in (self.brush, self.eraser, self.occluder):
            tool.set_radius(value)
        self.sync_tool_cursor()      # the ring is the size of the stroke
        self.update_status()         # the badge's r=, and the palette's numbers

    def palette_states(self, facts: G.GuideFacts) -> dict:
        """``{action name: (enabled, checked, why not)}`` for every button."""
        annotate = self.mode == A.MODE_ANNOTATE
        image = facts.has_image and facts.is_open
        if self.mode == A.MODE_REVIEW:
            closed = "复查模式画布只读：按 R 回标注模式返工"
        elif not annotate:
            closed = "只在标注模式可用：点顶部 Annotate"
        elif not image:
            closed = "这一帧没有图像"
        else:
            closed = ""
        armed = self.armed_tool_name()
        states: dict[str, tuple] = {}
        for name in A.PALETTE_TOOLS:
            action = A.action_named(name)
            tool = action.args[0] if action.slot == "act_tool" else "roi"
            if closed:
                states[name] = (False, False, closed)
                continue
            enabled, why = True, ""
            if tool in _SAM_TOOLS and not self.sam_available:
                enabled = False
                why = (f"SAM 还没准备好（{self.sam_reason}）：先用画笔 B；"
                       f"右下角出现 SAM ready 后就能用")
            elif tool == "roi" and facts.layer_dirty:
                enabled, why = False, "编辑层有未提交的像素：先 Enter 提交或 Esc 放弃"
            elif tool == "bench_box" and not self._bench_box_usable():
                # U2a's own sentence for a bare R, on the button too (round 3).
                enabled, why = False, BENCH_NEEDS_ITEM
            states[name] = (enabled, armed == tool, why)

        dirty = facts.layer_dirty
        owner = facts.ghost or facts.roi_editing or bool(facts.scope) or facts.warning
        nothing = "没有可提交的修改：先在任务卡上单击一个零件，画好再提交"
        states["commit"] = (bool(not closed and (owner or dirty)), False, closed or nothing)
        if facts.ghost:
            why_scope = "先处理草稿预览：Enter 采纳 / Esc 取消"
        elif facts.roi_editing:
            why_scope = "ROI 框开着：先 Enter 保存或 Esc 跳过"
        else:
            why_scope = nothing
        scoped = bool(not closed and dirty and not facts.ghost and not facts.roi_editing)
        # 只改这一帧 is an exception *on* a shape: with no keyframe under it
        # the part stays missing, so the button says so rather than write (U2d).
        no_shape = self.override_refusal() if scoped else ""
        states["commit_override"] = (scoped and not no_shape, False,
                                     closed or no_shape or why_scope)
        states["commit_split"] = (scoped, False, closed or why_scope)
        if dirty:
            why_confirm = "有未提交的修改：先 Enter 提交或 Esc 放弃"
        elif facts.ghost:
            why_confirm = "先处理草稿预览：Enter 采纳 / Esc 取消"
        else:
            why_confirm = ""
        states["confirm"] = (bool(not closed and not why_confirm), False,
                             closed or why_confirm)
        stack = getattr(self.session, "undo_stack", None)
        can_undo = bool(getattr(stack, "can_undo", False))
        can_redo = bool(getattr(stack, "can_redo", False))
        states["undo"] = (bool(not closed and can_undo), False, closed or "没有可以撤销的操作")
        states["redo"] = (bool(not closed and can_redo), False, closed or "没有可以重做的操作")
        heat_closed = "" if (image and self.mode in (A.MODE_ANNOTATE, A.MODE_REVIEW)) \
            else (closed or "这一帧没有图像")
        states["toggle_heat"] = (not heat_closed, bool(self.heat_visible), heat_closed)
        no_neighbour = "" if facts.neighbour is not None else "起点帧没有可以对比的上一帧"
        states["flash_compare"] = (bool(not closed and not no_neighbour), False,
                                   closed or no_neighbour)
        candidates = self._candidate_tool()
        many = candidates is not None and candidates.candidate_count > 1
        states["cycle_candidate"] = (bool(not closed and many), False,
                                     closed or "还没有 SAM 候选：先 S 点一下或 X 拖框")
        escapable = (facts.ghost or facts.warning or facts.roi_editing
                     or facts.bench is not None or facts.editing is not None)
        states["clear_edit"] = (bool(not closed and escapable), False,
                                closed or "没有可以放弃的编辑")
        # The Review queue's own keys (round 1, item 3); shown only in Review.
        review = self.mode == A.MODE_REVIEW and image
        not_review = "只在复查模式可用"
        states["review_accept"] = (bool(review), False, not_review)
        states["review_rework"] = (bool(review), False, not_review)
        conflict = review and self.review.selected_conflict() is not None
        why_conflict = not_review if not review else "先在 Conflicts 队列里选一条冲突"
        states["review_keep_old"] = (bool(conflict), False, why_conflict)
        states["review_accept_new"] = (bool(conflict), False, why_conflict)
        return states

    def _bench_box_usable(self) -> bool:
        """Is there a part on the bench for ``R`` to box (armed, or selected)?

        What :meth:`act_tool` checks before arming the box, read off the rows
        the panels already hold -- this runs on every refresh.
        """
        if getattr(self, "bench_instance", None):
            return True
        instance = (self.instances.selected_instance()
                    or self.task_card.current_instance())
        if not instance:
            return False
        return any(str(row.get("key")) == instance and row.get("placement") == ON_BENCH
                   for row in self.instances.rows())

    def key_refusal(self, name: str) -> str:
        """Why the key of palette action ``name`` must not run now; ``""`` if it may.

        One guard for the key and its greyed button (round 3): both read
        :meth:`palette_states`.  It applies on a frame with no image (every
        drawing and committing key) and to ``Alt+Enter`` / ``Ctrl+K`` on an
        untouched layer, where the button was grey with one reason and the
        key answered with another.
        """
        if not getattr(self, "_guidance_ready", False) or name not in _KEY_GUARD_NAMES:
            return ""
        facts = self.guide_facts()
        no_image = not (facts.is_open and facts.has_image)
        untouched = (name in _SCOPE_KEYS and not facts.layer_dirty and not facts.ghost
                     and not facts.roi_editing and not facts.scope and not facts.warning)
        # Alt+Enter on a part with no shape here (U2d): refused like its button.
        no_shape = name == "commit_override" and bool(self.override_refusal())
        if not (no_image or untouched or no_shape):
            return ""
        enabled, _checked, why = self.palette_states(facts).get(name, (True, False, ""))
        return "" if enabled else (why or "现在不能用")

    def override_refusal(self) -> str:
        """Why ``Alt+Enter`` cannot write this edit, or ``""`` (U2d).

        The session's answer to "does a keyframe of the part being edited
        apply here?" -- the question behind ``missing_shape`` -- in the words
        its own refusal uses (:data:`~tda.ui.session_api.OVERRIDE_NEEDS_SHAPE`).
        A dict lookup on the compiled frame, so it may run on every refresh.
        """
        instance = getattr(self.session, "editing_instance", None)
        applies = getattr(self.session, "keyframe_applies", None)
        if instance is None or not callable(applies):
            return ""
        try:
            return "" if applies(instance) else api.OVERRIDE_NEEDS_SHAPE
        except Exception:  # noqa: BLE001 - a grey button is never worth a failure
            return ""

    # --------------------------------------------------------- the guide
    def guide_facts(self) -> G.GuideFacts:
        """What :func:`tda.ui.guide.plan_for` needs, from state already held."""
        opened = compat.is_open(self.session)
        key = self.session.current() if opened else None
        editing = getattr(self.session, "editing_instance", None)
        if self._edit_facts[0] != editing:
            self._note_edit_facts(refresh=False)
        _inst, pixels, dirty = self._edit_facts
        items = tuple(
            G.CardItem(str(row.get("instance", "")), str(row.get("kind", "")),
                       bool(row.get("done", False)),
                       instance_label(str(row.get("instance", "")), row.get("cls"),
                                      row.get("attrs")))
            for row in self.task_card.rows()
        )
        roi_editing = bool(self.roi_editing)
        has_image = bool(self.tools_enabled) if opened else False
        roi_stored = bool(opened and has_image and self.roi() is not None)
        scope = str(self._pending_scope or "")
        warning = self._pending_warning
        neighbour = compat.task_neighbour(self.session) if opened else None
        forward_start = False
        if opened and key is not None and neighbour is None:
            steps = self.session.steps()
            forward_start = bool(steps) and int(key.step) == min(steps) != max(steps)
        return G.GuideFacts(
            mode=self.mode, is_open=opened, has_image=has_image,
            raw_missing="" if has_image or not opened else self._raw_missing_reason(),
            view=str(getattr(self.session, "view", "")),
            step=None if key is None else int(key.step),
            neighbour=neighbour,
            forward_start=forward_start,
            frame_confirmed=getattr(self, "_frame_status_seen", "") in _VERIFIED,
            roi_stored=roi_stored,
            roi_editing=roi_editing,
            roi_unanswered=bool(opened and has_image and not roi_stored
                                and not roi_editing and self.roi_unanswered()),
            editing=self._edit_label[1] if editing else None,
            editing_kind=self._card_kind(editing),
            layer_pixels=bool(pixels), layer_dirty=bool(dirty),
            warning=warning is not None,
            warning_scope=str(warning[0]) if warning is not None else "",
            scope=scope,
            scope_text=self.scope_text(scope),
            ghost=self.showing_draft_ghost(),
            bench=self._label_of(self.bench_instance)[0] if self.bench_instance else None,
            flashing=self.is_flashing(),
            sam_ready=bool(self.sam_available),
            items=items,
            # What the card's pane holds that makes Space refuse (U2d).
            blockers=self.task_card.problem_count() if opened else 0,
            blocker_hint=self.task_card.blocker_hint() if opened else "",
        )

    def _card_kind(self, instance: Optional[str]) -> str:
        """The task-card kind of ``instance`` on this frame, ``""`` when not listed."""
        if not instance:
            return ""
        for row in self.task_card.rows():
            if str(row.get("instance")) == str(instance):
                return str(row.get("kind", ""))
        return ""

    def scope_text(self, scope: str) -> str:
        """A commit scope as a sentence: ``zorder:above:chassis`` -> 它在 机箱 chassis 上面.

        The scope bar speaks the session's vocabulary; the guide speaks the
        annotator's (U2b round 2).  ``""`` for no scope; a scope this does not
        know is given back as it is.
        """
        scope = str(scope or "")
        if not scope:
            return ""
        split, _, rest = scope.partition("split+")
        body = rest if _ else scope
        prefix = "从这帧起新版本，并且" if _ else ""
        for token, where in (("zorder:above:", "上面"), ("zorder:below:", "下面")):
            if body.startswith(token):
                other = body[len(token):]
                return f"{prefix}它在 {self._label_of(other)[0] or other} {where}（层级）"
        words = {api.SCOPE_KEYFRAME: "改这个形状（它覆盖的每一帧都跟着变）",
                 api.SCOPE_SPLIT: "从这帧起新版本",
                 api.SCOPE_FRAME_OVERRIDE: "只改这一帧"}
        return words.get(scope, scope)

    def _raw_missing_reason(self) -> str:
        """Why this frame's pixels cannot be read (task U2a's two answers).

        Called only for a frame with no image:

        1. ``session.images.why_unreadable(key)`` -- the image cache's per-frame
           answer: the raw drive's own sentence plus the recorded path, or the
           file that could not be opened; ``None`` for a step that simply has
           no image in this view;
        2. the window's ``raw_root`` (:class:`tda.ui.app_rawdata.RawDataMixin`):
           configured and not connected means the drive is not there, and its
           ``message`` says which drive to plug in.
        """
        if compat.is_open(self.session):
            try:
                found = self.session.images.why_unreadable(self.session.current())
            except Exception:  # noqa: BLE001 - a hint is never worth a failure
                found = None
            if found:
                return str(found)
        raw = getattr(self, "raw_root", None)
        if raw is not None and raw.configured and not raw.connected:
            return str(raw.message or "原始数据盘没有接上 / raw drive missing")
        return ""

    def _cheat_for(self, mode: str) -> str:
        found = self._cheat_cache.get(mode)
        if found is None:
            found = A.mode_cheat_html(mode)
            self._cheat_cache[mode] = found
        return found

    def refresh_guidance(self) -> None:
        """Bring the palette, the guide, the banner and the card's mark up to date.

        Safe to call at any time and cheap to call often: see the module doc.
        """
        if not getattr(self, "_guidance_ready", False) or self.closed:
            return
        facts = self.guide_facts()
        plan = G.plan_for(facts)
        self.guide.show_plan(plan, self._cheat_for(self.mode))
        self.palette.setVisible(self.mode != A.MODE_STEPS)
        self.palette.set_mode(self.mode)
        states = self.palette_states(facts)
        self.palette.apply_states(states, plan.action)
        # The scope bar's 仅本帧 is the same action: the same grey, the same why (U2e).
        enabled, _checked, why = states.get("commit_override", (True, False, ""))
        button = getattr(self, "scope_override_button", None)
        if button is not None:
            button.setEnabled(bool(enabled))
            button.setToolTip("" if enabled else str(why))
        self.palette.set_radius(self.brush.radius)
        self.canvas.set_banner(self.banner_text(facts))
        editing = getattr(self.session, "editing_instance", None)
        if editing:
            self.task_card.set_editing(editing)
        else:
            self.task_card.set_editing(self.bench_instance,
                                       bench=self.bench_instance is not None)

    def guide_plan(self) -> G.GuidePlan:
        """The plan the panel is showing (for the tests and the report)."""
        return G.plan_for(self.guide_facts())

    def _on_bar_visible(self, _shown: bool) -> None:
        self.refresh_guidance()

    # --------------------------------------------------------- the banner
    def banner_text(self, facts: Optional[G.GuideFacts] = None) -> str:
        """What the canvas says across its top edge; ``""`` for nothing.

        ``正在画：导风罩 cover.01（日志：CPU Fan Cover）— 用 S 在零件上点一下，
        或 X 拖框`` while the layer is empty, ``… — Enter 提交 / Esc 放弃`` once
        it holds pixels (ruling U2b-4).
        """
        facts = facts or self.guide_facts()
        if self.mode != A.MODE_ANNOTATE or not facts.has_image or facts.flashing:
            return ""
        note, where = getattr(self, "_vis_note", ("", None))
        if note and where == self.session.current():
            return note       # a visibility just changed: said, briefly (round 3)
        instance = getattr(self.session, "editing_instance", None)
        if instance:
            _inst, label, raw = self._edit_label
            head = f"正在画：{label}" + (f"（日志：{raw}）" if raw else " ")
            split = facts.editing_kind == api.KIND_SPLIT_KEYFRAME
            if facts.layer_dirty:
                # The key a ✂ row is committed with; Enter would rewrite the
                # neighbour's version too (U2b round 2, I1).
                return f"{head}— {BANNER_SPLIT if split else BANNER_PIXELS}"
            if facts.layer_pixels:
                return f"{head}— {BANNER_LOADED_SPLIT if split else BANNER_LOADED}"
            return f"{head}— {BANNER_EMPTY if self.sam_available else BANNER_EMPTY_NO_SAM}"
        if self.bench_instance:
            label, raw = self._label_of(self.bench_instance)
            head = f"正在框：{label}" + (f"（日志：{raw}）" if raw else " ")
            return f"{head}— 在台面上拖一个框，松手就存好 / Esc 取消"
        return ""

    def _label_of(self, instance: Optional[str]) -> tuple[str, str]:
        """``("导风罩 cover.01", "CPU Fan Cover")`` for any instance.

        The card's rows already carry the class and the log's names; an
        instance picked from the instance table is looked up once, at the
        start of its edit, never per refresh.
        """
        if not instance:
            return ("", "")
        for row in self.task_card.rows():
            if str(row.get("instance")) == str(instance):
                raw = " / ".join(dict.fromkeys(str(n) for n in row.get("raw_names") or [] if n))
                return (instance_label(str(instance), row.get("cls"), row.get("attrs")), raw)
        try:
            rec = self.db.instances(int(self.session.desktop)).get(str(instance))
        except Exception:  # noqa: BLE001 - a label is never worth a failure
            rec = None
        if rec is None:
            return (instance_label(str(instance)), "")
        raw = " / ".join(dict.fromkeys(str(n) for n in (rec.raw_names or []) if n))
        return (instance_label(str(instance), rec.cls, rec.attrs), raw)

    # ------------------------------------------------ the editing layer's facts
    def _note_edit_facts(self, refresh: bool = True) -> None:
        """Re-read the two facts about the layer the guide and banner need.

        Called where the layer changes -- a stroke, a replacement, a commit,
        ``Esc``, an undo, an edit starting -- and nowhere else.  With nothing
        being edited it costs an attribute read.
        """
        instance = getattr(self.session, "editing_instance", None)
        if instance is None:
            facts = _NO_EDIT
        else:
            mask = self.session.editing_mask()
            pixels = mask is not None and bool(mask.any())
            dirty = mask is not None and self.has_uncommitted_edit()
            facts = (instance, pixels, bool(dirty))
        if instance != self._edit_label[0]:
            label, raw = self._label_of(instance)
            self._edit_label = (instance, label, raw)
        changed = facts != self._edit_facts
        self._edit_facts = facts
        if changed and refresh:
            self.refresh_guidance()

    def layer_facts(self) -> tuple:
        """``(instance, has pixels, uncommitted)`` as the guide last saw them."""
        return self._edit_facts

    # ------------------------------------------ starting a new shape (U2b-4)
    def note_draw_tool(self, name: str) -> None:
        """Remember the SAM tool the annotator last chose (S or X)."""
        if name in _SAM_TOOLS:
            self._last_sam_tool = name

    def pick_tool_for_new_shape(self) -> Optional[str]:
        """The tool an edit of an *empty* layer starts with, or ``None`` to stay.

        The SAM tool the annotator last chose -- ``S`` until they have chosen
        one -- when SAM is ready, otherwise whatever is armed (U2b round 2).
        A brush picked up to tidy the last part's edge says nothing about how
        they like to *start* a shape; which SAM prompt they prefer does.
        """
        if not self.sam_available:
            return None
        return self._last_sam_tool

    def switch_tool_for_empty_layer(self) -> None:
        """Arm :meth:`pick_tool_for_new_shape`'s answer when the layer is empty."""
        if self._edit_facts[1]:
            return
        wanted = self.pick_tool_for_new_shape()
        if wanted is None or wanted == self._tool_name:
            return
        self.note_tool_switch(wanted, via="new shape")
        self._tool_name = wanted

    # --------------------------------------------------- the card's hover
    @S.guard
    def on_card_hover(self, instance: str) -> None:
        """The pointer rests on a card row: outline where the part probably is.

        Display only -- no tool, prompt box, selection or layer is touched --
        and looked up only once the pointer has *rested* (:data:`HINT_DWELL_MS`),
        so sliding down the card costs nothing.
        """
        self._hint_for = str(instance or "")
        self._hint_timer.stop()
        if not self._hint_for:
            self.canvas.set_hint_boxes([])
            return
        self._hint_timer.start()

    @S.guard
    def _show_card_hint(self) -> None:
        if self.closed:
            return
        instance = self._hint_for
        if (not instance or self.mode != A.MODE_ANNOTATE or not self.tools_enabled
                or self.is_flashing() or not compat.is_open(self.session)):
            self.canvas.set_hint_boxes([])
            return
        self.canvas.set_hint_boxes(self.card_hints(instance))

    def card_hints(self, instance: str) -> list:
        """``[(box, label, rgb)]``: where a card item probably is on this frame.

        Three sources, all read-only: the instance's own shape on the nearest
        keyframe of this pose segment, the difference map's strongest change
        (for a part the card says has come back), and the team's old Label
        Studio drafts of the same class traced **anywhere in this pose
        segment** -- at most :data:`HINT_MAX_DRAFTS` of them, the ones nearest
        the difference map's box when there is one (round 1, item 4).  The
        window is wider than ``Shift+A``'s ±2 steps on purpose: this only
        points, it adopts nothing.  The two stored sources are cached per
        frame, instance and difference box.
        """
        key = self.session.current()
        row = next((r for r in self.task_card.rows()
                    if str(r.get("instance")) == str(instance)), {})
        anchor = self._frame_diff_box(key)
        centre = None if anchor is None else ((anchor[0] + anchor[2]) / 2.0,
                                              (anchor[1] + anchor[3]) / 2.0)
        if len(self._hint_cache) > 64:
            self._hint_cache.clear()
        shape = self._hint_cache.get(("shape", key, instance))
        if shape is None:
            shape = self._shape_hint(key, instance)
            self._hint_cache[("shape", key, instance)] = shape
        drafts = self._hint_cache.get(("drafts", key, instance, centre))
        if drafts is None:
            drafts = self._draft_hints(key, row, instance, centre)
            self._hint_cache[("drafts", key, instance, centre)] = drafts
        return _merge_hints(self._diff_hint(key, row) + list(shape) + list(drafts))

    def _frame_diff_box(self, key) -> Optional[tuple]:
        """The box SAM is armed with, else the strongest unexplained change."""
        from tda.ui.app_diff import best_unexplained

        payload = self.assist_result
        if not payload or payload.get("key") != key:
            return None
        if self._prompt_box is not None:
            return tuple(float(v) for v in self._prompt_box)
        blob = best_unexplained(payload)
        return None if blob is None else tuple(float(v) for v in blob.box)

    def _shape_hint(self, key, instance: str) -> list:
        seg = (self.db.pose_segment_for(key) or {}).get("seg")
        best = None
        try:
            frames = self.db.keyframes(int(key.desktop), str(key.view), str(instance))
        except Exception:  # noqa: BLE001 - a hint is never worth a failure
            return []
        for kf in frames:
            if seg is not None and int(kf.pose_segment) != int(seg):
                continue
            boxes = []
            for part in kf.parts or ():
                box = part.box if part.box is not None else _masks.rle_bbox(part.rle)
                if box is not None:
                    boxes.append(tuple(float(v) for v in box))
            if not boxes:
                continue
            union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes))
            distance = abs(int(kf.anchor_step) - int(key.step))
            if best is None or distance < best[0]:
                best = (distance, union, int(kf.anchor_step))
        if best is None:
            return []
        label = ("它现在的形状" if best[2] == int(key.step)
                 else f"第 {best[2]} 帧的形状")
        return [(best[1], label, HINT_SHAPE_RGB)]

    def _draft_hints(self, key, row: dict, instance: str,
                     centre: Optional[tuple] = None) -> list:
        """Same-class drafts of this pose segment, nearest first, at most five.

        ``drafts_for`` clamps its step window to the frame's pose segment, so
        an unbounded ``near_steps`` *is* "the whole segment".  With a
        ``centre`` (the difference box's) it ranks by distance to it; without
        one, by distance in steps.  A draft traced at several steps is one
        place, outlined once.
        """
        from tda.core import ls_adopt

        cls = str(row.get("cls") or self._class_of(instance) or "")
        if not cls or self.overlay is None:
            return []
        try:
            found = ls_adopt.drafts_for(self.db, self.session.tax, int(key.desktop),
                                        str(key.view), int(key.step), cls,
                                        hw=self.overlay.hw,
                                        near_steps=HINT_DRAFT_STEPS, cursor=centre)
        except Exception:  # noqa: BLE001 - a hint is never worth a failure
            return []
        out: list = []
        for candidate in found:
            if not candidate.same_class or candidate.box is None:
                continue          # the cursor also brings other classes under it
            box = tuple(float(v) for v in candidate.box)
            if any(_iou(box, kept) >= HINT_SAME_PLACE_IOU for kept, _l, _c in out):
                continue
            out.append((box, "旧草稿", HINT_DRAFT_RGB))
            if len(out) >= HINT_MAX_DRAFTS:
                break
        return out

    def _diff_hint(self, key, row: dict) -> list:
        """The difference map's box, outlined for a part the card says came back.

        The box SAM is armed with when there is one -- it may be a ``Shift+C``
        alternate -- and otherwise the strongest unexplained change.
        """
        if row.get("kind") != api.KIND_ADD_SHAPE:
            return []
        box = self._frame_diff_box(key)
        if box is None:
            return []
        label = "差异图的提示框" if self._prompt_box is not None else "差异最大处"
        return [(box, label, HINT_DIFF_RGB)]
