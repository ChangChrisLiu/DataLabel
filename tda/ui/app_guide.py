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
from tda.ui.class_names import instance_label
from tda.ui.panels.palette import RADIUS_MAX, RADIUS_MIN

__all__ = ["BANNER_EMPTY", "BANNER_PIXELS", "GuideMixin", "HINT_DWELL_MS"]

#: The banner's two endings (ruling U2b-4).
BANNER_EMPTY = "用 S 在零件上点一下，或 X 拖框"
BANNER_EMPTY_NO_SAM = "用 B 画笔涂出它（SAM 还没准备好）"
BANNER_PIXELS = "Enter 提交 / Esc 放弃"
#: How long the pointer rests on a card row before its outline is looked up.
#: Passing over rows on the way to another one should not cost a query each.
HINT_DWELL_MS = 120
#: At most this many old Label Studio drafts are outlined for one hover.
HINT_MAX_DRAFTS = 6
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
_DRAW_TOOLS = ("brush", "eraser", "sam_point", "sam_box")
_NO_EDIT = (None, False, False)
_VERIFIED = (api.STATUS_VERIFIED, api.STATUS_RECHECK)


class GuideMixin:
    """Palette, guide panel, canvas banner and task-card hover, for the window."""

    # ------------------------------------------------------------------ setup
    def _init_guide(self) -> None:
        """Wire the widgets :class:`ShellMixin` built; called once, last."""
        #: ``(instance, layer has pixels, layer differs from what was loaded)``.
        self._edit_facts: tuple = _NO_EDIT
        #: ``(instance, label, raw names)`` of the instance being edited.
        self._edit_label: tuple = (None, "", "")
        #: The last tool the annotator drew with (ruling U2b-4).
        self._last_draw_tool: Optional[str] = None
        self._hint_for = ""
        self._hint_cache: dict = {}
        self._hint_timer = QTimer(self)
        self._hint_timer.setSingleShot(True)
        self._hint_timer.setInterval(HINT_DWELL_MS)
        self._hint_timer.timeout.connect(self._show_card_hint)
        self._cheat_cache: dict[str, str] = {}
        self.palette.sigAction.connect(self.run_palette_action)
        self.palette.sigRadius.connect(self.set_brush_radius)
        self.task_card.sigHover.connect(self.on_card_hover)
        self.task_card.sigExplain.connect(self.report)
        for bar in (self.warn_bar, self.scope_bar, self.restore_bar, self.roi_bar):
            bar.sigVisible.connect(self._on_bar_visible)
        self._guidance_ready = True

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
        states["commit_override"] = (scoped, False, closed or why_scope)
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
        return states

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
        return G.GuideFacts(
            mode=self.mode, is_open=opened, has_image=has_image,
            raw_missing="" if has_image or not opened else self._raw_missing_reason(),
            view=str(getattr(self.session, "view", "")),
            step=None if key is None else int(key.step),
            neighbour=compat.task_neighbour(self.session) if opened else None,
            frame_confirmed=getattr(self, "_frame_status_seen", "") in _VERIFIED,
            roi_stored=roi_stored,
            roi_editing=roi_editing,
            roi_unanswered=bool(opened and has_image and not roi_stored
                                and not roi_editing and self.roi_unanswered()),
            editing=self._edit_label[1] if editing else None,
            layer_pixels=bool(pixels), layer_dirty=bool(dirty),
            warning=self._pending_warning is not None,
            scope=str(self._pending_scope or ""),
            ghost=self.showing_draft_ghost(),
            bench=self._label_of(self.bench_instance)[0] if self.bench_instance else None,
            flashing=self.is_flashing(),
            sam_ready=bool(self.sam_available),
            items=items,
        )

    def _raw_missing_reason(self) -> str:
        """Why no frame can be read at all, when the session can say (task U2a).

        Read if present, tolerated if not: the signal belongs to the data-access
        fix running in parallel, and the guide must work before and after it.
        """
        for owner in (self.session, self):
            for name in ("image_unavailable_reason", "raw_missing_reason"):
                found = getattr(owner, name, None)
                try:
                    value = found() if callable(found) else found
                except Exception:  # noqa: BLE001 - a hint is never worth a failure
                    value = None
                if value:
                    return str(value)
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
        self.palette.apply_states(self.palette_states(facts), plan.action)
        self.palette.set_radius(self.brush.radius)
        self.canvas.set_banner(self.banner_text(facts))
        editing = getattr(self.session, "editing_instance", None)
        if editing:
            self.task_card.set_editing(editing)
        else:
            self.task_card.set_editing(self.bench_instance, bench=True)

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
        instance = getattr(self.session, "editing_instance", None)
        if instance:
            _inst, label, raw = self._edit_label
            head = f"正在画：{label}" + (f"（日志：{raw}）" if raw else " ")
            if facts.layer_pixels:
                return f"{head}— {BANNER_PIXELS}"
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
        """Remember the tool the annotator last drew with."""
        if name in _DRAW_TOOLS:
            self._last_draw_tool = name

    def pick_tool_for_new_shape(self) -> Optional[str]:
        """The tool an edit of an *empty* layer starts with, or ``None`` to stay.

        The last drawing tool when it was a SAM tool, otherwise SAM point when
        SAM is ready, otherwise whatever is armed: a new shape is started with
        a click or a box, and the brush that ended the last one is the wrong
        tool to meet an empty layer with.
        """
        if not self.sam_available:
            return None
        last = self._last_draw_tool
        return last if last in _SAM_TOOLS else "sam_point"

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
        Studio drafts of the same class traced within two steps.  The two
        stored ones are cached per frame and instance.
        """
        key = self.session.current()
        row = next((r for r in self.task_card.rows()
                    if str(r.get("instance")) == str(instance)), {})
        cached = self._hint_cache.get((key, instance))
        if cached is None:
            cached = self._shape_hint(key, instance) + self._draft_hints(key, row, instance)
            if len(self._hint_cache) > 64:
                self._hint_cache.clear()
            self._hint_cache[(key, instance)] = cached
        return _merge_hints(self._diff_hint(key, row) + list(cached))

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

    def _draft_hints(self, key, row: dict, instance: str) -> list:
        from tda.core import ls_adopt

        cls = str(row.get("cls") or self._class_of(instance) or "")
        if not cls or self.overlay is None:
            return []
        try:
            found = ls_adopt.drafts_for(self.db, self.session.tax, int(key.desktop),
                                        str(key.view), int(key.step), cls,
                                        hw=self.overlay.hw)
        except Exception:  # noqa: BLE001 - a hint is never worth a failure
            return []
        out, seen = [], set()
        for candidate in found:
            box = candidate.box
            if box is None or tuple(box) in seen:
                continue
            seen.add(tuple(box))
            out.append((tuple(float(v) for v in box), "旧草稿", HINT_DRAFT_RGB))
            if len(out) >= HINT_MAX_DRAFTS:
                break
        return out

    def _diff_hint(self, key, row: dict) -> list:
        from tda.ui.app_diff import best_unexplained

        if row.get("kind") != api.KIND_ADD_SHAPE:
            return []
        payload = self.assist_result
        if not payload or payload.get("key") != key:
            return []
        # The box SAM is armed with, when there is one -- it may be a
        # ``Shift+C`` alternate -- and otherwise the difference map's own.
        if self._prompt_box is not None:
            return [(tuple(float(v) for v in self._prompt_box), "差异图的提示框",
                     HINT_DIFF_RGB)]
        blob = best_unexplained(payload)
        if blob is None:
            return []
        return [(tuple(float(v) for v in blob.box), "差异最大处", HINT_DIFF_RGB)]
