"""The window half of the model assist: SAM prompts, candidates, the heat map.

:mod:`tda.ui.app_diff` owns the comparison itself and its worker thread; this
mixes into :class:`tda.ui.app.MainWindow` and is the part that knows what a
task card is -- which instances a blob may be explained by, which item the
annotator activated, and where a prompt box should come from.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem

from tda.core.diffmap import DiffBlob, explain_blobs
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui import session_api as api
from tda.ui.app_diff import (
    AssistController,
    Box,
    best_unexplained,
    blob_boxes,
    expected_boxes,
    expected_payload,
    heat_rgba,
)

__all__ = ["ASSIST_CONFIRM_WAIT", "AssistMixin"]

class _SamLoader(QObject):
    """Carries the outcome of the background SAM load onto the GUI thread."""

    sigLoaded = Signal(object)


#: How long ``Space`` waits for an unfinished comparison before giving up on it.
ASSIST_CONFIRM_WAIT = 1.0


class AssistMixin:
    """The window half of the assist: SAM tools, prompt boxes, the heat map."""

    # ------------------------------------------------------------------ setup
    def _init_assist(self, sam_queue: Any) -> None:
        from tda.ui.canvas.sam_tools import SamBoxTool, SamPointTool

        self.sam_queue = sam_queue
        self._owns_queue = False
        self._sam_loading = False
        self.sam_available = sam_queue is not None
        self.sam_reason = "" if sam_queue is not None else "not loaded yet"
        self.sam_point = SamPointTool(self.canvas, None, queue=sam_queue, refine=True)
        self.sam_box = SamBoxTool(self.canvas, None, queue=sam_queue)
        #: Why the last prompt failed, until one is armed again.  Empty means
        #: the label may say "SAM ready" -- which it used to say after a submit
        #: raised, so the annotator clicked the same broken thing over and over.
        self._sam_failure = ""
        for tool in (self.sam_point, self.sam_box):
            tool.sigStroke.connect(self.on_stroke)
            tool.sigHint.connect(self.report)
            tool.sigError.connect(self.on_sam_error)

        self.assist = AssistController(self)
        self.assist.sigBlobs.connect(self._on_blobs)
        self.assist.sigFailed.connect(self.report_error)
        self.assist_result: Optional[dict] = None
        self._unexplained: dict[int, list[Box]] = {}
        #: Steps confirmed while the comparison had not landed (spec 4.4).
        self.unanalysed: set[int] = set()
        #: The diff-map box prompt of the frame, kept by the window because
        #: ``SamToolBase.detach()`` clears the tool's own copy: switching to the
        #: brush and back must not silently downgrade point+box to point-only.
        self._prompt_box: Optional[tuple[float, float, float, float]] = None

        self.heat_visible = False
        self.heat_item = QGraphicsPixmapItem()
        self.heat_item.setZValue(0.5)  # over the frame, under the label overlay
        self.heat_item.setVisible(False)
        self.canvas.scene().addItem(self.heat_item)
        self._sam_loader = _SamLoader(self)
        self._sam_loader.sigLoaded.connect(self._on_sam_loaded,
                                           Qt.ConnectionType.QueuedConnection)

    def _sam_tool(self):
        return self.sam_box if self._tool_name == "sam_box" else self.sam_point

    @S.guard
    def on_sam_error(self, text: str) -> None:
        """A SAM tool could not do what was asked; say it in both places.

        The status line is transient and the label is not: a prompt that failed
        has to leave a mark on the label, or the next click repeats it.  A
        dropped *result* (the frame moved on) is not a failure of the tool, so
        it only gets the line.
        """
        self.report_error(text)
        if not str(text).startswith("SAM result dropped"):
            self.note_sam_failure(text)

    def set_sam_instance(self, instance: Optional[str]) -> None:
        """Name the instance an applied mask belongs to.

        The tools have a ``"editing"`` fallback for when nothing says; relying
        on it would stamp two different edits with the same identity, so the
        window always answers explicitly.

        Changing it also throws the half-built prompt away: the points clicked
        so far were about the *previous* part, and sending them with the next
        click made every mask after the first commit a union of the two.
        """
        for tool in (self.sam_point, self.sam_box):
            if tool.instance != instance:
                tool.reset_prompt()
            tool.instance = instance

    def reset_sam_prompt(self) -> None:
        """Forget the points, the box drag and the candidates of both SAM tools.

        Called wherever the **editing layer is replaced from outside the tool**
        -- a commit, ``Esc``, an undo/redo, a restored sidecar.  That is the one
        rule worth remembering: a prompt refines the layer it produced, so the
        moment somebody else writes that layer the prompt describes nothing.
        """
        for tool in (self.sam_point, self.sam_box):
            tool.reset_prompt()

    def clear_prompt_box(self) -> None:
        """Forget the box prompt; the next frame's diff map proposes its own."""
        self._prompt_box = None

    def rearm_sam(self) -> None:
        """Give a freshly attached SAM tool its frame token and prompt box back.

        ``detach()`` cancels the in-flight prompt and clears the box, which is
        what makes a tool switch safe; the cost is that re-arming has to be
        explicit, or the next click would go out point-only.
        """
        tool = self._sam_tool()
        if not self.sam_available or self._tool_name not in ("sam_point", "sam_box"):
            return
        self.clear_sam_failure()     # arming a SAM tool is "try again"
        if compat.is_open(self.session) and self.session.image() is not None:
            tool.set_frame_token(self.session.current())
        if self._prompt_box is not None:
            tool.set_prompt_box(self._prompt_box)

    # ------------------------------------------------------------ frame hook
    def on_frame_changed_assist(self, key) -> None:
        """Re-stamp the SAM tools and start the comparison for the new frame.

        The token is **mandatory**: without one the tools refuse to prompt, so a
        frame with no image gets ``None`` (prompting is meaningless there) and
        every other frame gets its :class:`~tda.core.model.FrameKey`.
        """
        image = self.session.image()
        token = key if image is not None else None
        self.clear_prompt_box()
        for tool in (self.sam_point, self.sam_box):
            tool.overlay = self.overlay
            # The frame to crop from, explicitly: the canvas may be showing the
            # neighbour (``Tab``) and a prompt must never be about that one.
            tool.image = image
            tool.set_frame_token(token)
            # set_frame_token drops the box only when the token really changes,
            # and re-arming an attached tool may already have set it: say it.
            tool.set_prompt_box(None)
        self.set_sam_instance(getattr(self.session, "editing_instance", None))
        if not self.roi_editing:
            self.canvas.set_rubber_band(None)
        self.assist_result = None
        self.heat_item.setVisible(False)
        self.request_assist()

    def request_assist(self) -> None:
        """Compare the open frame with its task-card neighbour, off the GUI thread.

        The neighbour is the frame the card is written against -- ``j + 1``,
        the one the annotator came from -- so a blob marks the part that
        *re-appears* in the image on screen, which is exactly the part the card
        is asking for.
        """
        if not compat.is_open(self.session):
            return
        key = self.session.current()
        image = self.session.image()
        neighbour = compat.task_neighbour(self.session)
        self.report_neighbour_gap(key, neighbour)
        previous = None if neighbour is None else self.session.image_at(neighbour)
        self.assist.request(key, image, previous, self.roi(), self.expected_now())

    def report_neighbour_gap(self, key, neighbour: Optional[int]) -> None:
        """Say so when the neighbour is not the adjacent step.

        ``task_neighbour`` skips a step this view never photographed, and then
        the card spans more than one action -- which changes what the annotator
        is being asked to draw, so it has to be visible rather than inferred
        from a gap in the timeline.
        """
        if neighbour is None or abs(int(neighbour) - int(key.step)) == 1:
            return
        self.report(f"本视图缺帧：任务卡对照的是第 {neighbour} 帧，跨了 "
                    f"{abs(int(neighbour) - int(key.step))} 个步骤 / the card is "
                    f"diffed against step {neighbour}, not the adjacent one")

    def expected_now(self) -> dict:
        """What the frame already accounts for, as the worker wants it.

        A blob is explained when it sits on an instance whose geometry here
        differs from the neighbour's -- i.e. on one of the task card's own
        instances (and the children that come out with them) that already has a
        shape on this frame.  Reading it from the *visible masks* rather than
        from the stored ``box`` column is the whole fix: only bench rows carry a
        box, so the old version was always empty and nothing was ever explained.
        """
        wanted = self._card_instances()
        masks, _order = compat.overlay_layers(self.session)
        chosen = {key: mask for key, mask in masks.items() if key in wanted}
        boxes = []
        for key, inst in self.session.compiled().instances.items():
            if key in wanted and inst.visible is None and inst.box:
                boxes.append(tuple(int(round(float(v))) for v in inst.box))
        return expected_payload(chosen, boxes)

    def _card_instances(self) -> set[str]:
        """The task card's instances plus the children attached to them."""
        from tda.core.truth_inputs import instances_of

        wanted = {str(row["instance"]) for row in self.session.task_card()
                  if row.get("instance")}
        try:
            records = instances_of(self.db, int(self.session.desktop))
        except Exception:  # noqa: BLE001 - the card alone is still usable
            return wanted
        wanted |= {key for key, rec in records.items()
                   if getattr(rec, "attached", False) and rec.parent in wanted}
        return wanted

    # --------------------------------------------------------------- results
    @S.guard
    def _on_blobs(self, payload: object) -> None:
        """A comparison came back on the GUI thread."""
        self.assist_result = payload if isinstance(payload, dict) else None
        if self.assist_result is None or not compat.is_open(self.session):
            return
        if self.assist_result.get("key") != self.session.current():
            self.assist_result = None
            return
        if self.heat_visible:
            self._paint_heat()
        self._arm_from_card()

    def re_explain(self) -> Optional[dict]:
        """Re-split the blobs of this frame against what it holds *now*.

        Called after every commit and again at confirm time: a part that has
        just been drawn stops being an unexplained difference, and the review
        queue must not be told about it.
        """
        payload = self.assist_result
        if not payload or payload.get("key") != self.session.current():
            return payload
        boxes = expected_boxes(self.expected_now())
        explained, unexplained = explain_blobs(list(payload.get("blobs") or []), boxes)
        payload["explained"], payload["unexplained"] = explained, unexplained
        payload["expected"] = boxes
        return payload

    def unexplained_at_confirm(self) -> Optional[list[Box]]:
        """Boxes still unexplained on this frame, or ``None`` when not analysed.

        ``None`` is not the same as an empty list: an empty list says "this
        frame was compared and everything is accounted for", while ``None`` says
        "nobody looked", which the review queue has to be able to tell apart.
        """
        if not self._assist_ready():
            return None
        payload = self.re_explain()
        return blob_boxes((payload or {}).get("unexplained") or [])

    def _assist_ready(self, timeout: float = ASSIST_CONFIRM_WAIT) -> bool:
        """Make sure this frame's comparison has landed, within a short cap."""
        key = self.session.current()
        if (self.assist_result or {}).get("key") == key:
            return True
        if not self.assist.wait(timeout):
            return False
        QApplication.processEvents()          # let the queued result through
        return (self.assist_result or {}).get("key") == key

    def _arm_from_card(self) -> None:
        """Offer the strongest unexplained blob when the card wants a shape drawn."""
        rows = self.session.task_card()
        index = self.task_card.current_index()
        if not (0 <= index < len(rows)) or rows[index].get("kind") != api.KIND_ADD_SHAPE:
            return
        blob = best_unexplained(self.assist_result)
        if blob is not None:
            self.begin_add_shape(blob)

    def arm_prompt_box_for(self, instance: str) -> None:
        """Arm the box prompt for the instance the annotator actually activated.

        ``_on_blobs`` can only guess from the card's current row, and the
        annotator may well have double-clicked three rows further down; this is
        called from ``on_request_edit`` with the answer.
        """
        payload = self.assist_result
        if not payload or payload.get("key") != self.session.current():
            return
        rows = {str(r.get("instance")): r for r in self.session.task_card()}
        row = rows.get(str(instance))
        if row is None or row.get("kind") != api.KIND_ADD_SHAPE:
            return
        blob = best_unexplained(payload)
        if blob is not None:
            self.begin_add_shape(blob)

    def begin_add_shape(self, blob: DiffBlob) -> None:
        """Feed a changed region's box to SAM as the box half of point+box."""
        box = tuple(float(v) for v in blob.box)
        self._prompt_box = box
        for tool in (self.sam_point, self.sam_box):
            tool.set_prompt_box(box)
        if not self.roi_editing:
            self.canvas.set_rubber_band(box)
        self.report(f"prompt box from the difference map: "
                    f"{tuple(int(v) for v in blob.box)} ({blob.area} px)")

    def unexplained_boxes(self) -> list[Box]:
        """Boxes of the changes nothing on this frame accounts for."""
        return blob_boxes((self.assist_result or {}).get("unexplained") or [])

    def hand_over_unexplained(self, step: int, boxes: Optional[list[Box]]) -> None:
        """Give the review queue what the frame never explained (spec 4.4).

        ``None`` means the comparison never finished: the step is recorded as
        not analysed instead of being filed as "nothing to see", which would
        quietly claim a check that never happened.
        """
        if boxes is None:
            self.unanalysed.add(int(step))
            self.report_error(f"step {step}: the difference map did not finish; "
                              f"it is recorded as not analysed")
            return
        self.unanalysed.discard(int(step))
        if not compat.set_unexplained(self.session, step, boxes):
            self._unexplained[int(step)] = list(boxes)

    # ------------------------------------------------------------ SAM status
    def sam_status_text(self) -> str:
        """The SAM part of the status bar: state plus the candidate counter.

        The counter follows whichever tool actually *has* candidates, not the
        armed one: picking up the brush to tidy a proposal must not make the
        ``2/3`` disappear while ``C`` would still work.
        """
        if not self.sam_available:
            return f"SAM unavailable: {self.sam_reason}"
        if self._sam_failure:
            # A failed prompt with "SAM ready" still on the label is how an
            # annotator clicks the same broken thing twenty times.
            return f"SAM error: {self._sam_failure} (S to retry)"
        tool = self._candidate_tool()
        if tool is not None:
            return f"SAM ready · {tool.candidate_index + 1}/{tool.candidate_count}"
        return "SAM ready"

    def note_sam_failure(self, text: str) -> None:
        """Remember that the last prompt failed, until the next one is armed."""
        self._sam_failure = str(text).strip()[:60]
        self.update_status()

    def clear_sam_failure(self) -> None:
        """``S`` (or any fresh arming) means "try again"."""
        if self._sam_failure:
            self._sam_failure = ""
            self.update_status()

    def _candidate_tool(self):
        """The SAM tool holding candidates right now, preferring the armed one."""
        armed = self._sam_tool()
        for tool in (armed, self.sam_point, self.sam_box):
            if tool.candidate_count:
                return tool
        return None

    def set_sam_unavailable(self, reason: str) -> None:
        """Disable the SAM tools and say why; everything else keeps working."""
        self.sam_available = False
        self.sam_reason = str(reason)
        for tool in (self.sam_point, self.sam_box):
            tool.queue = None
        if self._tool_name in ("sam_point", "sam_box"):
            self._tool_name = "brush"
            self._attach_tool()
        self.update_status()

    def set_sam_queue(self, queue: Any, owns: bool = True) -> None:
        """Arm the SAM tools once a checkpoint has finished loading."""
        self.sam_queue = queue
        self._owns_queue = bool(owns)
        self.sam_available = queue is not None
        self.sam_reason = ""
        for tool in (self.sam_point, self.sam_box):
            tool.queue = queue
        self.update_status()

    def start_sam(self) -> None:
        """Load SAM on a background thread; the first frame must not wait for it."""
        if self.sam_queue is not None or self._sam_loading:
            return
        self._sam_loading = True
        self.sam_reason = "loading"
        self.update_status()
        loader = self._sam_loader

        def load() -> None:
            try:
                from tda.models.sam_service import SamQueue, SamService

                loader.sigLoaded.emit(SamQueue(SamService()))
            except Exception as exc:  # noqa: BLE001 - the app works without SAM
                loader.sigLoaded.emit(f"{type(exc).__name__}: {exc}")

        self._sam_loader_thread = threading.Thread(target=load, name="tda-sam-load",
                                                   daemon=True)
        self._sam_loader_thread.start()

    @S.guard
    def _on_sam_loaded(self, outcome: object) -> None:
        """The background load finished -- possibly after the window closed."""
        self._sam_loading = False
        if isinstance(outcome, str):
            self.set_sam_unavailable(outcome)
            return
        if self.closed:
            # Nobody is going to use it: a queue nobody stops keeps a thread and
            # a CUDA context alive for the life of the process.
            outcome.stop()
            return
        self.set_sam_queue(outcome, owns=True)
        self.report("SAM ready")

    # -------------------------------------------------------------- actions
    @S.guard
    def act_cycle_candidate(self) -> None:
        """``C``: the next of SAM's proposals for the same click."""
        tool = self._candidate_tool()
        if tool is None or tool.candidate_count < 2:
            self.report("no other SAM candidate"
                        + ("" if self.sam_available else f": {self.sam_reason}"))
            return
        tool.cycle_candidate()
        self.update_status()

    @S.guard
    def act_toggle_heat(self) -> None:
        """``D``: the frame-difference heat map over the image."""
        self.heat_visible = not self.heat_visible
        if self.heat_visible:
            self._paint_heat()
        else:
            self.heat_item.setVisible(False)
        self.report("difference heat map " + ("on" if self.heat_visible else "off"))

    def _paint_heat(self) -> None:
        payload = self.assist_result
        if not payload or payload.get("delta") is None:
            self.report("no difference map for this frame yet")
            self.heat_item.setVisible(False)
            return
        rgba = np.ascontiguousarray(heat_rgba(payload["delta"], payload.get("roi")))
        height, width = rgba.shape[:2]
        image = QImage(rgba.data, width, height, 4 * width,
                       QImage.Format.Format_RGBA8888)
        self.heat_item.setPixmap(QPixmap.fromImage(image.copy()))
        self.heat_item.setVisible(True)

    # ------------------------------------------------------------- lifecycle
    def shutdown_assist(self) -> None:
        """Stop every assist thread: the SAM queue, the loader, the diff worker."""
        for tool in (self.sam_point, self.sam_box):
            tool.queue = None       # nothing new may be submitted from here on
        if self._owns_queue and self.sam_queue is not None:
            try:
                self.sam_queue.stop()
            except Exception:  # pragma: no cover - a wedged GPU call
                pass
            self.sam_queue = None
        loader = getattr(self, "_sam_loader_thread", None)
        if loader is not None and loader.is_alive():
            loader.join(30.0)       # a half-built SamQueue must not outlive us
        self.assist.shutdown()
