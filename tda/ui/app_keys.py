"""The keyboard and the flash compare: what a key press means, and when.

Mixed into :class:`tda.ui.app.MainWindow`.  Two things live here because they
are the same question asked twice -- *is this gesture about the frame the
annotator is looking at?*

* **the keyboard.**  One table (:mod:`tda.ui.app_actions`) decides what a key
  does, in which mode, and what holding it means; this is the part that asks
  the table and refuses when the focus is in a text field, a modal dialog is
  up, or another window of ours has the keyboard.
* **the flash compare.**  While ``Tab`` is held the canvas shows the neighbour
  frame and everything else in the window still belongs to the open one, so
  every tool is inert and SAM refuses to prompt.  ``_flashing`` is the single
  piece of state that says so, and it is cleared from every direction a
  release might never arrive from.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication, QWidget

from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_support as S

__all__ = ["FLASH_UNNAMED", "KeysMixin"]

#: ``_flashing`` when a neighbour is on screen but its step number is unknown.
FLASH_UNNAMED = -1


class KeysMixin:
    """Key dispatch, the shortcut context, and the ``Tab`` flash compare."""

    # -------------------------------------------------------------- keyboard
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: D102
        kind = event.type()
        if self.closed:
            return False  # a window on its way out must not eat anybody's keys
        if kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            focus = QApplication.focusWidget()
            if focus is None or focus is self or self.isAncestorOf(focus):
                if self.handle_key(event):
                    return True
        return super().eventFilter(obj, event)

    def handle_key(self, event) -> bool:
        """Run the action bound to ``event``; ``True`` when it was consumed.

        Auto-repeat is answered per action, because holding a key means three
        different things:

        * a **repeat** action fires again on every repeat -- holding ``]`` grows
          the brush, holding ``PgDn`` walks back through the machine.  Swallowing
          the repeats for every binding (which is what used to happen) left the
          annotator pressing ``]`` forty times;
        * a **hold** action (``Tab``) consumes the repeat without firing: letting
          it through would walk Qt's focus chain into a combo box, after which
          :func:`~tda.ui.app_actions.blocks_shortcuts` switched the whole
          keyboard off until the annotator clicked somewhere;
        * everything else ignores the repeat: a held ``Enter`` commits once.

        An auto-repeat of a key that is *not* bound is left alone, so ordinary
        widgets keep their repeats.
        """
        if not self._shortcut_context_ok():
            return False
        focus = self._focus_widget()
        if A.blocks_shortcuts(focus) or A.navigates_a_list(focus, event.key()):
            return False
        action = A.action_for(event.key(), event.modifiers(), self.mode)
        if action is None:
            # Any other key is the annotator moving on: a flash that is still up
            # because its release went missing ends here.
            if event.type() == QEvent.Type.KeyPress:
                self.end_flash()
            return False
        if not action.hold and event.type() == QEvent.Type.KeyPress:
            self.end_flash()
        if event.isAutoRepeat() and not action.repeat:
            return True
        pressed = event.type() == QEvent.Type.KeyPress
        if action.hold:
            self.dispatch(action, pressed)
        elif pressed:
            self.dispatch(action)
        return True

    def _shortcut_context_ok(self) -> bool:
        """Are the window's shortcuts live at all right now?

        Not while a modal dialog is up, and not while the focus sits in another
        **visible** window of ours -- the cheat sheet is a child dialog, so
        without this its ``Esc`` would also discard the edit underneath it.

        The visibility check matters: Qt keeps the application focus on a widget
        of a window that has been closed but not yet deleted, so a torn-down
        window would otherwise switch off the keyboard of the one that replaced
        it -- which is exactly what made a whole suite fail when another suite
        had run first.
        """
        if QApplication.activeModalWidget() is not None:
            return False
        focus = QApplication.focusWidget()
        if focus is None:
            return True
        other = focus.window()
        if other is self._cheat_sheet and other is not None:
            # The sheet is read-only and non-modal, and it is precisely what
            # somebody has open while they are still learning the keys: it must
            # not be the reason none of them work.
            return True
        return other is self or not other.isVisible()

    def _focus_widget(self) -> Optional[QWidget]:
        """The focused widget *of this window*, or ``None``.

        ``QApplication.focusWidget()`` is authoritative while the window is
        active; when it is not (or nothing has been shown yet) the window's own
        ``focusWidget()`` still knows which child last took the focus, which is
        what makes the text-field guard work before the first activation.
        """
        focus = QApplication.focusWidget()
        if focus is not None and not (focus is self or self.isAncestorOf(focus)):
            focus = None
        return focus if focus is not None else self.focusWidget()

    def dispatch(self, action: A.Action, *extra) -> None:
        """Call the window slot an action names."""
        slot = getattr(self, action.slot, None)
        if slot is None:
            self.report_error(f"no slot {action.slot!r} for {action.name}")
            return
        slot(*(tuple(extra) if action.hold else tuple(action.args)))

    @S.guard
    def act_flash_compare(self, pressed: bool, other: bool = False) -> None:
        """Hold ``Tab`` to see the neighbour frame without moving the view.

        The neighbour is the frame the task card is written against -- the one
        the annotator came from, ``j + 1`` in reverse order.  ``Shift+Tab``
        shows the other side instead.

        While it is held the canvas is showing a frame that is **not** the one
        being annotated, so nothing may be drawn on it: every tool is detached
        and the SAM tools refuse to prompt.  A brush stroke or a SAM click on
        the flashed image asked about the neighbour's pixels and wrote the
        answer into this frame's layer -- and in reverse order the part the card
        asks for is *absent* in j+1, so the mask was confidently wrong.
        """
        if not pressed:
            self.end_flash()
            return
        if self._flashing is not None or not compat.is_open(self.session):
            return
        # Whatever was being offered was offered about *this* frame; the canvas
        # is about to show another one.
        self.forget_draft_ghost()
        image = compat.flash_image(self.session, other=other)
        if image is None:
            return
        # The step shown, so the status bar can name it; ``FLASH_UNNAMED`` when
        # the adapter cannot say which one it handed back.
        step = compat.flash_step(self.session, other=other)
        self._flashing = FLASH_UNNAMED if step is None else int(step)
        self._pause_tools(True)
        self._show_image(image)
        self.update_status()

    def end_flash(self) -> None:
        """Put the frame back on the canvas; safe to call at any time.

        Called from everywhere a release might never arrive: the key release,
        the window losing focus (``Alt+Tab`` while holding ``Tab`` is the one
        the reviewer hit), any other key, and every frame change.  Without it
        the canvas stayed on the neighbour's image with this frame's overlay and
        status, and every tool stayed live over it.
        """
        if self._flashing is None:
            return
        self._flashing = None
        self._pause_tools(False)
        image = self.session.image() if compat.is_open(self.session) else None
        if image is not None:
            self._show_image(image)
        self.update_status()

    def is_flashing(self) -> bool:
        """Is the canvas showing a neighbour frame rather than the open one?"""
        return self._flashing is not None

    def _show_image(self, image) -> None:
        """Swap the picture under the overlay, keeping zoom and centre."""
        zoom, centre = self.canvas.zoom_factor(), self._canvas_centre()
        self.canvas.set_image(image)
        self.canvas.set_zoom(zoom)
        self.canvas.center_on(centre)
        self.canvas.refresh()

    def _pause_tools(self, paused: bool) -> None:
        """Make every tool inert, or arm the chosen one again."""
        if paused:
            # The canvas is about to show another frame.  Whatever ``Shift+C``
            # was walking is an offer about the frame being annotated, and the
            # annotator is now looking at the other one; the armed box goes
            # back to the difference map's own, which is what ``rearm_sam``
            # will put back when ``Tab`` is released.
            self.reset_prompt_rank()
        for tool in (self.sam_point, self.sam_box):
            tool.paused = bool(paused)
        if paused:
            self._detach_tool()
        else:
            self._attach_tool()

    def event(self, ev) -> bool:  # noqa: D102 - Qt override
        # A lost key release (Alt+Tab, a focus steal, a system dialog) would
        # otherwise leave the canvas stuck on the neighbour for good.  Qt
        # delivers the deactivation here, not through ``changeEvent``.
        kind = ev.type()
        if kind in (QEvent.Type.WindowDeactivate, QEvent.Type.FocusOut) or (
            kind == QEvent.Type.ActivationChange and not self.isActiveWindow()
        ):
            self.end_flash()
        return super().event(ev)

    @S.guard
    def act_flash_other(self, pressed: bool) -> None:
        """``Shift+Tab``: flash the frame on the *other* side of this one."""
        self.act_flash_compare(pressed, other=True)
