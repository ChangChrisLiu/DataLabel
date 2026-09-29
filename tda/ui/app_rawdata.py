"""Saying, once and clearly, that the raw data drive is not where it was.

Mixed into :class:`tda.ui.app.MainWindow`. The raw dataset lives on an external
drive whose letter changes (:mod:`tda.core.rawroot`), and the day it came back
as ``G:`` instead of ``F:`` the window showed three blank views and said
"no image in this view" about frames that were on disk. What the window owes
the annotator is three things, and this module is all of them:

* a **non-modal bar** above the canvas while the drive is not connected (or
  ambiguous), with the one thing to do about it -- plug it in, press ``F5``;
* **view buttons** that say which views have no readable frame, before the
  annotator clicks one;
* a **placeholder** that names the file it could not read, instead of an empty
  canvas that looks like a missing step.

``F5`` looks for the drive again; everything that had failed to read is read
again when it is found.
"""
from __future__ import annotations

from typing import Optional

from tda.core import rawroot
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.app_widgets import Bar

__all__ = ["NO_IMAGE_TEXT", "RawDataMixin", "UNREADABLE_VIEW_STYLE"]

#: The placeholder when the step really has no image in this view.
NO_IMAGE_TEXT = "no image for this step in this view\n本视图在该步骤没有图像"
#: How a view button looks when none of its frames can be read right now.
UNREADABLE_VIEW_STYLE = "QToolButton { color: #8a8a8a; font-style: italic; }"


class RawDataMixin:
    """The raw-drive bar, the view buttons' readability and the placeholder."""

    def _init_rawdata(self) -> None:
        """Build the bar and read the resolver's answer (after the central widget)."""
        self.raw_bar = Bar(self)
        self.raw_bar.add_button("重新查找 F5 / look again", self.act_recheck_raw_data)
        self.raw_bar.setStyleSheet("QWidget { background: #5a2a1a; color: #fff2e0; }")
        # Above the canvas, not under it with the other bars: it is not about
        # the frame, it is about every frame.
        self._central_layout.insertWidget(0, self.raw_bar)
        self._marked_desktop: Optional[int] = None
        raw = rawroot.current()
        if raw is None:
            raw = rawroot.configure(self.paths)
        self.raw_root = raw
        if raw.log_line:
            self.logger.log(raw.log_level, raw.log_line)
        self.refresh_raw_bar()

    # ---------------------------------------------------------------- the bar
    def refresh_raw_bar(self) -> None:
        """Show the bar exactly while the raw drive is configured and not here."""
        raw = getattr(self, "raw_root", None)
        if raw is not None and raw.configured and not raw.connected:
            self.raw_bar.show_text(raw.message)
        else:
            self.raw_bar.hide()

    def raw_bar_text(self) -> str:
        """What the bar says, or ``""`` while it is hidden."""
        return "" if self.raw_bar.isHidden() else self.raw_bar.label.text()

    @S.guard
    def act_recheck_raw_data(self) -> bool:
        """Look for the raw drive again; ``True`` when the answer changed.

        What had failed to read is read again when the drive turns up: the
        decoded-image cache, the timeline's grey rows and the view buttons all
        remembered "nothing there".
        """
        before = getattr(self, "raw_root", None)
        raw = rawroot.configure(self.paths)
        self.raw_root = raw
        changed = before is None or (before.resolved, before.status) != (
            raw.resolved, raw.status)
        if changed:
            if raw.log_line:
                self.logger.log(raw.log_level, raw.log_line)
            images = getattr(self.session, "images", None)
            if images is not None:
                images.clear()
            forget = getattr(self.timeline, "forget_thumbnails", None)
            if callable(forget):
                forget()
        self.refresh_raw_bar()
        self._marked_desktop = None
        self.mark_view_buttons()
        if changed and compat.is_open(self.session):
            self.render_frame()
        return changed

    # ---------------------------------------------------------- view buttons
    def view_problem(self, view: str) -> Optional[str]:
        """Why ``view`` of the open machine shows nothing, or ``None``."""
        images = getattr(self.session, "images", None)
        desktop = getattr(self.session, "desktop", None)
        if images is None or desktop is None or not hasattr(images, "view_problem"):
            return None
        try:
            return images.view_problem(int(desktop), str(view))
        except Exception as exc:  # noqa: BLE001 - a marker must never break a paint
            self.logger.warning("could not check view %s: %s", view, exc)
            return None

    def mark_view_buttons(self, force: bool = False) -> None:
        """Grey the buttons of views with no readable frame, and say why.

        Once per machine (and on ``F5``): each view costs a few file checks,
        and a frame change is on a budget.
        """
        desktop = getattr(self.session, "desktop", None)
        if not force and desktop is not None and desktop == self._marked_desktop:
            return
        self._marked_desktop = desktop
        for view, button in self.view_buttons.items():
            problem = self.view_problem(view)
            button.setProperty("unreadable", bool(problem))
            button.setStyleSheet(UNREADABLE_VIEW_STYLE if problem else "")
            base = f"{view}"
            button.setToolTip(
                f"{base}：这一视角没有能读到的原图 / no readable image in this view\n"
                f"{problem}" if problem else base)

    # ------------------------------------------------------------ placeholder
    def no_image_text(self, key) -> str:
        """What the placeholder says about a frame with no pixels."""
        images = getattr(self.session, "images", None)
        why = None
        if images is not None and hasattr(images, "why_unreadable"):
            try:
                why = images.why_unreadable(key)
            except Exception:  # noqa: BLE001 - the default text is still true
                why = None
        return why or NO_IMAGE_TEXT

    def show_no_image(self, key) -> None:
        """Put the right sentence on the placeholder and in the status bar."""
        text = self.no_image_text(key)
        self.placeholder_label.setText(text)
        if text == NO_IMAGE_TEXT:
            self.report(f"step {key.step}: 本视图没有图像 / no image in this view")
        else:
            self.report(f"step {key.step}: {text}")
