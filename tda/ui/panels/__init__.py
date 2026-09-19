"""Dock panels of the annotator window, and the one thing they all need.

A panel is handed a session and then follows it; what none of them used to
handle is the session having **no frame open**, which happens for real: a
desktop/view with no frame rows leaves it closed, and asking such a session for
the current frame raises.  Since the panels answer Qt signals, that exception
escaped into the event loop -- and at start-up it killed
``MainWindow.__init__`` outright, so the app would not launch until somebody
edited the INI by hand.
"""
from __future__ import annotations

from typing import Any

__all__ = ["session_is_open"]


def session_is_open(session: Any) -> bool:
    """Is there a frame to show?  ``False`` for ``None`` and for a closed session.

    ``is_open`` is a property on :class:`~tda.ui.session.AnnotationSession` and
    a method on some stubs; both are accepted, and anything that raises counts
    as closed -- a panel deciding whether to draw is not the place to find out
    that the database is unhappy.
    """
    if session is None:
        return False
    flag = getattr(session, "is_open", None)
    try:
        if callable(flag):
            return bool(flag())
        if flag is not None:
            return bool(flag)
        session.current()
    except Exception:  # noqa: BLE001 - "cannot say" means "do not draw"
        return False
    return True
