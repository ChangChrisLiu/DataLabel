"""Offscreen tests for the canvas, label overlay, pixel tools and undo stack.

The asynchronous SAM prompt tools have their own file,
``tests/test_sam_tools.py``.

Everything here runs on the ``offscreen`` Qt platform plugin (see
``conftest.py``); the env var is also set at import time because the
``QApplication`` is created by a session fixture that may be built before the
function-scoped autouse fixture runs.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from tda.core import masks as M
from tda.ui.canvas.overlay import (
    EDIT_RGB,
    GHOST_RGB,
    OCCLUDER_RGB,
    PALETTE_64,
    LabelOverlay,
    palette_color,
)
from tda.ui.app_widgets import RoiBoxTool
from tda.ui.canvas.tools import BrushTool, EraserTool, OccluderTool, Tool
from tda.ui.canvas.view import CURSOR_MAX_PX, ImageCanvas, ToolCursor
from tda.ui.commands import KINDS, Op, UndoStack, edit_editing_mask_op


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def two_masks() -> tuple[dict[str, np.ndarray], list[str]]:
    a = np.zeros((40, 50), dtype=bool)
    a[5:15, 5:20] = True
    b = np.zeros((40, 50), dtype=bool)
    b[10:25, 15:35] = True  # overlaps a, painted on top
    return {"inst-a": a, "inst-b": b}, ["inst-a", "inst-b"]


def _rgb(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = np.arange(w, dtype=np.uint8)[None, :]
    img[..., 1] = np.arange(h, dtype=np.uint8)[:, None]
    img[..., 2] = 90
    return img


def _shown(canvas: ImageCanvas, w: int = 400, h: int = 400) -> ImageCanvas:
    canvas.resize(w, h)
    canvas.show()
    QApplication.processEvents()
    return canvas


def _press(x: float, y: float, button=Qt.MouseButton.LeftButton) -> QMouseEvent:
    pos = QPointF(x, y)
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        pos,
        button,
        button,
        Qt.KeyboardModifier.NoModifier,
    )


def _argb(img, x: int, y: int) -> tuple[int, int, int, int]:
    """(a, r, g, b) of one pixel of a QImage."""
    px = img.pixel(x, y)
    return ((px >> 24) & 0xFF, (px >> 16) & 0xFF, (px >> 8) & 0xFF, px & 0xFF)


def _pixels(img) -> np.ndarray:
    """The whole QImage as an HxW uint32 array (a copy)."""
    raw = np.frombuffer(bytes(img.constBits()), dtype=np.uint32)
    return raw.reshape(img.height(), img.width()).copy()


def _grow(rect, hw) -> tuple[int, int, int, int]:
    """The 1-px outline halo an outlined partial repaint must cover."""
    h, w = hw
    x0, y0, x1, y1 = rect
    return (max(0, x0 - 1), max(0, y0 - 1), min(w, x1 + 1), min(h, y1 + 1))



# ---------------------------------------------------------------------------
# LabelOverlay
# ---------------------------------------------------------------------------
def test_palette_has_64_distinct_colors():
    assert len(PALETTE_64) == 64
    assert len(set(PALETTE_64)) == 64
    assert all(len(c) == 3 and all(0 <= v <= 255 for v in c) for c in PALETTE_64)


def test_palette_color_is_stable_and_in_table():
    first = palette_color("D13/s010/psu#1")
    assert first == palette_color("D13/s010/psu#1")
    assert first in PALETTE_64
    assert palette_color("a") != palette_color("a-different-key")


def test_set_instances_builds_labelmap_and_palette(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)

    assert ov.labelmap.dtype == np.uint16
    assert ov.labelmap.shape == (40, 50)
    assert set(np.unique(ov.labelmap).tolist()) == {0, 1, 2}
    assert ov.labelmap[6, 6] == 1  # only in a
    assert ov.labelmap[20, 30] == 2  # only in b
    assert ov.labelmap[12, 17] == 2  # overlap: b is on top
    assert ov.id2key == {1: "inst-a", 2: "inst-b"}
    assert ov.palette[1] == palette_color("inst-a")
    assert ov.palette[2] == palette_color("inst-b")


def test_qimage_is_non_null_and_colors_match_palette(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)

    img = ov.qimage()
    assert not img.isNull()
    assert (img.width(), img.height()) == (50, 40)
    assert _argb(img, 0, 0) == (0, 0, 0, 0)  # background is transparent

    a_alpha, *a_rgb = _argb(img, 6, 6)
    assert tuple(a_rgb) == palette_color("inst-a")
    assert a_alpha == 110  # default fill alpha
    b_alpha, *b_rgb = _argb(img, 30, 20)
    assert tuple(b_rgb) == palette_color("inst-b")


def test_qimage_outline_is_opaque_and_toggleable(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)

    on = ov.qimage(outline=True)
    assert _argb(on, 5, 5)[0] == 255  # top-left corner of a: on the boundary
    assert _argb(on, 10, 10)[0] == 110  # interior of a

    off = ov.qimage(outline=False)
    assert _argb(off, 5, 5)[0] == 110


def test_qimage_alpha_argument_changes_fill(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    assert _argb(ov.qimage(alpha=200), 10, 10)[0] == 200


def test_paint_returns_dirty_rect_containing_the_stroke():
    ov = LabelOverlay((40, 50))
    ov.set_editing("inst-x", np.zeros((40, 50), dtype=bool))

    rect = ov.paint((25, 20), 5, True)
    assert rect == (20, 15, 31, 26)
    assert ov.editing[20, 25]
    assert not ov.editing[20, 31]  # just outside the disk
    ys, xs = np.nonzero(ov.editing)
    assert xs.min() >= rect[0] and xs.max() < rect[2]
    assert ys.min() >= rect[1] and ys.max() < rect[3]
    # a disk, not a square
    assert 0.6 < ov.editing.sum() / (11 * 11) < 0.9


def test_paint_clips_the_dirty_rect_to_the_image():
    ov = LabelOverlay((40, 50))
    rect = ov.paint((1, 1), 6, True)
    assert rect == (0, 0, 8, 8)
    assert ov.editing[0, 0]


def test_paint_with_add_false_erases():
    ov = LabelOverlay((40, 50))
    ov.paint((25, 20), 6, True)
    ov.paint((25, 20), 3, False)
    assert not ov.editing[20, 25]
    assert ov.editing[20, 31]


def test_paint_entirely_outside_the_image_returns_none():
    ov = LabelOverlay((40, 50))
    assert ov.paint((-40, -40), 3, True) is None
    assert ov.paint((200, 20), 3, True) is None
    assert not ov.editing.any()
    # nothing was marked dirty, so a later qimage() must not rebuild anything
    ov.qimage()
    ov.last_rebuild_rect = None
    assert ov.paint((-40, -40), 3, True) is None
    ov.qimage()
    assert ov.last_rebuild_rect is None


def test_paint_occluder_keeps_one_layer_per_type():
    ov = LabelOverlay((40, 50))
    assert ov.occluders == {}

    hand = ov.paint_occluder((10, 10), 3, True, "hand")
    cable = ov.paint_occluder((40, 30), 3, True, "cable")
    assert hand is not None and cable is not None
    assert set(ov.occluders) == {"hand", "cable"}
    assert ov.occluders["hand"][10, 10]
    assert not ov.occluders["hand"][30, 40], "a cable stroke leaked into hand"
    assert ov.occluders["cable"][30, 40]
    assert not ov.occluders["cable"][10, 10], "a hand stroke leaked into cable"
    assert not ov.editing.any()
    # every layer is rendered
    img = ov.qimage()
    assert tuple(_argb(img, 10, 10)[1:]) == OCCLUDER_RGB
    assert tuple(_argb(img, 40, 30)[1:]) == OCCLUDER_RGB


def test_paint_occluder_rejects_an_unknown_type():
    ov = LabelOverlay((40, 50))
    with pytest.raises(ValueError):
        ov.paint_occluder((10, 10), 3, True, "banana")


def test_set_occluder_replaces_one_type_only():
    ov = LabelOverlay((40, 50))
    ov.paint_occluder((10, 10), 3, True, "hand")
    mask = np.zeros((40, 50), dtype=bool)
    mask[30:35, 30:35] = True
    ov.set_occluder(mask, "cable")
    assert ov.occluders["hand"][10, 10]
    assert ov.occluders["cable"][32, 32]
    assert not ov.occluders["cable"][10, 10]


def test_editing_layer_draws_on_top_of_instances(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    edit = np.zeros((40, 50), dtype=bool)
    edit[8:12, 8:12] = True
    ov.set_editing("inst-c", edit)

    img = ov.qimage()
    assert tuple(_argb(img, 10, 10)[1:]) == EDIT_RGB
    assert tuple(_argb(img, 6, 6)[1:]) == palette_color("inst-a")


def test_qimage_rebuilds_only_the_dirty_rect(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    ov.qimage()
    assert ov.last_rebuild_rect == (0, 0, 50, 40)

    rect = ov.paint((40, 30), 3, True)
    img = ov.qimage(rect)
    # grown by 1 px so the outline of the newly exposed edge is correct
    assert ov.last_rebuild_rect == _grow(rect, (40, 50))
    assert tuple(_argb(img, 40, 30)[1:]) == EDIT_RGB
    # the untouched part of the buffer still carries the instance colours
    assert tuple(_argb(img, 6, 6)[1:]) == palette_color("inst-a")


def test_a_ghost_repaints_only_what_it_covers(two_masks):
    """A draft preview must not cost a whole-frame composite per keypress."""
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order)
    ov.qimage()

    ghost = np.zeros((40, 50), dtype=bool)
    ghost[10:14, 20:26] = True
    ov.set_ghost(ghost, (20, 10, 26, 14))
    img = ov.qimage()

    assert ov.ghost_rect == (20, 10, 26, 14)
    assert ov.last_rebuild_rect == _grow((20, 10, 26, 14), (40, 50))
    assert tuple(_argb(img, 22, 12)[1:]) == GHOST_RGB
    # walking to the next candidate repaints both regions and nothing else
    other = np.zeros((40, 50), dtype=bool)
    other[30:34, 4:8] = True
    ov.set_ghost(other, (4, 30, 8, 34))
    ov.qimage()
    assert ov.last_rebuild_rect == _grow((4, 10, 26, 34), (40, 50))
    # ... and dismissing it repaints only where it was
    ov.clear_ghost()
    ov.qimage()
    assert ov.last_rebuild_rect == _grow((4, 30, 8, 34), (40, 50))
    assert ov.has_ghost is False


def test_a_ghost_without_a_rect_still_repaints_everything(two_masks):
    """The safe fallback: a caller that cannot say marks the whole buffer."""
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order)
    ov.qimage()
    ghost = np.zeros((40, 50), dtype=bool)
    ghost[10:14, 20:26] = True
    ov.set_ghost(ghost)
    ov.qimage()
    assert ov.last_rebuild_rect == (0, 0, 50, 40)


def test_replacing_a_ghost_of_unknown_extent_repaints_everything(two_masks):
    """The one on screen has to be taken off, and nobody said where it is.

    Marking only the *new* rect left the previous proposal painted wherever it
    happened to be -- 450 px of it, measured -- because "I know where this one
    goes" says nothing about where the last one went.
    """
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order)
    first = np.zeros((40, 50), dtype=bool)
    first[2:8, 2:20] = True
    ov.set_ghost(first)                      # no rect: extent unknown
    ov.qimage()

    second = np.zeros((40, 50), dtype=bool)
    second[30:34, 40:46] = True
    ov.set_ghost(second, (40, 30, 46, 34))   # a rect for the new one only
    ov.qimage()
    assert ov.last_rebuild_rect == (0, 0, 50, 40)
    assert tuple(_argb(ov.qimage(), 4, 4)[1:]) != GHOST_RGB, \
        "the previous proposal is still on screen"


def test_a_first_ghost_with_a_rect_repaints_only_that_rect(two_masks):
    """Nothing was up, so nothing outside the new rect can be stale."""
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order)
    ov.qimage()
    ghost = np.zeros((40, 50), dtype=bool)
    ghost[30:34, 40:46] = True
    ov.set_ghost(ghost, (40, 30, 46, 34))
    ov.qimage()
    assert ov.last_rebuild_rect == _grow((40, 30, 46, 34), (40, 50))


def test_the_editing_layer_is_drawn_over_the_ghost(two_masks):
    """The annotator's own pixels are never hidden by a proposal."""
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order)
    both = np.zeros((40, 50), dtype=bool)
    both[10:14, 20:26] = True
    ov.set_ghost(both, (20, 10, 26, 14))
    ov.set_editing("inst-c", both)
    img = ov.qimage()
    assert tuple(_argb(img, 22, 12)[1:]) == EDIT_RGB


def test_partial_repaint_matches_a_full_repaint_after_an_erase(two_masks):
    """An eraser must not leave the old outline ring outside the dirty rect."""
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    filled = np.zeros((40, 50), dtype=bool)
    filled[10:30, 10:40] = True
    ov.set_editing("inst-c", filled)
    ov.qimage()

    rect = ov.paint((25, 20), 5, False)  # erase a disc out of the middle
    partial = _pixels(ov.qimage(rect))

    ov.force_full_rebuild()
    full = _pixels(ov.qimage())
    assert int(np.count_nonzero(partial != full)) == 0


def test_partial_repaint_matches_a_full_repaint_after_a_brush(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    ov.qimage()

    rect = ov.paint((25, 20), 4, True)
    partial = _pixels(ov.qimage(rect))
    ov.force_full_rebuild()
    assert int(np.count_nonzero(partial != _pixels(ov.qimage()))) == 0


def test_visible_false_renders_transparent_and_restores(two_masks):
    masks, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    shown = _pixels(ov.qimage())
    assert shown.any()

    ov.visible = False
    hidden = _pixels(ov.qimage())
    assert not hidden.any(), "visible=False did not invalidate the cache"

    ov.visible = True
    assert int(np.count_nonzero(_pixels(ov.qimage()) != shown)) == 0


def test_qimage_never_returns_a_stale_buffer():
    ov = LabelOverlay((40, 50))
    ov.qimage()
    ov.paint((10, 10), 2, True)
    img = ov.qimage()  # no rect given, but a paint is pending
    assert tuple(_argb(img, 10, 10)[1:]) == EDIT_RGB


def test_set_editing_rejects_a_wrong_shape():
    ov = LabelOverlay((40, 50))
    with pytest.raises(ValueError):
        ov.set_editing("x", np.zeros((10, 10), dtype=bool))


# ---------------------------------------------------------------------------
# what a repaint is allowed to cost (task B7)
# ---------------------------------------------------------------------------
def _windows(masks_: dict) -> dict:
    from tda.core import masks as M

    return {key: M.bbox(mask) for key, mask in masks_.items()}


def test_set_instances_with_the_same_arrays_repaints_nothing(two_masks):
    """A commit refreshes the overlay twice; the second time must be free.

    ``_commit`` repaints the layers, and the session's own re-announcement of
    the frame repaints them again from the *same* compiled frame -- the same
    numpy arrays, in the same order.  At 12 MP with forty instances that second
    pass was a 112 ms label map and a 95 ms composite for a picture that could
    not have changed.
    """
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order, windows=_windows(masks_))
    ov.qimage()

    ov.set_instances(dict(masks_), list(order), windows=_windows(masks_))
    ov.qimage()
    assert ov.last_rebuild_rect is None
    assert ov.id2key == {1: "inst-a", 2: "inst-b"}


def test_hiding_one_instance_repaints_only_where_it_was(two_masks):
    """``H`` takes one layer off; the other thirty-nine do not move."""
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order, windows=_windows(masks_))
    ov.qimage()

    kept = {"inst-a": masks_["inst-a"]}
    ov.set_instances(kept, ["inst-a"], windows=_windows(masks_))
    ov.qimage()
    assert ov.last_rebuild_rect == _grow((15, 10, 35, 25), (40, 50))
    assert ov.id2key == {1: "inst-a"}


def test_a_changed_layer_repaints_both_of_its_boxes(two_masks):
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order, windows=_windows(masks_))
    ov.qimage()

    moved = np.zeros((40, 50), dtype=bool)
    moved[30:36, 40:46] = True
    changed = {"inst-a": masks_["inst-a"], "inst-b": moved}
    ov.set_instances(changed, order, windows=_windows(changed))
    ov.qimage()
    assert ov.last_rebuild_rect == _grow((15, 10, 46, 36), (40, 50))


def test_a_reordered_stack_repaints_everything(two_masks):
    """Who is on top is not a local question; the safe answer is the frame."""
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order, windows=_windows(masks_))
    ov.qimage()
    ov.set_instances(masks_, list(reversed(order)), windows=_windows(masks_))
    ov.qimage()
    assert ov.last_rebuild_rect == (0, 0, 50, 40)


def test_a_layer_change_without_a_window_repaints_everything(two_masks):
    """No window, no promise: the whole buffer is the only safe answer."""
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order, windows=_windows(masks_))
    ov.qimage()
    moved = {"inst-a": masks_["inst-a"], "inst-b": masks_["inst-b"].copy()}
    ov.set_instances(moved, order)          # nothing said about where they are
    ov.qimage()
    assert ov.last_rebuild_rect == (0, 0, 50, 40)


def test_a_partial_layer_repaint_matches_a_full_one(two_masks):
    """Every localised layer change, checked against the frame it stands for."""
    masks_, order = two_masks
    scenes = [
        ({"inst-a": masks_["inst-a"]}, ["inst-a"]),                 # hide b
        (masks_, order),                                             # show it again
        ({"inst-b": masks_["inst-b"]}, ["inst-b"]),                 # hide a
        (masks_, order),
        (masks_, list(reversed(order))),                             # swap the stack
        (masks_, order),
    ]
    ov = LabelOverlay((40, 50))
    ref = LabelOverlay((40, 50))
    ov.set_instances(masks_, order, windows=_windows(masks_))
    ov.qimage()
    for layers, stack in scenes:
        ov.set_instances(layers, stack, windows=_windows(layers))
        here = _pixels(ov.qimage())
        ref.set_instances(layers, stack, windows=_windows(layers))
        ref.force_full_rebuild()
        assert int(np.count_nonzero(here != _pixels(ref.qimage()))) == 0


def test_clearing_the_editing_layer_repaints_only_where_it_was(two_masks):
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order, windows=_windows(masks_))
    edit = np.zeros((40, 50), dtype=bool)
    edit[8:12, 8:14] = True
    ov.set_editing("inst-c", edit)
    ov.qimage()

    ov.clear_editing()
    ov.qimage()
    assert ov.last_rebuild_rect == _grow((8, 8, 14, 12), (40, 50))
    assert not ov.editing.any()


def test_replacing_the_editing_layer_repaints_both_of_its_boxes(two_masks):
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order, windows=_windows(masks_))
    first = np.zeros((40, 50), dtype=bool)
    first[8:12, 8:14] = True
    ov.set_editing("inst-c", first)
    ov.qimage()

    second = np.zeros((40, 50), dtype=bool)
    second[30:34, 40:46] = True
    ov.set_editing("inst-c", second)
    ov.qimage()
    assert ov.last_rebuild_rect == _grow((8, 8, 46, 34), (40, 50))
    assert ov.editing[32, 42] and not ov.editing[10, 10]


def test_clearing_an_editing_layer_that_was_never_set_repaints_nothing(two_masks):
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order, windows=_windows(masks_))
    ov.qimage()
    ov.clear_editing()
    ov.qimage()
    assert ov.last_rebuild_rect is None


def test_a_clipped_qimage_leaves_the_rest_stale_and_finishes_it_later(two_masks):
    """The composite is paid for where the annotator is looking, when they look.

    A 12 MP ARGB buffer is 48 MB; at 59 % zoom the viewport holds a sixth of
    it.  ``clip`` is the canvas saying which part it is about to draw -- the
    rest stays on the stale list, so panning to it composites it then, and the
    buffer that results has to be the one a whole-frame composite would leave.
    """
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ref = LabelOverlay((40, 50))
    for target in (ov, ref):
        target.set_instances(masks_, order, windows=_windows(masks_))

    ov.qimage(clip=(0, 0, 25, 40))
    assert ov.last_rebuild_rect == (0, 0, 25, 40)
    left = _pixels(ov.qimage(clip=(0, 0, 25, 40)))
    assert ov.last_rebuild_rect is None, "the clipped region was rebuilt twice"

    full = _pixels(ref.qimage())
    assert int(np.count_nonzero(left[:, :25] != full[:, :25])) == 0

    ov.qimage(clip=(25, 0, 50, 40))
    assert ov.last_rebuild_rect == (25, 0, 50, 40)
    assert int(np.count_nonzero(_pixels(ov.qimage()) != full)) == 0
    assert ov.last_rebuild_rect is None, "nothing was left stale"


def test_a_clipped_repaint_of_a_stroke_matches_the_whole_frame(two_masks):
    """Strokes, ghosts and occluders through the clipped path, pixel for pixel."""
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ref = LabelOverlay((40, 50))
    for target in (ov, ref):
        target.set_instances(masks_, order, windows=_windows(masks_))
        target.paint((24, 20), 5, True)
        ghost = np.zeros((40, 50), dtype=bool)
        ghost[2:8, 30:40] = True
        target.set_ghost(ghost, (30, 2, 40, 8))
        target.paint_occluder((44, 34), 4, True, "hand")

    for clip in ((0, 0, 17, 40), (17, 0, 34, 40), (34, 0, 50, 40),
                 (0, 0, 50, 20), (0, 20, 50, 40)):
        ov.qimage(clip=clip)
    ref.force_full_rebuild()
    assert int(np.count_nonzero(_pixels(ov.qimage()) != _pixels(ref.qimage()))) == 0


def test_a_clip_outside_the_stale_region_rebuilds_nothing(two_masks):
    masks_, order = two_masks
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks_, order, windows=_windows(masks_))
    ov.qimage()
    rect = ov.paint((5, 5), 2, True)
    assert rect is not None
    ov.qimage(clip=(40, 30, 50, 40))
    assert ov.last_rebuild_rect is None
    ov.qimage(clip=(0, 0, 12, 12))
    assert ov.last_rebuild_rect == _grow(rect, (40, 50))


def test_overlay_uses_no_python_pixel_loops():
    """A 1 MP overlay must render fast enough to be numpy-only."""
    ov = LabelOverlay((1000, 1000))
    m = np.zeros((1000, 1000), dtype=bool)
    m[100:900, 100:900] = True
    ov.set_instances({"k": m}, ["k"])
    t0 = time.perf_counter()
    ov.qimage()
    assert (time.perf_counter() - t0) < 0.5


# ---------------------------------------------------------------------------
# ImageCanvas
# ---------------------------------------------------------------------------
def test_set_image_sets_scene_rect_and_keeps_the_array(qapp):
    canvas = _shown(ImageCanvas())
    rgb = _rgb(300, 400)
    canvas.set_image(rgb)
    assert canvas.image_hw() == (300, 400)
    assert canvas.sceneRect().width() == 400
    assert canvas.sceneRect().height() == 300
    np.testing.assert_array_equal(canvas.image_rgb(), rgb)


def test_zoom_to_box_zooms_in_and_shows_the_box(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(400, 400))
    canvas.zoom_to((100, 100, 200, 200))
    QApplication.processEvents()

    assert canvas.zoom_factor() > 1.0
    x0, y0, x1, y1 = canvas.viewport_image_rect()
    assert x0 <= 100 and y0 <= 100 and x1 >= 200 and y1 >= 200
    # 5% margin per side, square box in a square viewport
    assert (x1 - x0) < 150 and (y1 - y0) < 150


def test_viewport_image_rect_is_clipped_to_the_image(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(200, 300))
    canvas.set_zoom(0.2)  # whole image far smaller than the viewport
    QApplication.processEvents()
    assert canvas.viewport_image_rect() == (0, 0, 300, 200)


def test_wheel_zooms_about_the_cursor(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(400, 400))
    canvas.set_zoom(2.0)
    QApplication.processEvents()

    pos = QPointF(120.0, 90.0)
    before = canvas.image_pos(pos)
    event = QWheelEvent(
        pos,
        canvas.viewport().mapToGlobal(pos),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(canvas.viewport(), event)
    QApplication.processEvents()

    assert canvas.zoom_factor() == pytest.approx(2.5, rel=1e-3)  # 2.0 * 1.25
    after = canvas.image_pos(pos)
    assert after[0] == pytest.approx(before[0], abs=1.0)
    assert after[1] == pytest.approx(before[1], abs=1.0)


def test_zoom_is_clamped(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(100, 100))
    canvas.set_zoom(10_000.0)
    assert canvas.zoom_factor() <= canvas.MAX_ZOOM
    canvas.set_zoom(1e-6)
    assert canvas.zoom_factor() >= canvas.MIN_ZOOM


def test_mouse_signals_carry_image_coordinates(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(400, 400))
    canvas.set_zoom(1.0)
    canvas.center_on((200, 200))
    QApplication.processEvents()

    seen: list[tuple[float, float]] = []
    canvas.sigMousePress.connect(lambda x, y, ev: seen.append((x, y)))
    centre = QPointF(canvas.viewport().width() / 2, canvas.viewport().height() / 2)
    QApplication.sendEvent(canvas.viewport(), _press(centre.x(), centre.y()))

    assert len(seen) == 1
    assert seen[0][0] == pytest.approx(200.0, abs=1.5)
    assert seen[0][1] == pytest.approx(200.0, abs=1.5)


def test_middle_drag_pans_without_emitting_tool_signals(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(600, 600))
    canvas.set_zoom(4.0)
    QApplication.processEvents()

    presses: list[tuple[float, float]] = []
    canvas.sigMousePress.connect(lambda x, y, ev: presses.append((x, y)))
    before = canvas.viewport_image_rect()
    QApplication.sendEvent(
        canvas.viewport(), _press(200.0, 200.0, Qt.MouseButton.MiddleButton)
    )
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(240.0, 230.0),
        QPointF(240.0, 230.0),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas.viewport(), move)
    QApplication.processEvents()

    assert presses == []
    assert canvas.viewport_image_rect() != before


def test_canvas_renders_with_overlay_and_pixel_grid(qapp, two_masks):
    from PySide6.QtGui import QImage, QPainter

    masks, order = two_masks
    canvas = _shown(ImageCanvas(), 200, 200)
    canvas.set_image(_rgb(40, 50))
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    canvas.set_overlay(ov)
    canvas.set_zoom(8.0)  # > 4 => pixel grid is drawn
    QApplication.processEvents()
    assert canvas.zoom_factor() > 4.0

    target = QImage(200, 200, QImage.Format.Format_ARGB32)
    target.fill(0)
    painter = QPainter(target)
    canvas.render(painter)
    painter.end()
    assert target.constBits() is not None


def test_refresh_with_a_dirty_rect_updates_only_that_region(qapp, two_masks):
    masks, order = two_masks
    canvas = _shown(ImageCanvas(), 200, 200)
    canvas.set_image(_rgb(40, 50))
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    canvas.set_overlay(ov)
    canvas.refresh()

    rect = ov.paint((40, 30), 3, True)
    canvas.refresh(rect)
    assert ov.last_rebuild_rect == _grow(rect, (40, 50))
    # the item draws straight from the overlay buffer: no copy, full frame size
    assert canvas.overlay_image() is ov.qimage()
    assert canvas.overlay_image().size().toTuple() == (50, 40)


def test_refresh_ignores_an_empty_dirty_rect(qapp, two_masks):
    """A stroke entirely outside the image must not trigger a full repaint."""
    masks, order = two_masks
    canvas = _shown(ImageCanvas(), 200, 200)
    canvas.set_image(_rgb(40, 50))
    ov = LabelOverlay((40, 50))
    ov.set_instances(masks, order)
    canvas.set_overlay(ov)
    canvas.refresh()

    tool = BrushTool(canvas, ov, radius=3)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)
    ov.last_rebuild_rect = None
    tool.on_press(-50.0, -50.0, None)
    tool.on_release(-60.0, -50.0, None)
    assert strokes == []
    assert ov.last_rebuild_rect is None
    assert not ov.editing.any()


def test_minimap_stays_in_the_corner_while_the_view_scrolls(qapp):
    canvas = _shown(ImageCanvas(), 400, 400)
    canvas.set_image(_rgb(900, 900))
    mini = canvas.minimap()
    corner = (
        canvas.viewport().width() - mini.width() - 8,
        canvas.viewport().height() - mini.height() - 8,
    )
    assert (mini.x(), mini.y()) == corner

    # zooming and panning scroll the scene; the minimap must not ride along
    canvas.zoom_to((300, 300, 600, 600))
    QApplication.processEvents()
    assert (mini.x(), mini.y()) == corner
    canvas.center_on((700.0, 700.0))
    QApplication.processEvents()
    assert (mini.x(), mini.y()) == corner


def test_minimap_tracks_the_viewport(qapp):
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(400, 500))
    canvas.zoom_to((100, 100, 200, 200))
    QApplication.processEvents()
    mini = canvas.minimap()
    assert mini.isVisibleTo(canvas)
    assert mini.image_hw == (400, 500)
    x0, y0, x1, y1 = mini.view_rect
    assert x1 > x0 and y1 > y0
    assert (x1 - x0) < 500  # not the whole image


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@pytest.fixture
def rig(qapp):
    canvas = _shown(ImageCanvas(), 300, 300)
    canvas.set_image(_rgb(60, 80))
    ov = LabelOverlay((60, 80))
    ov.set_editing("inst-x", np.zeros((60, 80), dtype=bool))
    canvas.set_overlay(ov)
    return canvas, ov


def test_tool_base_is_a_no_op(rig):
    canvas, ov = rig
    tool = Tool(canvas, ov)
    tool.on_press(1.0, 2.0, None)
    tool.on_move(1.0, 2.0, None)
    tool.on_release(1.0, 2.0, None)
    assert not ov.editing.any()


def test_tool_attach_connects_the_canvas_signals(rig):
    canvas, ov = rig
    tool = BrushTool(canvas, ov, radius=3)
    tool.attach()
    QApplication.sendEvent(canvas.viewport(), _press(150.0, 150.0))
    QApplication.processEvents()
    assert ov.editing.any()

    ov.editing[:] = False
    tool.detach()
    QApplication.sendEvent(canvas.viewport(), _press(150.0, 150.0))
    QApplication.processEvents()
    assert not ov.editing.any()


def test_brush_paints_a_continuous_stroke_and_emits_the_dirty_rect(rig):
    canvas, ov = rig
    tool = BrushTool(canvas, ov, radius=3)
    strokes: list[tuple[int, int, int, int]] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(10.0, 10.0, None)
    tool.on_move(40.0, 10.0, None)  # one big jump: must be interpolated
    tool.on_release(40.0, 10.0, None)

    assert ov.editing[10, 10] and ov.editing[10, 40]
    assert ov.editing[10, 25], "the stroke has a gap between samples"
    assert len(strokes) == 1
    x0, y0, x1, y1 = strokes[0]
    assert x0 <= 7 and x1 >= 44 and y0 <= 7 and y1 >= 14


def test_brush_snapshots_the_mask_before_the_stroke(rig):
    canvas, ov = rig
    tool = BrushTool(canvas, ov, radius=2)
    tool.on_press(20.0, 20.0, None)
    tool.on_release(20.0, 20.0, None)
    assert tool.stroke_before is not None
    assert not tool.stroke_before.any()
    assert ov.editing.any()


def test_brush_radius_is_adjustable(rig):
    canvas, ov = rig
    tool = BrushTool(canvas, ov, radius=2)
    tool.set_radius(9)
    assert tool.radius == 9
    rect = ov.paint((30, 30), tool.radius, True)
    assert rect == (21, 21, 40, 40)


def test_eraser_removes_pixels(rig):
    canvas, ov = rig
    ov.editing[:] = True
    tool = EraserTool(canvas, ov, radius=4)
    tool.on_press(30.0, 30.0, None)
    tool.on_release(30.0, 30.0, None)
    assert not ov.editing[30, 30]
    assert ov.editing[0, 0]


def test_occluder_tool_paints_the_layer_of_its_type(rig):
    canvas, ov = rig
    tool = OccluderTool(canvas, ov, radius=3)
    assert tool.occluder_type == "hand"
    tool.occluder_type = "tool"
    tool.on_press(25.0, 25.0, None)
    tool.on_release(25.0, 25.0, None)
    assert ov.occluders["tool"][25, 25]
    assert set(ov.occluders) == {"tool"}
    assert not ov.editing.any()


def test_occluder_tool_switching_type_does_not_merge_strokes(rig):
    canvas, ov = rig
    tool = OccluderTool(canvas, ov, radius=3)
    tool.occluder_type = "hand"
    tool.on_press(10.0, 10.0, None)
    tool.on_release(10.0, 10.0, None)
    tool.occluder_type = "cable"
    tool.on_press(40.0, 40.0, None)
    tool.on_release(40.0, 40.0, None)

    assert set(ov.occluders) == {"hand", "cable"}
    assert ov.occluders["hand"][10, 10] and not ov.occluders["hand"][40, 40]
    assert ov.occluders["cable"][40, 40] and not ov.occluders["cable"][10, 10]
    # the before-snapshot follows the type too, so undo stays per layer
    assert tool.stroke_before is not None and not tool.stroke_before.any()


def test_occluder_tool_rejects_an_unknown_type(rig):
    canvas, ov = rig
    with pytest.raises(ValueError):
        OccluderTool(canvas, ov, occluder_type="banana")


# ---------------------------------------------------------------------------
# UndoStack
# ---------------------------------------------------------------------------
def _mask(hw=(20, 20), box=(2, 2, 8, 8)) -> np.ndarray:
    m = np.zeros(hw, dtype=bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


class MaskHolder:
    """Minimal undo target: one editing mask per instance."""

    def __init__(self, hw=(20, 20)) -> None:
        self.hw = hw
        self.masks: dict[str, np.ndarray] = {}
        self.calls = 0

    def apply(self, payload: dict) -> None:
        self.calls += 1
        self.masks[payload["instance"]] = M.decode_rle(payload["rle_after"])


def test_op_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        Op("not-a-kind", {}, {})
    assert "edit_editing_mask" in KINDS


def test_edit_editing_mask_op_stores_rle_not_arrays():
    before, after = _mask(), _mask(box=(2, 2, 12, 12))
    op = edit_editing_mask_op("inst-a", before, after)
    assert op.kind == "edit_editing_mask"
    assert set(op.payload) == {"instance", "rle_before", "rle_after"}
    for value in op.payload.values():
        assert not isinstance(value, np.ndarray)
    assert isinstance(op.payload["rle_after"], dict)
    assert isinstance(op.payload["rle_after"]["counts"], str)
    # the inverse is the same op with the two masks swapped
    assert op.inverse["rle_after"] == op.payload["rle_before"]
    assert op.inverse["rle_before"] == op.payload["rle_after"]
    np.testing.assert_array_equal(M.decode_rle(op.payload["rle_after"]), after)


def test_push_applies_and_undo_redo_restore_masks_exactly():
    holder = MaskHolder()
    stack = UndoStack()
    stack.register("edit_editing_mask", holder.apply, holder.apply)

    before, after = _mask(), _mask(box=(5, 5, 15, 15))
    stack.push(edit_editing_mask_op("inst-a", before, after))
    np.testing.assert_array_equal(holder.masks["inst-a"], after)

    op = stack.undo()
    assert op is not None and op.kind == "edit_editing_mask"
    np.testing.assert_array_equal(holder.masks["inst-a"], before)

    op = stack.redo()
    assert op is not None
    np.testing.assert_array_equal(holder.masks["inst-a"], after)


def test_push_without_apply_only_records():
    holder = MaskHolder()
    stack = UndoStack()
    stack.register("edit_editing_mask", holder.apply, holder.apply)
    stack.push(edit_editing_mask_op("inst-a", _mask(), _mask(box=(1, 1, 3, 3))), apply=False)
    assert holder.calls == 0
    assert stack.can_undo
    stack.undo()
    assert holder.calls == 1


def test_undo_and_redo_are_empty_at_the_edges():
    stack = UndoStack()
    assert stack.undo() is None
    assert stack.redo() is None
    assert not stack.can_undo and not stack.can_redo


def test_a_new_push_clears_the_redo_branch():
    holder = MaskHolder()
    stack = UndoStack()
    stack.register("edit_editing_mask", holder.apply, holder.apply)
    stack.push(edit_editing_mask_op("a", _mask(), _mask(box=(1, 1, 4, 4))))
    stack.undo()
    assert stack.can_redo
    stack.push(edit_editing_mask_op("a", _mask(), _mask(box=(6, 6, 9, 9))))
    assert not stack.can_redo
    assert stack.redo() is None


def test_stack_is_capped_at_200_entries():
    holder = MaskHolder()
    stack = UndoStack()
    stack.register("edit_editing_mask", holder.apply, holder.apply)
    for i in range(250):
        stack.push(edit_editing_mask_op("a", _mask(), _mask(box=(1, 1, 2 + i % 10, 3))))
    assert stack.LIMIT == 200
    assert len(stack) == 200


def test_push_needs_a_registered_handler():
    stack = UndoStack()
    with pytest.raises(LookupError):
        stack.push(Op("set_zorder", {"a": 1}, {"a": 0}))


def test_other_kinds_round_trip_through_their_handlers():
    seen: list[tuple[str, dict]] = []
    stack = UndoStack()
    for kind in ("set_zorder", "set_pair_override", "set_frame_override",
                 "set_occluder", "commit_keyframe"):
        stack.register(
            kind,
            lambda p, k=kind: seen.append((f"do:{k}", p)),
            lambda p, k=kind: seen.append((f"undo:{k}", p)),
        )
        stack.push(Op(kind, {"v": 1}, {"v": 0}))
    assert [name for name, _ in seen] == [
        "do:set_zorder", "do:set_pair_override", "do:set_frame_override",
        "do:set_occluder", "do:commit_keyframe",
    ]
    stack.undo()
    assert seen[-1] == ("undo:commit_keyframe", {"v": 0})


def test_clear_drops_both_branches():
    holder = MaskHolder()
    stack = UndoStack()
    stack.register("edit_editing_mask", holder.apply, holder.apply)
    stack.push(edit_editing_mask_op("a", _mask(), _mask(box=(1, 1, 4, 4))))
    stack.clear()
    assert not stack.can_undo and not stack.can_redo and len(stack) == 0


# ---------------------------------------------------------------------------
# what the annotator sees is what a whole-frame composite would have drawn
# ---------------------------------------------------------------------------
def _screen(canvas: ImageCanvas) -> np.ndarray:
    """The canvas as the annotator sees it, as an HxW uint32 ARGB array.

    The **viewport** is grabbed, not the view: ``QGraphicsView.grab()``
    renders its child viewport through a path that clips the scene items away,
    so it produces a picture with no overlay in it at all -- which would make
    every comparison below pass without proving anything.
    """
    from PySide6.QtGui import QImage

    QApplication.processEvents()
    image = canvas.viewport().grab().toImage().convertToFormat(
        QImage.Format.Format_ARGB32
    )
    w, h = image.width(), image.height()
    raw = np.frombuffer(bytes(image.constBits()), dtype=np.uint32)
    return raw.reshape(h, image.bytesPerLine() // 4)[:, :w].copy()


def _fidelity_layers() -> tuple[dict, list]:
    """Four shapes: one on the left border, one in the middle, one in the
    bottom-right corner, and a 3 px column crossing the whole frame."""
    hw = (160, 200)
    layers = {
        "edge-left": np.zeros(hw, dtype=bool),
        "middle": np.zeros(hw, dtype=bool),
        "corner": np.zeros(hw, dtype=bool),
        "thin": np.zeros(hw, dtype=bool),
    }
    layers["edge-left"][20:140, 0:40] = True
    layers["middle"][40:120, 60:150] = True
    layers["corner"][130:160, 170:200] = True
    layers["thin"][0:160, 96:99] = True
    return layers, ["edge-left", "middle", "corner", "thin"]


@pytest.fixture
def fidelity_scene(qapp):
    """A 200x160 frame on a 130x100 viewport, plus a factory for fresh overlays.

    The factory matters: an overlay's buffer starts at zero, so a region the
    lazy path never composites shows *through* -- which is how a comparison can
    tell "composited on demand" apart from "composited earlier and still
    right".  Reusing one overlay hides every mistake this file exists to find.
    """
    layers, order = _fidelity_layers()
    canvas = _shown(ImageCanvas(), 130, 100)
    canvas.set_image(_rgb(160, 200))

    def fresh(shown=None, stack=None) -> LabelOverlay:
        given = layers if shown is None else shown
        ov = LabelOverlay((160, 200))
        ov.set_instances(given, order if stack is None else stack,
                         windows=_windows(given))
        return ov

    return canvas, fresh, layers, order


def _walk(canvas, fresh, moves, *, whole: bool, dress=None) -> list[np.ndarray]:
    """Take a *new* overlay through ``moves``, grabbing the screen at each one.

    ``whole=False`` is the shipped path: the overlay is attached (one clipped
    composite) and from then on only what is about to be drawn is composited.
    ``whole=True`` composites the entire frame before every grab.  The two
    lists of screens have to be identical.
    """
    canvas.set_overlay(None)
    # Attached on a close-up, so the one composite the attachment makes covers
    # a corner of the frame and everything the moves below reach is genuinely
    # uncomposited.  Attaching while the whole frame is on screen composites
    # the whole frame, and then there is nothing left for the comparison to
    # catch.
    canvas.set_zoom(8.0)
    canvas.center_on((100.0, 80.0))
    ov = fresh()
    if dress is not None:
        dress(ov)
    canvas.set_overlay(ov)
    shots = []
    for move in moves:
        move()
        if whole:
            ov.force_full_rebuild()
            ov.qimage(alpha=canvas.overlay_alpha, outline=canvas.overlay_outline)
        shots.append(_screen(canvas))
    return shots


def _same(canvas, fresh, moves, what: str, dress=None) -> None:
    lazy = _walk(canvas, fresh, moves, whole=False, dress=dress)
    whole = _walk(canvas, fresh, moves, whole=True, dress=dress)
    for index, (here, there) in enumerate(zip(lazy, whole)):
        wrong = int(np.count_nonzero(here != there))
        assert wrong == 0, f"{wrong} px differ at {what} step {index}"


ZOOMS = (0.37, 0.5, 1.0, 1.0 / 3.0, 2.5, 3.7, 7.25)
CENTRES = ((0.0, 0.0), (99.0, 80.0), (97.5, 80.5), (199.0, 159.0),
           (40.0, 20.0), (150.0, 130.0))


def test_the_screen_is_the_same_at_every_zoom_and_pan(fidelity_scene):
    """Fractional zooms, the four borders, and a shape crossing the viewport."""
    canvas, fresh, _layers, _order = fidelity_scene
    for zoom in ZOOMS:
        moves = []
        for centre in CENTRES:
            moves.append(lambda z=zoom, c=centre: (canvas.set_zoom(z),
                                                   canvas.center_on(c)))
        _same(canvas, fresh, moves, f"zoom {zoom}")


def test_the_screen_is_the_same_with_every_display_setting(fidelity_scene):
    """``Q`` outline, ``,``/``.`` opacity and ``A`` overlays-off, at two zooms."""
    canvas, fresh, _layers, _order = fidelity_scene
    try:
        for zoom, centre in ((1.0 / 3.0, (99.0, 80.0)), (3.7, (60.0, 45.0))):
            for outline in (True, False):
                for alpha in (0, 70, 110, 255):
                    canvas.overlay_outline = outline
                    canvas.overlay_alpha = alpha

                    def move(z=zoom, c=centre):
                        canvas.set_zoom(z)
                        canvas.center_on(c)

                    _same(canvas, fresh, [move],
                          f"outline={outline} alpha={alpha} zoom={zoom}")
    finally:
        canvas.overlay_outline = True
        canvas.overlay_alpha = 110


def test_the_screen_is_the_same_with_the_overlays_switched_off(fidelity_scene):
    """``A``: the layers go, and what is left has to be the same either way."""
    canvas, fresh, _layers, _order = fidelity_scene

    def hide(ov):
        ov.visible = False

    _same(canvas, fresh,
          [lambda: (canvas.set_zoom(2.5), canvas.center_on((99.0, 80.0)))],
          "overlays off", dress=hide)


def test_the_screen_is_the_same_with_the_edit_layers_up(fidelity_scene):
    """The editing layer, the draft ghost under it and two occluder types."""
    canvas, fresh, _layers, _order = fidelity_scene

    def dress(ov):
        edit = np.zeros((160, 200), dtype=bool)
        edit[70:110, 80:130] = True        # crosses the viewport border
        ghost = np.zeros((160, 200), dtype=bool)
        ghost[60:100, 70:120] = True       # under the editing layer, overlapping
        ov.set_ghost(ghost, (70, 60, 120, 100))
        ov.set_editing("being-drawn", edit)
        ov.paint_occluder((100, 80), 9, True, "hand")
        ov.paint_occluder((30, 140), 6, True, "cable")

    moves = []
    for zoom, centre in ((0.37, (99.0, 80.0)), (1.0, (100.0, 80.0)),
                         (2.5, (95.5, 79.5)), (7.25, (99.0, 80.0))):
        moves.append(lambda z=zoom, c=centre: (canvas.set_zoom(z),
                                               canvas.center_on(c)))
    _same(canvas, fresh, moves, "the edit layers", dress=dress)


def test_the_screen_is_the_same_after_hiding_and_showing_instances(fidelity_scene):
    """``H`` on an instance, then back: each step against the whole composite."""
    canvas, fresh, layers, order = fidelity_scene

    def hide(name):
        shown = {k: v for k, v in layers.items() if k != name}
        stack = [k for k in order if k != name]

        def move():
            canvas.overlay().set_instances(shown, stack, windows=_windows(shown))
            canvas.refresh()

        return move

    moves = [lambda: (canvas.set_zoom(2.5), canvas.center_on((99.0, 80.0)))]
    moves += [hide(name) for name in
              ("middle", "thin", "edge-left", None, "corner", None)]
    _same(canvas, fresh, moves, "hiding an instance")


def test_panning_one_step_at_a_time_never_shows_a_stale_strip(fidelity_scene):
    """The strip a pan exposes is composited before it is drawn, every time."""
    canvas, fresh, _layers, _order = fidelity_scene

    def step():
        hbar, vbar = canvas.horizontalScrollBar(), canvas.verticalScrollBar()
        hbar.setValue(hbar.value() + 17)
        vbar.setValue(vbar.value() + 11)

    moves = [lambda: (canvas.set_zoom(3.0), canvas.center_on((40.0, 40.0)))]
    moves += [step] * 12
    _same(canvas, fresh, moves, "a pan")


def test_zooming_out_composites_the_frame_that_comes_into_view(fidelity_scene):
    """``F`` from a close-up: everything the smaller scale reveals is drawn."""
    canvas, fresh, _layers, _order = fidelity_scene
    moves = [lambda: (canvas.set_zoom(6.0), canvas.center_on((99.0, 80.0)))]
    moves += [lambda z=z: canvas.set_zoom(z) for z in (5.0, 3.0, 1.0, 0.5, 0.37)]
    _same(canvas, fresh, moves, "zooming out")


def test_a_stroke_lands_where_the_image_coordinates_say_at_any_zoom(qapp):
    """Hit testing is untouched: a click is the same pixel it always was."""
    canvas = _shown(ImageCanvas(), 130, 100)
    canvas.set_image(_rgb(160, 200))
    ov = LabelOverlay((160, 200))
    canvas.set_overlay(ov)
    brush = BrushTool(canvas, ov, radius=0)
    brush.attach()
    checked = 0
    for zoom, centre in ((1.0, (100.0, 80.0)), (3.7, (60.0, 45.0)),
                         (0.5, (100.0, 80.0)), (7.25, (120.5, 90.5))):
        canvas.set_zoom(zoom)
        canvas.center_on(centre)
        ov.clear_editing()
        for pos in (QPointF(11.0, 7.0), QPointF(64.0, 52.0), QPointF(129.0, 99.0)):
            x, y = canvas.image_pos(pos)
            expected = (int(round(x)), int(round(y)))
            if not (0 <= expected[0] < 200 and 0 <= expected[1] < 160):
                continue
            before = int(ov.editing.sum())
            brush.on_press(x, y, _press(pos.x(), pos.y()))
            brush.on_release(x, y, _press(pos.x(), pos.y()))
            if int(ov.editing.sum()) == before:
                continue
            painted = {tuple(p) for p in np.argwhere(ov.editing)}
            assert (expected[1], expected[0]) in painted, (
                f"a click at {pos.toTuple()} (zoom {zoom}) painted "
                f"{sorted(painted)[:4]}, not {expected}"
            )
            checked += 1
    assert checked >= 8, f"only {checked} clicks were inside the frame"


def test_the_prompt_point_is_drawn_without_touching_hit_testing(qapp):
    """``Shift+C`` marks where to click; the mark must not *be* clickable.

    It is painted in ``drawForeground`` like the rubber band rather than added
    to the scene, so there is nothing under the cursor that could swallow a
    press or shift the coordinate a tool is handed.
    """
    canvas = _shown(ImageCanvas())
    canvas.set_image(_rgb(400, 400))
    canvas.set_zoom(1.0)
    canvas.center_on((200, 200))
    QApplication.processEvents()
    assert canvas.prompt_point() is None

    seen: list[tuple[float, float]] = []
    canvas.sigMousePress.connect(lambda x, y, ev: seen.append((x, y)))
    centre = QPointF(canvas.viewport().width() / 2, canvas.viewport().height() / 2)
    QApplication.sendEvent(canvas.viewport(), _press(centre.x(), centre.y()))
    without = list(seen)

    canvas.set_prompt_point((200, 200))
    assert canvas.prompt_point() == (200, 200)
    QApplication.processEvents()
    seen.clear()
    QApplication.sendEvent(canvas.viewport(), _press(centre.x(), centre.y()))

    assert seen == without
    canvas.set_prompt_point(None)
    assert canvas.prompt_point() is None


# ---------------------------------------------------------------------------
# the armed tool is visible under the mouse (task U1, report 1, ruling R1)
# ---------------------------------------------------------------------------
def _blank_canvas() -> ImageCanvas:
    canvas = ImageCanvas()
    canvas.resize(300, 300)
    canvas.set_image(np.zeros((200, 200, 3), dtype=np.uint8))
    return canvas


def test_a_circle_cursor_is_the_size_of_the_stroke_and_follows_the_zoom(qapp):
    canvas = _blank_canvas()
    canvas.set_zoom(1.0)
    canvas.set_tool_cursor(ToolCursor("circle", EDIT_RGB, radius=8))
    assert canvas.cursor_diameter() == 17, "2r + 1 image px at 100 %"

    canvas.set_zoom(4.0)
    assert canvas.cursor_diameter() == 68, "the ring has to follow the wheel"
    canvas.set_tool_cursor(ToolCursor("circle", EDIT_RGB, radius=2))
    assert canvas.cursor_diameter() == 20, "... and the bracket keys"


def test_an_unusable_circle_falls_back_to_a_crosshair(qapp):
    canvas = _blank_canvas()
    canvas.set_zoom(1.0)
    canvas.set_tool_cursor(ToolCursor("circle", EDIT_RGB, radius=400))
    assert canvas.cursor_diameter() > CURSOR_MAX_PX
    assert canvas.viewport().cursor().shape() == Qt.CursorShape.CrossCursor


def test_the_eraser_cursor_is_not_the_brush_cursor(qapp):
    brush = ToolCursor("circle", EDIT_RGB, radius=8)
    eraser = ToolCursor("circle", (245, 245, 245), radius=8, dashed=True)
    assert brush != eraser, "colour and dash are what tell them apart"


def test_the_tool_cursor_survives_a_middle_drag_pan(qapp):
    from PySide6.QtTest import QTest

    canvas = _blank_canvas()
    canvas.show()
    QApplication.processEvents()
    canvas.set_tool_cursor(ToolCursor("cross"))
    vp = canvas.viewport()
    QTest.mousePress(vp, Qt.MouseButton.MiddleButton,
                     Qt.KeyboardModifier.NoModifier, QPoint(50, 50))
    QTest.mouseRelease(vp, Qt.MouseButton.MiddleButton,
                       Qt.KeyboardModifier.NoModifier, QPoint(60, 60))
    assert vp.cursor().shape() == Qt.CursorShape.CrossCursor, (
        "the pan put the arrow back and the armed tool became invisible"
    )
    canvas.close()


# ---------------------------------------------------------------------------
# the chassis rectangle (task U1, report 2, ruling U-ROI-2)
# ---------------------------------------------------------------------------
def test_the_roi_rectangle_offers_eight_handles_in_image_coordinates(qapp):
    canvas = _blank_canvas()
    canvas.set_roi((20, 40, 120, 140), editing=True)
    handles = canvas.roi_handle_points()
    assert set(handles) == {"nw", "n", "ne", "w", "e", "sw", "s", "se"}
    assert handles["nw"] == (20.0, 40.0)
    assert handles["se"] == (120.0, 140.0)
    assert handles["n"] == (70.0, 40.0)
    assert handles["e"] == (120.0, 90.0)


def test_a_stored_roi_is_drawn_and_follows_the_overlays_key(qapp):
    canvas = _blank_canvas()
    canvas.set_roi((20, 40, 120, 140), editing=False)
    assert canvas.roi_rect() == (20.0, 40.0, 120.0, 140.0)
    assert canvas.roi_outline_visible is True
    canvas.roi_outline_visible = False
    canvas.viewport().update()          # must not raise with the outline hidden
    QApplication.processEvents()


def test_the_roi_tool_resizes_moves_and_redraws(qapp):
    """Every gesture ruling U-ROI-2 asks for, on one rectangle."""
    canvas = _blank_canvas()
    canvas.set_zoom(1.0)
    tool = RoiBoxTool(canvas, None)
    tool.set_rect((20.0, 40.0, 120.0, 140.0))
    canvas.set_roi(tool.rect, editing=True)
    boxes: list[object] = []
    tool.sigBox.connect(boxes.append)

    assert tool.hit(20.0, 40.0) == "nw"
    assert tool.hit(120.0, 90.0) == "e"
    assert tool.hit(70.0, 90.0) == "inside"
    assert tool.hit(5.0, 5.0) is None

    tool.on_press(120.0, 140.0, None)      # the se corner
    tool.on_move(150.0, 170.0, None)
    tool.on_release(150.0, 170.0, None)
    assert boxes[-1] == (20.0, 40.0, 150.0, 170.0)

    tool.on_press(70.0, 90.0, None)        # inside: move the whole rectangle
    tool.on_release(80.0, 100.0, None)
    assert boxes[-1] == (30.0, 50.0, 160.0, 180.0)

    tool.on_press(2.0, 2.0, None)          # empty canvas: draw a new one
    tool.on_move(12.0, 14.0, None)
    tool.on_release(12.0, 14.0, None)
    assert boxes[-1] == (2.0, 2.0, 12.0, 14.0)


def test_a_click_outside_the_rectangle_does_not_shrink_it_to_a_sliver(qapp):
    """A drag too small to be a rectangle puts the draft back on screen."""
    canvas = _blank_canvas()
    canvas.set_zoom(1.0)
    tool = RoiBoxTool(canvas, None)
    tool.set_rect((20.0, 40.0, 120.0, 140.0))
    boxes: list[object] = []
    previews: list[object] = []
    tool.sigBox.connect(boxes.append)
    tool.sigPreview.connect(previews.append)

    tool.on_press(2.0, 2.0, None)
    tool.on_move(3.0, 3.0, None)           # one image pixel: not a rectangle
    tool.on_release(3.0, 3.0, None)

    assert boxes == [], "a click stored a sliver"
    assert previews[-1] == (20.0, 40.0, 120.0, 140.0), "the draft was not put back"
    assert tool.rect == (20.0, 40.0, 120.0, 140.0)


def test_the_roi_tool_cursor_says_where_the_handles_are(qapp):
    canvas = _blank_canvas()
    canvas.set_zoom(1.0)
    tool = RoiBoxTool(canvas, None)
    tool.set_rect((20.0, 40.0, 120.0, 140.0))
    assert tool.cursor_for(20.0, 40.0) == Qt.CursorShape.SizeFDiagCursor
    assert tool.cursor_for(120.0, 40.0) == Qt.CursorShape.SizeBDiagCursor
    assert tool.cursor_for(70.0, 40.0) == Qt.CursorShape.SizeVerCursor
    assert tool.cursor_for(70.0, 90.0) == Qt.CursorShape.SizeAllCursor
    assert tool.cursor_for(2.0, 2.0) == Qt.CursorShape.CrossCursor


def test_a_resize_never_leaves_the_frame_or_an_inside_out_rectangle(qapp):
    canvas = _blank_canvas()
    canvas.set_zoom(1.0)
    tool = RoiBoxTool(canvas, None)
    tool.set_rect((20.0, 40.0, 120.0, 140.0))
    boxes: list[object] = []
    tool.sigBox.connect(boxes.append)

    tool.on_press(20.0, 40.0, None)        # drag nw far past se
    tool.on_release(400.0, 400.0, None)
    x0, y0, x1, y1 = boxes[-1]
    assert x0 < x1 and y0 < y1, "a resize turned the rectangle inside out"

    tool.set_rect((20.0, 40.0, 120.0, 140.0))
    tool.on_press(70.0, 90.0, None)        # move it off the top-left corner
    tool.on_release(-500.0, -500.0, None)
    assert boxes[-1] == (0.0, 0.0, 100.0, 100.0)
