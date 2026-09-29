"""The frozen-row comparison works inside the part and says what it always said (U2f).

``truth_conflicts.disagreement`` decides which conflicts are queued, so making
it cheap may not change one answer (U2f ruling 1).  Every test here holds the
new code against a reference that cannot move with it:

* ``disagreement_oracle.disagreement`` -- main's full-canvas comparison at
  ``df8333d``, copied verbatim with everything it reached;
* for the primitives, pycocotools itself (``encode``/``decode``) and the
  oracle's ``tolerant_sym_diff``.

The inputs are seeded random pairs on canvases with the OAK frame's 4:3 shape
(small ones, so that thousands of pairs cost seconds) plus a few odd shapes and
a few true 12 MP cases, covering empty and full masks, masks touching every
border, tiny parts, identical pairs, one-pixel differences, differences at
exactly the tolerance and exactly the threshold, shifted masks and label-only
differences -- stored as canonical, ``bytes`` and non-canonical run lengths,
compiled as Fortran or C order with tight, loose or no windows.  Every
``encode_rle_windowed`` and ``bbox_in`` window promise is checked on every
call (``tests/conftest.py`` turns :data:`tda.core.masks.CHECK_ENCODE_WINDOW`
on).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from pycocotools import _mask as coco_low

import disagreement_oracle as oracle
from tda.core import masks as M
from tda.core import truth_conflicts as TC
from tda.core.compiler import CompiledInstance

#: The OAK frame (spec 2.4) and canvases of the same 3:4 shape.
OAK_HW = (3040, 4032)
SMALL = [(30, 40), (57, 76), (114, 152), (228, 304)]
#: Shapes where a column is a row, a row is a column, or both are one pixel.
ODD = [(1, 1), (1, 9), (9, 1), (2, 3), (13, 17)]
PAIRS_PER_CANVAS = 900


# --------------------------------------------------------------------------- #
# builders -- independent of the code under test
# --------------------------------------------------------------------------- #
def _runs_of(mask: np.ndarray) -> np.ndarray:
    """The column-major run lengths of ``mask``, starting with background."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F").astype(np.int8)
    change = np.flatnonzero(np.diff(np.concatenate(([0], flat))) != 0)
    return np.diff(np.concatenate(([0], change, [flat.size]))).astype(np.int64)


def _rle_from_runs(runs, hw) -> dict:
    h, w = int(hw[0]), int(hw[1])
    out = coco_low.frUncompressedRLE(
        [{"counts": np.asarray(runs, dtype=np.uint32), "size": [h, w]}], h, w)[0]
    return {"size": [h, w], "counts": out["counts"].decode("ascii")}


def _noncanonical(mask: np.ndarray, rng) -> dict:
    """The same pixels, written with empty runs a hand-made RLE may carry."""
    runs = list(_runs_of(mask))
    for _ in range(int(rng.integers(1, 4))):
        i = int(rng.integers(0, len(runs)))
        cut = int(rng.integers(0, runs[i] + 1))
        runs[i:i + 1] = [cut, 0, runs[i] - cut]
    rle = _rle_from_runs(runs, mask.shape)
    assert np.array_equal(oracle.decode_rle(rle), mask)
    return rle


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = mask[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def _shape(rng, hw) -> np.ndarray:
    """A random mask: empty, full, a border ring, specks, noise or blobs."""
    h, w = hw
    mask = np.zeros(hw, dtype=bool)
    kind = int(rng.integers(0, 10))
    if kind == 0:
        return mask
    if kind == 1:
        mask[:] = True
        return mask
    if kind == 2:                                   # touches every border
        mask[:] = True
        t = int(rng.integers(1, max(2, min(h, w) // 3 + 1)))
        mask[t:h - t, t:w - t] = False
        return mask
    if kind == 3:                                   # a tiny part: 1-3 px
        n = int(rng.integers(1, 4))
        mask[rng.integers(0, h, n), rng.integers(0, w, n)] = True
        return mask
    if kind == 4:
        mask[:] = rng.random(hw) < rng.uniform(0.02, 0.7)
        return mask
    canvas = np.zeros(hw, dtype=np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        # corners may lie off the canvas: parts are cut by the frame edge
        x0, x1 = sorted(int(v) for v in rng.integers(-w // 4, w + w // 4, 2))
        y0, y1 = sorted(int(v) for v in rng.integers(-h // 4, h + h // 4, 2))
        if rng.random() < 0.5:
            canvas[max(0, y0):max(0, y1 + 1), max(0, x0):max(0, x1 + 1)] = 1
        else:
            cv2.ellipse(canvas, ((x0 + x1) // 2, (y0 + y1) // 2),
                        (max(0, (x1 - x0) // 2), max(0, (y1 - y0) // 2)),
                        float(rng.uniform(0, 180)), 0, 360, 1, -1)
    return canvas.astype(bool)


def _changed(rng, old: np.ndarray) -> np.ndarray:
    """What a re-compilation might make of ``old``."""
    h, w = old.shape
    how = int(rng.integers(0, 9))
    if how == 0:
        return old.copy()                           # identical
    if how == 1:                                    # one pixel, anywhere
        new = old.copy()
        new[int(rng.integers(0, h)), int(rng.integers(0, w))] ^= True
        return new
    if how == 2:                                    # shifted, up to past the band
        return _shift(old, int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))
    if how in (3, 4):                               # re-traced thinner / fatter
        size = 2 * int(rng.integers(1, 4)) + 1
        kernel = np.ones((size, size), np.uint8)
        op = cv2.erode if how == 3 else cv2.dilate
        return op(old.astype(np.uint8), kernel, borderValue=0).astype(bool)
    if how == 5:                                    # a hole or a patch toggled
        new = old.copy()
        x0, x1 = sorted(int(v) for v in rng.integers(0, w + 1, 2))
        y0, y1 = sorted(int(v) for v in rng.integers(0, h + 1, 2))
        new[y0:y1, x0:x1] ^= True
        return new
    if how == 6:
        return np.zeros_like(old)
    return _shape(rng, old.shape)                   # something else entirely


def _stored_row(rng, mask) -> dict:
    """A frozen truth row holding ``mask`` (``None``: no geometry)."""
    labels = {"visibility": str(rng.choice(["visible", "occluded_partial", "too_small"])),
              "placement": str(rng.choice(["in_chassis", "on_bench"])),
              "occlusion_ratio": float(rng.random()), "status": "verified"}
    if mask is None:
        return {"geom_type": "mask", "visible_rle": None, "box": None, **labels}
    pick = rng.random()
    if pick < 0.2:
        rle = _noncanonical(mask, rng)
    else:
        rle = M.encode_rle(mask)
        if pick < 0.3:
            rle = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    return {"geom_type": "mask", "visible_rle": rle, "box": None, **labels}


def _window_for(rng, mask: np.ndarray):
    """A window ``mask`` is empty outside: tight, loose (maybe off the canvas) or none."""
    tight = oracle.bbox(mask)
    pick = rng.random()
    if pick < 0.2:
        return None
    if tight is None:
        return (0, 0, 0, 0) if pick < 0.6 else (0, 0, mask.shape[1], mask.shape[0])
    if pick < 0.6:
        return tight
    grow = [int(v) for v in rng.integers(0, 6, 4)]
    return (tight[0] - grow[0], tight[1] - grow[1], tight[2] + grow[2], tight[3] + grow[3])


def _compiled(rng, mask) -> CompiledInstance:
    """A re-compilation whose visible mask is ``mask`` (``None``: none)."""
    visibility = str(rng.choice(["visible", "occluded_partial", "too_small"]))
    placement = str(rng.choice(["in_chassis", "on_bench"]))
    if mask is None:
        amodal = None if rng.random() < 0.5 else np.zeros((3, 3), bool)
        return CompiledInstance("p", None, None, amodal, 1.0, "out_of_view", placement,
                                None)
    arr = np.asfortranarray(mask) if rng.random() < 0.8 else np.ascontiguousarray(mask)
    if rng.random() < 0.5:
        arr.flags.writeable = False                 # what the compiler publishes
    return CompiledInstance("p", arr, oracle.bbox(arr), arr, float(rng.random()),
                            visibility, placement, 1, window=_window_for(rng, arr))


def _random_box(rng, hw) -> list[float]:
    h, w = hw
    x0, x1 = sorted(float(v) for v in rng.uniform(-3, w + 3, 2))
    y0, y1 = sorted(float(v) for v in rng.uniform(-3, h + 3, 2))
    return [x0, y0, x1, y1]


def _case(rng, hw):
    """One ``(row, compiled)`` pair of any kind the truth table can meet."""
    pick = rng.random()
    if pick < 0.06:                                  # stored without geometry
        return _stored_row(rng, None), _compiled(rng, _shape(rng, hw))
    if pick < 0.12:                                  # compiled without geometry
        return _stored_row(rng, _shape(rng, hw)), _compiled(rng, None)
    if pick < 0.14:                                  # neither
        return _stored_row(rng, None), _compiled(rng, None)
    if pick < 0.18:                                  # another canvas size
        other = (hw[0] + int(rng.integers(1, 4)), hw[1])
        return _stored_row(rng, _shape(rng, other)), _compiled(rng, _shape(rng, hw))
    if pick < 0.26:                                  # a box on either side
        side = rng.random()
        row = (_stored_row(rng, _shape(rng, hw)) if side < 0.33 else
               {"geom_type": "box", "visible_rle": None, "box": _random_box(rng, hw)})
        if side >= 0.33 and rng.random() < 0.3:
            row["box"] = None
        box = CompiledInstance("p", None, tuple(_random_box(rng, hw)), None, 0.0,
                               "visible", "on_bench", 1)
        compiled = _compiled(rng, _shape(rng, hw)) if side >= 0.66 else box
        return row, compiled
    old = _shape(rng, hw)
    return _stored_row(rng, old), _compiled(rng, _changed(rng, old))


def _outcome(fn, row, compiled):
    """``fn``'s answer, or the type of what it raised -- both must agree."""
    try:
        return ("ok", fn(row, compiled))
    except Exception as exc:  # noqa: BLE001 - the type is the answer here
        return ("raised", type(exc))


# --------------------------------------------------------------------------- #
# disagreement == the full-canvas reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hw", SMALL + ODD)
def test_disagreement_is_the_full_canvas_answer_for_every_kind_of_pair(hw):
    rng = np.random.default_rng(1000 * hw[0] + hw[1])
    decided = {"agree": 0, "disagree": 0}
    for trial in range(PAIRS_PER_CANVAS):
        row, compiled = _case(rng, hw)
        want = _outcome(oracle.disagreement, row, compiled)
        got = _outcome(TC.disagreement, row, compiled)
        assert got == want, (
            f"pair {trial} on {hw}: reference {want}, got {got}; row "
            f"{ {k: v for k, v in row.items() if k != 'visible_rle'} } window "
            f"{getattr(compiled, 'window', None)}")
        if want[0] == "ok":
            decided["agree" if want[1] is None else "disagree"] += 1
    # the generator exercised both answers, not one of them a thousand times
    assert decided["agree"] > PAIRS_PER_CANVAS // 10, decided
    assert decided["disagree"] > PAIRS_PER_CANVAS // 10, decided


def _line(hw, n: int, row: int) -> np.ndarray:
    """``n`` pixels in a line along ``row``: far from anything, no band around it."""
    mask = np.zeros(hw, dtype=bool)
    mask[row, 1:1 + n] = True
    return mask


@pytest.mark.parametrize("side", [6, 40])
def test_a_difference_of_exactly_the_threshold_agrees_and_one_more_pixel_does_not(side):
    """``tolerant_sym_diff > max(2 % of the area, 20)``: the boundary itself agrees.

    A 6x6 part has the 20-pixel floor, a 40x40 one its 2 % (32 px); the extra
    pixels are a line far from the part, where no tolerance band reaches.
    """
    hw = (120, 160)
    old = np.zeros(hw, dtype=bool)
    old[60:60 + side, 80:80 + side] = True
    threshold = max(0.02 * side * side, 20.0)
    exactly = int(threshold)
    assert exactly == threshold
    for extra, verdict in ((exactly, None), (exactly + 1, exactly + 1)):
        new = old | _line(hw, extra, 2)
        row = {"geom_type": "mask", "visible_rle": M.encode_rle(old), "box": None}
        compiled = CompiledInstance("p", np.asfortranarray(new), oracle.bbox(new), new,
                                    0.0, "visible", "in_chassis", 1,
                                    window=oracle.bbox(new))
        assert oracle.disagreement(row, compiled) == verdict
        assert TC.disagreement(row, compiled) == verdict


@pytest.mark.parametrize("shift", [1, 2, 3, 4])
def test_a_shift_at_the_tolerance_and_just_past_it(shift):
    """Up to ``tol_px`` (2) of translation is inside the band; 3 is not."""
    hw = (60, 80)
    old = np.zeros(hw, dtype=bool)
    old[10:50, 10:70] = True
    for dy, dx in ((shift, 0), (0, -shift), (shift, shift)):
        new = _shift(old, dy, dx)
        row = {"geom_type": "mask", "visible_rle": M.encode_rle(old), "box": None}
        compiled = CompiledInstance("p", np.asfortranarray(new), oracle.bbox(new), new,
                                    0.0, "visible", "in_chassis", 1,
                                    window=oracle.bbox(new))
        assert TC.disagreement(row, compiled) == oracle.disagreement(row, compiled)
        if shift <= 2:
            assert oracle.tolerant_sym_diff(old, new) == 0


def test_a_label_only_difference_is_no_pixel_disagreement_either_way():
    """The labels are :func:`label_changes`' business; the pixels still agree."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        mask = _shape(rng, (57, 76))
        row = _stored_row(rng, mask)
        row["visibility"], row["placement"] = "occluded_partial", "in_chassis"
        compiled = _compiled(rng, mask)
        compiled.visibility, compiled.placement = "visible", "on_bench"
        assert TC.disagreement(row, compiled) is None
        assert oracle.disagreement(row, compiled) is None
        labels = TC.label_changes(row, compiled)
        assert [c["field"] for c in labels][-1:] == ["placement"]


def _oak(mask_fn) -> np.ndarray:
    mask = np.zeros(OAK_HW, dtype=bool, order="F")
    mask_fn(mask)
    return mask


def _oak_cases():
    """Six true 12 MP pairs: what the Enter on a confirmed frame compares."""
    def chassis(m):
        m[1:434, 1:434] = True
    def carved(m):
        chassis(m)
        m[100:300, 100:300] ^= True
    def screw(m):
        m[2000:2020, 3000:3020] = True
    def screw_moved(m):
        m[2003:2023, 3000:3020] = True
    def frame(m):
        m[:, :] = True
        m[5:-5, 5:-5] = False
    def frame_corner(m):
        frame(m)
        m[-1, -1] = False
    def nothing(m):
        pass
    return [
        ("identical chassis", chassis, chassis),
        ("the Enter: a hole carved in the chassis", chassis, carved),
        ("a 20 px screw moved 3 px", screw, screw_moved),
        ("a frame touching every border, one corner pixel off", frame, frame_corner),
        ("a screw that vanished", screw, nothing),
        ("the whole frame against a screw", frame, screw),
    ]


@pytest.mark.parametrize("name,old_fn,new_fn", _oak_cases(),
                         ids=[c[0] for c in _oak_cases()])
def test_disagreement_is_the_full_canvas_answer_at_12_mp(name, old_fn, new_fn):
    old, new = _oak(old_fn), _oak(new_fn)
    new.flags.writeable = False
    row = {"geom_type": "mask", "visible_rle": M.encode_rle(old), "box": None}
    compiled = CompiledInstance("p", new, oracle.bbox(new), new, 0.0, "visible",
                                "in_chassis", 1, window=oracle.bbox(new))
    assert TC.disagreement(row, compiled) == oracle.disagreement(row, compiled)


def test_the_comparison_never_decodes_a_readable_row(monkeypatch):
    """Agreeing or not, a stored row is read off its run lengths (ruling 2, 3).

    Only an RLE the reader cannot read goes to pycocotools, and a truth row
    written by this tool never is one.
    """
    def refuse(*_a, **_k):
        raise AssertionError("the canvas was decoded")

    monkeypatch.setattr(M, "decode_rle", refuse)
    monkeypatch.setattr(M.coco_mask, "decode", refuse)
    rng = np.random.default_rng(11)
    for _ in range(300):
        old = _shape(rng, (114, 152))
        row = {"geom_type": "mask", "visible_rle": M.encode_rle(old), "box": None}
        TC.disagreement(row, _compiled(rng, _changed(rng, old)))
        TC.disagreement(row, _compiled(rng, old))


# --------------------------------------------------------------------------- #
# the primitives, against pycocotools
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hw", SMALL + ODD)
def test_encode_rle_windowed_writes_encode_rles_bytes(hw):
    rng = np.random.default_rng(hw[0] * 7919 + hw[1])
    for _ in range(300):
        mask = _shape(rng, hw)
        arr = np.asfortranarray(mask) if rng.random() < 0.7 else mask
        want = M.encode_rle(arr)
        assert M.encode_rle_windowed(arr, _window_for(rng, arr)) == want
        assert M.encode_rle_windowed(arr, None) == want


def test_encode_rle_windowed_on_runs_that_wrap_from_one_column_into_the_next():
    """The one place a window's column edge is not a run edge: full-height windows."""
    hw = (5, 4)
    for pattern in range(0, 1 << 20, 997):
        mask = np.array([(pattern >> i) & 1 for i in range(20)], dtype=bool).reshape(hw)
        want = M.encode_rle(mask)
        assert M.encode_rle_windowed(mask, (0, 0, 4, 5)) == want
        assert M.encode_rle_windowed(mask, oracle.bbox(mask) or (0, 0, 0, 0)) == want
    full = np.ones(hw, dtype=bool)
    assert M.encode_rle_windowed(full, (0, 0, 4, 5)) == M.encode_rle(full)


@pytest.mark.parametrize("hw", SMALL + ODD)
def test_run_lengths_answer_what_the_decoded_canvas_answers(hw):
    rng = np.random.default_rng(hw[0] * 104729 + hw[1])
    h, w = hw
    for _ in range(300):
        mask = _shape(rng, hw)
        rle = _stored_row(rng, mask)["visible_rle"]
        runs = M.RunLengths(rle)
        assert not runs.decoded
        full = oracle.decode_rle(rle)
        assert runs.hw == hw
        assert runs.area() == oracle.area(full)
        assert runs.bbox() == oracle.bbox(full)
        for _ in range(4):
            x0, x1 = sorted(int(v) for v in rng.integers(-3, w + 4, 2))
            y0, y1 = sorted(int(v) for v in rng.integers(-3, h + 4, 2))
            got = runs.window((x0, y0, x1, y1))
            want = full[max(0, y0):max(0, min(y1, h)), max(0, x0):max(0, min(x1, w))]
            assert got.dtype == bool and got.shape == want.shape
            assert np.array_equal(got, want)


def test_run_lengths_leave_what_they_cannot_read_to_pycocotools():
    """Same exception, or the same decode: never a different answer."""
    good = M.encode_rle(np.eye(6, dtype=bool))
    short = _rle_from_runs([5, 3], (6, 6))          # 8 of 36 pixels spelled
    assert M.RunLengths(short).decoded
    for broken in ({"size": [6, 6], "counts": [30, 6]},
                   {"size": [6, 6], "counts": good["counts"] + "é"}):
        with pytest.raises(Exception) as ours:
            M.RunLengths(broken)
        with pytest.raises(Exception) as theirs:
            oracle.decode_rle(broken)
        assert type(ours.value) is type(theirs.value)


@pytest.mark.parametrize("tol_px", [0, 1, 2, 3])
def test_tolerant_sym_diff_is_the_reference_whatever_the_layout(tol_px):
    """Fortran, C, windows of either, mixed: the same count, no copies needed."""
    rng = np.random.default_rng(31 + tol_px)
    for _ in range(400):
        hw = SMALL[int(rng.integers(0, len(SMALL)))]
        a = _shape(rng, hw)
        b = _changed(rng, a)
        want = oracle.tolerant_sym_diff(a, b, tol_px)
        fa, fb = np.asfortranarray(a), np.asfortranarray(b)
        for x, y in ((a, b), (fa, fb), (fa, b), (a, fb), (fa.T.T, fb[:, :])):
            assert M.tolerant_sym_diff(x, y, tol_px) == want
        assert M.tolerant_sym_diff(a.T, b.T, tol_px) == want      # the transpose
        assert M.is_conflict(fa, fb, tol_px=tol_px) == oracle.is_conflict(a, b, tol_px=tol_px)
        # windows of a Fortran canvas are what the comparison hands over
        y0, x0 = int(rng.integers(0, hw[0])), int(rng.integers(0, hw[1]))
        assert (M.tolerant_sym_diff(fa[y0:, x0:], fb[y0:, x0:], tol_px)
                == oracle.tolerant_sym_diff(a[y0:, x0:], b[y0:, x0:], tol_px))


def test_the_window_answer_is_the_canvas_answer_with_or_without_the_margin():
    """Both halves of ruling 3's proof, on the primitive itself.

    Cropped to the union of the two boxes the count is already the canvas'
    (every XOR pixel is inside, and the border is 0 exactly where the canvas
    has no pixels); grown by ``tolerance_reach`` it is too, and there no
    border value is ever consulted for a pixel that counts.
    """
    rng = np.random.default_rng(5)
    for _ in range(600):
        hw = SMALL[int(rng.integers(0, len(SMALL)))]
        a = _shape(rng, hw)
        b = _changed(rng, a)
        union = TC._union(oracle.bbox(a), oracle.bbox(b))
        if union is None:
            continue
        want = oracle.tolerant_sym_diff(a, b)
        for reach in (0, M.tolerance_reach()):
            x0, y0, x1, y1 = TC._grown(union, reach, hw)
            assert M.tolerant_sym_diff(a[y0:y1, x0:x1], b[y0:y1, x0:x1]) == want


def test_bbox_in_refuses_a_window_the_mask_spills_out_of():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:9, 5:9] = True
    assert M.bbox_in(mask, (4, 4, 12, 12)) == (5, 5, 9, 9)
    assert M.bbox_in(mask, None) == (5, 5, 9, 9)
    assert M.CHECK_ENCODE_WINDOW is True, "tests/conftest.py should have set this"
    with pytest.raises(ValueError, match="pixels outside it"):
        M.bbox_in(mask, (6, 6, 12, 12))
    with pytest.raises(ValueError, match="pixels outside it"):
        M.encode_rle_windowed(mask, (6, 6, 12, 12))
