"""The candidate guessers, M0-M5.  Pure numpy/opencv, no database access.

Every method takes the frame being annotated (``img_j``, part present), the
next frame (``img_k``, part gone), the ROI and whatever *legitimate* side
information it needs, and returns a ranked list of :class:`Prop` in full-frame
coordinates.  "Legitimate" means: known to the app at the moment the annotator
lands on frame ``j`` in the reverse walk -- the step log's target class, the
class's area band, and the shapes already drawn on frames ``>= j+1``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from tda.core.diffmap import diff_blobs, diff_delta_e

Box = tuple[int, int, int, int]

#: The production diff scale (``tda.ui.app_diff.MAX_DIFF_SIDE``).
MAX_SIDE = 1600


@dataclass
class Prop:
    """What a method arms: a box and a point, best first."""

    box: Box
    point: tuple[float, float]
    score: float
    area: int = 0                    # region pixels (blob methods) or box area
    extra: dict = field(default_factory=dict)


def blob_centroid(blob) -> tuple[float, float]:
    x0, y0 = blob.box[:2]
    ys, xs = np.nonzero(blob.mask)
    if xs.size == 0:
        return (0.5 * (blob.box[0] + blob.box[2]), 0.5 * (blob.box[1] + blob.box[3]))
    return (float(x0 + xs.mean()), float(y0 + ys.mean()))


def blobs_to_props(blobs, offset=(0, 0), n: int = 8) -> list[Prop]:
    ox, oy = offset
    out = []
    for b in blobs[:n]:
        cx, cy = blob_centroid(b)
        x0, y0, x1, y1 = b.box
        out.append(Prop(box=(x0 + ox, y0 + oy, x1 + ox, y1 + oy),
                        point=(cx + ox, cy + oy), score=float(b.score),
                        area=int(b.area)))
    return out


# --------------------------------------------------------------------------- #
# M0 -- today's rank-1 blob (B2's baseline, byte for byte)
# --------------------------------------------------------------------------- #
def m0(img_j, img_k, roi, min_area):
    """``(props, delta)``: diff_blobs on the full-frame dE map, B2's call."""
    delta = diff_delta_e(img_j, img_k, roi=roi, max_side=MAX_SIDE)
    blobs = diff_blobs(delta, min_area=min_area, max_blobs=8)
    return blobs_to_props(blobs), delta, blobs


# --------------------------------------------------------------------------- #
# M1 -- no box when the top blob's area does not fit the class
# --------------------------------------------------------------------------- #
def outside_factor(area: float, band: Optional[tuple[float, float]]) -> float:
    """How far outside ``band`` (pixels) ``area`` is, as a factor >= 1."""
    if band is None or area <= 0:
        return 1.0
    lo, hi = band
    if area < lo:
        return lo / area
    if area > hi:
        return area / hi
    return 1.0


# --------------------------------------------------------------------------- #
# M2 -- register j to k inside the ROI before the dE map
# --------------------------------------------------------------------------- #
def _roi_window(roi: Box, shape, margin: int) -> Box:
    h, w = shape[:2]
    x0, y0, x1, y1 = roi
    return (max(0, x0 - margin), max(0, y0 - margin),
            min(w, x1 + margin), min(h, y1 + margin))


def _grey_small(rgb: np.ndarray, scale: float) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return g.astype(np.float32) / 255.0


def estimate_shift(crop_j, crop_k, mode: str, work_side: int = 800):
    """2x3 matrix that maps ``crop_k`` onto ``crop_j`` (crop coordinates).

    ``phase``: global translation by phase correlation (sub-pixel);
    ``ecc``: euclidean ECC, initialised by the phase estimate.
    Both are estimated on a ``work_side`` copy and scaled back.
    Returns ``(M, info)``; ``M`` is ``None`` when the estimate failed.
    """
    h, w = crop_j.shape[:2]
    s = min(1.0, work_side / float(max(h, w)))
    gj, gk = _grey_small(crop_j, s), _grey_small(crop_k, s)
    win = cv2.createHanningWindow((gj.shape[1], gj.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(gk, gj, win)
    # phaseCorrelate(a, b) returns the shift of b relative to a
    m = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    info = {"dx": dx / s, "dy": dy / s, "rot_deg": 0.0, "resp": float(resp)}
    if mode == "ecc":
        warp = m.copy()
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
            cc, warp = cv2.findTransformECC(gj, gk, warp, cv2.MOTION_EUCLIDEAN,
                                            crit, None, 5)
            m = warp
            info.update(dx=float(warp[0, 2]) / s, dy=float(warp[1, 2]) / s,
                        rot_deg=float(np.degrees(np.arctan2(warp[1, 0],
                                                            warp[0, 0]))),
                        cc=float(cc))
        except cv2.error:
            info["ecc_failed"] = 1
    full = m.astype(np.float64).copy()
    full[0, 2] /= s
    full[1, 2] /= s
    return full, info


def m2(img_j, img_k, roi, min_area, mode: str, max_shift_frac: float = 0.05,
       app_order: bool = False):
    """Register k onto j inside the ROI, then B2's dE map + blobs on the crop.

    The dE map is computed at **the same scale** as M0 (``1600 / long side of
    the full frame``), so the only thing that differs from M0 is the warp.
    ``app_order`` passes (k, j) to ``diff_delta_e`` as the app does, instead of
    B2's (j, k); the map is not symmetric.
    """
    H, W = img_j.shape[:2]
    margin = int(0.05 * max(roi[2] - roi[0], roi[3] - roi[1]))
    win = _roi_window(roi, img_j.shape, margin)
    x0, y0, x1, y1 = win
    cj = np.ascontiguousarray(img_j[y0:y1, x0:x1])
    ck = np.ascontiguousarray(img_k[y0:y1, x0:x1])
    t0 = time.perf_counter()
    info: dict = {}
    if mode != "none":
        M, info = estimate_shift(cj, ck, mode)
        shift = math.hypot(info["dx"], info["dy"])
        # a wild estimate (textureless pair, a hand) is worse than none
        if shift > max_shift_frac * max(x1 - x0, y1 - y0):
            info["rejected"] = 1
        else:
            # M maps k-crop coordinates to j-crop coordinates
            ck = cv2.warpAffine(ck, M.astype(np.float32), (cj.shape[1], cj.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    info["reg_s"] = time.perf_counter() - t0
    scale = MAX_SIDE / float(max(H, W)) if max(H, W) > MAX_SIDE else 1.0
    side = int(round(max(cj.shape[:2]) * scale)) if scale < 1.0 else None
    inner = (roi[0] - x0, roi[1] - y0, roi[2] - x0, roi[3] - y0)
    if app_order:
        delta = diff_delta_e(ck, cj, roi=inner, max_side=side)
    else:
        delta = diff_delta_e(cj, ck, roi=inner, max_side=side)
    blobs = diff_blobs(delta, min_area=min_area, max_blobs=8)
    return blobs_to_props(blobs, offset=(x0, y0)), delta, (x0, y0), info, ck


# --------------------------------------------------------------------------- #
# M3 -- same-class appearance search: present in j, gone in k
# --------------------------------------------------------------------------- #
@dataclass
class Template:
    """A sibling's appearance, cut from the frame it was drawn on."""

    rgb: np.ndarray          # padded crop, native pixels
    box_wh: tuple[int, int]  # the sibling's own box size (unpadded)
    pad: tuple[int, int]     # padding (x, y) inside ``rgb``
    step: int
    instance: str


def cut_template(img: np.ndarray, box: Box, step: int, instance: str,
                 pad_frac: float = 0.25, min_pad: int = 3) -> Optional[Template]:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bw < 3 or bh < 3:
        return None
    px = max(min_pad, int(round(pad_frac * bw)))
    py = max(min_pad, int(round(pad_frac * bh)))
    h, w = img.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)
    crop = np.ascontiguousarray(img[Y0:Y1, X0:X1])
    if crop.size == 0:
        return None
    return Template(rgb=crop, box_wh=(bw, bh), pad=(x0 - X0, y0 - Y0),
                    step=step, instance=instance)


def features(rgb: np.ndarray, kind: str) -> np.ndarray:
    """Float32 feature image for NCC: gray, gradient magnitude or colour."""
    if kind == "color":
        return rgb.astype(np.float32)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if kind == "gray":
        return g
    if kind == "grad":
        g = cv2.GaussianBlur(g, (3, 3), 0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)
    raise ValueError(kind)


@dataclass
class M3Config:
    """Knobs of the appearance search (and, with a dE term, of M4)."""

    feats: tuple[str, ...] = ("gray",)
    scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    n_templates: int = 4
    #: context around the sibling's box in the template, as a fraction of it
    pad_frac: float = 0.25
    #: working scale: the template's sqrt(w*h) is brought to about this many
    #: pixels (never upsampled)
    target_side: float = 24.0
    #: grey dilation radius on the k-frame score, in working pixels -- a small
    #: misregistration must not look like "gone"
    dilate: int = 2
    #: weight of the k-frame match that is subtracted ("present now, gone
    #: next"); 0 = appearance in j only
    k_weight: float = 1.0
    #: weight of ``(1 - ZNCC(j patch, k patch)) / 2`` at template size, with a
    #: +-1 working-pixel shift tolerance: "the patch here changed", by
    #: appearance alone
    zncc_weight: float = 0.0
    #: weight of ``min(1, local max dE / 24)`` from a (registered) dE map -- M4
    de_weight: float = 0.0
    #: ``max``: local max dE over the template footprint; ``cs``: mean dE over
    #: the footprint minus the mean over a 3x surround (a compact change of
    #: the template's size, not the inside of a big one)
    de_mode: str = "max"
    #: weight of the appearance (NCC) term; 0 = the dE term alone, at the
    #: sibling's size -- the ablation "does appearance help at all"
    ncc_weight: float = 1.0
    #: drop every location whose local max dE is under this (M4 "restrict")
    de_gate: float = 0.0
    #: a window whose texture is under this share of the template's is not a
    #: match (TM_CCOEFF_NORMED is unstable on flat windows: 1.0 on white table)
    guard: float = 0.1
    suppress_known: bool = False
    top: int = 5
    #: two-stage: take this many appearance peaks, then re-rank them by how
    #: much the patch changed between j and (registered) k at native
    #: resolution -- "looks like a sibling" first, "and left" second
    rerank: int = 0
    #: weight of the appearance score in the re-ranking (the change term is
    #: ``mean native dE in the box / 24``)
    rerank_app: float = 0.0
    #: sequence prior: ``prior_weight * exp(-d^2 / 2 sigma^2)`` with ``d`` the
    #: distance to the part removed at ``k+1`` (drawn on frame ``k`` just
    #: before) and ``sigma`` in ROI diagonals; only when that part is known
    prior_weight: float = 0.0
    prior_sigma: float = 0.15


def _local_std(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Std of every ``th x tw`` window, on matchTemplate's valid grid."""
    s1 = cv2.boxFilter(img, cv2.CV_32F, (tw, th), anchor=(0, 0),
                       normalize=True, borderType=cv2.BORDER_CONSTANT)
    s2 = cv2.boxFilter(img * img, cv2.CV_32F, (tw, th), anchor=(0, 0),
                       normalize=True, borderType=cv2.BORDER_CONSTANT)
    h, w = img.shape[:2]
    var = (s2 - s1 * s1)[: h - th + 1, : w - tw + 1]
    return np.sqrt(np.maximum(var, 0.0))


def _zncc_change(gj: np.ndarray, gk: np.ndarray, th: int, tw: int,
                 shift: int = 1) -> np.ndarray:
    """``(1 - max_shift ZNCC(j window, k window)) / 2``, centre-indexed.

    Two flat windows are "unchanged" (0); a flat window against a textured one
    is "changed" (a part that left bare board behind).
    """
    def box(a):
        return cv2.boxFilter(a, cv2.CV_32F, (tw, th), normalize=True,
                             borderType=cv2.BORDER_REFLECT)

    mj = box(gj)
    vj = np.maximum(box(gj * gj) - mj * mj, 0.0)
    best = None
    pad = cv2.copyMakeBorder(gk, shift, shift, shift, shift, cv2.BORDER_REFLECT)
    h, w = gj.shape
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            k = pad[shift + dy:shift + dy + h, shift + dx:shift + dx + w]
            mk = box(k)
            vk = np.maximum(box(k * k) - mk * mk, 0.0)
            cov = box(gj * k) - mj * mk
            z = cov / np.sqrt(vj * vk + 1e-6)
            flat = (vj < 4.0) & (vk < 4.0)     # both std < 2 grey levels
            z = np.where(flat, 1.0, z)
            best = z if best is None else np.maximum(best, z)
    return np.clip((1.0 - best) * 0.5, 0.0, 1.0).astype(np.float32)


def m3(img_j, img_k, roi, templates: Sequence[Template], cfg: M3Config,
       known_boxes: Sequence[Box] = (), delta: Optional[np.ndarray] = None,
       delta_origin: tuple[int, int] = (0, 0),
       k_reg: Optional[np.ndarray] = None, k_origin: tuple[int, int] = (0, 0),
       prior_point: Optional[tuple[float, float]] = None):
    """Template search; returns ``(props, info)``.

    Score at a candidate centre ``x``::

        max over templates/scales of [NCC_j(x) - k_weight * dilate(NCC_k)(x)]
          + zncc_weight * change_appearance(x)
          + de_weight   * min(1, local max dE(x) / 24)

    ``delta`` is a dE map whose pixel (0, 0) sits at ``delta_origin`` in the
    frame; it is needed only when ``de_weight`` or ``de_gate`` is set (M4).
    """
    t0 = time.perf_counter()
    temps = list(templates)[: cfg.n_templates]
    if not temps:
        return [], {"m3_s": 0.0}
    sides = [math.sqrt(t.box_wh[0] * t.box_wh[1]) for t in temps]
    f = min(1.0, cfg.target_side / float(np.median(sides)))
    x0, y0, x1, y1 = roi
    cj = img_j[y0:y1, x0:x1]
    ck = img_k[y0:y1, x0:x1]
    if f < 1.0:
        cj = cv2.resize(cj, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        ck = cv2.resize(ck, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    H, W = cj.shape[:2]
    feats_j = {k: features(cj, k) for k in cfg.feats}
    feats_k = ({k: features(ck, k) for k in cfg.feats}
               if cfg.k_weight > 0 else {})
    best = np.full((H, W), -np.inf, np.float32)
    arg = np.full((H, W), -1, np.int32)
    tags: list[tuple[int, float]] = []
    kernel = (cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (2 * cfg.dilate + 1, 2 * cfg.dilate + 1))
              if cfg.dilate > 0 else None)
    if cfg.ncc_weight <= 0:
        # dE term alone: one "template" (the median sibling), no matching
        best[:] = 0.0
        arg[:] = 0
        tags.append((len(temps) // 2, 1.0))
    for ti, t in enumerate(temps if cfg.ncc_weight > 0 else []):
        trgb = t.rgb
        if f < 1.0:
            trgb = cv2.resize(trgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        for sc in cfg.scales:
            tr = trgb if sc == 1.0 else cv2.resize(
                trgb, None, fx=sc, fy=sc,
                interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
            th, tw = tr.shape[:2]
            if th < 4 or tw < 4 or th >= H or tw >= W:
                continue
            acc = None
            for kind in cfg.feats:
                ft = features(tr, kind)
                tstd = float(ft.std())
                if tstd < 1e-3:
                    continue
                fj = feats_j[kind]
                rj = cv2.matchTemplate(fj, ft, cv2.TM_CCOEFF_NORMED)
                if cfg.guard > 0:
                    g1 = fj if fj.ndim == 2 else fj.mean(axis=2)
                    rj[_local_std(g1, th, tw) < cfg.guard * tstd] = 0.0
                s = rj
                if cfg.k_weight > 0:
                    fk = feats_k[kind]
                    rk = cv2.matchTemplate(fk, ft, cv2.TM_CCOEFF_NORMED)
                    if cfg.guard > 0:
                        g2 = fk if fk.ndim == 2 else fk.mean(axis=2)
                        rk[_local_std(g2, th, tw) < cfg.guard * tstd] = 0.0
                    if kernel is not None:
                        rk = cv2.dilate(rk, kernel)
                    s = rj - cfg.k_weight * rk
                acc = s if acc is None else acc + s
            if acc is None:
                continue
            acc *= cfg.ncc_weight / float(len(cfg.feats))
            tag = len(tags)
            tags.append((ti, sc))
            oy, ox = th // 2, tw // 2
            h, w = acc.shape
            view = best[oy:oy + h, ox:ox + w]
            better = acc > view
            np.copyto(view, acc, where=better)
            np.copyto(arg[oy:oy + h, ox:ox + w], tag, where=better)
    info = {"m3_scale": round(f, 4), "m3_templates": len(temps)}
    if not tags:
        info["m3_s"] = time.perf_counter() - t0
        return [], info

    score = np.where(np.isfinite(best), best, -10.0).astype(np.float32)
    tw_m = max(3, int(round(float(np.median([t.rgb.shape[1] for t in temps])) * f)))
    th_m = max(3, int(round(float(np.median([t.rgb.shape[0] for t in temps])) * f)))
    if cfg.zncc_weight > 0:
        gj = cv2.cvtColor(cj, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gk = cv2.cvtColor(ck, cv2.COLOR_RGB2GRAY).astype(np.float32)
        score = score + cfg.zncc_weight * _zncc_change(gj, gk, th_m, tw_m)
    if delta is not None and (cfg.de_weight > 0 or cfg.de_gate > 0):
        ox, oy = delta_origin
        d = delta[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        if d.shape[:2] != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_AREA)
        rad = max(1, int(round(0.5 * min(tw_m, th_m))))
        if cfg.de_mode == "cs":
            inner = cv2.boxFilter(d, cv2.CV_32F, (tw_m, th_m))
            outer = cv2.boxFilter(d, cv2.CV_32F, (3 * tw_m, 3 * th_m))
            # mean of the 3x window's ring = (9 * outer - inner) / 8
            dloc = np.maximum(inner - (9.0 * outer - inner) / 8.0, 0.0)
        else:
            dloc = cv2.dilate(d, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)))
        if cfg.de_weight > 0:
            score = score + cfg.de_weight * np.clip(dloc / 24.0, 0.0, 1.0)
        if cfg.de_gate > 0:
            score = np.where(dloc >= cfg.de_gate, score, -10.0).astype(np.float32)
    prior_map = None
    if cfg.prior_weight > 0 and prior_point is not None:
        diag = math.hypot(x1 - x0, y1 - y0)
        sig = max(1.0, cfg.prior_sigma * diag * f)
        px0, py0 = (prior_point[0] - x0) * f, (prior_point[1] - y0) * f
        gx = np.exp(-((np.arange(W, dtype=np.float32) + 0.5 - px0) ** 2)
                    / (2 * sig * sig))
        gy = np.exp(-((np.arange(H, dtype=np.float32) + 0.5 - py0) ** 2)
                    / (2 * sig * sig))
        prior_map = np.outer(gy, gx).astype(np.float32)
        score = score + cfg.prior_weight * prior_map
    if cfg.suppress_known and known_boxes:
        for bx in known_boxes:
            a0 = int((bx[0] - x0) * f) - 1
            b0 = int((bx[1] - y0) * f) - 1
            a1 = int(math.ceil((bx[2] - x0) * f)) + 1
            b1 = int(math.ceil((bx[3] - y0) * f)) + 1
            score[max(0, b0):max(0, b1), max(0, a0):max(0, a1)] = -10.0

    props: list[Prop] = []
    work = score.copy()
    for _ in range(max(cfg.top, cfg.rerank)):
        idx = int(np.argmax(work))
        py, px = divmod(idx, W)
        val = float(work[py, px])
        if val <= -9.0:
            break
        ti, sc = tags[int(arg[py, px])] if arg[py, px] >= 0 else (0, 1.0)
        bw, bh = temps[ti].box_wh
        bw, bh = bw * sc, bh * sc
        cx, cy = x0 + (px + 0.5) / f, y0 + (py + 0.5) / f
        # the template's padded crop is centred on the box only when the pad
        # is symmetric; correct for a crop clipped at the frame edge
        tpx, tpy = temps[ti].pad
        tw_full = temps[ti].rgb.shape[1]
        th_full = temps[ti].rgb.shape[0]
        cx += sc * (tpx + 0.5 * temps[ti].box_wh[0] - 0.5 * tw_full)
        cy += sc * (tpy + 0.5 * temps[ti].box_wh[1] - 0.5 * th_full)
        box = (int(round(cx - bw / 2)), int(round(cy - bh / 2)),
               int(round(cx + bw / 2)), int(round(cy + bh / 2)))
        pr = float(prior_map[py, px]) if prior_map is not None else 0.0
        props.append(Prop(box=box, point=(cx, cy),
                          score=val, area=int(bw * bh),
                          extra={"tmpl": ti, "scale": sc, "prior": pr,
                                 "app": val - cfg.prior_weight * pr}))
        r = max(2, int(round(0.75 * max(bw, bh) * f)))
        work[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1] = -10.0
    if cfg.rerank and props:
        kimg, korg = (k_reg, k_origin) if k_reg is not None else (img_k, (0, 0))
        for p in props:
            p.extra["change"] = box_change(img_j, kimg, korg, p.box)
            p.score = (p.extra["change"] / 24.0 + cfg.rerank_app * p.extra["app"]
                       + cfg.prior_weight * p.extra["prior"])
        props.sort(key=lambda p: -p.score)
    info["m3_s"] = time.perf_counter() - t0
    return props, info


def box_change(img_j: np.ndarray, k_img: np.ndarray, k_origin: tuple[int, int],
               box: Box) -> float:
    """Mean native dE inside ``box`` between j and k (1 px shift tolerance).

    ``k_img`` may be a (registered) crop whose pixel (0, 0) is ``k_origin`` in
    frame coordinates.
    """
    ox, oy = k_origin
    kh, kw = k_img.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    pad = max(3, int(round(0.25 * max(bw, bh))))
    x0 = max(box[0] - pad, ox, 0)
    y0 = max(box[1] - pad, oy, 0)
    x1 = min(box[2] + pad, ox + kw, img_j.shape[1])
    y1 = min(box[3] + pad, oy + kh, img_j.shape[0])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    cj = np.ascontiguousarray(img_j[y0:y1, x0:x1])
    ck = np.ascontiguousarray(k_img[y0 - oy:y1 - oy, x0 - ox:x1 - ox])
    d = diff_delta_e(cj, ck, blur=3, shift_px=1)
    ix0, iy0 = max(0, box[0] - x0), max(0, box[1] - y0)
    ix1, iy1 = min(x1 - x0, box[2] - x0), min(y1 - y0, box[3] - y0)
    inner = d[iy0:iy1, ix0:ix1]
    return float(inner.mean()) if inner.size else 0.0


# --------------------------------------------------------------------------- #
# M5 -- a compact change of the class's size (no sibling needed)
# --------------------------------------------------------------------------- #
def m5(delta: np.ndarray, delta_origin: tuple[int, int], roi: Box,
       side: float, target_side: float = 24.0, top: int = 5,
       mode: str = "cs", prior_point=None, prior_weight: float = 0.0,
       prior_sigma: float = 0.15):
    """Peaks of a size-matched filter on the (registered) dE map.

    ``side`` is the class's typical width in native pixels (``sqrt`` of its
    median draft area).  ``cs`` scores mean dE over a ``side`` window minus the
    mean over the 3x ring around it, so a screw-sized change beats the inside
    of a cable-sized one; ``mean`` is the plain window mean.
    """
    t0 = time.perf_counter()
    side = max(3.0, float(side))
    f = min(1.0, target_side / side)
    ox, oy = delta_origin
    x0, y0, x1, y1 = roi
    d = delta[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    if f < 1.0:
        d = cv2.resize(d, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    w = max(3, int(round(side * f)))
    inner = cv2.boxFilter(d, cv2.CV_32F, (w, w))
    if mode == "cs":
        outer = cv2.boxFilter(d, cv2.CV_32F, (3 * w, 3 * w))
        score = inner - (9.0 * outer - inner) / 8.0
    else:
        score = inner
    H, W = score.shape
    score = score / 24.0
    if prior_weight > 0 and prior_point is not None:
        sig = max(1.0, prior_sigma * math.hypot(x1 - x0, y1 - y0) * f)
        px0, py0 = (prior_point[0] - x0) * f, (prior_point[1] - y0) * f
        gx = np.exp(-((np.arange(W, dtype=np.float32) + 0.5 - px0) ** 2) / (2 * sig * sig))
        gy = np.exp(-((np.arange(H, dtype=np.float32) + 0.5 - py0) ** 2) / (2 * sig * sig))
        score = score + prior_weight * np.outer(gy, gx)
    props: list[Prop] = []
    work = score.copy()
    for _ in range(top):
        idx = int(np.argmax(work))
        py, px = divmod(idx, W)
        val = float(work[py, px])
        if val <= 0.0:
            break
        cx, cy = x0 + (px + 0.5) / f, y0 + (py + 0.5) / f
        box = (int(round(cx - side / 2)), int(round(cy - side / 2)),
               int(round(cx + side / 2)), int(round(cy + side / 2)))
        props.append(Prop(box=box, point=(cx, cy), score=val,
                          area=int(side * side)))
        r = max(2, int(round(0.75 * w)))
        work[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1] = -1e9
    return props, {"search_s": time.perf_counter() - t0}
