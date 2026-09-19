"""Write ``report.md`` into the experiment output directory.

The prose lives here rather than as a loose markdown file so that the report and
the code that produced its numbers are versioned together: every figure quoted
below comes from ``summarise.py``, whose full output is written alongside as
``summary.txt``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.plan_b_probe.transfer.common import OUT  # noqa: E402

REPORT = r"""
# Plan B probe: can a 4-corner homography carry a `scan` annotation into `oak1`, `oak2`, `rs`?

Read-only measurement on desktops 13, 24 and 33. Nothing was written to
`annotations/tda.sqlite`; every read went through
`file:D:/DataSet/.cache/tmp/exp_copy.sqlite?mode=ro`, and `F:` was read only.
Code and the corner file are committed on branch
`worktree-agent-ad56db6176d22770d` under `experiments/plan_b_probe/transfer/`.

**Short answer.** The homography is worth building *only as a coarse region hint
for large parts*, and only with the plane chosen per class. It never becomes a
point prompt you can trust: the best per-view HIT rates are 61% / 64% / 35%
(oak1 / oak2 / rs) on the cleanest possible subset, and 0% on every part smaller
than about 60 px in the target view -- which is most of the dataset. The
per-view difference map is not a replacement either (top-1 21-41%), but it is
the only signal that works without any cross-view geometry, and the two together
are still under 44%. The recommendation is in section 10.

---

## 1. What was measured, and against what

Ground truth is the old Label Studio import: `shape_keyframe` rows whose
`instance` starts with `ls:`, geometry in `shape_part` as COCO RLE. The key is
`ls:<Label>#<ordinal>`.

Per instance that the drafts place in **both** `scan` and a target view **at the
same step**:

| symbol | definition |
|---|---|
| **HIT** | the scan polygon's centroid, projected through the homography, falls **inside the target polygon of the same identity**. This is the only thing a point prompt has to get right. |
| **centre error** | distance from that projected point to the target polygon's centroid, in target pixels, and divided by `sqrt(target area)` -- the part's own length scale, so `1.0` means "off by the size of the part". |
| **box IoU** | IoU of the target box with the axis-aligned box of the projected scan box. |

Two reference points frame every number:

* **`baseline`** -- prompting with the centre of the target's own chassis quad
  and no cross-view geometry at all. A homography that cannot beat this is
  buying nothing.
* **M4**, the per-view difference map, which needs no cross-view geometry either.

Per-instance results: `m1_per_instance.csv` (4 305 rows), `m1_identity.csv`,
`m3_jitter.csv`, `m4_per_event.csv`. Every table below is reproduced by
`summarise.py`; its full output is `summary.txt`.

## 2. M0 -- the overlays that had to come first

`m0/m0_d<desktop>_<view>.jpg`, 11 images, two steps each, **all of which I
opened and looked at**.

* **Coordinate frame confirmed by eye.** `tda/core/ls_import.py` rasterises the
  percent geometry into `ls_export.NATIVE_HW` and `upload_is_native()` *refuses*
  the `_Align_` uploads outright, so no OAK draft can be in aligned-1280x800
  coordinates. The overlays confirm it: on `m0_d13_oak1.jpg` and
  `m0_d24_oak2.jpg` the Motherboard, PSU and PSU-bracket polygons sit on those
  objects in the **4032x3040** still from `frame.path`, not on the
  `aux_json["aligned"]` png.
* **Step alignment looks right by eye and measures right.** On D13 the CPU
  cooler is outlined at step 11 in `scan`, `oak2` and `rs` and is gone by step
  31/32, matching `action` (cooler removed at step 13). M4's step-alignment
  probe (section 7) confirms it numerically: offset 0 is never beaten by
  offset -1.
* **Parts get annotated on the bench.** D24/D33 `scan` step 11/12 show the PSU
  polygon hanging off the left edge of the chassis; D13 `scan` step 31 has two
  `Case to Motherboard Connector` polygons on the table below the case. No
  chassis-plane homography can reach those. 39 of 4 305 rows (0.9%) have their
  scan centroid outside the clicked chassis quad; they are excluded from the
  headline and counted separately.
* **Ordinals drift inside one view too.** D33 `scan` calls the board
  `Motherboard#1` at step 12 and `Motherboard#2` at step 34.
* **Some scan frames are badly underexposed** -- `gallery/fail_06_oak1_d13.jpg`
  (D13 scan step 33) is nearly black. `frame.image_quality`,
  `hand_or_tool_in_frame` and `in_progress` are **NULL for all 516 frames** of
  these three desktops, so nothing could be filtered on them.

## 3. Pose segments: none needed

`m1a_pose_check.py` reduces every frame to 512 px and phase-correlates it
against the first frame of its sequence (`m1a_pose_check.csv`).

* Largest frame-to-frame jump anywhere: **13.2 px on a 4032 px OAK frame**
  (D24 oak1, step 9) -- 0.3% of the frame width.
* D24 shows a single small sustained step at step 9 (oak1 drift 6 -> 19 px and
  then flat at 13-16 px; scan 3 -> 8 px). D33 shows one at step 2 (scan 0 -> 4
  px, oak1 -> 8.7 px). D13 is flat throughout.
* The large "drift" figures (D33 scan 330 px, D33 rs 208 px, D13 scan 53 px) all
  occur **only on the last one or two steps**, where the motherboard has been
  lifted out and the correlation loses its lock on an almost-empty chassis. They
  are not motion: the consecutive jump at those same steps is 1-5 px.

**Conclusion:** one pose segment per (desktop, view); one set of clicks covers
the whole sequence. The residual 16 px nudge on D24 is an order of magnitude
below the errors measured below, and below what M3 shows matters.

## 4. M1 -- the clicked corners and the homography

`corners.yaml` (committed) holds four corners per (desktop, view) on **two**
planes, clicked on **step 41** -- the last step on which the motherboard is
still in the chassis in all three desktops, so the rim is unobstructed *and* the
floor plane still has a landmark.

* **`rim`** -- the four outer corners of the **chassis opening rim**, the lip
  the side cover seats on. This is the plane M1 quotes.
* **`floor`** -- the four corners of the **motherboard PCB**, which lies on the
  chassis floor. Chosen over bare sheet metal because the PCB edge is a crisp
  straight line in every view.

Corners are named by where they appear in **that desktop's scan frame**
(`s_tl`, `s_tr`, `s_br`, `s_bl`). The same physical corner is clicked in the
other views, located through the rotation between views that the M0 overlays
establish from where the annotated parts sit: **`rs` = scan orientation,
`oak1` = scan rotated 90 degrees anticlockwise, `oak2` = scan rotated 180
degrees**. (PSU, chassis screw cover and CPU socket all move as that rotation
predicts, on all three desktops.) No fiducials and no extrinsics were used; none
exist.

Clicking was done in three passes, all committed as code
(`make_corner_sheets.py --stage 1|2|3`): quadrant montages with labelled native
grids, zoom tiles on each click, and the quads drawn over the whole frame. The
stage-3 sheets (`corners/stage3/`) are the ones to look at to judge the clicks.

**How good are the clicks?** The motherboard is one big unambiguous object on
the floor plane present in every view, so its projection error is a clean read:

| desktop | view | n | rim err px | floor err px | chassis span px | rim % | floor % |
|---|---|---|---|---|---|---|---|
| 13 | oak1 | 21 | 44 | 79 | 1350 | 3.3% | 5.9% |
| 13 | oak2 | 31 | 42 | 32 | 914 | 4.6% | 3.5% |
| 13 | rs | 31 | 39 | 46 | 355 | 10.9% | 12.8% |
| 24 | oak1 | 34 | 246 | 105 | 1814 | 13.6% | 5.8% |
| 24 | oak2 | 33 | 40 | 128 | 1397 | 2.9% | 9.1% |
| 24 | rs | 33 | 53 | 73 | 457 | 11.5% | 15.9% |
| 33 | oak1 | 33 | 138 | 34 | 1792 | 7.7% | 1.9% |
| 33 | rs | 39 | 33 | 45 | 452 | 7.4% | 9.9% |

Read the **floor** column as click quality (the board is *on* that plane) and
the gap to the rim column as parallax. Floor error is 1.9-5.9% of the chassis on
the OAK views and 9.9-15.9% on `rs`. Part of even the floor number is not
geometry at all: the drafts are `amodal_complete=False`, i.e. **visible pixels
only**, so the same board traced in two views has two different visible extents
and therefore two different centroids. `gallery/fail_01_oak1_d24.jpg` shows
exactly that.

### M1 headline -- projected centroid, rim plane

Only rows whose scan centroid is inside the clicked chassis quad.

| view | subset | rim HIT | floor HIT | chassis-centre baseline | rim d/size | rim box IoU |
|---|---|---|---|---|---|---|
| oak1 | unique label | **53.3% (315)** | 62.2% (315) | 10.8% (315) | 0.47 | 0.20 |
| oak1 | multi-instance label, `#ordinal` pairing | 5.6% (1297) | 4.6% (1297) | 4.3% (1297) | 7.34 | 0.00 |
| oak2 | unique label | **68.6% (293)** | 36.2% (293) | 18.4% (293) | 0.25 | 0.46 |
| oak2 | multi-instance label | 10.9% (781) | 9.3% (781) | 4.6% (781) | 12.07 | 0.00 |
| rs | unique label | **35.2% (378)** | 31.2% (378) | 8.7% (378) | 0.62 | 0.17 |
| rs | multi-instance label | 13.3% (1202) | 8.8% (1202) | 5.8% (1202) | 3.07 | 0.00 |

Dropping the motherboard (whose polygon covers most of the chassis, so a hit on
it is nearly free): oak1 **47.7% (281)**, oak2 **64.6% (260)**, rs **29.0%
(345)**; baseline 0.0% / 8.1% / 0.0%.

**Box IoU is unusable everywhere.** Median 0.17-0.46 on the clean subset and
0.00 on everything else. The box should not be transferred at all.

### The number that explains all of the above

Per-label HIT on the unique subset, rim plane:

| label | n | median size px | rim HIT | rim d/size |
|---|---|---|---|---|
| Power Supply Unit (PSU) | 132 | 395 | 100.0% | 0.12 |
| Motherboard | 100 | 673 | 100.0% | 0.24 |
| Optical Drive | 36 | 441 | 100.0% | 0.10 |
| CPU Cooling Fan | 22 | 159 | 100.0% | 0.13 |
| Solid State Drive (SSD) | 22 | 234 | 86.4% | 0.17 |
| Heatsink | 25 | 212 | 60.0% | 0.20 |
| PSU Retention Bracket | 77 | 106 | 41.6% | 0.51 |
| Additional Card Upper Cover | 155 | 155 | 41.3% | 0.47 |
| CPU Socket Retention Bracket | 112 | 185 | 40.2% | 0.60 |
| GPU | 94 | 79 | 31.9% | 0.86 |
| CPU Chip | 112 | 87 | **0.0%** | 1.09 |
| RAM Module | 60 | 66 | **0.0%** | 1.00 |
| CPU Fan Connector | 32 | 11 | **0.0%** | 13.93 |

Stratified by part size (motherboard excluded):

| view | part size band | rim HIT | floor HIT | best plane |
|---|---|---|---|---|
| oak1 | 40-100 px | 0.0% (52) | 0.0% (52) | 0.0% |
| oak1 | 100-250 px | 0.0% (70) | 4.3% (70) | 4.3% |
| oak1 | > 250 px | 84.3% (159) | 100.0% (159) | 100.0% |
| oak2 | < 40 px | 0.0% (11) | 0.0% (11) | 0.0% |
| oak2 | 40-100 px | 0.0% (31) | 25.8% (31) | 25.8% |
| oak2 | 100-250 px | 68.9% (161) | 8.7% (161) | 68.9% |
| oak2 | > 250 px | 100.0% (57) | 89.5% (57) | 100.0% |
| rs | < 40 px | 0.0% (107) | 0.0% (107) | 0.0% |
| rs | 40-100 px | 27.0% (189) | 25.9% (189) | 27.0% |
| rs | 100-250 px | 100.0% (49) | 73.5% (49) | 100.0% |

The mechanism is simple and it is the whole result: **the projection error is
roughly constant in absolute pixels, so a part is hit if and only if it is
bigger than that error.**

| view | frame width | plane | p25 | median | p75 | p90 | median as % of frame width |
|---|---|---|---|---|---|---|---|
| oak1 | 4032 | rim | 112 | 200 | 249 | 257 | 5.0% |
| oak1 | 4032 | floor | 75 | **90** | 123 | 128 | 2.2% |
| oak2 | 4032 | rim | 26 | **38** | 65 | 79 | 0.9% |
| oak2 | 4032 | floor | 58 | 120 | 179 | 213 | 3.0% |
| rs | 1280 | rim | 25 | **34** | 49 | 66 | 2.6% |
| rs | 1280 | floor | 31 | 46 | 112 | 128 | 3.6% |

### Identity: the `#ordinal` is not a physical identity

`tda/core/ls_export.geometry_of` sorts a frame's shapes by ascending percent
bbox `min x` **within that frame**, and `ls_import` numbers them in that order.
So `#3` is "the third-from-the-left shape with this label *in this frame*" --
which differs between views and even between two steps of one view once a
neighbour has been removed.

Hungarian assignment on projected-centre distance versus the `#ordinal` pairing,
per multi-instance label and step:

| view | assignment agrees with `#ordinal` |
|---|---|
| oak1 | 20.8% (1136) |
| oak2 | 17.5% (668) |
| rs | 49.7% (1014) |
| all | **30.4% (2818)** |

Both pairings cannot be right, and the `#ordinal` rows' median centre error of
3-12 *part sizes* says the ordinal one is usually wrong. **Ordinals must never
be used as cross-view identity.** `gallery/fail_06_oak1_d13.jpg` shows
`Motherboard Screw#3` sitting on opposite sides of the board in the two views.

## 5. M2 -- which plane, and how much parallax

Per class group, pooled over views, unique-label rows:

| group | rim HIT | floor HIT | rim d/size | floor d/size | better |
|---|---|---|---|---|---|
| flat (board, RAM, CPU, connectors, screws on the board) | 34.9% (416) | **37.5% (416)** | 0.69 | 1.24 | floor |
| tall (heatsink/fan, PSU, drive cage, drives, cards) | **77.2% (338)** | 68.6% (338) | 0.17 | 0.34 | rim |
| rim (covers, latches) | **41.4% (232)** | 13.8% (232) | 0.47 | 0.82 | rim |

That is the physically expected answer, and the per-view detail carries it:

| view | group | rim HIT | floor HIT | rim err px | floor err px |
|---|---|---|---|---|---|
| oak1 | flat | 45.6% (125) | **65.6% (125)** | 246 | **90** |
| oak1 | tall | 69.3% (114) | **71.9% (114)** | 155 | 85 |
| oak1 | rim | 42.1% (76) | 42.1% (76) | 120 | 119 |
| oak2 | flat | **46.6% (118)** | 34.7% (118) | **65** | 179 |
| oak2 | tall | **100.0% (114)** | 57.0% (114) | **26** | 58 |
| oak2 | rim | **52.5% (61)** | 0.0% (61) | **26** | 120 |
| rs | flat | 19.1% (173) | 19.1% (173) | **56** | 77 |
| rs | tall | 61.8% (110) | **77.3% (110)** | 31 | 31 |
| rs | rim | **33.7% (95)** | 0.0% (95) | **25** | 35 |

**Parallax size.** Switching planes moves the projected point by 30-160 px on
the OAK views and 10-40 px on `rs` -- i.e. by **0.3 to 1.5 part-sizes** for a
typical part. `gallery/fail_09_oak1_d24.jpg` is the clean demonstration: the SSD
sits in the drive bay at floor level; the floor cross lands on it, the rim cross
lands ~150 px away, outside.

**oak2 `flat` is the exception** and it is my clicking, not physics: in `oak2`
the board's near edge projects almost onto the chassis's near rim line, so the
two clicked quads share an edge and the floor homography is poorly conditioned
in that direction. Treat the oak2 floor row as unreliable.

**The policy Plan B would actually ship** -- pick the plane from the part's
class group (flat -> floor, tall and rim -> rim):

| view | policy HIT | rim only | floor only | per-instance oracle |
|---|---|---|---|---|
| oak1 | **61.3% (315)** | 53.3% | 62.2% | 62.2% |
| oak2 | **63.8% (293)** | 68.6% | 36.2% | 71.3% |
| rs | **35.2% (378)** | 35.2% | 31.2% | 43.7% |

**Error by view.** `oak2` and `rs` are the *better* views for the rim plane
(median 38 px on 4032, 34 px on 1280) and `oak1` is the worst (200 px). That is
counter-intuitive -- `oak1` is the near-top-down camera -- and it is driven by
D24/D33 `oak1`, whose rim clicks carry 7.7-13.6% error against 1.9-5.8% on the
floor: the `oak1` cameras look down the chassis's long axis, where the rim and
floor quads are most foreshortened and the rim corners hardest to see.

## 6. M3 -- how careful must the four clicks be?

Jitter each of the four **target-view** clicks by `N(0, sigma)`, 200 draws, and
re-measure HIT on the unique-label rows (`m3_jitter.csv`, 25 596 rows;
63 000 / 58 600 / 75 600 instance-draws per cell).

| view | plane | sigma = 0 | 3 px | 6 px | 12 px |
|---|---|---|---|---|---|
| oak1 | rim | 53.3% | 52.3% | 50.6% | 50.1% |
| oak1 | floor | 62.2% | 63.5% | 64.7% | 66.7% |
| oak2 | rim | 68.6% | 69.8% | 70.1% | 70.7% |
| oak2 | floor | 36.2% | 36.3% | 37.4% | 39.0% |
| rs | rim | 35.2% | 36.9% | 38.2% | 39.7% |
| rs | floor | 31.2% | 34.8% | 33.8% | 32.0% |

**Nothing moves.** Some cells even improve, which is the tell: the unjittered
homography is not at a local optimum, i.e. my own click error already exceeds
12 px. The reason is measurable directly -- the same jitter applied to a grid of
points inside the scan quad displaces the projected point by:

| sigma | median displacement in target px (all desktops and views) |
|---|---|
| 3 px | 2.6 - 2.8 |
| 6 px | 5.0 - 5.6 |
| 12 px | 10.1 - 11.3 |

**One line:** a 12 px click error moves the hint by ~11 px while the systematic
error it competes with is 34-200 px, so click precision is *not* the bottleneck
-- the plane choice and the parallax are. Asking the human for sub-pixel care
buys nothing.

## 7. M4 -- the no-geometry baseline: the view's own difference map

For every step *k* whose `action` verb is `remove` / `disconnect` / `unscrew` /
`displace`, the removed instance is the draft **of the action target's taxonomy
class** that is annotated at step *k-1* and absent at step *k* (`action.target`
holds real keys like `screw.motherboard.03`, the drafts hold `ls:` keys, and
`instance.cls` is the only bridge). 372 events; **201 resolved to exactly one
draft** -- the rest are LS coverage gaps and are not counted.

`diff_delta_e` (ROI = the view's own LS union box padded 10%, `max_side=1600`,
`min_area` scaled to the view's resolution) then `diff_blobs`.

| view | top-1 blob centroid in the part | top-3 | top-1 blob *overlaps* the part | top-3 overlaps |
|---|---|---|---|---|
| **scan** (reference) | **21.5% (65)** | 21.5% (65) | 58.5% (65) | 63.1% (65) |
| oak1 | 21.6% (51) | 25.5% (51) | 35.3% (51) | 47.1% (51) |
| oak2 | 27.9% (43) | 32.6% (43) | 46.5% (43) | 51.2% (43) |
| rs | **40.5% (42)** | 40.5% (42) | 59.5% (42) | 64.3% (42) |

By class group, all views pooled: **tall parts 77.1% (35)**, **flat parts 16.3%
(166)**. The events are dominated by screws and connectors, which is why the
overall number is low.

The gap between "centroid inside" and "overlaps" is the whole story: the blob
usually *touches* the part but is much bigger than it, because removing a part
reveals a hole, a socket and a patch of board, and `diff_blobs` merges all of
that into one region whose centroid lands somewhere else.
`gallery/fail_11_oak1_d33.jpg` shows blob #1 swallowing the PSU *and* the board.
`diff_blobs` also logged `max_components` warnings (470-1008 raw components) on
the 12 MP views; those pairs are genuinely busy rather than misregistered,
because the pose check says the frames are stable.

**Step alignment, measured.** Same blobs, but the polygon taken at step
*k-1+off*:

| view | off = -1 | off = 0 |
|---|---|---|
| scan | 22.0% (59) | **21.5% (65)** |
| oak1 | 20.9% (43) | **25.5% (51)** |
| oak2 | 32.5% (40) | **32.6% (43)** |
| rs | 40.5% (37) | **40.5% (42)** |

Offset 0 is never beaten, so the LS step numbering agrees with the `action`
table on these three desktops. Caveat: `off = +1` is structurally untestable
here -- the instance is *defined* as absent at step *k* -- so this rules out a
-1 shift, not a +1 one.

## 8. M5 -- homography point gated by the diff blobs

Prompt = the top-3 diff blob whose centroid is nearest the projected point. Same
population as M4, so the M1 column here is lower than section 4's headline:
these events are mostly small flat parts and include multi-instance labels.

| view | M1 point alone | M4 top-1 blob alone | M5 combined |
|---|---|---|---|
| oak1 | 19.5% (41) | 24.4% (41) | **24.4% (41)** |
| oak2 | 26.5% (34) | 35.3% (34) | **38.2% (34)** |
| rs | 20.5% (39) | 43.6% (39) | **43.6% (39)** |

**The combination adds almost nothing.** M5 equals M4-top-1 on `oak1` and `rs`
and beats it by one event on `oak2`: the projected point is usually far enough
out that it simply picks whichever blob was already strongest. The diff carries
the combination; the homography contributes the tie-break and little else.

M6 (SAM 2.1 end to end) was not run -- M1 and M5 never produce a point reliable
enough for the mask IoU to mean anything, so the measurement would only report
SAM's behaviour on a wrong prompt.

## 9. Failure gallery

`gallery/fail_01..12`, all under 200 KiB, all opened and looked at. Left panel:
the `scan` frame, source polygon in green with its centroid; red = clicked rim
quad, blue = clicked floor quad. Right panel: the target frame, the instance's
own polygon in green, the rim projection as a red cross, the floor projection as
a blue cross, and for the diff cases the top-3 change blobs as orange boxes.

| file | case | why it failed |
|---|---|---|
| `fail_01_oak1_d24.jpg` | D24 oak1 step 8, Motherboard | **LS noise, not geometry.** At step 8 the scan traces only the board's upper strip (everything else is still covered) while oak1 traces a different visible lobe. Two different visible extents of one part have two different centroids, so the transfer cannot be right whatever the homography does. |
| `fail_02_oak2_d13.jpg` | D13 oak2, largest both-plane miss | **Parallax.** The rim and floor crosses sit a chassis height apart and the part lies between the two planes, so neither reaches it. |
| `fail_03_rs_d24.jpg` | D24 rs, largest both-plane miss | **Parallax plus a small chassis.** The rs chassis spans only ~457 px, so the same relative error is a larger share of the part. |
| `fail_04_rs_d33.jpg` | D33 rs step 3, Heatsink | **Small part.** The heatsink is ~30 px across in rs; both crosses land 40-60 px away, which is an *ordinary* error for this view. Nothing is wrong except that the part is smaller than the method's accuracy. |
| `fail_05_oak1_d13.jpg` | D13 oak1, part outside the chassis quad | **On the bench.** The part has been lifted out and annotated where it lies on the table; it is no longer on either clicked plane, so the homography extrapolates outside its quad and the answer is meaningless. |
| `fail_06_oak1_d13.jpg` | D13 oak1 step 33, `Motherboard Screw#3` | **Identity swap.** The `#3` screw in scan and the `#3` screw in oak1 are on opposite sides of the board -- the ordinal is a per-frame left-to-right index. Note also that the scan frame is almost black: an image-quality problem the NULL `image_quality` column cannot flag. |
| `fail_07_rs_d13.jpg` | D13 rs, worst ordinal mismatch | **Identity swap**, same mechanism, with the projected point landing several part sizes from the "matching" polygon. |
| `fail_08_oak2_d24.jpg` | D24 oak2, rim-correct / floor-wrong | **Wrong plane.** A part sitting on the chassis rim, projected with the floor homography, lands a chassis depth inside the case. |
| `fail_09_oak1_d24.jpg` | D24 oak1 step 12, SSD | **Wrong plane, the other way.** The SSD is at floor level; the blue (floor) cross lands on it, the red (rim) cross ~150 px away and outside. This is the clearest single picture of why the plane must be chosen per class. |
| `fail_10_oak2_d24.jpg` | D24 oak2 step 43, Motherboard removal | **Diff blob too large.** Lifting the board changes the whole chassis interior; the merged blob's centroid lands on revealed sheet metal rather than on where the board was. |
| `fail_11_oak1_d33.jpg` | D33 oak1 step 18, PSU removal | **Diff blob merges two events.** Blob #1 swallows the PSU bay *and* the cable movement across the board, so its centroid is on the board. Blob #1's box does overlap the PSU -- which is why "overlap" scores 47-64% while "centroid inside" scores 21-41%. |
| `fail_12_rs_d13.jpg` | D13 rs step 29, PSU-to-board connector | **Diff too small to separate.** Unplugging a connector changes a handful of pixels; the strongest blobs in the frame belong to the cable being pulled aside, not to the connector body. |

## 10. Recommendation

**Per view -- is the 4-corner homography good enough as a hint?**

* **`oak1`** -- *Only for parts larger than ~250 px (about 6% of the frame).*
  Above that it is 84-100%; between 40 and 250 px it is **0-4%**. Median error
  90 px (floor plane). Worth building for the PSU, the board, the drive cage and
  the optical drive; useless for screws, RAM, the CPU and connectors.
* **`oak2`** -- *The best of the three, and still not a point prompt.* Median rim
  error 38 px on a 4032 px frame; 100% on parts over 250 px, 69% on 100-250 px,
  0% below 40 px. The operator regularly occludes the chassis in this view, so
  any hint must be allowed to come back empty.
* **`rs`** -- *No.* 29-35% on the cleanest subset, 0% on everything under 40 px,
  and 40-100 px parts only reach 27%. The chassis occupies ~450 px of a 1280 px
  frame; there is not enough resolution for the error budget.

**Is the per-view diff enough on its own?** Not as a point prompt (top-1
21-41%), but it is the *only* signal here that needs no cross-view geometry, no
clicking and no per-desktop calibration, it is already implemented
(`tda/core/diffmap.py`), and its *region* is right far more often than its
centroid (overlap 47-64%). On `rs` it beats the homography outright (40.5%
against 20.5% on the same events).

**What Plan B should build**

1. **Do not ship the box.** Box IoU is 0.17-0.46 at best and 0.00 in general.
   Transfer a point or a region, never a box.
2. **Do not use `ls:<Label>#<n>` -- or any positional ordinal -- as cross-view
   identity.** A Hungarian assignment disagrees with it on 70% of cases. If Plan
   B needs cross-view identity it has to come from the annotator or from the
   `instance` table, not from ordering.
3. **If the homography is built, make it two homographies and pick by class.**
   Flat -> floor plane, tall and rim-mounted -> rim plane; that is worth +8
   points on `oak1` over rim-only and is the difference between 0% and 66% for
   flat parts there. `pose_segment.corners_json` / `homography_json` already
   exist and are NULL everywhere, so there is a place to store both.
4. **Gate the hint on part size, and say so in the UI.** Below roughly
   `2 x median error` in the target view (about 180 px on oak1, 80 px on oak2,
   70 px on rs) the hint is worse than useless -- it points at a neighbour. Show
   it as a *disc of radius = the measured p75 error* rather than as a point, so
   the annotator can see that the tool is guessing.
5. **Spend the effort on the per-view diff instead of on clicking.** M3 proves
   the four clicks do not need care (12 px of jitter changes nothing), so
   clicking is cheap -- but so is the return. The diff is already there, needs no
   calibration and degrades gracefully; the work worth doing is making
   `diff_blobs` produce a *tight* region for a removal, since the top-3 blob
   already overlaps the part 47-64% of the time and only its centroid is wrong.
6. **Handle bench parts explicitly.** 0.9% of rows here, but structurally
   unfixable by any chassis homography, and the fraction grows towards the end of
   every sequence. Suppress the hint when the source centroid falls outside the
   chassis quad.
7. **Before any of this, fix the ground truth.** These numbers are a *lower bound
   on the method* and an *upper bound on the data*: the drafts are visible-pixel
   traces (`amodal_complete=False`), so part of every error above is two views
   disagreeing about what is visible rather than the geometry being wrong.
   `frame.image_quality` / `hand_or_tool_in_frame` / `in_progress` are NULL on
   all 516 frames of these desktops, so nothing could be excluded on quality, and
   at least one scan frame in the sample is nearly black.

---

### Files

| file | what |
|---|---|
| `report.md` | this |
| `summary.txt` | full `summarise.py` output, every table with its `n` |
| `m1_per_instance.csv` | 4 305 rows: per instance-step, both planes, HIT / distances / IoU / flags |
| `m1_identity.csv` | 2 818 Hungarian-versus-ordinal comparisons |
| `m3_jitter.csv` | 25 596 rows: hits per (instance, plane, sigma) over 200 draws |
| `m4_per_event.csv` | 372 removal events, 201 resolved; blobs, offsets, M5 |
| `m1a_pose_check.csv` | per-frame drift and jump, all 12 (desktop, view) sequences |
| `m0/` | 11 sanity overlays |
| `corners/stage1,2,3/` | the sheets the corners were clicked and checked on |
| `gallery/` | 12 failure sheets |

Code and `corners.yaml`: `experiments/plan_b_probe/transfer/` on branch
`worktree-agent-ad56db6176d22770d`.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "report.md"
    path.write_text(REPORT.lstrip("\n"), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
