"""Write the E1 deliverable report into experiments_out/.../camera_moves/report.md.

The prose lives here so the report is regenerated from the repo rather than
hand-edited in the output directory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TEXT = r"""# E1 - Did the rig hold still? Camera motion from table-fixed landmarks

Read-only measurement over all 4 views x 66 desktops x ~2,830 steps (11,320
frames). Nothing was written to `F:`, and the database was read only through the
read-only copy `D:\DataSet\.cache\tmp\exp_copy.sqlite`.

Code: `experiments/plan_b_probe/` on branch `worktree-agent-ac593d562118e7242`.
Outputs: this directory (`events.csv`, `rig_epochs.csv`,
`desktop_boundaries.csv`, `tag_board.csv`, `overlays/`).

---

## 1. Headline

| view | rig epochs | between-desktop moves | within-sequence moves | unresolved boundaries |
|------|-----------:|----------------------:|----------------------:|----------------------:|
| oak1 | **12**     | 6                     | 5                     | 3 |
| oak2 | **4**      | 2                     | 1                     | 3 |
| rs   | **1**      | 0                     | 0                     | 1 |
| scan | **1**      | 0                     | 0                     | 2 |

* **14 camera moves confirmed in total** (8 between desktops, 6 inside a
  sequence).
* **The RealSense and the scanner never moved relative to the table**, over all
  10 capture days - no confirmed break in either view.
* **oak1 is the unstable one.** It moved 11 times, including one very large move
  *in the middle of desktop 36's teardown* (step 18->19, ~190 px).
* The cam1<->cam2 relation from the fiducial board is **not constant**: it jumps
  between desktop 35 and desktop 36 - independent corroboration of that oak1
  move.

Validation: **14/14** flagged moves confirmed by eye, **12/12** checked
non-events correctly quiet (details in section 4).

---

## 2. Method

### 2.1 What counts as a landmark

Every frame shares exactly one thing: the white work table, with its yellow tape
and its black marks and screw heads. The chassis, hands, tools and removed parts
all move, so none of them may vote.

The table mask is built per frame as:

1. **colour gate** - white (low saturation, high value) union yellow tape, with
   per-view HSV thresholds (the scanner's tape reads tan and its table is blown
   out, so it gets its own).
2. **tape, not stickers** - yellow components are kept only if they are long and
   thin (perimeter^2/area >= 30). Hardware carries yellow warranty and drive
   labels that pass any colour gate and then move with the part.
3. **fill holes** - black table marks and screw heads are enclosed by table, so
   they are filled back in; otherwise step 6 would erase the very corners that
   make the best landmarks.
4. **drop the chassis hull** - bright hardware (bare optical-drive metal, a PSU
   lid) passes the white gate and sits flush against the table, so neither
   colour nor connectivity separates it. The chassis as a whole is dark and
   saturated, so its convex hull is found from the central dark mass and
   everything inside it is removed, bright parts included. The hull is discarded
   if it would swallow more than 55 % of the frame (an oblique view can merge
   the chassis with the floor).
5. **border test** - a surviving table component must reach the image border.
6. **pull back** - erode away from everything that is not table, so no chassis
   outline, hand or removed part contributes a keypoint.

ORB keypoints are then detected **inside that mask only**, at a 640-px working
scale, and cached (`.cache/tmp/e1`, deleted at the end).

### 2.2 Measuring the motion

Frame pairs are matched (Hamming + ratio test) and fitted with a RANSAC
similarity. The magnitude reported is the **median displacement over an image
grid under the refitted similarity**, converted back to original-image pixels.

Raw keypoint coordinates are quantised, so the median raw displacement bottoms
out at ~1 working pixel and cannot resolve a small move; a similarity fitted in
closed form (Umeyama) to a few hundred inliers is good to well under a pixel.

**Noise floor**: consecutive frames of a static camera give a median of
**0.11-0.19 px** (p90 approx 0.24-0.42 px) across the four views. The reporting
threshold is 2.0 px, roughly 10x the floor.

### 2.3 The arbiter: an independent tape test

ORB alone is a *candidate generator*, not a decision. It will happily fit a
transform to a bright part that slid across the table - and in the first pass it
did exactly that: at threshold 2 px only **9 of 24** flags survived scrutiny.

Every candidate is therefore re-tested on the tape shape, which cannot be
confused with hardware: warp frame A's tape mask by the proposed transform and
measure the overlap with frame B's tape, before and after.

* `iou_after > iou_before + 0.02` -> the transform corrects a real table
  displacement -> **confirmed**.
* otherwise -> **rejected**: whatever moved, it was not the table.
* if the tape barely overlaps under either hypothesis (`max(iou) < 0.05`) ->
  **undetermined**; calling that a rejection would be inventing a result.

This test shares no failure mode with the matching that raised the flag. On 100
quiet control pairs the gain was **exactly 0.000 in 100/100** - the test never
manufactures a move.

Candidates the matcher cannot resolve at all get a scale-invariant SIFT retry on
the same mask, then a tape-mask ECC registration. What none of those settle is
reported as undetermined and carried to the overlays for a human verdict; five
such pairs were decided by eye and are marked `decided_by=eye` in the CSVs, with
the reason recorded.

### 2.4 Camera vs chassis

Chassis motion is measured from ORB features taken *inside* the chassis, not
from its outline. A teardown removes parts at nearly every step, which moves the
blob's centroid and changes its area without the chassis having been touched -
the outline measure produced ~760 spurious "chassis moved" events per view; the
rigid fit produces 5 in total.

* `camera` - table landmarks moved (confirmed).
* `chassis` - table static, chassis has a coherent rigid motion >= 12 px.
* `both` - both.

---

## 3. Ordering note (important)

Q1 asks for capture-time order. For `oak1`, `oak2` and `rs` the capture
timestamps are genuine and run **monotonically with desktop number** (desktop 1
= 2025-05-31 10:47, desktop 66 = 2025-06-09 16:57), so capture order *is*
desktop order.

`frame.ts` for the **scan** view is **not a capture time**: it is 2025-06-15/16
for every row and orders the desktops 10, 11, ..., 66, 1, ..., 9 - the signature
of a bulk export sorted lexicographically by folder name. Scanner frames were
therefore ordered by desktop number, justified by the OAK/RS clocks, not by
their own `ts`.

---

## 4. Validation

All overlays are in `overlays/` (99 PNGs, each < 1.5 MB). Each has the two
frames side by side, a full-frame red/cyan anaglyph, and a x2.6 zoom on a tape
region that lies on the table in *both* frames, with the same reference cross
drawn on each crop. A camera move doubles the tape edges; a chassis-only change
leaves them grey.

**Every verdict below comes from opening the PNG and looking at it.**

### Flagged events - 14/14 correct (precision 1.00)

| overlay | measured | verdict |
|---|---:|---|
| `conf_oak1_d03-d04` | 7.58 px | tape strips visibly doubled - correct |
| `conf_oak1_d07-d08` | 3.36 px | distinct red/cyan tape streaks - correct |
| `conf_oak1_d10-d11` | 5.71 px | tape edge doubled in zoom - correct |
| `undet_oak1_d20-d21` | undetermined | tape grid grossly offset - **real move**, magnitude not measurable |
| `conf_oak1_d24-d25` | 3.13 px | tape corner shifted ~9 zoom px - correct |
| `conf_oak1_d29-d30` | 5.40 px | tape corner clearly shifted - correct |
| `conf_oak2_d03-d04` | 28.75 px | whole table grossly doubled - correct |
| `conf_oak2_d31-d32` | 3.27 px | subtle but tape edges split - correct |
| `conf_oak1_d02_s005-006` | 8.60 px | two tape strips shifted together - correct |
| `conf_oak1_d04_s006-007` | 9.02 px | three tape strips shifted together - correct |
| `conf_oak1_d29_s046-047` | 5.95 px | red/cyan on both tape edges - correct |
| `conf_oak1_d32_s038-039` | 2.34 px | smallest confirmed; tape edge split visible - correct |
| `conf_oak1_d36_s018-019` | 190.45 px | tape rectangle moved right out of place - correct |
| `conf_oak2_d01_s033-034` | 14.09 px | tape, table and fixed machine all doubled - correct |

### Non-events - 12/12 correct

Eight quiet pairs (chosen where the *chassis* changed most, so only the table is
being tested) and four pairs the tape test rejected:

`still_oak1_d24_s008-009`, `still_oak1_d33_s039-040`, `still_oak2_d24_s008-009`,
`still_oak2_d48_s010-011`, `still_rs_d20_s028-029`, `still_rs_d49_s024-025`,
`still_scan_d14_s017-018`, `still_scan_d48_s035-036` - in all eight the tape
sits at an identical position and the zoom anaglyph is pure grey, while the
hardware is strongly doubled.

`rej_scan_d55_s021-022` (46.7 px) - the tape is pixel-identical; a **cable
connector** lying on the tape slid. Correct rejection.
`rej_scan_d56-d57` (40.8 px) - the black table dots are single, not doubled; the
40 px came from the chassis and from per-desktop corner tape. Correct.
`rej_scan_d24_s041-042` (3.07 px) and `rej_rs_d24_s008-009` (2.55 px) - table
and tape aligned, only hardware doubled. Correct.

### Cases inspected and demoted to undetermined

* `rej_oak1_d64_s019-020` (610 px) and `rej_oak1_d39_s023-024` - too little
  table in frame to judge; reported as undetermined rather than as a rejection.
* `undet_scan_d63_s031-032` (960 px) - looking at it, the **chassis was rotated
  ~90 deg** while the visible table sliver stayed aligned. Recorded as
  `chassis`.

---

## 5. Q1 - Rig epochs per view

Full table in `rig_epochs.csv`; per-boundary numbers in
`desktop_boundaries.csv`. An epoch is a span over which the camera held still,
in `(desktop, step)` coordinates - a mid-sequence move splits its desktop.

### oak1 - 12 epochs

| epoch | from | to | broken by | px |
|---|---|---|---|---:|
| 1 | d1 | d2 s5 | d2 step 5->6 | 8.60 |
| 2 | d2 s6 | d3 end | after desktop 3 | 7.58 |
| 3 | d4 s1 | d4 s6 | d4 step 6->7 | 9.02 |
| 4 | d4 s7 | d7 end | after desktop 7 | 3.36 |
| 5 | d8 | d10 end | after desktop 10 | 5.71 |
| 6 | d11 | d20 end | after desktop 20 | large, undetermined |
| 7 | d21 | d24 end | after desktop 24 | 3.13 |
| 8 | d25 | d29 s46 | d29 step 46->47 | 5.95 |
| 9 | d29 s47 | d29 end | after desktop 29 | 5.40 |
| 10 | d30 | d32 s38 | d32 step 38->39 | 2.34 |
| 11 | d32 s39 | d36 s18 | d36 step 18->19 | 190.45 |
| 12 | d36 s19 | d66 end | - | - |

Rotations are tiny throughout (|theta| <= 0.22 deg) and scale stays within
1.000 +/- 0.007, except the d20->d21 and d36 moves, which change the framing
substantially (the d20->d21 pair goes from a wide table view to a tight one).

### oak2 - 4 epochs

| epoch | from | to | broken by | px |
|---|---|---|---|---:|
| 1 | d1 | d1 s33 | d1 step 33->34 | 14.09 |
| 2 | d1 s34 | d3 end | after desktop 3 | 28.75 |
| 3 | d4 | d31 end | after desktop 31 | 3.27 |
| 4 | d32 | d66 end | - | - |

A marginal case: oak2 d15->d16 measures 1.46 px with a small positive tape gain
(0.022). It is below the 2 px reporting threshold and is *not* treated as an
epoch break; if you want maximum conservatism, treat it as a possible fifth
epoch boundary.

### rs - 1 epoch, scan - 1 epoch

No confirmed between-desktop or within-sequence move in either view. The largest
between-desktop measurement that survived the tape test is 1.67 px (rs) and
3.25 px (scan, rejected: tape gain -0.011).

### Note on d3->d4

Both OAK views break at the same boundary (oak1 7.58 px, oak2 28.75 px), across
a two-day gap (desktop 3 = 31 May, desktop 4 = 2 June). Two cameras moving by
different amounts at the same moment is more consistent with the **shared mount
or the table** being disturbed than with two independent knocks.

This is a general limit of the method: it measures the **camera-to-table
relation**. It cannot tell whether the camera moved or the table moved - and for
reusing annotations that distinction does not matter, because either one
invalidates a stored pose.

---

## 6. Q2 - Within-sequence events

Full table in `events.csv` (`kind` in camera | chassis | undetermined;
`magnitude_px`; plus the inlier counts and the tape-test numbers behind each
decision).

### Camera moves inside a sequence (6)

| view | desktop | step | px | rot | scale |
|---|---:|---|---:|---:|---:|
| oak1 | 2 | 5->6 | 8.60 | 0.22 deg | 0.9986 |
| oak1 | 4 | 6->7 | 9.02 | 0.13 deg | 1.0067 |
| oak1 | 29 | 46->47 | 5.95 | 0.22 deg | 0.9986 |
| oak1 | 32 | 38->39 | 2.34 | 0.08 deg | 0.9994 |
| oak1 | 36 | 18->19 | **190.45** | -0.89 deg | 0.9480 |
| oak2 | 1 | 33->34 | 14.09 | -0.13 deg | 0.9942 |

**Desktops whose sequences contain a camera move:** oak1 -> 2, 4, 29, 32, 36;
oak2 -> 1; rs -> none; scan -> none.

### Chassis moves (5)

| view | desktop | step | px | note |
|---|---:|---|---:|---|
| scan | 36 | 40->41 | 13.1 | table static, chassis repositioned |
| scan | 61 | 1->2 | 18.0 | table static, chassis repositioned |
| scan | 63 | 31->32 | 964.8 | chassis rotated ~90 deg (confirmed by eye) |
| scan | 64 | 15->16 | 933.1 | tower manipulated; low confidence |
| oak1 | 64 | 16->17 | 570.7 | tower manipulated; low confidence |

The chassis measure only catches a *rigid* re-placement. A chassis flip, where
the visible surface changes entirely, breaks the feature match and is reported
as undetermined rather than as a chassis move. Ordinary part removal is
correctly **not** reported.

### Comparison with the stored `pose_segment`

Two things to know first:

* `frame.pose_segment` is **NULL for all 11,320 rows** - the column is unused.
  The segmentation lives in the `pose_segment` table.
* In that table, `homography_json` and `corners_json` are **NULL for every row**
  in all four views: segments are declared but carry no geometry.

The stored segmentation has 73 rows per view = 65 desktops with a single segment
plus **desktop 64 split into 8 segments** (steps 1-15, then 16, 17, 18, 19, 20,
21, 22 each alone). It is **identical across all four views**.

**Where we agree.** Desktop 64 is the one place the stored annotation claims the
pose changes, and it is exactly where my method runs out of evidence: in that
desktop the camera is looking inside a tower chassis and almost no table is in
frame, giving undetermined verdicts at oak1 18->19/19->20/20->21, oak2
17->18/20->21, rs 15->16/16->17/17->18/20->21, scan
17->18/18->19/20->21/21->22 - steps 15-22 in every view. Both accounts agree
that desktop 64 steps ~15-22 are not a single stable pose.

**Where we disagree.**

1. The stored annotation marks **no** pose change in desktops 2, 4, 29, 32, 36
   (oak1) or desktop 1 (oak2), where the camera demonstrably moved - the
   desktop 36 move is ~190 px and visible at a glance. These are missed.
2. The stored segmentation is **the same for all four views**, which cannot be
   right: a camera move affects only its own view. oak1 moved during desktop 36
   while rs and scan did not; oak2 moved during desktop 1 while oak1 did not.
   Pose segments have to be per view.
3. Conversely, the per-step splitting of desktop 64 into 7 single-step segments
   is finer than anything I can support - I can only say that region is
   unresolvable from table landmarks.

---

## 7. Q3 - The fiducial board (time-boxed feasibility, then continued)

The brief said to stop unless detection was clearly solid. It is, comfortably,
so the cam1<->cam2 question was answered too. Per-capture numbers:
`tag_board.csv`.

### Detection

* **Dictionary: AprilTag 36h11**, tag ids 1...40 (8x5). Of the 21 predefined
  OpenCV dictionaries swept, only 36h11 gives a coherent result; 4x4 and 16h5
  return 2 spurious tags.
* **12 MP jpg: 40/40 tags in both cameras on 43 of 44 captures** (d41 cam1: 39;
  one capture reported 41 in cam2, i.e. one false positive).
* **1280x800 aligned png**: cam2 40/40 on all 44; cam1 32-40 (median 40, mean
  39).
* Detection was solid on the first 10-capture sample (40/40 in both cameras,
  10/10), which passed the ">= 30/40 in both cameras on most captures" gate.
* **44 of 66 desktops** have a calibration capture. Two folder naming
  conventions are in use - `Camera_1`/`Camera_2` (39 desktops) and `C1`/`C2`
  (5 desktops); counting only the first gives a misleading 39.

### Intrinsics, recovered from the data

No intrinsics ship with the dataset, but they can be recovered. The `.ply` is
not organized (923,927 points != 1,024,000), **but its vertex count equals the
number of non-zero depth pixels** - Open3D dropped the invalid ones and kept
row-major order. So the i-th vertex is the i-th non-zero pixel of the 800x1280
depth map.

Two checks confirm it: the ply's colour equals the aligned png's colour at the
mapped pixel for **100.00 %** of points, and a pinhole model fits the resulting
pixel<->3D correspondence with a **median residual of 0.000 px**.

Recovered (desktop 30): cam1 fx 1148.6, fy 1148.5, cx 632.1, cy 399.9;
cam2 fx 1146.5, fy 1145.6, cx 629.7, cy 391.1. Depth `.npy` is in mm; ply z is
in metres with the sign flipped.

### Is cam1<->cam2 constant? No - it jumps between desktop 35 and desktop 36.

Two independent estimates, both per capture:

* **Plane-induced homography** from the 2D tag corners (reprojection
  0.28-0.49 px on all 44). For a planar scene H depends on the relative camera
  pose and the board plane, not on where the board was put down.
* **Rigid transform** by Kabsch, with each corner's 3D position obtained by
  fitting the board plane to the cloud and intersecting the corner ray with it
  (per-pixel depth is too noisy at a corner). Usable on 26 of 44 - on the other
  18 the cam2 board-plane residual is 33-76 mm, because the board sits in a
  glossy plastic sleeve and glare destroys the depth. Those are gated out, not
  rescued.

| | desktops 22-35 | desktops 36-66 |
|---|---|---|
| H vs the d22 reference | 0.0 - 9.3 px | 97 - 143 px |
| rigid rotation vs the d24 reference | 0.0 - 2.4 deg | 6.5 - 16.9 deg |
| rigid translation vs the d24 reference | 0 - 31 mm | 28 - 104 mm |

`dH` from one capture to the next is <= 10 px through d35, then **128 px** at
d35->d36, and stays displaced thereafter. Because the rigid transform is
board-independent and shows the same step, the jump is a genuine change in the
camera pair, not the board being re-placed.

**This agrees with Q1/Q2 from a completely independent measurement.**
Calibration captures are taken 1-25 minutes *after* each desktop's sequence, so
they reflect that desktop's rig state - and the table-landmark method
independently found a 190 px oak1 camera move at desktop 36 step 18->19, between
the two calibrations that bracket the jump.

Residual scatter within each regime (H +/-20 px, rotation +/-5 deg) is *not*
resolved: it is confounded with board re-placement and with the ~2 deg / 25 mm
noise floor of the depth-derived 3D. Do not read small changes in
`tag_board.csv` as rig motion.

### Does the board appear in the scanner or RealSense data?

Searched folder and file names for
calib/tag/board/aruco/april/chess/charuco/marker.

* **RealSense: yes.** `Exp\Desktop NN\Calibration\001\` exists for desktops
  **22-66** (45 desktops), each with `original_color.png` (1280x720),
  `depth_raw.npy`, `pointcloud_bg_removed.ply`, `background_removed.png` and
  `capture_info.txt`. There is also `Dataset Information\CalibrationALL\Aruco\`
  with **44 PNGs** (`1.png`...`44.png`, 1280x720). The board is clearly visible
  but much smaller in frame than in the OAK views; these were **not**
  processed - out of scope once the OAK pair answered the question.
* **Scanner: no.** Nothing under `UGA DATA` matches any of those keywords; the
  tree is `TAMU_B2.3_<range>_RGB\<desktop>\RGB<step>\P_*.png` only. No board
  shot exists for the scanner.

---

## 8. Limits and uncertainties

1. **Camera vs table is not separable.** The measurement is the camera-to-table
   relation. Both OAK views breaking at d3->d4 suggests the table or a shared
   mount moved, not two cameras independently.
2. **Unresolved boundaries.** oak1 after d49, d58, d62; oak2 after d9, d62, d63;
   rs after d63; scan after d62, d63. The epoch counts are **lower bounds** -
   each unresolved boundary could hide a move. They are listed in
   `rig_epochs.csv:uncertain_breaks`.
3. **Desktops 63 and 64 are largely unmeasurable.** The chassis is a tower that
   fills the frame and the table is barely visible, so most step pairs there are
   undetermined in all four views. Any pose claim about those two desktops needs
   a different method.
4. **Tape is not always table-fixed.** In the scanner view, small tape pieces are
   placed around each chassis and move between desktops; only the large tape
   rectangle and the black table marks are truly fixed. This is why the tape
   test is a *differential* check (does the proposed transform improve the
   overlap?) rather than an absolute one.
5. **Sub-2 px moves are not reported.** The floor is ~0.2 px and the threshold is
   2.0 px, so a genuine 1 px drift would be recorded as static. oak2 d15->d16
   (1.46 px, positive tape gain) is the one near-miss.
6. **Chassis flips are invisible** to the chassis measure (see section 6).
7. **The board's residual scatter is not rig motion** (see section 7).
8. **The 12 MP OAK jpgs were not used** for Q1/Q2; the 1280x800
   `*_rgb_aligned.png` was, as the brief allowed. Magnitudes are in aligned-png
   pixels for OAK (1280 wide), RealSense pixels (1280 wide) and scanner pixels
   (1600 wide).

---

## 9. Files

| file | contents |
|---|---|
| `events.csv` | every non-quiet step pair: view, desktop, step_from, step_to, kind, magnitude_px, rot/scale, verdict, how it was decided, inlier counts, tape-test IoUs |
| `rig_epochs.csv` | epochs per view in (desktop, step) coordinates, what broke each one, and the unresolved boundaries |
| `desktop_boundaries.csv` | all 65 between-desktop comparisons per view with their verdicts |
| `tag_board.csv` | per calibration capture: tags found per camera on both resolutions, homography reprojection and drift, Kabsch rmse, plane residuals, whether depth was usable |
| `overlays/` | 99 validation PNGs (`conf_` / `rej_` / `undet_` / `still_`) |
"""


if __name__ == "__main__":
    os.makedirs(C.OUT, exist_ok=True)
    path = os.path.join(C.OUT, "report.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(TEXT)
    print("wrote", path, len(TEXT), "chars")
