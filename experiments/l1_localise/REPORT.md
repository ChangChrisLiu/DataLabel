# L1 — can the reverse-walk guess ("where is the part to add back?") be made much better?

Measured on B2's 201 removal events (desktops 13/24/33, old Label Studio drafts as
ground truth), read-only on a copy of the DB. Code: this folder. Per-event CSVs,
tables, overlays and probes: `experiments_out/l1_localise/` (git-ignored). Written
up by the controller from the worker's hand-back (the worker could not write .md
files in its sandbox). "Hit" = the armed point lies inside the part's mask
(point-in-part).

**Verdict: do not wire M3/M4.** The best heuristic lifts parts under 40 px from
~8 % to ~18 %; the gate (+20 points on scan *and* oak1) fails on scan (+4 to +8),
and the runtime sits on the 0.3 s line. **M1 (no box when unsure) is worth wiring
for scan only.** The most important finding is that **B2's ground truth is wrong on
about a quarter of the events.**

## 1. B2's ground truth is wrong on 48 / 201 events

`ls:` instance ordinals are re-numbered per frame by bbox min-x, so the key that
"disappears" between frame k-1 and k is always the last ordinal, not necessarily the
part that was removed (`probe_gt.py`). On 48 events the B2 ground-truth part is still
in place at k (33 of them < 40 px, 14 of 40–100 px); 17 more cannot be resolved by
position. Native-resolution ΔE inside the GT box, median: B2 GT 1.6, position GT 4.2;
41 / 48 B2 GT boxes have ΔE < 3. Seven check sheets opened by eye: in all seven the
B2 GT is unchanged between j and k, and in the six resolvable ones the position GT is
exactly the part that left (screw head → hole, RAM stick pulled).

- **GT-A**: B2 as is (kept as asked).
- **GT-B**: position-resolved, 184 / 201 events. **Conclusions use GT-B.** GT-A
  rewards "found another, still-present part of the same class" (appearance-only M3j
  top-3 on small parts: 33 % under GT-A, 5 % under GT-B).

## 2. B2's baseline is not the call the app makes

The app calls `diff_delta_e(k, j)` (neighbour first) with `diff_blobs`' default
`min_area=80`; B2 called `diff_delta_e(j, k)` with a per-view min_area, and
`diff_delta_e` is asymmetric (the ±1 px tolerance and blur act on the second frame).
`M0app` reproduces the app exactly; M0 still reproduces B2 (21.5 / 21.6 / 27.9 /
40.5 %). Totals are close (GT-A 27.4 % vs 26.9 %), single events differ.

## 3. Headline — hit rate %, parts < 40 px, GT-B

Variants chosen on **D24 only**, reported on all events and on D13+D33.
M3 = `M3js_rr10` (sibling appearance, skip already-drawn siblings, re-rank top 10 by
native ΔE); M4 = `M4j_rr10` (same re-rank, registered k, no skip); fallback to M0
where no sibling exists. P4 = M4 with a sibling, else M0 when M1(k=3) keeps the box,
else M5.

| view (n) | M0 | M0app | M2e | M3 spec \| M0 | **M3js_rr10 \| M0** | M4 \| M0 | P4 |
|---|---|---|---|---|---|---|---|
| scan (40) | 7.5 | 7.5 | 7.5 | 2.5 | **15.0** | 12.5 | 15.0 |
| oak1 (19) | 0.0 | 0.0 | 0.0 | 0.0 | **21.1** | 21.1 | 21.1 |
| oak2 (19) | 0.0 | 0.0 | 0.0 | 0.0 | 15.8 | 15.8 | 21.1 |
| rs (24) | 20.8 | 16.7 | 20.8 | 16.7 | 20.8 | 20.8 | 20.8 |
| all (102) | 7.8 | 6.9 | 7.8 | 4.9 | **17.6** | 16.7 | 18.6 |

Other widths (GT-B, all views): M3js does not hurt (40–100: 37.8 → 45.9; 100–250:
87.5 → 87.5; > 250: 71.4 → 71.4); M4 loses one 100–250 event; P4 loses one > 250.
Held out (D13+D33 only, GT-B, < 40 px): scan 12.0 → 24.0, oak1 0 → 20.0.
Median distance from the armed point to a small part stays ~14 part-widths for every
method (M0 355 px, M3js 287 px, P4 224 px): more hits, but misses are not closer.

**Ceiling** (any of the top 3, GT-B, < 40 px): M0 9.8 %, M3js / M4 20.6 %, M5cs
17.8 %; with every candidate counted, ~19–21 %. Why (`probe_rank.py`, 76 small
events with a sibling): the true part ranks first 10.5 % by sibling NCC, 2.6 % by
registered ΔE, 15.8 % combined (top-10: 27.6 %). **The bottleneck is generating
screw-like candidates**: at 15–20 px, board texture, capacitor tops and empty
standoffs match a screw template as well as a screw does. Conversely, when the
candidates are the real same-class parts in frame j, "largest change" picks the
removed one 67 % of the time, and skipping drawn siblings leaves it as the only one.

## 4. Per method

**M1 — no box when unsure** (area gate on the app's own blob, bands fitted leave-one-
desktop-out). All views, GT-A: k=3 withholds 43.3 % of boxes, precision of kept boxes
29 % → 47.2 %, **cuts 57.6 % of wrong boxes, withholds 9.1 % of good ones** (GT-B:
54.1 % / 10.2 %). Per view (GT-B, k=3): **scan 72.5 % / 5.9 %**, oak1 36.1 % / 0 %,
oak2 62.5 % / **35.7 %** (RAM, cooler, motherboard), rs 27.3 % / 0 %. The shipped
`configs/area_priors.yaml` looks better but is fitted in-sample on the same drafts.

**M2 — registration**: residual shift (ECC) median / p90 is 0.10–0.45 / 0.8–1.8 px;
hit rates and blob counts unchanged in every bucket — `diff_delta_e`'s ±1 px
tolerance already covers it. Dropped.

**M3 — sibling appearance**: a drawn sibling exists for 74 % of < 40 px events.
"NCC(j) − NCC(k)" as specified is ~0 (the template matches its own sibling in j; an
empty standoff in k still looks like a screw); fixed by skipping drawn siblings and
re-ranking by native ΔE (`M3js_rr10`).

**Also tried**: a sequence prior (the part removed at k+1 is the nearest same-class
part 71 % of the time, within two 96 %) hurt as an additive term (14.8 → 11.1 %);
M5 (class-scale centre-surround ΔE) has the best scan top-3 (25.6 %) but a weak top-1.

## 5. D13 / scan frames 37 and 38 (both in the set)

The annotator's boxes are reproduced exactly with the app's argument order and ROI
[627, 294, 1385, 1144] (`repro_d13.txt`): 6,332 px and 121 px.
- **37**: the 6,332 px box is on the black cable (M1 withholds it); the screw to add
  back is at (1303, 928); M3/M4 land on capacitor tops; every method misses.
- **38**: the 121 px box centres at ~(1101, 942), **a capacitor next to the cable,
  not the screw**. The screw that became a hole between 38 and 39 is at (918, 921),
  ~185 px away; the app's second blob (108 px) is on it. B2's GT for this event is a
  hole that was already empty (927, 363).

## 6. Runtime (30 OAK 12 MP events, decoded frames, median of 3; another suite was
running, so these are upper bounds)

| step | median | p90 |
|---|---|---|
| app diff today (full frame + blobs) | 0.164 s | 0.192 s |
| same ΔE on the ROI crop only | 0.129 s | 0.176 s |
| ECC registration | 0.050 s | 0.059 s |
| M3 as specified | 0.269 s | 0.630 s |
| M3js_rr10 | 0.138 s | 0.301 s |
| M4j_rr10 | 0.140 s | 0.312 s |
| M5 | 0.011 s | 0.024 s |
| M1 | ~0 | |

Diff + M3js: ~0.30 s median (on the line), ~0.49 s p90 (over).

## 7. Gate

| gate | result |
|---|---|
| < 40 px +20 points on scan and oak1 | **fails** — scan +7.5 (GT-B); oak1 +21.1 |
| no ≥ 100 px bucket loses > 3 points | M3js passes; M4 and P4 lose one event |
| OAK runtime ≤ 0.3 s | M3js on the line (p90 over); M4 over |
| M1: cut > 50 % wrong boxes, withhold < 15 % good | all views pass at k=3; per view only scan passes; oak2 withholds 36 % |

## 8. Recommendations

1. Do not wire M3 or M4.
2. Wire M1 for **scan only** (k=3, bands fitted per class): show no box instead of a
   likely-wrong one; re-measure on the annotator's own D13 labels once they exist.
3. Fix the harness before the next localisation experiment: position-resolved events
   (`probe_gt.py`), or better the annotator's confirmed labels; the app's diff
   argument order.
4. The lever is **candidate generation**: an L2 experiment training a small-part
   detector (YOLO26 / RF-DETR, already in the env) on the drafts with leave-one-
   machine-out validation; candidates = detections not yet drawn, ranked by native
   ΔE, sequence prior only as a tie-breaker. Needs its own runtime budget.
5. The heuristic line is a negative result: the difference map does not see most
   screws, and at 15–40 px sibling appearance cannot tell a screw from the board.
6. DINOv3 ViT-L/16 exists locally under `D:\DataSet\models\hf` (1.2 GB, downloaded
   by the user on 2026-09-18); it was not loaded. A screw is about one 16 px patch,
   so it would need ~2× upsampling and ~0.3–1 s per frame on the 5090 — over budget
   for OAK; possible as an offline upper bound with the user's consent.

Note: `frame_quality.csv` flags two nearly black frames (D13 scan 34, D24 scan 16) —
to be checked against what the app shows.
