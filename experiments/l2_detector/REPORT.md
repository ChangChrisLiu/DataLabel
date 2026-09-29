# L2 — can a detector trained on the old drafts generate the small-part candidates?

Code: this folder. Tiles, weights, detections, per-event CSVs, tables and overlays:
`D:\DataSet\experiments_out\l2_detector\` (git-ignored). Read-only: a DB copy made
from `u2b3_copy.sqlite`; the raw drive only read; nothing downloaded; all caches on
D:. Events and ground truth are L1's: B2's 201 events scored against the
position-resolved **GT-B** (184 events, recomputed with `probe_gt.check`, identical
to L1's `gt_check.csv`); `M0app` (the app's own call) matches L1 on 184 / 184.
"Hit" = the armed point lies inside the removed part's mask. Written up by the
controller from the worker's hand-back (the worker's sandbox refused .md files).

**Verdict: yes for motherboard screws on scan and oak1; the gate passes, with a
thin oak1 margin.** Parts < 40 px: scan 7.5 → 50.0 % (+42.5, 95 % CI +25 to +60,
n = 40); oak1 0 → 36.8 % (+36.8, CI +16 to +58, n = 19). No ≥ 100 px bucket
changes. 0.04 s (scan) / 0.09 s (oak1) per frame change, or precomputable. The
gain is narrow: motherboard screws hit 86 % on scan (21 events) and 62 % on oak1
(13); connectors 0–29 %; cooler screws 0 % (they are captive and stay on the
cooler); oak2 and rs gain little. **The limit is recall, not ranking**: of 114
small-class events the removed part was a candidate in 40, and native ΔE ranked it
first in 35 of them.

## 1. Data

All `labelstudio` keyframes (11 191 drafts); bbox and area from the COCO RLE; class
from the `instance` table (L1's `siblings.py` join). Detector classes: `screw`,
`connector`, `ram_latch`, `psu_latch`, `cpu_socket_lever`, `drive_latch`, plus
context `ram_module`, `cpu`. Counts per class × view × desktop: `label_counts.md`.

| view | desktops with drafts | screw | connector | ram_latch | screw width p50 |
|---|---|---|---|---|---|
| scan | D2, D8, D10, D11, D13, D15, D16, D18, D19, D24, D33, D64, D65, D66 | 1 500 | 1 877 | 1 509 | 18 px |
| oak1 | D13, D24, D33 | 427 | 561 | 338 | 38 px |
| oak2 | D13, D24 | 263 | 329 | 206 | 23 px |
| rs | D13, D24, D33 | 337 | 379 | 399 | 7 px |

Training uses desktop-views with ≥ 20 screws + connectors, which drops the clearly
partial scan drafts of D2, D11, D15, D16, D64, D65 (their unlabeled screws would be
taught as background). Tiling (`tiles.py`, same for training and inference): crop
to the view's ROI (drafts' union box + 10 %), work scale 1.0 for scan/oak1/oak2 and
2.0 for rs (its screws are 6–10 px), 640² tiles at stride 512, grey padding, a label
kept if ≥ half of its box is in the tile. 4 873 tiles (1 302 empty). The drafts are
noisy: unlabeled SATA connectors and RAM latches, an rs "connector" covering the
PSU, partial RAM boxes on oak1.

## 2. Split, model, weights

Leave-one-desktop-out: fold `holdN` trains on every other desktop's tiles in all
views; every event is scored by the fold that held its desktop out (oak1 folds
train on two event desktops; oak2 folds on one). **YOLO26n fine-tuned from COCO
`yolo26n.pt`** (the only YOLO26 detection weight on disk), one a-priori recipe for
all folds: 60 epochs, batch 32, imgsz 640, mosaic off for the last 10, h/v flips,
scale ±25 %, 4 loader workers / 4 threads, `last.pt` used. 694–898 s per fold.
hold13 crashed once at close-mosaic (Windows re-pickling a ~4.9 GB RAM cache) and
was retrained with `cache=False`, as was hold33 (same pixels, same recipe).

On disk: `yolo26n.pt`, `yolo26n-seg.pt`, `rf-detr-seg-preview.pt` (loads offline),
SAM 2.1, SAM 3, DINOv3. Missing: YOLO26 s/m/l/x detection weights (AGPL-3.0) and
RF-DETR detection checkpoints (Apache-2.0). **RF-DETR was not trained**: rfdetr
1.10.1's trainer needs PyTorch Lightning (`pip install "rfdetr[train,loggers]"`),
which is a download; `train_rfdetr.py` and `build_rfdetr.py` exist but were not run.

## 3. Detector metrics (held-out desktops, drafts as GT; `det_metrics.md`)

AP50 at IoU 0.5; hit = a same-class box centre inside the draft box at conf ≥ 0.10;
FP/frame = unmatched detections in the ROI (inflated by missing labels).

| class | view | n | AP50 | R50 @0.10 | hit @0.10 | FP/frame |
|---|---|---|---|---|---|---|
| screw | scan | 805 | 49.8 | 64.5 | **72.8** | 1.9 |
| screw | oak1 | 427 | 64.8 | 60.9 | **62.8** | 0.8 |
| screw | oak2 | 263 | 11.6 | 16.0 | 18.6 | 1.9 |
| screw | rs | 337 | 5.9 | 5.9 | 5.9 | 0.5 |
| connector | scan | 895 | 11.0 | 16.2 | 27.5 | 1.7 |
| connector | oak1 | 561 | 10.5 | 22.3 | 32.6 | 2.0 |
| ram_latch | scan | 848 | 34.4 | 47.2 | 61.0 | 1.0 |
| ram_latch | oak1 | 338 | 25.0 | 30.8 | 41.4 | 0.4 |
| cpu_socket_lever | scan | 65 | 87.7 | 84.6 | 84.6 | 0.0 |
| psu_latch | scan (D13) | 32 | 100 | 96.9 | 96.9 | 0.0 |

Screw hit per held-out desktop — scan: D13 68.9, D24 69.4, D33 79.4; oak1: D13
**7.5**, D24 66.7, D33 81.8; oak2: D13 7.9, D24 25.3.

## 4. The guess on the events (GT-B; `guess_tables.md`, `guess_events.csv`)

On frame j (step to undo k = j + 1): candidates = the fold's detections of the
step's class in frame j (conf ≥ 0.10); skip a candidate whose centre lies within
±2 px of, or whose box has IoU ≥ 0.3 with, a same-class draft on frame k (already
drawn); rank by native mean ΔE j → k in the box; sequence prior as a tie-break
within 0.5 ΔE; arm the box centre; no candidate → M0app. `L2small` (declared
primary before any result) uses the detector for the six small classes; `L2all`
also for `ram_module` and `cpu`.

Point-in-part top-1 %:

| view | method | < 40 | 40–100 | 100–250 | > 250 | all |
|---|---|---|---|---|---|---|
| scan | M0app | 7.5 (40) | 57.1 (7) | 100 (5) | 83.3 (6) | 29.3 |
| scan | L1 M3js_rr10 | 15.0 | 71.4 | 100 | 66.7 | 34.5 |
| scan | **L2small** | **50.0** | 57.1 | 100 | 83.3 | 58.6 |
| oak1 | M0app | 0.0 (19) | 0.0 (14) | 75.0 (4) | 80.0 (10) | 23.4 |
| oak1 | L1 M3js_rr10 | 21.1 | 21.4 | 50.0 | 80.0 | 36.2 |
| oak1 | **L2small** | **36.8** | 21.4 | 75.0 | 80.0 | 44.7 |
| oak2 | M0app | 0.0 (19) | 22.2 | 100 | 80.0 | 34.1 |
| oak2 | L2small | 15.8 | 22.2 | 100 | 80.0 | 41.5 |
| rs | M0app | 16.7 (24) | 100 | 85.7 | – | 44.7 |
| rs | L2small | 20.8 | 100 | 85.7 | – | 47.4 |
| all | M0app | 6.9 (102) | 35.1 | 91.7 | 81.0 | 32.1 |
| all | **L2small** | **34.3** | 43.2 | 91.7 | 81.0 | 48.9 |
| all | L2all | 34.3 | 59.5 | 91.7 | 81.0 | 52.2 |

Top-3, < 40 px: scan 52.5, oak1 42.1, oak2 21.1, rs 20.8 (M0app 12.5 / 10.5 / 0 /
16.7). Ceiling (removed part among the candidates, conf ≥ 0.10; 0.01 in brackets):
scan 53.8 (56.4), oak1 42.1 (52.6), oak2 21.1 (31.6), rs 10.5 (31.6). The skip never
removed the true part. When the true part was a candidate it ranked 1st in 35 of 40
and never below 3rd; no top-1 miss lands within half a part width of the part.

By what was removed (top-1 %, M0app → L2small, ceiling): scan motherboard screw
(21) 14.3 → **85.7**, 90.5; scan connector (13) 0 → 15.4; scan cooler screw (5)
0 → 0; oak1 motherboard screw (13) 0 → **61.5**, 76.9; oak1 connector (14) 0 → 14.3;
oak2 motherboard screw (8) 0 → 25.0; oak2 connector 0 → 0; rs small (19) 5.3 → 10.5.

Per held-out desktop, < 40 px (M0app → L2small): scan D13 14 → 100 (7), D24 0 →
46.7 (15), D33 11 → 33.3 (18); oak1 D13 0 → 0 (2), D24 0 → 44.4 (9), D33 0 → 37.5 (8).
L2small beats M0app on 33 events and loses 2 (D33 scan k = 35, 36: a wrong screw
found, the removed one missed); against L1's M3js it wins 31 and loses 12 (6 of those
used M0app by policy or for lack of a candidate) — the approaches partly complement.

Sensitivity (all views, < 40 px, L2small = 34.3): no sequence prior 34.3; no skip
28.4; conf 0.05 32.4; conf 0.25 35.3. Keep the skip, drop the prior. Post hoc, not
validated: armed boxes whose top candidate has ΔE < 3 are mostly wrong (withholding
them would drop 18 of 51 wrong boxes and 2 of 35 hits).

## 5. D13 / scan frames 37 and 38

Frame 37 (k = 38): 2 candidates after the skip; the screw at (1305, 930) has ΔE
6.65 vs 1.89 — top-1 is the screw L1 said every method missed; M0app arms
(756, 774). Frame 38 (k = 39): 3 candidates; the screw at (918, 922) has ΔE 7.42 vs
1.82 / 1.67 — top-1 is the screw that became a hole; M0app arms the capacitor at
(1105, 948). Cooler screws (`D33_scan_k05_miss.jpg`): the "removed" draft is a
captive screw still visible at k — the event is a loosening, not a removal.

## 6. Runtime (RTX 5090, fp16, 30 events per view, median of 3; `timing.md`)

| view | ROI (median) | tiles | detector median / p90 | ΔE ranking | guess median / p90 | today's diff |
|---|---|---|---|---|---|---|
| scan 1600² | 1362 × 1026 | 6 | 0.038 / 0.047 s | 0.001 s | **0.039 / 0.048 s** | 0.195 s |
| oak1 12 MP | 2170 × 2497 | 20 | 0.086 / 0.100 s | 0.001 s | **0.089 / 0.104 s** | 0.215 s |
| oak2 | 1002 × 1106 | 4 | 0.033 / 0.056 s | 0.001 s | 0.042 / 0.059 s | 0.134 s |
| rs (2×) | 630 × 399 | 6 | 0.030 / 0.035 s | < 0.001 s | 0.031 / 0.035 s | 0.036 s |

Full 12 MP OAK frame without a ROI: 0.131 / 0.156 s. Precomputable: a background
decode + detect pass costs 0.10 s per scan frame and 0.16 s per oak1 frame (4.5–7 s
for a 45-step view), after which a frame change pays only the ~1 ms ranking.

## 7. Gate

| gate | result |
|---|---|
| < 40 px +20 points over M0app on scan and oak1, held out | **passes** — scan +42.5 (CI +25.0 to +60.0), oak1 +36.8 (CI +15.8 to +57.9; oak1 is mostly D24 + D33 and the detector fails on D13 oak1) |
| no ≥ 100 px bucket loses > 3 points | **passes** (+0.0 everywhere; large classes fall back to M0app) |
| runtime ≤ 0.3 s or precomputable | **passes** |

## 8. Recommendations (nothing wired)

1. Worth wiring for **scan and oak1, screws only**, gated by class; precompute
   detections per frame in the background; keep the skip, drop the prior. Do not
   expect it to find connectors or cooler screws.
2. Re-measure on the annotator's own labels once they exist; validate a ΔE-confidence
   gate (§4) before using it.
3. The next lever is recall: more and cleaner screw and connector labels (oak2 has
   one training desktop per fold), and a larger model — `yolo26s/m` needs a weight
   download, RF-DETR needs the `rfdetr[train]` install; both are the user's call.
4. rs stays out (screw hit 6 % even at 2×).

Overlays (`experiments_out/l2_detector/overlays/`): `D13_scan_k38_hit.jpg` (frame
37), `D13_scan_k39_hit.jpg` (frame 38), `D24_scan_k36_hit.jpg`, `D24_oak1_k38_hit.jpg`,
`D24_oak1_k12_hit.jpg`, `D33_oak1_k39_miss.jpg`, `D33_scan_k05_miss.jpg` (captive
cooler screw), `D33_scan_k35_miss.jpg`, `D24_scan_k18_miss.jpg`, `D24_oak2_k40_miss.jpg`.
