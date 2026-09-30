# L3 — do larger detectors find more of the small parts?

Follows L2 with the same tiles (hard-linked, byte-identical labels), folds, classes,
GT-B events and evaluation code (`env.bind_l2`); re-scoring L2's YOLO26n detections
through this path reproduces L2's `det_metrics.csv` / `guess_events.csv` exactly.
Code: this folder. Outputs (~7 GB, mostly checkpoints): `experiments_out/l3_detector/`
(`compare.md`, `pairwise_*.md`, `decision_rule.txt` written before any RF-DETR
result, overlays). Written up by the controller from the worker's hand-back.

**Verdict: larger YOLOs do not help; RF-DETR does.** YOLO26s / m are within noise
of YOLO26n on the guess (top-1 on parts < 40 px, all views: n 34.3, s 32.4, m 32.4 %;
yolo26l therefore not trained). RF-DETR small, fine-tuned from its COCO checkpoint
with its own defaults for the same number of optimizer steps, raises screw recall
(hit@0.10, held out) scan 72.8 → 87.7, oak1 62.8 → 78.7, oak2 18.6 → 62.7, rs
5.9 → 34.7, the candidate ceiling 35 → 58 % (89 % at conf 0.01) and the guess on
parts < 40 px 34.3 → 51.0 % (+16.7, CI +6.9..+26.5). The gain is mostly oak2 (+36.8)
and rs (+25.0); on scan + oak1 motherboard screws it is 76 → 85 % (+8.8, CI
−11.8..+26.5, not significant). RF-DETR medium ties with small. With a 0.10 score
floor before L2's merge, RF-DETR small costs 0.027 s per scan frame and 0.053 s per
oak1 ROI frame — no slower than YOLO26n. **The limit has moved from recall to
ranking**: the removed part is a candidate in 66 / 114 small-class events and ΔE
ranks it first in 55.

## 1. Environment, downloads, safety (the user approved larger models on 2026-09-29)

- `tda_l3` = `conda create --clone tda` (42 s, no conda downloads). The annotator's
  `tda` env is untouched (`pip freeze` identical, 0 newer files).
- Caches (`PIP_CACHE_DIR`, `TORCH_HOME`, `HF_HOME`, `XDG_CACHE_HOME`,
  `YOLO_CONFIG_DIR`, `TMP`/`TEMP`) under `D:\DataSet\.cache\l3\`; threads capped at 4;
  one GPU job at a time. The DB copy was opened `mode=ro`; the raw drive only read.
- Weights downloaded to `D:\DataSet\models\weights\` (ledger `downloads.json`):
  `yolo26s.pt` 20.4 MB and `yolo26m.pt` 44.3 MB (github ultralytics/assets, AGPL-3.0);
  `rf-detr-small.pth` 386 MB and `rf-detr-medium.pth` 405 MB (storage.googleapis.com
  /rfdetr, MD5-checked, Apache-2.0); plus an UNPLANNED `yolo26n.pt` 5.5 MB fetched by
  ultralytics' AMP self-check at the first YOLO26s training despite `YOLO_OFFLINE=1`
  (byte-identical to the existing `models/ultralytics/weights/yolo26n.pt`; left in
  place). Not downloaded: `yolo26l.pt`, RF-DETR base.
- pip into `tda_l3` only (14 wheels, ~4.5 MB): pytorch-lightning 2.6.6, torchmetrics
  1.8.2 (downgraded from 1.9.0; rfdetr pins < 1.9), peft, accelerate, faster-coco-eval,
  torch-hungarian, aiohttp and its deps. The dry run of `rfdetr[train,loggers]` wanted
  `opencv-python-headless` (a second cv2, via roboflow) and ~60 logger packages, so the
  `train` requirements were installed minus roboflow (only used by a cloud upload) and
  `loggers` was skipped; torch / torchvision / numpy / opencv unchanged.
- C: checked: pip cache 0 files, `~/.cache` unchanged, `%APPDATA%\Ultralytics`
  unchanged; conda's env registry `~/.conda/environments.txt` gained the `tda_l3` line
  and two conda notice files were rewritten (conda itself).

## 2. Models and recipes (fixed a priori, one recipe per family for all folds)

- YOLO26s / m: L2's recipe verbatim (60 epochs, batch 32, imgsz 640, mosaic off last
  10, h/v flips, scale ±25 %, `last.pt`); 688–905 s (s) and 1095–1407 s (m) per fold.
- RF-DETR small / medium: RF-DETR defaults from the COCO detection checkpoints (AdamW,
  lr 1e-4, encoder 1.5e-4, wd 1e-4, EMA, bf16), effective batch 16, 30 epochs (= the
  YOLO optimizer steps at half the image passes), resolution 640, library-default
  train-time resize to 800² with random crops, `last_ema.pth`; 2146–3071 s per fold
  (CPU-bound on SciPy Hungarian matching at 4 threads).
- Inference: L2's tiling and merge unchanged; RF-DETR swaps the per-tile forward
  (fp16), its extra (num_classes + 1) head slot dropped. Evaluation uses L2's protocol
  (every box ≥ 0.001 into the merge).

## 3. Detector metrics (held-out desktops; AP50 / hit@0.10 / FP per frame)

| class | view | n | YOLO26n | YOLO26s | YOLO26m | RF-DETR small | RF-DETR medium |
|---|---|---|---|---|---|---|---|
| screw | scan | 805 | 49.8 / 72.8 / 1.9 | 54.7 / 75.2 / 1.2 | 53.5 / 74.2 / 1.1 | 61.1 / 87.7 / 4.4 | 60.9 / 86.7 / 3.9 |
| screw | oak1 | 427 | 64.8 / 62.8 / 0.8 | 70.8 / 68.6 / 0.3 | 63.8 / 55.7 / 0.2 | 78.1 / 78.7 / 1.1 | 81.4 / 83.1 / 0.9 |
| screw | oak2 | 263 | 11.6 / 18.6 / 1.9 | 11.4 / 13.3 / 0.6 | 5.9 / 11.0 / 0.6 | 47.0 / 62.7 / 0.9 | 50.5 / 55.5 / 0.8 |
| screw | rs | 337 | 5.9 / 5.9 / 0.5 | 10.0 / 13.4 / 0.3 | 9.4 / 11.6 / 0.1 | 26.1 / 34.7 / 0.7 | 25.1 / 39.2 / 1.2 |
| connector | scan | 895 | 11.0 / 27.5 / 1.7 | 11.3 / 28.5 / 1.0 | 14.9 / 23.6 / 0.9 | 26.8 / 62.6 / 7.1 | 25.6 / 60.1 / 5.7 |
| connector | oak1 | 561 | 10.5 / 32.6 / 2.0 | 3.1 / 13.4 / 1.3 | 5.5 / 15.3 / 0.7 | 9.9 / 47.8 / 4.9 | 14.2 / 48.0 / 4.5 |
| ram_latch | scan | 848 | 34.4 / 61.0 / 1.0 | 28.5 / 54.4 / 1.0 | 24.4 / 40.8 / 0.7 | 53.5 / 79.7 / 3.2 | 53.5 / 89.0 / 2.4 |

Screw hit per held-out desktop (D13 / D24 / D33): scan n 68.9/69.4/79.4, RF-DETR small
90.8/75.1/95.5; oak1 n 7.5/66.7/81.8, small 36.2/82.6/92.6, medium 51.2/84.7/94.6;
oak2 n 7.9/25.3, small 52.5/69.1. RF-DETR's FP/frame is higher (DETR calibration and
unlabeled parts); since the guess ranks by ΔE the extra boxes cost ranking, not recall.
Paired bootstrap, RF-DETR small vs YOLO26n screw hit: scan +14.9 (CI +11.8..+18.3),
oak1 +15.9 (+12.1..+19.9) (frames within a desktop are correlated, so optimistic).

## 4. The guess on GT-B events (L2small policy, held-out folds)

Parts < 40 px, top-1 / top-3 %:

| view | n | M0app | YOLO26n | s | m | RF-DETR small | RF-DETR medium |
|---|---|---|---|---|---|---|---|
| scan | 40 | 7.5 / 12.5 | 50.0 / 52.5 | 47.5 / 52.5 | 50.0 / 57.5 | 55.0 / 65.0 | 52.5 / 67.5 |
| oak1 | 19 | 0 / 10.5 | 36.8 / 42.1 | 47.4 / 52.6 | 36.8 / 36.8 | 47.4 / 57.9 | 73.7 / 78.9 |
| oak2 | 19 | 0 / 0 | 15.8 / 21.1 | 5.3 / 5.3 | 10.5 / 10.5 | 52.6 / 52.6 | 42.1 / 47.4 |
| rs | 24 | 16.7 / 16.7 | 20.8 / 20.8 | 16.7 / 16.7 | 16.7 / 20.8 | 45.8 / 58.3 | 37.5 / 54.2 |
| all | 102 | 6.9 / 10.8 | 34.3 / 37.3 | 32.4 / 35.3 | 32.4 / 36.3 | 51.0 / 59.8 | 51.0 / 62.7 |

All 184 events, all sizes, top-1: M0app 32.1, n 48.9, s 47.3, m 47.3, RF-DETR small
59.8, medium 59.2; no ≥ 100 px bucket changes. L2's gate (< 40 px, over M0app): RF-DETR
small scan +47.5 (CI +32.5..+62.5), oak1 +47.4 (+26.3..+68.4), oak2 +52.6, rs +29.2.
Paired vs YOLO26n (< 40 px): s −2.0, m −2.0, RF-DETR small +16.7 (+6.9..+26.5, 23 wins
/ 6 losses), medium +16.7; scan + oak1 motherboard screws (34): small +8.8 (−11.8..
+26.5), medium +14.7 (−2.9..+32.4). Medium − small: oak1 +26.3 (0..+52.6), oak2 −10.5,
rs −8.3, all 0.0.

By what was removed, top-1 % (ceiling): scan motherboard screw (21) n 85.7 (90.5),
RF-DETR small 90.5 (100); oak1 motherboard screw (13) n 61.5 (76.9), small 76.9 (84.6),
medium 100; oak2 motherboard screw (8) n 25.0, small 87.5; rs motherboard screw (7)
n 0, small 57.1; connectors stay 15–29 % even when in the pool (unplugging moves the
cable, raising ΔE in neighbouring boxes); cooler screws remain captive loosenings.
Candidate ceiling (small-class events, conf ≥ 0.10 after the skip; ranked first when
present): n 35.1 % (35/40), RF-DETR small 57.9 % (55/66), medium 61.4 % (54/70).
L2's two losses vs M0app (D33 scan k = 35, 36) are hits with RF-DETR small.

## 5. Runtime (RTX 5090, fp16, idle GPU, L2's 30 events per view, median of 3 s)

| model | protocol | scan det / guess | oak1 ROI det / guess | oak1 full 12 MP | precompute per frame scan / oak1 |
|---|---|---|---|---|---|
| YOLO26n | L2 (≥ 0.001) | 0.031 / 0.033 | 0.077 / 0.078 | 0.131 | 0.060 / 0.120 |
| YOLO26n | floor 0.10 | 0.030 / 0.031 | 0.057 / 0.059 | 0.098 | 0.053 / 0.103 |
| RF-DETR small | L2 | 0.052 / 0.055 | 0.241 / 0.247 | 0.801 | 0.083 / 0.291 |
| **RF-DETR small** | **floor 0.10** | **0.025 / 0.027** | **0.050 / 0.053** | **0.094** | **0.060 / 0.103** |
| RF-DETR medium | floor 0.10 | 0.024 / 0.026 | 0.050 / 0.053 | 0.094 | 0.061 / 0.106 |

Under L2's protocol the cost is L2's pure-Python merge of 300 queries per tile; the
floor (dropping boxes below the guess's own 0.10 threshold before the merge) changes
0 of 105 scan/oak1 events and 4 oak2/rs events (3 wins, 1 loss). GPU memory with three
fold models loaded: 2.5 GB RF-DETR, 0.6 GB YOLO26n.

## 6. Recommendation

1. **Ship RF-DETR small for scan and oak1 screws** (the a-priori rule selects it:
   best pooled scan + oak1 screw hit 84.6 % vs 69.3 % YOLO26n / 72.9 % YOLO26s; no
   worse on the guess; within budget; medium ties → small; Apache-2.0 vs AGPL-3.0).
   Keep L2's settings (skip, conf ≥ 0.10, no sequence prior, small classes only) and
   add the 0.10 floor before the merge.
2. **Enable oak2 screws** (top-1 < 40 px 0 → 52.6 %, CI +31.6..+73.7; motherboard
   screws 87.5 %, n = 8; only D13/D24 have oak2 drafts, so each fold learned from one
   desktop).
3. rs screws promising (+29.2, CI +12.5..+50) but not ready (hit 34.7 % at 6–10 px).
4. No model makes connectors worth enabling (recall up to 62.6 % but 5.7–7 FP/frame
   and 15–29 % guess): the problem is ΔE ranking.
5. Next lever: ranking and labels, not model size (top-3 display, a ranking feature
   beyond mean ΔE, cleaner labels); re-measure on the annotator's own labels.

## 7. Production candidate

RF-DETR small trained on all 4 873 tiles (no hold-out), fold recipe unchanged
(3 294 s): `D:\DataSet\models\weights\tda_det_rfdetr_small_all.pt` (128 438 745 bytes,
sha256 bd0698e5…722b) with `tda_det_rfdetr_small_all.json` (tiles hash
4586bbca…38a9, classes, recipe, tiling/merge/guess rule, init checkpoint + sha256,
source commit c1fcfe2, date 2026-09-30, licence; no held-out score of its own). Smoke:
loads through `RFDetector`, 0.023 s per scan frame, 0.044 s per oak1 ROI frame at the
0.10 floor. Weights are not committed. Each RF-DETR run keeps a 513 MB `last.ckpt`
under `experiments_out/l3_detector/*/runs/*` (~4 GB reclaimable).
