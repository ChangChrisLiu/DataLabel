# Task B2 BASELINE — the per-view diff as it ships today

Produced by `experiments/plan_b_probe/diff_eval.py` before any production change,
so that a later table can be compared against something that was written down
first. `experiments_out/` is git-ignored, hence this copy lives in the tree.

```
D:\Anaconda\envs\tda\python.exe -m experiments.plan_b_probe.diff_eval \
    --db D:/DataSet/.cache/tmp/planb_ro.sqlite \
    --views scan,oak1,oak2,rs --methods baseline --sam --tag baseline
```

Ground truth: the old Label Studio drafts (`ls:*`) on desktops 13, 24, 33 —
noisy, visible-pixels-only. 201 of 372 removal events resolve to exactly one
draft, the same 201 the E2 probe used. The frame being annotated is `step - 1`
(the part is still there); the diff compares it with `step`.

`baseline` = what the app arms today: `diff_delta_e` → `diff_blobs`, box = the
top blob's box, point = that blob's **centroid**
(`tda/ui/app_assist.py:begin_add_shape`). SAM IoU is the IoU of
`candidates[0]` for a `box + point` prompt on a ROI crop (≤1024 px long side,
`multimask=True`) against the draft mask.

## Per view

| view | method | n | point-in-part top-1 | top-3 | box IoU p25/med/p75 | SAM IoU p25/med/p75 |
|---|---|---|---|---|---|---|
| scan | baseline | 65 | 21.5% | 21.5% | 0.000 / **0.002** / 0.125 | 0.000 / **0.001** / 0.051 (n=64) |
| oak1 | baseline | 51 | 21.6% | 25.5% | 0.000 / **0.000** / 0.157 | 0.000 / **0.000** / 0.016 (n=49) |
| oak2 | baseline | 43 | 27.9% | 32.6% | 0.000 / **0.018** / 0.249 | 0.000 / **0.000** / 0.180 (n=36) |
| rs | baseline | 42 | 40.5% | 40.5% | 0.000 / **0.073** / 0.480 | 0.000 / **0.073** / 0.624 (n=34) |

The `point-in-part` column reproduces the E2 M4 table
(`experiments_out/plan_b_probe/transfer/report.md` §7) **exactly** — 21.5 / 21.6
/ 27.9 / 40.5 % with n = 65 / 51 / 43 / 42 — which is the check that the event
resolution and the ground truth are the same population.

## Per class group

| group | method | n | point-in-part | box IoU med | SAM IoU med |
|---|---|---|---|---|---|
| tall | baseline | 35 | 77.1% | 0.372 | 0.624 |
| flat | baseline | 166 | 16.3% | 0.001 | 0.000 |

(E2 reports the same 77.1 % / 16.3 % split.)

## Runtime, median seconds per event

| view | method | n | frame load | ΔE map | propose | SAM |
|---|---|---|---|---|---|---|
| scan | baseline | 65 | 0.028 | 0.174 | 0.013 | 0.040 |
| oak1 | baseline | 51 | 0.057 | 0.155 | 0.038 | 0.041 |
| oak2 | baseline | 43 | 0.050 | 0.058 | 0.037 | 0.041 |
| rs | baseline | 42 | 0.023 | 0.030 | 0.003 | 0.037 |

SAM: 183 calls, 8.4 s total, 46 ms each (RTX 5090, shared). 18 of the 201 events
produce no armed prompt at all (no blob, or the top blob's centroid falls
outside the ROI crop), which is why the SAM column's `n` is below the event `n`.
