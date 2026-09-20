# TDA Plan B (wave 1–2) — Per-View Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Each task is one worker in its own git worktree, followed by an independent review. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make annotating *every* view from scratch fast and safe — per-view pose segments, usable speed on 12 MP OAK frames, better per-view change localisation, reuse of old Label Studio drafts — and add the constraint editor and the full VLM generator.

**Architecture:** No cross-view transfer (spec v1.5 §1.3, §2.5). Identity comes from the step table and is shared by the four views; shapes, z-order, pose segments and occluders are per view and are produced with the same reverse-order flow that already works on `scan`. Everything new in `tda/core` stays Qt-free and is driven by the existing `AnnotationSession`; the window's `ACTIONS` table remains the only key map.

**Tech Stack:** Python 3.11 (`D:\Anaconda\envs\tda`), PySide6 6.11, SQLite (WAL), numpy/opencv, torch cu128 + SAM 2.1, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-teardown-annotator-design.md` (v1.5). Evidence: `experiments_out/plan_b_probe/camera_moves/report.md`, `experiments_out/plan_b_probe/transfer/report.md`.

## Global Constraints

- Never open `D:\DataSet\annotations\tda.sqlite` from worker or reviewer code. Real-data probes run on a copy the controller provides under `D:\DataSet\.cache\tmp\`, opened read-only unless the task says otherwise; delete your own copies. Only the controller runs reviewed CLI commands on the real DB (backup first).
- `F:` is read-only raw data. Nothing new on `C:`. Temp files under `D:\DataSet\.cache\tmp`.
- Tests: `QT_QPA_PLATFORM=offscreen D:\Anaconda\envs\tda\python.exe -m pytest tests -o addopts=` from the worktree root; check the **exit code**; the suite (2106 tests at plan time) stays green. TDD: failing test first.
- Invariants that no task may weaken: (I1) a VERIFIED compiled row changes only through the conflict queue; (I2) compiled rows of unverified frames are a cache guarded by per-frame input digests; (I3) state events are derived from actions at read time; (I4) one transaction per user action; (I5) no Qt in `tda/core`; (I6) instance keys never change. **Never lose an uncommitted edit**: anything that navigates or reloads goes through the window's `leave_frame()` gate / the session's `SessionRefusal`.
- Masks are stored in full-frame native coordinates (spec §2.4). An ROI is a window for display, diffing and inference, never a coordinate system for stored geometry.
- `ls:*` instances are draft material (`tda.core.model.is_provisional`): never need geometry, never compiled, never exported.
- User-facing strings in the app are English + the existing Simplified-Chinese task-card phrases; `docs/annotation_guide.md` is Simplified Chinese and has a prose-line cap test (212) — new guide text must fit by tightening, not by raising the cap. The shortcut table in the guide is generated from `tda/ui/app_actions.py`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Workers do not push and do not touch `main`.
- Every worker writes `task-<id>-report.md` in its worktree root (uncommitted): what changed, files, measured numbers, full-suite count + exit code, doubts. Literal reporting: "confirmed by eye" only for an image actually opened.

## Review Focus

1. **A pose break inserted into a segment that already holds shapes, z-order, pair overrides, occluders and VERIFIED frames** — the annotator expects nothing to disappear silently: rows keyed by `pose_segment` must be renumbered in the same transaction, verified frames in the affected range must go to the re-check queue (→ conflict queue), never be rewritten.
2. **`load-index` / `import-logs --force` re-running `split_pose_segments` after manual per-view breaks exist** — breaks must survive; a `reorient` step and a manual break at the same step must not create an empty segment.
3. **12 MP frames with 40+ instances** — commit, Space and frame change must stay inside the budgets *with the sweeper ON and at the late steps* (the F1 lesson: the shipped configuration, not the convenient one).
4. **An adopted Label Studio draft of the wrong size / wrong view / already-adopted** — must be refused with a status-bar reason, never written; adopting must go through the normal editing layer so Esc/undo/sidecar all apply.
5. **A manual constraint edge that creates a cycle, duplicates a rule edge, or points at a deleted / provisional instance** — refused at edit time with the reason; `graph_version` re-stamped only on Apply; rule edges survive `constraints` re-runs unchanged and manual edges are never dropped by them.

---

## File structure (new / touched)

| File | Responsibility |
|---|---|
| `tda/core/schema.sql`, `tda/core/dbconn.py`, `tda/core/db.py` (`SCHEMA_VERSION = 4`) | new table `pose_break`; migration v3→v4 |
| `tda/core/pose_breaks.py` (new) | pure logic: boundaries per view, re-cut plan, renumbering map |
| `tda/core/db_pose.py` | `pose_breaks()`, `add_pose_break()`, `remove_pose_break()`, `apply_recut()` (transactional renumbering) |
| `tda/pipeline.py` | `split_pose_segments` uses reorient ∪ accepted breaks per view |
| `tda/cli_pose.py` (new) + `tda/cli.py` | `pose-breaks list|import|accept|reject` |
| `tda/ui/app_pose.py` (new), `tda/ui/app_actions.py` | "Split pose segment here" action, proposal bar |
| `tda/core/diffmap.py`, `tda/core/diff_split.py` (new), `tda/ui/app_diff.py` | blob splitting, box prompts |
| `experiments/plan_b_probe/diff_eval.py` (new) | offline metric harness on LS polygons |
| `tda/core/compiler*.py`, `tda/core/truth*.py`, `tda/core/cache_roi*.py`, `tda/ui/session*.py`, `scripts/mvp_smoke.py` | 12 MP performance, OAK ROI proposal |
| `tda/core/ls_adopt.py` (new), `tda/ui/app_adopt.py` (new) | adopt a draft into the editing layer |
| `tda/ui/panels/relations.py` (new), `tda/ui/steps_relations.py` (new), `tda/core/graph_edit.py` (new) | constraint editor |
| `tda/core/export/vlm*.py` | full VLM task set |

---

### Task B1: Per-view pose breaks

**Why:** pose segments are cut only at `reorient` steps and are identical in all four views. The camera audit found 6 within-sequence camera moves (oak1 D2 s5→6, D4 s6→7, D29 s46→47, D32 s38→39, D36 s18→19 ≈190 px; oak2 D1 s33→34) and a chassis rotation that is typed `compound` (scan D63 s31→32). Shapes cannot carry across those frames.

**Files:** Create `tda/core/pose_breaks.py`, `tda/cli_pose.py`, `tda/ui/app_pose.py`, `tests/test_pose_breaks.py`, `tests/test_cli_pose.py`, `tests/test_app_pose.py`. Modify `tda/core/schema.sql`, `tda/core/dbconn.py` (migration), `tda/core/db.py` (`SCHEMA_VERSION = 4`), `tda/core/db_pose.py`, `tda/pipeline.py:318-420`, `tda/cli.py`, `tda/ui/app_actions.py`, `docs/annotation_guide.md`.

**Interfaces (Produces):**

```sql
CREATE TABLE IF NOT EXISTS pose_break (
    desktop INTEGER NOT NULL,
    view    TEXT    NOT NULL,
    step    INTEGER NOT NULL,          -- the new segment STARTS at this step
    status  TEXT    NOT NULL DEFAULT 'accepted',   -- proposed | accepted | rejected
    kind    TEXT,                      -- camera | chassis | manual
    magnitude_px REAL,
    source  TEXT NOT NULL,             -- 'manual:<annotator>' | 'audit:<file>'
    note    TEXT,
    PRIMARY KEY (desktop, view, step)
);
```

```python
# tda/core/pose_breaks.py  (pure, no Db)
def boundaries(n_steps: int, reorients: Iterable[int], accepted: Iterable[int]) -> list[int]: ...
    # sorted unique starts > 1 and <= n_steps; reorient ∪ accepted
def recut_plan(old: list[tuple[int, int, int]], new_bounds: list[int], n_steps: int) -> RecutPlan: ...
    # old = [(seg, start, end)]; RecutPlan.ranges = [(seg, start, end)],
    # RecutPlan.renumber = {old_seg: new_seg_of_its_FIRST_step}, RecutPlan.split = {old_seg: [new_seg, ...]}
# tda/core/db_pose.py
def pose_breaks(self, desktop: int, view: str | None = None, status: str | None = None) -> list[dict]: ...
def add_pose_break(self, desktop: int, view: str, step: int, *, status: str, kind: str | None,
                   magnitude_px: float | None, source: str, note: str = "") -> None: ...
def set_pose_break_status(self, desktop: int, view: str, step: int, status: str) -> None: ...
# tda/pipeline.py
def split_pose_segments(db: Db, desktop: int) -> dict[str, int]:   # signature unchanged
```

**Semantics of a re-cut (the heart of the task):** inside ONE transaction, for one (desktop, view): compute the new ranges; every row keyed by `pose_segment` — `shape_keyframe`, `zorder`, `pair_override`, `occluder_mask`, `pose_segment` itself (ROI, bench ROI, corners) — is moved to the new segment that contains **its own step** (`anchor_step` for keyframes; an occluder/override's frame step) or, for per-segment rows (`zorder`, `pair_override`, ROI), **copied** to every new segment the old one was split into (the layer order and the chassis ROI are still the best first guess on the other side of a small camera nudge; the annotator re-confirms the ROI). A keyframe whose coverage `(previous anchor, anchor]` straddles the new boundary stays with its anchor's segment; the frames on the other side become `missing_shape` — that is the intended "redraw in the new pose" signal, unless the annotator chooses **"carry shapes across"** (checkbox in the split dialog, default ON for `magnitude_px < 25`, OFF otherwise), which duplicates each straddling keyframe into the earlier segment with `anchor_step = boundary - 1` and `source = 'carried'`. VERIFIED frames in the two affected segments are added to the re-check queue (`db.add_rechecks(desktop, view, steps)`); nothing verified is rewritten (I1). The op is logged (`log_op(..., "pose_recut", payload)`), and one Ctrl+Z is **not** offered for it (it is a structural edit like S1 Apply): removing the break (`set_pose_break_status(..., 'rejected')` + re-cut) is the undo, and merges the two segments back (rows of the later one re-keyed; duplicate `carried` keyframes deleted).

- [ ] **Step 1 — pure logic, failing tests first** (`tests/test_pose_breaks.py`): `boundaries(42, [16], [16, 19])  == [16, 19]`; `boundaries(42, [], [1, 43]) == []`; `recut_plan([(1,1,42)], [19], 42).ranges == [(1,1,18),(2,19,42)]`; inserting `[10]` into `[(1,1,15),(2,16,42)]` gives `[(1,1,9),(2,10,15),(3,16,42)]` with `renumber == {1:1, 2:3}` and `split == {1:[1,2]}`; removing a boundary merges and renumbers down.
- [ ] **Step 2 — schema v4 + migration**: v3 DB opens, gains the empty table, `meta.schema_version == 4`; a v4 DB opened by v3 code is refused by the existing version guard (test both directions with a temp DB).
- [ ] **Step 3 — transactional re-cut in `db_pose.apply_recut`**, tests on a temp DB built with `tests/truth_scenes.py`: keyframes/zorder/pair overrides/occluders land in the right segments; per-segment rows are copied; `carry=True` duplicates straddling keyframes; a VERIFIED frame in range is queued for re-check and its compiled row is byte-identical afterwards; a failure in the middle (monkeypatch one UPDATE to raise) leaves every table unchanged.
- [ ] **Step 4 — `split_pose_segments`** = reorient ∪ accepted breaks **per view**; idempotent; survives `load-index` and `import-logs --force` (call both, assert breaks and renumbered rows unchanged); a reorient and a manual break at the same step give one boundary.
- [ ] **Step 5 — CLI** `python -m tda.cli pose-breaks list [--desktop N]`, `import <events.csv> [--min-px 2] [--dry-run]` (reads the audit's `events.csv`: rows with `kind in (camera, chassis)` and `verdict in (confirmed, chassis_manual)` become `status='proposed'`, `source='audit:events.csv'`; the break step is `step_to`; chassis events are proposed for **all four views**, camera events for their own view only; never overwrites an existing row), `accept|reject --desktop N --view V --step S`. Backup before any write, like `infer-relations`. Dry-run prints what would be written.
- [ ] **Step 6 — UI** (`tda/ui/app_pose.py`, one new entry in `ACTIONS`: `Ctrl+Shift+B` "Split pose segment at this frame"): goes through `leave_frame()`; dialog shows the two frames (this and the previous one) side by side, the number of shapes that would straddle, the `carry shapes across` checkbox, OK/Cancel; after OK the session reloads **on the same frame**. When the current frame has a `proposed` break, a non-modal bar (same widget family as the restore bar) says `possible camera move before step 19 (≈190 px) — Tab to compare — Accept / Reject`; Accept opens the same dialog. Timeline draws a thin vertical mark at every accepted break of the current view.
- [ ] **Step 7 — guide**: one short section in `docs/annotation_guide.md` (fit the cap), shortcut table regenerated.
- [ ] **Step 8 — real-copy acceptance** (controller provides `.cache/tmp/b1_copy.sqlite`): `pose-breaks import experiments_out/plan_b_probe/camera_moves/events.csv --dry-run` → report the proposals per view; accept D63/scan/32 on the copy and show `pose_segments(63,'scan')` before/after; full suite + exit code.

### Task B2: Per-view change localisation that can prompt SAM

**Why:** measured on LS polygons (201 removal events, D13/24/33): the top diff blob *overlaps* the removed part 47–64 % of the time but its *centroid* is inside it only 21–41 % — removing a part reveals a hole, a socket and a patch of board, and `diff_blobs` merges them. Per-view diff is now the primary localiser in every view, so this is where prompt quality comes from.

**Files:** Create `tda/core/diff_split.py`, `experiments/plan_b_probe/diff_eval.py`, `tests/test_diff_split.py`. Modify `tda/core/diffmap.py` (only to expose what `diff_split` needs), `tda/ui/app_diff.py`, `tda/ui/canvas/sam_prompt.py`.

**Interfaces (Produces):**

```python
# tda/core/diff_split.py  (pure numpy/opencv, no Qt)
@dataclass(frozen=True)
class PartProposal:
    box: tuple[int, int, int, int]      # x0, y0, x1, y1 in full-frame coords
    point: tuple[int, int]              # a point INSIDE the proposed region (distance-transform peak, not the centroid)
    score: float
    blob_index: int
def propose_parts(prev_rgb: np.ndarray, cur_rgb: np.ndarray, roi: Box | None, *,
                  expect_area: tuple[float, float] | None = None,   # from configs/area_priors.yaml, fraction of ROI
                  known_masks: Sequence[np.ndarray] = (),           # shapes already explained on this frame
                  max_proposals: int = 3) -> list[PartProposal]: ...
```

- [ ] **Step 1 — harness first**: `experiments/plan_b_probe/diff_eval.py --db <copy> --views scan,oak1,oak2,rs` reproduces the E2 M4 baseline (centroid-in-part, box IoU with the part, and **SAM IoU when prompted with box+point**) on the same 201 events; commit the baseline numbers in the report. No production change may be merged without this table before/after.
- [ ] **Step 2 — splitting**: candidates = connected components of the ΔE map at two thresholds + watershed on the distance transform; rank by (a) agreement with the class's area prior, (b) "appearance" direction — in reverse-order annotation the part is present in the CURRENT frame and absent in the next one, so prefer the component whose interior in the current frame has higher edge/texture energy than in the other frame (a revealed hole is the opposite), (c) not already covered by `known_masks`. `point` = distance-transform argmax of the chosen component. Unit tests on synthetic pairs (part + hole + shadow), incl. the degenerate cases: identical frames → `[]`; whole-ROI change (lighting jump) → `[]`, never a full-ROI proposal; ROI `None`.
- [ ] **Step 3 — wire in**: the canvas' armed prompt = `PartProposal.box` + `PartProposal.point` (replaces blob bbox + blob centroid); `C` still cycles SAM's 3 candidates; **new**: `Shift+C` cycles the top-3 proposals. Existing rule kept: blobs > 60 % of ROI are not armed.
- [ ] **Step 4 — acceptance**: table before/after per view; target: centroid/point-in-part ≥ +15 points on scan and oak1, SAM IoU (box+point) median ≥ baseline + 0.10. A negative or flat result is a valid outcome — report it and do not merge the production change (the harness is merged either way).

### Task B3: Usable speed on 12 MP OAK frames + OAK ROI proposal

**Why (measured by the controller on main, D13/oak1, 4032×3040, real SAM):** frame change ≈190 ms (first 850 ms), timeline jump 0.83–1.28 s, commit 0.81–1.18 s **with 1–4 shapes** (grows with instance count), Space 0.30–0.45 s, SAM round trip 0.72 s with the full frame as prompt box, **ROI proposal empty** (`cache_roi_detect` finds nothing on the white table). Scanner budgets for comparison: frame change 80 ms, commit 0.2 s, Space 0.36 s, SAM 0.13 s.

**Files:** Modify `tda/core/cache_roi_detect.py`, `tda/core/cache_roi.py`, `tda/core/compiler.py`, `tda/core/compiler_layers.py`, `tda/core/truth.py`, `tda/ui/session_commits.py`, `tda/ui/session_truth.py`, `tda/ui/canvas/sam_crop.py`, `scripts/mvp_smoke.py`, `tests/test_session_perf.py`, `tests/test_cache_roi*.py`.

- [ ] **Step 1 — measure before changing**: extend `scripts/mvp_smoke.py` with `--instances N` (draw N synthetic shapes inside the ROI before timing) and run D13 on `oak1` and `oak2` with N = 10 and 40; profile commit and Space (`cProfile`, top 15 cumulative) and put the table in the report. The fix follows the profile, not this plan's guess.
- [ ] **Step 2 — compile inside the ROI window**: masks stay full-frame in storage, but painting, visibility and RLE encoding run on the union bbox of (pose-segment ROI padded 5 %, every selected shape's bbox) and are pasted back; `input_hash` and digests must be **unchanged** (same inputs → same hash; assert on the truth scenes that every stored `input_hash` is identical before/after, and that compiled RLEs are byte-identical to the full-frame path on 50 random scenes incl. shapes partly outside the ROI and bench boxes).
- [ ] **Step 3 — OAK ROI proposal**: chassis on a white table with yellow tape: dark-object segmentation (Otsu on luminance inside the table region, largest component, tape excluded by HSV) → padded bbox; accept only if it covers 8–70 % of the frame, else fall back to "no proposal" (full frame) as today. Verify on the first and last frame of 12 desktops spread over the rig epochs (incl. D36 before/after the camera move, D63/64 towers) by writing overlays and **looking** at them; report x/12 by eye.
- [ ] **Step 4 — SAM on the ROI crop** for OAK views (the existing `sam_crop` path, prompt box clipped to the ROI), and downscaled display pyramid for the 12 MP canvas if the profile says frame change is paint-bound.
- [ ] **Step 5 — budgets as tests** (`tests/test_session_perf.py`, marked `slow`, sweeper ON, late step, 40 instances, 4032×3040 synthetic scene): commit ≤ 0.45 s, Space ≤ 0.6 s, frame change ≤ 0.15 s, timeline jump ≤ 0.5 s. Re-run the real smoke on oak1/oak2 and report before/after.

### Task B4: Adopt a Label Studio draft

**Why:** 11,191 draft keyframes exist (scan: 14 desktops; oak1/rs: 3; oak2: 2). They are noisy visible-only polygons, but a draft that is 80 % right is faster to fix than to redraw, and adoption is per view — it fits "annotate each view fresh".

**Files:** Create `tda/core/ls_adopt.py`, `tda/ui/app_adopt.py`, `tests/test_ls_adopt.py`, `tests/test_app_adopt.py`. Modify `tda/ui/app_actions.py`, `docs/annotation_guide.md`.

**Interfaces (Produces):**

```python
# tda/core/ls_adopt.py (pure + Db reads)
@dataclass(frozen=True)
class DraftCandidate:
    key: str            # 'ls:<Label>#<n>'
    step: int           # the draft's own anchor step
    mask: np.ndarray    # bool, full-frame, this view
    label: str
    iou_with_editing: float
def drafts_for(db: Db, tax: Taxonomy, desktop: int, view: str, step: int, cls: str, *,
               hw: tuple[int, int], near_steps: int = 2,
               editing: np.ndarray | None = None) -> list[DraftCandidate]: ...
    # drafts of this view whose label maps (configs/ls_label_map.yaml) to `cls`, at `step` first,
    # then |Δstep| <= near_steps; wrong-size RLEs are skipped and counted; sorted by step distance then IoU
```

- [ ] **Step 1** tests for `drafts_for`: label→class mapping, step window, wrong-size skipped (never resized), other views excluded, provisional instances never written to.
- [ ] **Step 2** UI: while editing an instance, `Shift+A` ("Adopt draft") cycles the candidates as a **ghost overlay**; `Enter` on the ghost copies it into the editing layer as ONE undoable stroke (then the normal brush/SAM/commit flow applies; sidecar covers it); `Esc` dismisses the ghost only. No candidates → status bar says so. The adopted draft's key is recorded in the commit's `extra` (`{"adopted_from": key}`) so the paper can report draft reuse. Drafts are never deleted or modified.
- [ ] **Step 3** guide sentence (fit the cap) + shortcut table regenerated; full suite.

### Task B5 (wave 2): Constraint editor

**Why:** `reports/constraints_report.md` lists 18 violations (15 likely log gaps, 3 failed attempts that want a manual `blocked_by`); today the graph can only be regenerated, not edited.

**Files:** Create `tda/core/graph_edit.py`, `tda/ui/steps_relations.py`, `tda/ui/panels/relations.py`, `tests/test_graph_edit.py`, `tests/test_relations_panel.py`. Modify `tda/ui/panels/steptable.py` (third tab "Relations"), `tda/core/graph.py` (only if `constraint_edges` needs a source filter), `docs/annotation_guide.md`.

**Interfaces (Produces):**

```python
# tda/core/graph_edit.py
MANUAL = "manual"
def add_manual_edge(edges: list[Edge], src: str, kind: str, dst: str, *, necessity: str = "required",
                    mode: str | None = None, note: str = "", instances: dict[str, InstanceRec]) -> list[Edge]: ...
    # raises GraphEditError(reason) on: unknown/provisional/deleted endpoint, self edge, duplicate of ANY existing
    # edge (rule or manual), kind not in the spec-7 set, or a cycle among hard constraints (reason names the cycle)
def remove_manual_edge(edges: list[Edge], src: str, kind: str, dst: str) -> list[Edge]: ...   # rule edges: GraphEditError
def violations(db_like, desktop: int) -> list[Violation]: ...   # the same replay `constraints --validate` prints
```

- [ ] **Step 1** core tests: every refusal above; rule edges untouchable; a `constraints` re-run keeps manual edges byte-identical and re-derives rule edges; `graph_version` changes iff the edge set changes.
- [ ] **Step 2** panel: table (source badge rule/manual/labelstudio, kind, from, to, necessity, note), "Add edge" row with instance pickers limited to real instances, violations list below (double-click jumps the Steps table to that step); edits are staged like the step table and written by the existing `Apply` (one transaction, `graph_version` re-stamped, op-logged); `Revert` discards.
- [ ] **Step 3** acceptance on a DB copy: fix one of the three failed-attempt violations by adding `blocked_by`, show the violation list shrink, re-run `constraints` and show the manual edge survived.

### Task B6 (wave 2): Full VLM task generator

**Why:** Plan A exports a minimal V1/V2/V3; the paper needs the P0 set of spec §8 (V1, V2, V3, V4, V5, V6, V8, V10, V12, V14, V15, V16) with programmatically checkable answers.

**Files:** Create `tda/core/export/vlm_tasks.py`, `tda/core/export/vlm_reasoning.py`, `tests/test_vlm_tasks.py`. Modify `tda/core/export/vlm.py`, `tda/cli_app.py` (`export-vlm --tasks`).

- [ ] **Step 1** one generator function per task id with the signature `gen_Vn(ctx: DesktopCtx, frame: FrameKey) -> Iterator[VlmRecord]`; every record carries `task`, `view`, `tier`, `verified`, `graph_version`, the structured `answer` and an `answer_check` spec (exact / set / order-consistent-with-graph / abstain). Records are emitted only from VERIFIED frames for perception tasks (V1/V2/V8/V15) and from the state machine + graph for planning tasks (V4/V5/V6/V16), so planning tasks are exportable today.
- [ ] **Step 2** legality: V5/V6/V16 answers come from `graph_plan` (legal-action set at state k); hard negatives per spec §8 (blocked action with the blocking edge as the explanation). V14 (answerability): questions about instances with `visibility in (occluded_full, out_of_view)` must have the abstain answer. V15 (cross-view): same step, two views, built only from per-view VERIFIED `visibility` — no geometry.
- [ ] **Step 3** tests on the synthetic truth scenes: every answer re-derivable by an independent checker in the test; no `ls:*`, no implied-only leakage without `attributes.implied`; deterministic output order; export refuses on open conflicts exactly like COCO.
- [ ] **Step 4** acceptance on a DB copy: planning-task counts per desktop for all 66 (there are no verified frames yet, so perception counts are 0 — say so).

---

## Out of this plan (next plan, needs real annotations first)

Per-view detector training loop (YOLO26 on verified ROI crops, retrain every ~10 desktops, per-view cold start after 3–4 desktops), frame registration P1 (RANSAC similarity inside a pose segment, residual shown to the human), bench ROI editor, polygon vertex editing, full §9.1 auto-checks, optional coarse cross-view hint (large parts only), zero-shot / fine-tune evaluation pipeline.

## Self-review notes

Spec v1.5 §2.5 per-view pose segments → B1; §3.2 "time + per-view diff" identity → B2; §4.1 S3/S4 "same flow on OAK" → B3 (speed, ROI) + B1; drafts (§5, Label Studio) → B4; §7 editing → B5; §8 task set → B6. Registration P1, detector loop and §9.1 checks are deliberately deferred (listed above). Review Focus items 1–2 are pinned by B1 steps 3–4, item 3 by B3 step 5, item 4 by B4 steps 1–2, item 5 by B5 step 1.
