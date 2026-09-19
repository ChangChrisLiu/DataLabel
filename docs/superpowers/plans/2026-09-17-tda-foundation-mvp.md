# TDA Plan A — Data Foundation + Scanner MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data foundation (index, log import, taxonomy, SQLite store, layer compiler, truth table) and a working PySide6 annotator for the scanner view, so real annotation of Desktop 13 can start and an end-to-end rehearsal (annotate → COCO → VLM JSONL) can run.

**Architecture:** `tda/core` is pure Python (no Qt) — index/logs/taxonomy/db/masks/states/compiler/truth/export; `tda/models` wraps SAM 2.1 in a worker thread; `tda/ui` is a thin PySide6 layer over core. Layered masks (shape keyframes + z-order + per-frame overrides) are compiled into a per-frame truth table that is frozen on human verification. Everything is keyed by `(desktop_id, step_k, view)`.

**Tech Stack:** Python 3.11 (`D:\Anaconda\envs\tda\python.exe`), PySide6, numpy, opencv-python, pycocotools, SQLite (stdlib `sqlite3`, WAL), PyYAML, pytest, sam2 (SAM 2.1 Hiera-L), ultralytics (later plans).

**Spec:** `docs/superpowers/specs/2026-09-17-teardown-annotator-design.md` (v1.3). Executors read both. Section references below (§) point into the spec.

## Global Constraints

- All environments, caches, weights and temp files live under `D:\` (`D:\Anaconda\envs\tda`, `D:\DataSet\...`). Nothing new on `C:\`.
- Raw data on `F:\PHD Data Backup\Desktop Dataset\` is **read-only**. Never write, move or delete there (except the backup folder `F:\PHD Data Backup\Desktop Dataset\TDA_backups\` created by the backup task).
- `F:` is a slow external drive: never `find`/`du` whole trees; read only the files you need; cache to `D:\DataSet\cache\`.
- Masks are stored in **original image coordinates** as COCO RLE (`{"size":[h,w],"counts":str}`), never in ROI coordinates.
- Frame key = `(desktop_id:int, step:int, view:str)` with `view ∈ {"scan","oak1","oak2","rs"}`; `step` is the 1-based logical step (step 1 = initial state).
- Image at step k shows the state **after** action k (§C9).
- Run tests with `D:\Anaconda\envs\tda\python.exe -m pytest tests -q` from `D:\DataSet`. Use `QT_QPA_PLATFORM=offscreen` for UI tests.
- Commit after every task with a message ending in `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; remote is `origin` = `https://github.com/ChangChrisLiu/DataLabel.git`, branch `main`.
- Code comments/docstrings in English; user-facing UI strings in English (annotators are bilingual); docs may be Chinese.
- No file over ~600 lines; split by responsibility.
- Language/wording of the taxonomy must follow §6 exactly (class names, states, verbs, tools).

## Reference material (read-only inputs)

- Drive log exports (66 desktops): `C:\Users\61908\AppData\Local\Temp\claude\D--DataSet\3f6dd74d-e1dc-46a6-bbc2-2fe9e89c0835\scratchpad\drive_logs\` (`desktop_NN.xlsx/.csv`, `desktop_meta.csv`, `steps_long.csv`, `vocab_target_canon.csv`, `canon_rules.py`). Task 2 copies the xlsx/csv into `D:\DataSet\raw_logs\drive\`.
- OAK survey: same scratchpad `oakd_survey\desktop_survey.csv`, `anomalies.txt`, `listing.tsv`.
- Label Studio export: `D:\DataSet\raw_logs\labelstudio\humansignal_annotated_projects_export.json` (26 projects; task `data.image` basenames look like `3e99d760-Rs_13_42.png`, `83334756-Front_OAK_13_42.jpg`, `696bb253-33_39.png`; the number pair is `<desktop>_<step>`).
- Sample images for tests/smoke: scratchpad `views\scanner.png` (900×900 downscale), `views\scanner_crop_native.png` (900×900 native crop), `views\d13_s1_3.png`.
- Source layout on F: (§2.1): scanner `UGA DATA\TAMU_B2.3_<a>-<b>_RGB\<TAMU_B2.3_<a>-<b>_RGB | tamu_color_50-66_Bright2.3>\<desktop>\RGB<step>1\P_0..P_9.png`; OAK `OAKD Capture\Desktop_Datacollection\DesktopData\Desktop <n>\Disassemble\Camera_<1|2>\<NNN>\<ts>_camera_<1|2>_{rgb_12mp.jpg,rgb_aligned.png,depth_raw.npy,depth_raw.png,pointcloud.ply}` (ts = `YYYYMMDD_HHMMSS_mmm`); RealSense `Realsense Capture\Dataset Information\Exp\Desktop <n>\<Disassemble|Disassembly>\<NNN>\{original_color.png,depth_raw.npy,...}`.

## File structure (created by this plan)

```
D:\DataSet\
  pyproject.toml                 # package metadata, pytest config
  tda\__init__.py
  tda\core\__init__.py
  tda\core\model.py              # shared dataclasses/enums (FrameKey, InstanceRec, ShapeKeyframe, ...)
  tda\core\taxonomy.py           # loads configs/taxonomy.yaml + taxonomy_map.yaml; raw-name parsing
  tda\core\logs.py               # Drive xlsx → Step/Action/Instance drafts
  tda\core\index.py              # F: scan → frames; fixes; gap report
  tda\core\masks.py              # RLE, morphology, tolerant diff, polygon conversion
  tda\core\states.py             # state machine, attached children, placement, needs_mask
  tda\core\compiler.py           # layer compiler (pure)
  tda\core\db.py                 # SQLite schema + repository + backup + lock
  tda\core\truth.py              # CompiledMask refresh / verify / conflicts
  tda\core\cache.py              # local image cache, burst metrics, ROI suggestion
  tda\core\ls_import.py          # Label Studio export → draft shapes
  tda\core\export\__init__.py
  tda\core\export\coco.py        # minimal COCO export (rehearsal)
  tda\core\export\vlm.py         # minimal V1/V2/V3 JSONL (rehearsal)
  tda\models\__init__.py
  tda\models\sam_service.py      # SAM 2.1 worker
  tda\ui\__init__.py
  tda\ui\app.py                  # main window wiring
  tda\ui\session.py              # AnnotationSession: current desktop/view/step, dirty state, save
  tda\ui\canvas\__init__.py
  tda\ui\canvas\view.py          # ImageCanvas (QGraphicsView): zoom/pan/pixel grid/minimap
  tda\ui\canvas\overlay.py       # LabelOverlay: uint16 labelmap → QImage via palette, dirty rects
  tda\ui\canvas\tools.py         # BrushTool, EraserTool, SamPointTool, SamBoxTool, OccluderTool
  tda\ui\commands.py             # Op log, undo/redo
  tda\ui\panels\__init__.py
  tda\ui\panels\timeline.py      # TimelinePanel
  tda\ui\panels\taskcard.py      # TaskCardPanel
  tda\ui\panels\instances.py     # InstanceListPanel
  tda\ui\panels\steptable.py     # StepTablePanel (S1)
  tda\ui\panels\review.py        # ReviewPanel
  tda\cli.py                     # tda build-index | import-logs | build-cache | import-ls | check | backup | export-coco | export-vlm | app
  configs\taxonomy.yaml
  configs\taxonomy_map.yaml
  configs\index_fixes.yaml
  configs\paths.yaml             # F: roots, D: cache/db locations
  tests\conftest.py              # fixtures: tmp db, synthetic images, fixture F-tree
  tests\fixtures\...             # small synthetic data
  tests\test_*.py
```

## Dependency waves (for parallel workers)

- **Wave 1 (parallel):** T1 taxonomy, T3 index, T5 masks, T10 SAM service.
- **Wave 2 (parallel, after wave 1):** T2 logs (needs T1), T4 db (needs T0 model), T6 states (needs T1), T9 cache (needs T3).
- **Wave 3:** T7 compiler (needs T5, T6), T11 canvas (needs T5, T10), T14 LS import (needs T4, T5).
- **Wave 4:** T8 truth (needs T4, T7), T12 panels + T13 app/cli (need all), T15 rehearsal export (needs T8).

T0 (scaffold + `model.py`) is done first by a single worker before wave 1.

---

### Task 0: Scaffold, shared model types, config paths

**Files:**
- Create: `pyproject.toml`, `tda/__init__.py`, `tda/core/__init__.py`, `tda/core/model.py`, `configs/paths.yaml`, `tests/conftest.py`, `tests/test_model.py`

**Interfaces:**
- Produces (used by every later task):

```python
# tda/core/model.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

VIEWS = ("scan", "oak1", "oak2", "rs")

class Visibility(str, Enum):
    VISIBLE = "visible"; OCCLUDED_PARTIAL = "occluded_partial"; OCCLUDED_FULL = "occluded_full"
    OUT_OF_VIEW = "out_of_view"; TOO_SMALL = "too_small"; VISIBLE_TINY = "visible_tiny"; MOTION_BLUR = "motion_blur"

class Placement(str, Enum):
    IN_CHASSIS = "in_chassis"; ON_BENCH = "on_bench"; ELSEWHERE = "elsewhere"

class StepType(str, Enum):
    INITIAL = "initial"; NORMAL = "normal"; DUPLI = "dupli"; COMPOUND = "compound"; FAILED = "failed"
    AUXILIARY = "auxiliary"; REORIENT = "reorient"; IGNORE = "ignore"

@dataclass(frozen=True, order=True)
class FrameKey:
    desktop: int
    step: int
    view: str

@dataclass
class InstanceRec:
    key: str                      # e.g. "screw.cpu_cooler.03"
    desktop: int
    cls: str                      # taxonomy class
    attrs: dict = field(default_factory=dict)   # role, head, head_source, kind, captive, of, ...
    parent: Optional[str] = None
    attached: bool = False
    mounted_on: Optional[str] = None
    fastens: Optional[str] = None
    socket_host: Optional[str] = None
    cable: Optional[str] = None
    slot_id: Optional[str] = None
    group_id: Optional[str] = None
    group_order: str = "unordered"
    removal_direction: Optional[str] = None
    raw_names: list[str] = field(default_factory=list)

@dataclass
class StepRec:
    desktop: int
    step: int
    step_type: str
    raw_name: str
    dupli: bool = False
    notes: str = ""
    duration_s: Optional[float] = None

@dataclass
class ActionRec:
    desktop: int
    step: int
    idx: int                      # order within the step
    target: str                   # instance key or virtual node id ("cable:psu_harness")
    verb: str                     # unscrew|disconnect|open|release|remove|displace|reorient
    tool: str = "none"
    direction: str = "none"
    result: str = "success"       # success|failed
    failure_reason: Optional[str] = None
    difficulty: Optional[int] = None

@dataclass
class StateEvent:
    desktop: int
    step: int
    target: str
    attr: str                     # "state" | "placement"
    old: str
    new: str
    evidence_view: Optional[str] = None
    auto: bool = True

@dataclass
class ShapePart:
    name: str                     # "main" or a named part e.g. "floor", "wall"
    rle: Optional[dict] = None    # COCO RLE in reference-frame coords
    box: Optional[tuple[float, float, float, float]] = None  # x0,y0,x1,y1

@dataclass
class ShapeKeyframe:
    id: Optional[int]
    instance: str
    desktop: int
    view: str
    pose_segment: int
    anchor_step: int
    placement: str = Placement.IN_CHASSIS.value
    geom_type: str = "mask"       # mask|box
    parts: list[ShapePart] = field(default_factory=list)
    amodal_complete: bool = True
    source: str = "manual"        # manual|sam|model:<name>@<ver>|labelstudio
    draft_id: Optional[int] = None
    version: int = 1
    edit_count: int = 0
    edit_time_ms: int = 0

@dataclass
class ZOrderRec:
    desktop: int
    view: str
    pose_segment: int
    order: list[tuple[str, str]]  # (instance_key, part_name) bottom → top
    version: int = 1

@dataclass(frozen=True)
class PairOverride:
    desktop: int; view: str; pose_segment: int; above: str; below: str

@dataclass
class OccluderMask:
    frame: FrameKey
    occluder_type: str            # hand|arm|body|tool|cable|other
    rle: dict

@dataclass
class FrameOverride:
    frame: FrameKey
    instance: str
    visible_rle: Optional[dict] = None
    visibility: Optional[str] = None

@dataclass
class Similarity:
    scale: float = 1.0; theta: float = 0.0; tx: float = 0.0; ty: float = 0.0
    def is_identity(self) -> bool: ...
```

- `configs/paths.yaml`:

```yaml
f_root: "F:/PHD Data Backup/Desktop Dataset"
scanner_root: "F:/PHD Data Backup/Desktop Dataset/UGA DATA"
oak_root: "F:/PHD Data Backup/Desktop Dataset/OAKD Capture/Desktop_Datacollection/DesktopData"
rs_root: "F:/PHD Data Backup/Desktop Dataset/Realsense Capture/Dataset Information/Exp"
cache_dir: "D:/DataSet/cache"
db_path: "D:/DataSet/annotations/tda.sqlite"
backup_dir: "F:/PHD Data Backup/Desktop Dataset/TDA_backups"
raw_logs_dir: "D:/DataSet/raw_logs"
weights_dir: "D:/DataSet/models/weights"
```

- [ ] **Step 1: Write `pyproject.toml`** with `[project] name="tda" version="0.1.0" requires-python=">=3.11"`, `[tool.pytest.ini_options] testpaths=["tests"]`, and package discovery for `tda*`.
- [ ] **Step 2: Write `tda/core/model.py`** exactly as the interface above (complete `Similarity.is_identity` as `abs(scale-1)<1e-9 and abs(theta)<1e-9 and abs(tx)<1e-9 and abs(ty)<1e-9`).
- [ ] **Step 3: Write `tests/conftest.py`** with fixtures: `tmp_db_path(tmp_path)`, `small_img()` (a 64×64 RGB numpy image with a bright square), `paths_cfg()` loading `configs/paths.yaml`.
- [ ] **Step 4: Write `tests/test_model.py`**: `FrameKey(13,2,"scan") < FrameKey(13,3,"scan")`; `Similarity().is_identity()`; `Visibility("too_small")` roundtrip.
- [ ] **Step 5: Run** `D:\Anaconda\envs\tda\python.exe -m pytest tests/test_model.py -q` → PASS.
- [ ] **Step 6: Commit** `chore: scaffold tda package and shared model types`.

---

### Task 1: Taxonomy and raw-name mapping

**Files:**
- Create: `configs/taxonomy.yaml`, `configs/taxonomy_map.yaml`, `tda/core/taxonomy.py`, `tests/test_taxonomy.py`

**Interfaces:**
- Produces:

```python
# tda/core/taxonomy.py
@dataclass
class ParsedTarget:
    cls: str                      # taxonomy class or "virtual"
    attrs: dict                   # role/kind/of/... inferred from the raw name
    instance_no: Optional[int]    # trailing instance number if present
    virtual: Optional[str]        # e.g. "cable" for cable-routing steps
    verb: Optional[str]           # verb implied by the name (open/displace/release/...) or None
    multi: bool                   # names several targets at once ("screws 1/2/3")
    attempt: bool                 # "try to ..." / "failed"
    step_type_hint: Optional[str] # "initial"|"dupli"|"reorient"|"auxiliary"|"ignore"|None
    canon_group: str              # the 52-group label for review
    matched_rule: str

class Taxonomy:
    classes: dict[str, dict]      # class -> {"group":..., "attrs": {...}, "states": [...], "default_state":..., "needs_mask": {state: bool}}
    verbs: dict[str, dict]        # verb -> {"applies_to": [...], "effect": {...}}
    tools: list[str]
    def states_of(self, cls: str) -> list[str]
    def default_state(self, cls: str) -> str
    def needs_mask(self, cls: str, state: str, placement: str) -> bool
    def apply_verb(self, cls: str, attrs: dict, verb: str) -> tuple[str, str] | None  # (attr, new_value) or None (no state change)

def load_taxonomy(path="configs/taxonomy.yaml") -> Taxonomy
def parse_raw_name(raw: str, nest_group: str = "", rules_path="configs/taxonomy_map.yaml") -> ParsedTarget
def map_tool(raw: str) -> str      # "Philips PH2"/"PH 2"/"PH2"→"PH2"; "T15"/"Screw Driver T15"→"T15"; "Hand"→"hand"; ""→"none"
```

- `configs/taxonomy.yaml` must encode §6.1 (23 classes with attrs), §6.2 (states + which states need masks: `needs_mask` true for bold states; `unplugged` false; `removed` false unless placement on_bench), §6.3 (verbs and effects; `unscrew` → `loosened` if `captive` else `removed`), §6.5 tools.
- `configs/taxonomy_map.yaml`: ordered regex rules ported from scratchpad `canon_rules.py` + `vocab_target_canon.csv`. Each rule: `{pattern, cls, attrs, virtual, verb, step_type_hint, canon_group}`. Instance numbers are captured by a generic trailing-number regex `(\d+)\s*$` and patterns like `RAM1`, `screw 3`, `1&2&3`, `1/2/3` (multi).

- [ ] **Step 1: Write failing tests** `tests/test_taxonomy.py`:

```python
import pytest
from tda.core.taxonomy import load_taxonomy, parse_raw_name, map_tool

CASES = [
    ("CPU fan screw 3", "", dict(cls="screw", role="cpu_cooler", n=3)),
    ("Heatsink screw 2", "", dict(cls="screw", role="cpu_cooler", n=2)),
    ("Motherboard screw 6", "", dict(cls="screw", role="motherboard", n=6)),
    ("RAM clip 1", "", dict(cls="ram_latch", n=1)),
    ("RAM1", "", dict(cls="ram_module", n=1)),
    ("CPU locker", "", dict(cls="cpu_socket_lever")),
    ("Power module", "", dict(cls="psu")),
    ("Case - motherboard connector 2", "", dict(cls="connector", kind="front_panel", n=2)),
    ("Power module - motherboard connector 1", "", dict(cls="connector", cable_owner="psu", n=1)),
    ("Connector 3", "Power Supply Unit (PSU)", dict(cls="connector", cable_owner="psu", n=3)),
    ("SSD SATA connector", "", dict(cls="connector", kind="sata_data")),
    ("Optical drive power connector", "", dict(cls="connector", kind="sata_power")),
    ("SSD shield", "", dict(cls="drive_cage")),
    ("SSD shield locker", "", dict(cls="drive_latch")),
    ("Case cover for motherboard screws", "", dict(cls="cover", of="motherboard_screws")),
    ("Initial Conditions", "", dict(step_type_hint="initial")),
    ("dupli", "", dict(step_type_hint="dupli")),
    ("change a direction", "", dict(step_type_hint="reorient")),
    ("remove cable from cable locker", "", dict(virtual="cable", verb="release")),
    ("Try to remove the power module", "", dict(cls="psu", attempt=True)),
    ("Motherboard screws 1/2/3", "", dict(cls="screw", role="motherboard", multi=True)),
    ("Open power module", "", dict(cls="psu", verb="displace")),
    ("Heatsink locker base", "", dict(cls="cooler_bracket")),
    ("all components", "", dict(step_type_hint="ignore")),
]

@pytest.mark.parametrize("raw,nest,exp", CASES)
def test_parse_raw_name(raw, nest, exp):
    p = parse_raw_name(raw, nest)
    for k, v in exp.items():
        if k == "n": assert p.instance_no == v
        elif k in ("cls", "virtual", "verb", "multi", "attempt", "step_type_hint"): assert getattr(p, k) == v
        else: assert p.attrs.get(k) == v

def test_taxonomy_states_and_verbs():
    t = load_taxonomy()
    assert set(t.classes) == {"chassis","drive_cage","cover","cooler_bracket","motherboard","cpu","cpu_cooler","ram_module","psu","storage_drive","optical_drive","expansion_card","case_fan","misc_part","screw","ram_latch","cpu_socket_lever","psu_latch","drive_latch","card_latch","cooler_latch","cable_clip","connector"}
    assert t.states_of("screw") == ["fastened","loosened","removed"]
    assert t.apply_verb("screw", {"captive": True}, "unscrew") == ("state","loosened")
    assert t.apply_verb("screw", {"captive": False}, "unscrew") == ("state","removed")
    assert t.apply_verb("connector", {}, "disconnect") == ("state","unplugged")
    assert t.needs_mask("connector", "unplugged", "in_chassis") is False
    assert t.needs_mask("screw", "loosened", "in_chassis") is True
    assert t.needs_mask("psu", "removed", "on_bench") is True
    assert t.needs_mask("psu", "removed", "elsewhere") is False

@pytest.mark.parametrize("raw,exp", [("Philips PH2","PH2"),("PH 2","PH2"),("Philip PH2","PH2"),("T15","T15"),("Screw Driver T15","T15"),("Hand","hand"),("","none"),("PH1","PH1")])
def test_map_tool(raw, exp):
    assert map_tool(raw) == exp
```

- [ ] **Step 2: Run** → FAIL (module missing).
- [ ] **Step 3: Write `configs/taxonomy.yaml`** covering §6.1–6.5 (all 23 classes; states per class; verbs with `applies_to` and `effect`; tools list). Put `needs_mask` per class as a mapping `state → bool` and a top-level rule `on_bench_needs_geom: true`.
- [ ] **Step 4: Write `configs/taxonomy_map.yaml`** — port every group in `vocab_target_canon.csv` (52 groups) into ordered rules; the first matching rule wins; keep `canon_group` text identical to the csv's `target_canon` column.
- [ ] **Step 5: Implement `tda/core/taxonomy.py`** (`load_taxonomy`, `parse_raw_name`, `map_tool`). Number extraction: strip `(…)` qualifiers first (keep them in `attrs["qualifier"]`), then match `(\d+)\s*$`, `RAM(\d)`, `(\d+)\s*[&/,]\s*\d+` → `multi=True`. Lowercase, collapse whitespace, fix known typos (`Coonn`, `Conenctor`, `Compoq`) before matching.
- [ ] **Step 6: Run tests** → PASS. Also run a coverage script over `steps_long.csv` (all 2,830 rows) and print rows with `matched_rule == "fallback"`; add rules until fallback < 1% of rows; commit the list of remaining fallbacks to `configs/taxonomy_map_unmatched.txt`.
- [ ] **Step 7: Commit** `feat(core): taxonomy config and raw-name parser`.

---

### Task 2: Drive log import → steps, actions, instances

**Files:**
- Create: `tda/core/logs.py`, `tests/test_logs.py`, `tests/fixtures/logs/desktop_13.csv`, `tests/fixtures/logs/desktop_63.csv`, `tests/fixtures/logs/desktop_01.csv`
- Copy: scratchpad `drive_logs/desktop_NN.xlsx|csv|_meta.csv` → `D:\DataSet\raw_logs\drive\`

**Interfaces:**
- Consumes: `parse_raw_name`, `map_tool`, `Taxonomy` (T1); `StepRec`, `ActionRec`, `InstanceRec` (T0).
- Produces:

```python
# tda/core/logs.py
@dataclass
class LogImport:
    desktop: int
    meta: dict                    # brand_model_raw, size_raw, collection_date, notes
    steps: list[StepRec]
    actions: list[ActionRec]
    instances: dict[str, InstanceRec]
    issues: list[str]             # human-readable problems (skipped seq numbers, duplicate numbers, fallback names)

def read_desktop_csv(path) -> tuple[list[dict], dict]   # rows (Sequence Number, Sequence Name, Target Nest Group, Tool Utility, Notes, Complexity), meta
def import_log(desktop: int, rows: list[dict], meta: dict, taxonomy: Taxonomy) -> LogImport
def instance_key(cls: str, attrs: dict, ordinal: int) -> str   # "screw.cpu_cooler.03", "connector.front_panel.02", "ram_latch.01", "psu.01", "chassis"
```

Rules (from §2.3, §3.2):
- Logical step numbers = row order (1-based), **not** the sheet's Sequence Number (D56 skips 41, D63 repeats 11–13); record mismatches in `issues`.
- `step_type`: `initial` for row 1; `dupli` when name is dupli; `reorient`, `ignore`, `auxiliary` from `step_type_hint`; `failed` when `attempt`; `compound` when `multi` or the name contains " and "; else `normal`.
- Instance ordinal = order of **first operation** within the desktop for the same `(cls, role/kind)` (ignore the sheet's own numbers). A `compound` row with `multi` creates N placeholder targets `…?` flagged in `issues` for manual resolution in S1 (do not guess N).
- `verb` = `step_type_hint` verb if present else default by class: screw→unscrew, connector→disconnect, latch/lever/cover→open, cable_clip→release, parts→remove.
- `tool` via `map_tool`; screws with empty tool get `"unknown"` and an issue entry.
- Every desktop gets an implicit `chassis` instance (`cls=chassis`, key `chassis`).
- `attrs["captive"]` default: True for `role=cpu_cooler` on Dell models (meta brand contains "Dell"/"Optiplex"), else False; record `attrs["captive_source"]="heuristic"`.

- [ ] **Step 1: Write fixtures** by copying the three real CSVs from the scratchpad into `tests/fixtures/logs/`.
- [ ] **Step 2: Write failing tests**:

```python
from tda.core.logs import read_desktop_csv, import_log, instance_key
from tda.core.taxonomy import load_taxonomy

def test_instance_key():
    assert instance_key("screw", {"role":"cpu_cooler"}, 3) == "screw.cpu_cooler.03"
    assert instance_key("psu", {}, 1) == "psu.01"

def test_import_d13_counts():
    rows, meta = read_desktop_csv("tests/fixtures/logs/desktop_13.csv")
    li = import_log(13, rows, meta, load_taxonomy())
    assert len(li.steps) == 42 and li.steps[0].step_type == "initial"
    screws = [k for k in li.instances if k.startswith("screw.cpu_cooler.")]
    assert screws == ["screw.cpu_cooler.01","screw.cpu_cooler.02","screw.cpu_cooler.03","screw.cpu_cooler.04"]
    a = [x for x in li.actions if x.step == 3][0]
    assert a.target == "screw.cpu_cooler.02" and a.verb == "unscrew" and a.tool == "PH2"
    assert "chassis" in li.instances

def test_import_d63_duplicate_seq_numbers_reported():
    rows, meta = read_desktop_csv("tests/fixtures/logs/desktop_63.csv")
    li = import_log(63, rows, meta, load_taxonomy())
    assert any("Sequence Number" in s for s in li.issues)
    assert li.steps[-1].step == len(rows)

def test_import_d01_dupli_and_connectors():
    rows, meta = read_desktop_csv("tests/fixtures/logs/desktop_01.csv")
    li = import_log(1, rows, meta, load_taxonomy())
    dupli = [s for s in li.steps if s.step_type == "dupli"]
    assert len(dupli) == 2 and all(not [a for a in li.actions if a.step == s.step] for s in dupli)
    c = [a for a in li.actions if a.step == 14][0]           # "Connector 1" nest=PSU
    assert li.instances[c.target].cls == "connector" and li.instances[c.target].cable == "cable:psu"
```

- [ ] **Step 3: Run** → FAIL.
- [ ] **Step 4: Implement `tda/core/logs.py`** per the rules; `read_desktop_csv` must handle the two-table layout (steps table, then metadata table) as exported (see scratchpad `s2_load.py` for the parsing already proven to work — port it).
- [ ] **Step 5: Run tests** → PASS. Then run the importer over all 66 CSVs (`python -m tda.core.logs --all D:/DataSet/raw_logs/drive`) and write `D:\DataSet\raw_logs\drive\import_issues.md` (per-desktop issue list). Expected: step counts equal the scratchpad `n_logged_steps` column for all 66.
- [ ] **Step 6: Commit** `feat(core): import Drive disassembly logs into steps/actions/instances`.

---

### Task 3: Unified frame index with anomaly fixes

**Files:**
- Create: `tda/core/index.py`, `configs/index_fixes.yaml`, `tests/test_index.py`, `tests/fixtures/ftree/` (synthetic mini tree built by a fixture function, not checked-in images)

**Interfaces:**
- Consumes: `FrameKey`, `paths.yaml`.
- Produces:

```python
# tda/core/index.py
@dataclass
class FrameFile:
    key: FrameKey
    path: str                     # main image (scan: P_0.png; oak: rgb_12mp.jpg; rs: original_color.png)
    aux: dict[str, str]           # {"aligned": ..., "depth_npy": ..., "burst": [P_0..P_9 paths]}
    ts: Optional[str]             # capture timestamp (oak filename ts; file mtime otherwise)
    src_step_dir: str             # original folder name (e.g. "016")

@dataclass
class DesktopIndex:
    desktop: int
    n_steps: int                  # logical steps = OAK cam1 captures after fixes
    frames: dict[FrameKey, FrameFile]
    missing: list[FrameKey]
    issues: list[str]

def scan_desktop(desktop: int, roots: dict, fixes: dict) -> DesktopIndex
def build_index(desktops: list[int], roots: dict, fixes_path="configs/index_fixes.yaml") -> dict[int, DesktopIndex]
def save_index(idx: dict[int, DesktopIndex], path: str) -> None       # JSON
def load_index(path: str) -> dict[int, DesktopIndex]
```

Fix rules (`configs/index_fixes.yaml`, all from §2.2; each entry has a `reason` string):

```yaml
oak:
  60: {cam2_shift: {from_dir: "004", to_step: 3, note: "cam2 folder 003 missing; 004 holds step 3; 005 holds two captures → split by timestamp"}, cam1_reorder_by_ts: true}
  42: {cam2_swap: [["016","017"]]}
  24: {cam2_extra_from: "Components/C2/043", as_step: 43}
  10: {step1_keep: "latest"}          # two captures in 001 → keep the later one
scan:
  1: {missing_steps: [2,3,4,28,29,30,31,32,34,35]}
rs:
  42: {missing_steps_range: [1,12]}
  folder_aliases: ["Disassemble","Disassembly"]
general:
  step_order_source: "oak1_timestamp"
```

- Logical step order = OAK cam1 capture timestamps ascending (after `cam1_reorder_by_ts`). Scanner step = `RGB<n>1` → n; RealSense step = folder number; OAK cam2 matched to cam1 by identical timestamp (after fixes).
- Scanner `burst` lists all existing `P_k.png` (9 or 10).

- [ ] **Step 1: Write a fixture builder** in `tests/test_index.py` that creates a temp tree mimicking the real layout for a fake desktop 99 with 4 steps: OAK cam1/cam2 with timestamps, scanner `RGB11`,`RGB21`,`RGB41` (step 3 missing), RS `Disassembly/001..004` (alias test), empty PNG/JPG files.
- [ ] **Step 2: Write failing tests**: `scan_desktop(99, roots, fixes)` → `n_steps == 4`; `FrameKey(99,3,"scan") in idx.missing`; `frames[FrameKey(99,2,"oak2")].ts == frames[FrameKey(99,2,"oak1")].ts`; RS alias resolved; `save_index`/`load_index` roundtrip equality.
- [ ] **Step 3: Run** → FAIL.
- [ ] **Step 4: Implement** `index.py` (use `os.scandir`, never recursive walks beyond the known depth; sort by ts).
- [ ] **Step 5: Run tests** → PASS. Then run `build_index(range(1,67))` against F: (takes minutes; print progress) and save to `D:\DataSet\cache\index.json`; write `D:\DataSet\cache\index_report.md` with per-desktop `n_steps`, missing frames per view, issues. Expected: OAK cam1 n_steps matches scratchpad `desktop_survey.csv` (`cam1_steps`) for all 66; D60/D42/D24 fixes applied (assert in a script, not only by eye).
- [ ] **Step 6: Commit** `feat(core): unified frame index with anomaly fixes`.

---

### Task 4: SQLite schema, repository, backup, lock

**Files:**
- Create: `tda/core/db.py`, `tests/test_db.py`

**Interfaces:**
- Consumes: T0 dataclasses.
- Produces:

```python
# tda/core/db.py
class Db:
    def __init__(self, path: str): ...            # opens with WAL, foreign_keys=ON, creates schema if missing
    def close(self) -> None
    # desktops / frames
    def upsert_desktop(self, desktop: int, meta: dict) -> None
    def upsert_frame(self, key: FrameKey, path: str, aux: dict, ts: str|None, flags: dict|None=None) -> None
    def set_frame_flags(self, key: FrameKey, **flags) -> None      # hand_or_tool_in_frame, in_progress, image_quality, bench_annotated, review_status
    def get_frame(self, key: FrameKey) -> dict | None
    def frames_for(self, desktop: int, view: str) -> list[dict]
    # steps / actions / instances / events
    def replace_steps(self, desktop: int, steps: list[StepRec], actions: list[ActionRec]) -> None
    def steps(self, desktop: int) -> list[StepRec]
    def actions(self, desktop: int, step: int|None=None) -> list[ActionRec]
    def upsert_instance(self, inst: InstanceRec) -> None
    def instances(self, desktop: int) -> dict[str, InstanceRec]
    def replace_events(self, desktop: int, events: list[StateEvent], auto_only=True) -> None
    def events(self, desktop: int) -> list[StateEvent]
    # geometry
    def add_keyframe(self, kf: ShapeKeyframe) -> int
    def update_keyframe(self, kf: ShapeKeyframe) -> None            # bumps version
    def keyframes(self, desktop: int, view: str, instance: str|None=None) -> list[ShapeKeyframe]
    def set_zorder(self, z: ZOrderRec) -> None
    def zorder(self, desktop: int, view: str, pose_segment: int) -> ZOrderRec
    def set_pair_override(self, po: PairOverride) -> None
    def pair_overrides(self, desktop: int, view: str, pose_segment: int) -> list[PairOverride]
    def set_occluder(self, om: OccluderMask) -> None
    def occluders(self, key: FrameKey) -> list[OccluderMask]
    def set_frame_override(self, fo: FrameOverride) -> None
    def frame_overrides(self, key: FrameKey) -> dict[str, FrameOverride]
    def set_pose_segment(self, desktop: int, view: str, seg: int, start: int, end: int, ref_step: int, corners: list|None, homography: list|None) -> None
    def pose_segment_for(self, key: FrameKey) -> dict
    def set_transform(self, key: FrameKey, sim: Similarity) -> None
    def transform(self, key: FrameKey) -> Similarity
    # truth
    def put_compiled(self, key: FrameKey, instance: str, visible_rle: dict|None, occlusion_ratio: float, visibility: str, placement: str, status: str, input_hash: str, verified_by: str|None=None) -> None
    def compiled(self, key: FrameKey) -> dict[str, dict]
    def add_conflict(self, key: FrameKey, instance: str, old_rle: dict|None, new_rle: dict|None, sym_diff_px: int) -> int
    def conflicts(self, desktop: int, view: str|None=None, open_only=True) -> list[dict]
    def resolve_conflict(self, cid: int, resolution: str) -> None    # keep_old|accept_new|edited
    # ops
    def log_op(self, desktop: int, view: str, kind: str, payload: dict, inverse: dict, annotator: str) -> int
    def ops(self, desktop: int, view: str, limit: int=100) -> list[dict]
    # misc
    def backup(self, dest_dir: str) -> str                           # sqlite3 backup API → "<dest>/tda_YYYYMMDD_HHMMSS.sqlite", returns path
    def acquire_lock(self, annotator: str) -> None                   # writes <db>.lock with annotator+timestamp; raises RuntimeError if held by another annotator < 12h old
    def release_lock(self) -> None
```

Schema (DDL executed in `Db.__init__`, table per entity in spec §3.1): `desktop`, `frame`, `step`, `action`, `instance`, `state_event`, `pose_segment`, `frame_transform`, `shape_keyframe` (+ `shape_part` with `rle_json`, `box_json`), `zorder`, `pair_override`, `occluder_mask`, `frame_override`, `instance_frame_flags`, `compiled_mask`, `conflict`, `relation` (columns: type, target, blocker, necessity, mode, reason, source, evidence_step, status), `op_log`, `meta` (schema_version). All JSON columns are TEXT. Primary keys as in the interfaces.

- [ ] **Step 1: Write failing tests** `tests/test_db.py` covering: schema creation idempotent; `upsert_frame`/`get_frame` roundtrip; `replace_steps` then `steps()` ordered by step; keyframe add/update bumps `version`; `zorder` default (empty order) when unset; `put_compiled` upsert then `compiled()`; `add_conflict`/`resolve_conflict`; `backup()` creates a readable copy (open it and count tables); `acquire_lock` twice with different annotators raises.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement `db.py`** (one connection, `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`; dataclass ↔ row helpers; keep under 600 lines by moving DDL into `tda/core/schema.sql` loaded at init).
- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** `feat(core): sqlite repository, backup and lock`.

---

### Task 5: Mask utilities

**Files:**
- Create: `tda/core/masks.py`, `tests/test_masks.py`

**Interfaces:**

```python
# tda/core/masks.py  (all masks are np.ndarray bool HxW unless stated)
def encode_rle(mask: np.ndarray) -> dict            # COCO RLE with str counts
def decode_rle(rle: dict) -> np.ndarray
def bbox(mask) -> tuple[int,int,int,int] | None     # x0,y0,x1,y1 inclusive-exclusive
def min_side(mask) -> int
def area(mask) -> int
def fill_holes(mask) -> np.ndarray
def remove_small_components(mask, min_px: int) -> np.ndarray
def tolerant_sym_diff(a, b, tol_px: int = 2) -> int  # |(a XOR b) minus dilate(boundary(a)|boundary(b), tol_px)|
def is_conflict(old, new, area_frac=0.02, min_px=20, tol_px=2) -> bool   # tolerant_sym_diff > max(area_frac*area(old), min_px)
def mask_to_polygons(mask, tol: float = 1.0) -> list[list[float]]        # cv2 approxPolyDP, [x0,y0,x1,y1,...]
def polygons_to_mask(polys, hw) -> np.ndarray
def warp_mask(mask, sim: Similarity, hw_out) -> np.ndarray               # nearest-neighbour
def crop(mask, box) -> np.ndarray; def paste(dst, src, xy) -> None
def labelmap_from_masks(masks: dict[str, np.ndarray], order: list[str], hw) -> tuple[np.ndarray, dict[int,str]]  # uint16 labelmap, id→instance
```

- [ ] **Step 1: Write failing tests**: RLE roundtrip on random mask; `bbox`/`min_side` on a 5×9 rectangle; `fill_holes` fills a ring; `remove_small_components` removes a 3-px speck; `tolerant_sym_diff` of a mask vs itself shifted by 1 px == 0 and shifted by 5 px > 0; `is_conflict` false for 1-px shift on a 30×30 square, true for a 40% erosion; polygon roundtrip IoU ≥ 0.97 on a blob; `warp_mask` with identity is equal; `labelmap_from_masks` gives later instances higher ids.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** `feat(core): mask utilities`.

---

### Task 6: State machine (`states.py`)

**Files:**
- Create: `tda/core/states.py`, `tests/test_states.py`

**Interfaces:**

```python
# tda/core/states.py
@dataclass
class InstState:
    state: str
    placement: str

FrameState = dict[str, InstState]        # instance key → state at step k

def initial_state(instances: dict[str, InstanceRec], tax: Taxonomy) -> FrameState
def events_from_actions(instances, actions: list[ActionRec], tax: Taxonomy) -> list[StateEvent]
    # success actions → (attr,new) via tax.apply_verb; "remove" also emits placement in_chassis→on_bench;
    # parent removed → every child with attached=True gets state→removed & placement→on_bench (auto=True)
    # failed actions → no events
def state_at(instances, events: list[StateEvent], step: int, tax) -> FrameState
def needs_geom(instances, fs: FrameState, tax) -> dict[str, str]   # key → "mask"|"box" for instances needing geometry at this step
def diff_states(a: FrameState, b: FrameState) -> list[tuple[str,str,str,str]]   # (key, attr, old, new)
def validate_events(instances, events, tax) -> list[str]   # e.g. "screw.x: removed → fastened at step 12 is illegal"
```

Legal transitions (§6.2): screw fastened→loosened→removed (also fastened→removed); latch closed↔open; cover closed→open→removed; connector plugged→unplugged→removed; parts installed→displaced→removed and installed→removed; chassis fixed. Manual events may reverse (e.g. open→closed) but validation flags `removed → anything`.

- [ ] **Step 1: Write failing tests**: initial state has all defaults; captive screw `unscrew` → loosened; non-captive → removed + on_bench; cooler `remove` at step 13 → attached captive screws removed at 13 (auto event); `state_at(...,12)` still has cooler installed; `needs_geom` excludes unplugged connectors and includes on_bench psu as "box"; `validate_events` flags removed→fastened.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** `feat(core): state machine and geometry policy`.

---

### Task 7: Layer compiler (pure)

**Files:**
- Create: `tda/core/compiler.py`, `tests/test_compiler.py`

**Interfaces:**
- Consumes: T5 masks, T6 `FrameState`, T0 dataclasses.
- Produces:

```python
# tda/core/compiler.py
@dataclass
class CompiledInstance:
    instance: str
    visible: Optional[np.ndarray]     # None when no geometry (e.g. box-only or hidden)
    box: Optional[tuple]
    amodal: Optional[np.ndarray]
    occlusion_ratio: float
    visibility: str
    placement: str
    keyframe_id: Optional[int]

@dataclass
class CompiledFrame:
    key: FrameKey
    instances: dict[str, CompiledInstance]
    problems: list[str]              # "missing_shape:<key>" etc.
    input_hash: str

def select_keyframe(kfs: list[ShapeKeyframe], step: int) -> Optional[ShapeKeyframe]
    # among keyframes in the same (view, pose_segment): the one with the smallest anchor_step >= step
def above(a: tuple[str,str], b: tuple[str,str], zorder: ZOrderRec, overrides: list[PairOverride]) -> bool
def derive_visibility(visible, amodal, occlusion_ratio) -> str
    # occlusion_ratio<0.3→visible; 0.3–0.95→occluded_partial; ≥0.95→occluded_full; then min_side<6→too_small; <12→visible_tiny
def compile_frame(key: FrameKey, hw: tuple[int,int], needs: dict[str,str], keyframes: dict[str, list[ShapeKeyframe]],
                  zorder: ZOrderRec, overrides: list[PairOverride], occluders: list[OccluderMask],
                  frame_overrides: dict[str, FrameOverride], transform: Similarity, compiler_version: str = "1") -> CompiledFrame
```

Algorithm exactly as spec §3.3 (steps 3–7). `input_hash` = sha1 over sorted (keyframe id+version, zorder version, overrides, occluder rles, frame_override contents, transform, compiler_version). Bench instances (`placement=on_bench`) only interact with other bench instances in z-order.

- [ ] **Step 1: Write failing tests** on 64×64 synthetic masks:
  - `select_keyframe`: anchors [10, 20, 40] with step 15 → anchor 20; step 45 → None.
  - two overlapping squares A (bottom) and B (top): `visible[A] == A − B`; `occlusion_ratio[A] > 0`; with `PairOverride(above=A, below=B)` flips.
  - occluder mask removes pixels from both; `FrameOverride(visibility="out_of_view")` → visible None and visibility set.
  - missing keyframe for a required instance → `"missing_shape:<key>"` in problems.
  - bench instance and chassis instance overlapping in pixels do **not** occlude each other.
  - `derive_visibility` thresholds; a 4×4 blob → too_small.
  - identical inputs → identical `input_hash`; bumping a keyframe version changes it.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** `feat(core): layer compiler`.

---

### Task 8: Truth table refresh, verification and conflicts

**Files:**
- Create: `tda/core/truth.py`, `tests/test_truth.py`

**Interfaces:**
- Consumes: T4 `Db`, T7 `compile_frame`, T6 `state_at/needs_geom`, T5 `is_conflict`.
- Produces:

```python
# tda/core/truth.py
class TruthService:
    def __init__(self, db: Db, tax: Taxonomy): ...
    def compile(self, key: FrameKey) -> CompiledFrame              # gathers inputs from db, calls compile_frame
    def refresh(self, key: FrameKey) -> dict                       # updates auto rows; for verified rows: compares with is_conflict → add_conflict; returns {"updated":n,"conflicts":m,"problems":[...]}
    def refresh_range(self, desktop: int, view: str, steps: Iterable[int]) -> dict
    def verify_frame(self, key: FrameKey, annotator: str) -> None  # requires no problems; sets all rows status=verified, frame review_status="verified"
    def demote_frame(self, key: FrameKey, reason: str) -> None     # review_status="needs_review"
    def affected_steps(self, desktop: int, view: str, instance: str, keyframe: ShapeKeyframe) -> list[int]   # steps whose compile depends on this keyframe (for "affects N frames")
```

- [ ] **Step 1: Write failing tests** using an in-memory/tmp Db seeded with 1 desktop, 1 view, 3 steps, 2 instances, 1 keyframe each: `refresh` writes 3×2 auto rows; `verify_frame(step 2)` freezes; editing the keyframe (shift by 1 px) then `refresh` → no conflict, auto rows for steps 1,3 updated, step 2 unchanged; editing with a 40% erosion → 1 conflict for step 2, its row unchanged; adding a new instance to the verified frame → `demote_frame` called (review_status needs_review); `affected_steps` returns [1,2,3] for an anchor at step 3.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** `feat(core): truth table service with conflicts`.

---

### Task 9: Local cache, burst metrics, ROI suggestion

**Files:**
- Create: `tda/core/cache.py`, `tests/test_cache.py`

**Interfaces:**

```python
# tda/core/cache.py
def burst_metrics(paths: list[str]) -> list[dict]      # per image: mean, sat_frac, lap_var, dist_to_median (mean abs diff to the pixel-wise median of the burst, downscaled 8x)
def choose_scan_image(metrics: list[dict]) -> tuple[int, str]   # index, reason: P_0 unless (mean<20 or sat_frac>0.2 or dist_to_median > 2*median(dist)) → best remaining by lap_var
def cache_path(cache_dir, key: FrameKey, ext: str) -> str        # "<cache>/<view>/D13/s042.png"
def build_cache(index: dict[int, DesktopIndex], cache_dir: str, views=("scan",), desktops=None, progress=None) -> dict   # copies chosen images; writes <cache>/<view>/D13/manifest.json with per-step chosen index, metrics, reason
def suggest_roi(img: np.ndarray, view: str) -> tuple[int,int,int,int]  # scan: largest dark rectangular blob inside the yellow-tape frame (HSV yellow mask → inner region → dark object bbox, padded 3%); fallback central 70%
```

- [ ] **Step 1: Write failing tests** with synthetic bursts (10 identical images, one dark, one with a bright blob): `choose_scan_image` returns 0 normally; returns non-dark best when P_0 is dark; `suggest_roi` on a synthetic white image with a yellow tape square and a dark rectangle returns a box containing the dark rectangle; `cache_path` format.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (cv2). **Step 4: Run** → PASS. Then run `build_cache` for `scan` on all 66 desktops (≈2,800 PNGs, ~10 GB) → `D:\DataSet\cache\scan\`; report time and any failures.
- [ ] **Step 5: Commit** `feat(core): image cache and burst selection`.

---

### Task 10: SAM 2.1 service

**Files:**
- Create: `tda/models/__init__.py`, `tda/models/sam_service.py`, `tests/test_sam_service.py`

**Interfaces:**

```python
# tda/models/sam_service.py
@dataclass
class SamRequest:
    image_crop: np.ndarray                 # HxWx3 RGB uint8 (viewport crop, ≤1024 long side; caller scales)
    points: list[tuple[float,float,int]]   # (x,y,label) in crop coords; label 1=pos, 0=neg
    box: Optional[tuple[float,float,float,float]]
    mask_input: Optional[np.ndarray]       # bool HxW in crop coords (prior mask for local refinement)
    multimask: bool = False

@dataclass
class SamResult:
    mask: np.ndarray                       # bool HxW crop coords
    score: float
    ms: float

class SamService:
    def __init__(self, checkpoint: str, config: str = "configs/sam2.1/sam2.1_hiera_l.yaml", device="cuda"): ...
    def set_image(self, image_crop: np.ndarray) -> str       # returns image id; caches embedding until a different array is set
    def predict(self, req: SamRequest) -> SamResult
    @staticmethod
    def available() -> bool                                  # torch+cuda+sam2 importable and checkpoint exists

class SamWorker(QObject):   # in tda/ui later; here only the thread-safe queue wrapper without Qt:
class SamQueue:
    def __init__(self, service: SamService): ...             # background thread, single in-flight request, latest-wins for set_image
    def submit(self, req: SamRequest, cb: Callable[[SamResult], None]) -> None
    def stop(self) -> None
```

- Local refinement: when `mask_input` is given, feed it as SAM's `mask_input` (low-res logits from the prior mask) and only accept changes within a radius of 48 px around the new points; elsewhere keep the prior mask.

- [ ] **Step 1: Write tests** (skip if `not SamService.available()`): load service; `set_image` on scratchpad `scanner_crop_native.png`; point at (450,450) → mask area > 500 px and score > 0.5; `predict` with box covering the fan gives IoU > 0.5 with the point mask; `mask_input` refinement with one negative point reduces area but keeps ≥ 80% of pixels farther than 48 px from the point; `SamQueue.submit` returns result via callback within 5 s.
- [ ] **Step 2: Implement** using `sam2.build_sam.build_sam2` + `SAM2ImagePredictor`; checkpoint path from `paths.yaml: weights_dir` (`sam2.1_hiera_large.pt`, downloaded by the environment task).
- [ ] **Step 3: Run tests** → PASS on this workstation. **Step 4: Commit** `feat(models): SAM 2.1 service with local refinement`.

---

### Task 11: Canvas, overlay, tools, undo

**Files:**
- Create: `tda/ui/__init__.py`, `tda/ui/canvas/__init__.py`, `tda/ui/canvas/view.py`, `tda/ui/canvas/overlay.py`, `tda/ui/canvas/tools.py`, `tda/ui/commands.py`, `tests/test_canvas.py`

**Interfaces:**

```python
# tda/ui/canvas/overlay.py
class LabelOverlay:
    def __init__(self, hw: tuple[int,int]): ...
    labelmap: np.ndarray                  # uint16 HxW, 0 = background
    palette: dict[int, tuple[int,int,int]]
    def set_instances(self, masks: dict[str, np.ndarray], order: list[str]) -> None   # rebuild labelmap (bottom→top)
    def set_editing(self, instance: str, mask: np.ndarray) -> None                    # separate bool layer drawn on top
    def paint(self, xy: tuple[int,int], radius: int, add: bool) -> tuple[int,int,int,int]   # edits editing layer; returns dirty rect
    def qimage(self, rect=None, alpha=110, outline=True) -> QImage                    # ARGB32 of labelmap+editing (dirty-rect aware, cached)

# tda/ui/canvas/view.py
class ImageCanvas(QGraphicsView):
    sigMousePress = Signal(float, float, object)   # image coords, QMouseEvent
    sigMouseMove  = Signal(float, float, object)
    sigMouseRelease = Signal(float, float, object)
    def set_image(self, rgb: np.ndarray) -> None
    def set_overlay(self, overlay: LabelOverlay) -> None
    def zoom_to(self, box: tuple[int,int,int,int]) -> None
    def zoom_factor(self) -> float
    def viewport_image_rect(self) -> tuple[int,int,int,int]   # visible region in image coords (for SAM crops)
    def refresh(self, rect=None) -> None
    # wheel = zoom about cursor; space+drag or middle-drag = pan; pixel grid when zoom_factor()>4; minimap widget in corner

# tda/ui/canvas/tools.py
class Tool(QObject):  # base: on_press/on_move/on_release(x, y, ev) → None
class BrushTool(Tool):    # radius adjustable; paints editing layer; emits sigStroke(dirty_rect) on release
class EraserTool(Tool)
class SamPointTool(Tool): # left = positive, right = negative; collects points; calls SamQueue with viewport crop; result → editing layer; supports refine mode (mask_input = current editing mask)
class SamBoxTool(Tool)
class OccluderTool(Tool)  # paints into the frame occluder layer with a type selector

# tda/ui/commands.py
@dataclass
class Op: kind: str; payload: dict; inverse: dict
class UndoStack:
    def push(self, op: Op, apply: bool=True) -> None
    def undo(self) -> Op | None
    def redo(self) -> Op | None
    # kinds: "edit_editing_mask" (payload: instance, rle_before, rle_after), "set_zorder", "set_pair_override", "set_frame_override", "set_occluder", "commit_keyframe"
```

- [ ] **Step 1: Write tests** (offscreen): `LabelOverlay.set_instances` with two masks → labelmap ids 1,2 and `qimage()` non-null; `paint` returns a dirty rect containing the stroke; `ImageCanvas.set_image` then `zoom_to` box → `zoom_factor()` > 1 and `viewport_image_rect()` ≈ box; `UndoStack` push/undo/redo restores masks.
- [ ] **Step 2: Implement** (QGraphicsPixmapItem for image; a second pixmap item for the overlay, updated with dirty rects; `QImage` built from a preallocated ARGB buffer via numpy; palette from a fixed 64-color table, instance id → color stable across frames using a hash of the instance key).
- [ ] **Step 3: Run tests** → PASS. Manual check: run `python -m tda.ui.canvas.view --image D:/DataSet/cache/scan/D13/s010.png` and confirm smooth zoom/paint at 800%.
- [ ] **Step 4: Commit** `feat(ui): canvas, label overlay, brush/SAM tools, undo stack`.

---

### Task 12: Session, panels, task card, review mode

**Files:**
- Create: `tda/ui/session.py`, `tda/ui/panels/__init__.py`, `tda/ui/panels/timeline.py`, `tda/ui/panels/taskcard.py`, `tda/ui/panels/instances.py`, `tda/ui/panels/steptable.py`, `tda/ui/panels/review.py`, `tests/test_session.py`

**Interfaces:**

```python
# tda/ui/session.py
class AnnotationSession(QObject):
    sigFrameChanged = Signal(FrameKey); sigDirty = Signal(bool); sigProblems = Signal(list)
    def __init__(self, db: Db, tax: Taxonomy, truth: TruthService, cache_dir: str, annotator: str): ...
    def open(self, desktop: int, view: str) -> None           # loads index frames from db, pose segments, instances, events; sets step = last available frame
    def goto(self, step: int) -> None
    def prev(self) / next(self)                               # reverse-order default: "advance" = step-1
    def current(self) -> FrameKey
    def image(self) -> np.ndarray                             # cached image
    def compiled(self) -> CompiledFrame
    def task_card(self) -> list[dict]                         # for step k (going to k-1): changes from diff_states(state_at(k), state_at(k-1)) mapped to instructions:
                                                              # {"instance","kind": "add_shape"|"split_keyframe"|"state_only"|"remove_bench_box"|"confirm", "text": ...}
    def begin_edit(self, instance: str) -> None               # loads amodal shape (current keyframe warped) into editing layer
    def commit_edit(self, scope: str) -> None                 # scope: "keyframe"|"frame_override"|"split"; writes db, logs op, refreshes affected frames, emits problems
    def set_visibility(self, instance: str, vis: str) -> None
    def set_zorder_move(self, instance: str, above_of: str) -> None
    def confirm_frame(self) -> bool                           # verify_frame; False if problems
    def flash_compare(self) -> np.ndarray                     # returns image of step k-1 for Tab flash
    def save(self) -> None; def close(self) -> None
```

Panels (each a `QDockWidget` content):
- `TimelinePanel`: vertical thumbnails per step (from cache, 96 px), colors: grey unlabeled / yellow auto / green verified / red conflict-or-needs_review; click → `goto`.
- `TaskCardPanel`: list from `task_card()`, current item highlighted; Enter = commit current edit; Space = confirm frame (`confirm_frame()`; on False show problems).
- `InstanceListPanel`: instances of the current frame with class, state, visibility, placement; drag reorder writes z-order; keys `H` toggle hide, `V` cycle visibility, `1..7` set visibility values.
- `StepTablePanel` (S1): table of steps (thumbnail k−1 / k, raw name, parsed target, verb, tool, difficulty, notes); editable combo boxes; "Apply" writes steps/actions/instances and regenerates auto events; shows import issues.
- `ReviewPanel`: queue tabs (conflicts / needs_review / missing_shape / unexplained); Enter accept, `R` rework, opens the frame in the canvas.

- [ ] **Step 1: Write tests** (offscreen, tmp db seeded from the D13 fixture log + 3 synthetic frames): `open(13,"scan")` sets current step to the last frame; `task_card()` at step 13 lists `add_shape` for `cpu_cooler.01` and its attached screws when going to 12; `begin_edit`/`commit_edit("keyframe")` creates a keyframe with anchor = last step; `confirm_frame()` False when a required instance has no shape and True after adding it; `set_visibility` writes a FrameOverride.
- [ ] **Step 2: Implement** session + panels. Keyboard map (document in `docs/annotation_guide.md` §快捷键): `B` brush, `E` eraser, `S` SAM point, `X` SAM box, `O` occluder, `[`/`]` radius, `Enter` commit, `Space` confirm frame, `Tab` (hold) flash k−1, `Ctrl+Z/Y`, `Ctrl+K` split keyframe, `Alt+Enter` commit as frame override, `PgUp/PgDn` step, `F` fit ROI, `G` toggle pixel grid.
- [ ] **Step 3: Run tests** → PASS. **Step 4: Commit** `feat(ui): annotation session, timeline, task card, instances, step table, review`.

---

### Task 13: Main window, CLI, backup on exit

**Files:**
- Create: `tda/ui/app.py`, `tda/cli.py`, `tests/test_cli.py`

**Interfaces:**

```
python -m tda.cli build-index [--desktops 1-66]        → D:/DataSet/cache/index.json + report
python -m tda.cli import-logs [--desktops ...]         → db steps/actions/instances/events (+ issues md)
python -m tda.cli load-index                            → db frames/pose segments from index.json
python -m tda.cli build-cache --views scan [--desktops] → cache images + manifests
python -m tda.cli check --desktop 13 --view scan       → runs TruthService.refresh_range + checks, prints problems
python -m tda.cli backup                                → Db.backup(paths.backup_dir)
python -m tda.cli app --desktop 13 --view scan --annotator chang
```

- `app.py`: `QMainWindow` with top bar (desktop combo, view buttons 1–4, mode tabs: Steps / Annotate / Review), central `ImageCanvas`, docks for the panels, status bar (zoom %, frame status, SAM busy indicator). On start: `acquire_lock`; on close: `save()`, `backup()`, `release_lock()`. First open of a desktop/view: if no ROI stored → run `suggest_roi`, show an adjustable rectangle, store into `pose_segment.corners` placeholder `roi` field (a 4-tuple) and `zoom_to` it.

- [ ] **Step 1: Write tests**: `cli check` on the seeded tmp db returns exit 0 and prints "problems: 0"; `cli backup` creates a file in a tmp backup dir (override via `--paths tests/fixtures/paths_tmp.yaml`).
- [ ] **Step 2: Implement.** **Step 3: Run tests** → PASS; launch the app on D13 scan and annotate 3 frames end to end (screenshot to `docs/img/mvp_d13.png`).
- [ ] **Step 4: Commit** `feat(app): main window and CLI`.

---

### Task 14: Label Studio import as draft shapes

**Files:**
- Create: `tda/core/ls_import.py`, `tests/test_ls_import.py`, `tests/fixtures/ls_small.json` (2 tasks extracted from the real export)

**Interfaces:**

```python
# tda/core/ls_import.py
def parse_task_image(name: str) -> tuple[int, int, str] | None   # "3e99d760-Rs_13_42.png" → (13, 42, "rs"); "…Front_OAK_13_42.jpg" → oak1; "…Side_OAK_…" → oak2; "…33_39.png" → scan
def ls_polygon_to_mask(points_pct: list[list[float]], hw) -> np.ndarray     # LS percent coords → mask
def import_ls_export(path: str, db: Db, tax: Taxonomy, index: dict[int, DesktopIndex]) -> dict
    # for each annotated task: map view/step; for each polygon result with label L: map L to (cls, attrs) via configs/ls_label_map.yaml (33 old labels → new taxonomy; "(open)/(closed)" → state attr);
    # create ShapeKeyframe(source="labelstudio", anchor_step=step, instance=<provisional key "ls:<label>#<n>">) — provisional keys are resolved in S1 by the annotator;
    # also store text fields (seq_name_value, task_desc_value, target_desc_value, ...) into step.notes as "LS: ..."; return counts
def ls_reference_masks(path: str, view: str) -> Iterable[tuple[FrameKey, str, np.ndarray]]   # for model evaluation (Plan B)
```

- [ ] **Step 1: Write tests**: `parse_task_image` for the three patterns; polygon → mask area sanity; import of the 2-task fixture creates keyframes with `source == "labelstudio"`.
- [ ] **Step 2: Implement** (+ `configs/ls_label_map.yaml` covering all 33 old labels from the old config in the spec's history: e.g. `"RAM Module Retention Clip (open)"` → `ram_latch` state open; `"Storage Drive Retention Bracket"` → `drive_cage`; `"Screw Locker (Big Screw)"` → `screw` role other; `"Frame Screw (NRS)"` → `screw` role other).
- [ ] **Step 3: Run tests** → PASS; run the importer on the full export and report counts per view/desktop.
- [ ] **Step 4: Commit** `feat(core): import Label Studio annotations as drafts`.

---

### Task 15: Rehearsal exports (COCO + minimal VLM JSONL)

**Files:**
- Create: `tda/core/export/__init__.py`, `tda/core/export/coco.py`, `tda/core/export/vlm.py`, `tests/test_export.py`

**Interfaces:**

```python
# tda/core/export/coco.py
def export_coco(db: Db, tax: Taxonomy, desktops: list[int], view: str, out_json: str, only_verified=True, roi_crop=False) -> dict
    # images: one per frame with a compiled row; annotations: segmentation RLE, bbox, area, category_id (23 classes), attributes {state, placement, visibility, occlusion_ratio, amodal_complete, implied, tier: "gold"|"silver"|"bronze" (the VIEW's standard, from configs/taxonomy.yaml's view_tiers), verified: bool (a human confirmed this row)}
# tda/core/export/vlm.py
def export_vlm(db, tax, desktops, view, out_jsonl, tasks=("V1","V2","V3")) -> dict
    # V1: {"images":[frame], "q": "List all visible components with boxes", "a": {...}}
    # V2: per instance state questions ("Is <x> open/closed?", "How many motherboard screws remain fastened?")
    # V3: pairs (k-1, k): "What action was just performed?" → {verb, target_class, tool}
    # each record: id, task, desktop, step, view, images, question, answer (JSON), evidence (instance keys + boxes)
```

- [ ] **Step 1: Write tests** on the seeded tmp db: COCO file validates with `pycocotools.coco.COCO`; category count 23; VLM JSONL has ≥1 record per task with parseable `answer`.
- [ ] **Step 2: Implement.** **Step 3: Run tests** → PASS.
- [ ] **Step 4: Rehearsal (with the user):** after D13 scan is annotated, run `export-coco` and `export-vlm`, then a 50-question zero-shot check with one VLM API (script `scripts/eval_zero_shot.py`, model configurable); write findings to `docs/rehearsal-2026-09-30.md`.
- [ ] **Step 5: Commit** `feat(export): COCO and minimal VLM exports for rehearsal`.

---

## Self-review

**Spec coverage:** §2.2 index (T3), §2.3 logs (T2), §2.4 cache/burst/ROI (T9, T13), §3.1 entities (T0, T4), §3.2 identity ordinal (T2), §3.3 compiler (T7), §3.4 truth (T8), §3.5 storage/backup/lock (T4, T13), §4.2 reverse-order task card (T12), §4.3 edit scopes (T12 `commit_edit` scope + `Ctrl+K`, `Alt+Enter`), §4.4 review (T12), §4.5 layout (T13), §4.6 MVP tools (T11), §5.1 SAM (T10), §6 taxonomy (T1), §8.1/8.2 minimal exports (T15), §9.1 checks partially (T8 problems + `cli check`; full checks in Plan B), Label Studio drafts (T14). **Not in this plan (Plan B/C):** registration RANSAC (§2.5 P1), cross-view homography and OAK/RS flows, constraints module (§7), YOLO training loop (§5.2), polygon vertex editing, full auto-checks (§9.1), full VLM task set.

**Placeholder scan:** no TBD/TODO; every code step has interfaces or code; UI tasks specify behavior and tests.

**Type consistency:** `FrameKey(desktop, step, view)` everywhere; `ShapeKeyframe.anchor_step` used by `select_keyframe`; `Db.put_compiled` fields match `CompiledInstance`; `needs_geom` returns `"mask"|"box"` consumed by `compile_frame(needs=...)`; `TruthService.refresh` returns problems consumed by `AnnotationSession.confirm_frame`.
