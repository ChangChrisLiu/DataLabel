-- TDA SQLite schema (schema_version = 1).
-- One table per entity of design spec section 3.1. Every JSON column is TEXT
-- holding json.dumps(..., ensure_ascii=False); RLE dicts are stored as JSON.
-- All statements are IF NOT EXISTS so that Db.__init__ stays idempotent.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS desktop (
    id               INTEGER PRIMARY KEY,
    brand            TEXT,
    model_family     TEXT,
    chassis_platform TEXT,
    chassis_type     TEXT,
    size             TEXT,
    date             TEXT,
    split            TEXT,
    notes            TEXT,
    meta_json        TEXT
);

CREATE TABLE IF NOT EXISTS frame (
    desktop               INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    step                  INTEGER NOT NULL,
    view                  TEXT    NOT NULL,
    path                  TEXT,
    aux_json              TEXT,
    ts                    TEXT,
    hand_or_tool_in_frame INTEGER,
    in_progress           INTEGER,
    image_quality         TEXT,
    missing               INTEGER,
    bench_annotated       INTEGER,
    review_status         TEXT DEFAULT 'unlabeled',
    burst_json            TEXT,
    pose_segment          INTEGER,
    PRIMARY KEY (desktop, step, view)
);

CREATE TABLE IF NOT EXISTS step (
    desktop    INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    step       INTEGER NOT NULL,
    step_type  TEXT    NOT NULL,
    raw_name   TEXT,
    dupli      INTEGER NOT NULL DEFAULT 0,
    notes      TEXT    NOT NULL DEFAULT '',
    duration_s REAL,
    PRIMARY KEY (desktop, step)
);

CREATE TABLE IF NOT EXISTS action (
    desktop        INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    step           INTEGER NOT NULL,
    idx            INTEGER NOT NULL,
    target         TEXT    NOT NULL,
    verb           TEXT    NOT NULL,
    tool           TEXT    NOT NULL DEFAULT 'none',
    direction      TEXT    NOT NULL DEFAULT 'none',
    result         TEXT    NOT NULL DEFAULT 'success',
    failure_reason TEXT,
    difficulty     INTEGER,
    PRIMARY KEY (desktop, step, idx)
);

CREATE TABLE IF NOT EXISTS instance (
    desktop           INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    key               TEXT    NOT NULL,
    cls               TEXT    NOT NULL,
    attrs_json        TEXT,
    parent            TEXT,
    attached          INTEGER NOT NULL DEFAULT 0,
    mounted_on        TEXT,
    fastens           TEXT,
    socket_host       TEXT,
    cable             TEXT,
    slot_id           TEXT,
    group_id          TEXT,
    group_order       TEXT    NOT NULL DEFAULT 'unordered',
    removal_direction TEXT,
    raw_names_json    TEXT,
    PRIMARY KEY (desktop, key)
);

CREATE TABLE IF NOT EXISTS state_event (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    desktop       INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    step          INTEGER NOT NULL,
    target        TEXT    NOT NULL,
    attr          TEXT    NOT NULL,
    old           TEXT,
    new           TEXT,
    evidence_view TEXT,
    auto          INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_state_event_desktop ON state_event(desktop, step);

CREATE TABLE IF NOT EXISTS pose_segment (
    desktop         INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    view            TEXT    NOT NULL,
    seg             INTEGER NOT NULL,
    start_step      INTEGER,
    end_step        INTEGER,
    ref_step        INTEGER,
    corners_json    TEXT,
    homography_json TEXT,
    roi_json        TEXT,
    PRIMARY KEY (desktop, view, seg)
);

CREATE TABLE IF NOT EXISTS frame_transform (
    desktop INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    step    INTEGER NOT NULL,
    view    TEXT    NOT NULL,
    scale   REAL    NOT NULL DEFAULT 1.0,
    theta   REAL    NOT NULL DEFAULT 0.0,
    tx      REAL    NOT NULL DEFAULT 0.0,
    ty      REAL    NOT NULL DEFAULT 0.0,
    PRIMARY KEY (desktop, step, view)
);

CREATE TABLE IF NOT EXISTS shape_keyframe (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    desktop          INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    view             TEXT    NOT NULL,
    instance         TEXT    NOT NULL,
    pose_segment     INTEGER NOT NULL DEFAULT 0,
    anchor_step      INTEGER NOT NULL,
    placement        TEXT    NOT NULL DEFAULT 'in_chassis',
    geom_type        TEXT    NOT NULL DEFAULT 'mask',
    amodal_complete  INTEGER NOT NULL DEFAULT 1,
    source           TEXT    NOT NULL DEFAULT 'manual',
    draft_id         INTEGER,
    version          INTEGER NOT NULL DEFAULT 1,
    edit_count       INTEGER NOT NULL DEFAULT 0,
    edit_time_ms     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_keyframe_lookup
    ON shape_keyframe(desktop, view, instance, pose_segment, anchor_step);

CREATE TABLE IF NOT EXISTS shape_part (
    keyframe_id INTEGER NOT NULL REFERENCES shape_keyframe(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    rle_json    TEXT,
    box_json    TEXT,
    PRIMARY KEY (keyframe_id, idx)
);

CREATE TABLE IF NOT EXISTS zorder (
    desktop      INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    view         TEXT    NOT NULL,
    pose_segment INTEGER NOT NULL,
    order_json   TEXT    NOT NULL DEFAULT '[]',
    version      INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (desktop, view, pose_segment)
);

CREATE TABLE IF NOT EXISTS pair_override (
    desktop      INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    view         TEXT    NOT NULL,
    pose_segment INTEGER NOT NULL,
    above        TEXT    NOT NULL,
    below        TEXT    NOT NULL,
    PRIMARY KEY (desktop, view, pose_segment, above, below)
);

CREATE TABLE IF NOT EXISTS occluder_mask (
    desktop       INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    step          INTEGER NOT NULL,
    view          TEXT    NOT NULL,
    occluder_type TEXT    NOT NULL,
    rle_json      TEXT,
    PRIMARY KEY (desktop, step, view, occluder_type)
);

CREATE TABLE IF NOT EXISTS frame_override (
    desktop         INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    step            INTEGER NOT NULL,
    view            TEXT    NOT NULL,
    instance        TEXT    NOT NULL,
    visible_rle_json TEXT,
    visibility      TEXT,
    PRIMARY KEY (desktop, step, view, instance)
);

CREATE TABLE IF NOT EXISTS instance_frame_flags (
    desktop               INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    step                  INTEGER NOT NULL,
    view                  TEXT    NOT NULL,
    instance              TEXT    NOT NULL,
    difficulty_flags_json TEXT,
    PRIMARY KEY (desktop, step, view, instance)
);

CREATE TABLE IF NOT EXISTS compiled_mask (
    desktop          INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    step             INTEGER NOT NULL,
    view             TEXT    NOT NULL,
    instance         TEXT    NOT NULL,
    visible_rle_json TEXT,
    occlusion_ratio  REAL,
    visibility       TEXT,
    placement        TEXT,
    status           TEXT    NOT NULL DEFAULT 'auto',
    input_hash       TEXT,
    verified_by      TEXT,
    verified_at      TEXT,
    -- schema_version 2: box rows (a bench part) keep their rectangle here and
    -- leave visible_rle_json NULL; mask rows do the opposite.
    geom_type        TEXT    NOT NULL DEFAULT 'mask',
    box_json         TEXT,
    PRIMARY KEY (desktop, step, view, instance)
);

CREATE TABLE IF NOT EXISTS conflict (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    desktop      INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    step         INTEGER NOT NULL,
    view         TEXT    NOT NULL,
    instance     TEXT    NOT NULL,
    old_rle_json TEXT,
    new_rle_json TEXT,
    sym_diff_px  INTEGER,
    status       TEXT    NOT NULL DEFAULT 'open',
    resolution   TEXT,
    created_at   TEXT,
    resolved_at  TEXT
);
CREATE INDEX IF NOT EXISTS ix_conflict_open ON conflict(desktop, view, status);

CREATE TABLE IF NOT EXISTS relation (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    desktop       INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    type          TEXT    NOT NULL,
    target        TEXT    NOT NULL,
    blocker       TEXT    NOT NULL,
    necessity     TEXT    NOT NULL DEFAULT 'required',
    mode          TEXT,
    reason        TEXT,
    source        TEXT    NOT NULL DEFAULT 'manual',
    evidence_step INTEGER,
    status        TEXT    NOT NULL DEFAULT 'active'
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_relation_edge
    ON relation(desktop, type, target, blocker);

CREATE TABLE IF NOT EXISTS op_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    desktop      INTEGER NOT NULL REFERENCES desktop(id) ON DELETE CASCADE,
    view         TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    payload_json TEXT,
    inverse_json TEXT,
    annotator    TEXT,
    ts           TEXT
);
CREATE INDEX IF NOT EXISTS ix_op_log_scope ON op_log(desktop, view, id);
