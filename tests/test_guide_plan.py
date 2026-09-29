"""The "现在做什么 / What now" panel's reasoning, without any Qt (task U2b).

The second trial ended with "on the last frame I don't know what the other four
tasks are" and a double-click that "did nothing" -- it had started an edit,
and the only thing on screen that said so was one word in the status bar.  The
guide names the phase the annotator is in and the exact next key or button;
these tests pin one phase per state the ruling lists.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from tda.ui import guide as G
from tda.ui import session_api as api

COVER = G.CardItem("cover.01", api.KIND_ADD_SHAPE, False, "导风罩 cover.01")
LATCH = G.CardItem("psu_latch.01", api.KIND_ADD_SHAPE, False, "电源卡扣 psu_latch.01")
DONE = G.CardItem("chassis", api.KIND_ADD_SHAPE, True, "机箱 chassis")

#: A fresh start frame: ROI stored, nothing selected, two parts left to draw.
FRESH = G.GuideFacts(mode="annotate", step=42, neighbour=None, roi_stored=True,
                     items=(DONE, COVER, LATCH))


def now_step(plan: G.GuidePlan) -> int:
    """Index of the highlighted step of the numbered list, or -1."""
    states = [state for state, _text in plan.steps]
    return states.index(G.NOW) if G.NOW in states else -1


def test_nothing_selected_asks_for_a_click_on_the_first_open_item():
    plan = G.plan_for(FRESH)
    assert plan.phase == G.PHASE_PICK
    assert "单击" in plan.now and "导风罩 cover.01" in plan.now
    assert "2" in plan.now                      # how many are left
    assert now_step(plan) == 1                  # ② pick a part
    assert plan.steps[0][0] == G.DONE           # ① the ROI is answered
    assert len(plan.steps) == 5


def test_the_start_frame_says_it_is_the_start_frame():
    assert "起点" in G.plan_for(FRESH).title
    middle = replace(FRESH, step=40, neighbour=41)
    assert "起点" not in G.plan_for(middle).title
    assert "41" in G.plan_for(middle).title


def test_an_unanswered_roi_rectangle_is_the_first_thing_to_do():
    plan = G.plan_for(replace(FRESH, roi_stored=False, roi_editing=True,
                              roi_unanswered=True))
    assert plan.phase == G.PHASE_ROI
    assert now_step(plan) == 0
    assert "Enter" in plan.now and "Esc" in plan.now
    assert plan.action == "commit"


def test_a_skipped_roi_question_stays_on_the_list_without_blocking():
    plan = G.plan_for(replace(FRESH, roi_stored=False, roi_unanswered=True))
    assert plan.phase == G.PHASE_PICK
    assert plan.steps[0][0] == G.WARN
    assert "确认建议框" in plan.steps[0][1]


def test_editing_an_empty_shape_names_the_sam_click():
    plan = G.plan_for(replace(FRESH, editing="导风罩 cover.01"))
    assert plan.phase == G.PHASE_DRAW_EMPTY
    assert now_step(plan) == 2
    assert "S" in plan.now and "X" in plan.now and "导风罩 cover.01" in plan.now
    assert plan.action == "tool_sam_point"


def test_without_sam_the_empty_shape_is_painted_with_the_brush():
    plan = G.plan_for(replace(FRESH, editing="导风罩 cover.01", sam_ready=False))
    assert plan.phase == G.PHASE_DRAW_EMPTY
    assert "B" in plan.now
    assert plan.action == "tool_brush"


def test_editing_with_pixels_asks_for_enter():
    plan = G.plan_for(replace(FRESH, editing="导风罩 cover.01", layer_pixels=True,
                              layer_dirty=True))
    assert plan.phase == G.PHASE_DRAW_PIXELS
    assert now_step(plan) == 3
    assert "Enter" in plan.now
    assert plan.action == "commit"


def test_the_area_warning_owns_the_next_enter():
    plan = G.plan_for(replace(FRESH, editing="x", layer_pixels=True,
                              layer_dirty=True, warning=True))
    assert plan.phase == G.PHASE_WARNING
    assert "再按一次 Enter" in plan.now and "Esc" in plan.now


def test_a_scope_suggestion_names_the_three_answers():
    plan = G.plan_for(replace(FRESH, editing="x", layer_pixels=True,
                              layer_dirty=True, scope="zorder:above:chassis"))
    assert plan.phase == G.PHASE_SCOPE
    for key in ("Enter", "Alt+Enter", "Ctrl+K"):
        assert key in plan.now


def test_a_draft_ghost_owns_enter_and_esc():
    plan = G.plan_for(replace(FRESH, editing="x", ghost=True))
    assert plan.phase == G.PHASE_GHOST
    assert "Enter" in plan.now and "Esc" in plan.now and "Shift+A" in plan.now


def test_an_armed_bench_box_asks_for_a_drag():
    plan = G.plan_for(replace(FRESH, bench="硬盘 storage_drive.ssd.01"))
    assert plan.phase == G.PHASE_BENCH
    assert "拖" in plan.now
    assert plan.action == "tool_bench_box"


def test_all_card_items_done_asks_for_space():
    plan = G.plan_for(replace(FRESH, items=(DONE,)))
    assert plan.phase == G.PHASE_CONFIRM
    assert now_step(plan) == 4
    assert "Space" in plan.now
    assert plan.action == "confirm"


def test_a_confirmed_frame_points_onwards():
    plan = G.plan_for(replace(FRESH, items=(DONE,), frame_confirmed=True))
    assert plan.phase == G.PHASE_CONFIRMED
    assert "PgDn" in plan.now
    assert now_step(plan) == -1


def test_review_mode_is_read_only():
    plan = G.plan_for(replace(FRESH, mode="review"))
    assert plan.phase == G.PHASE_REVIEW
    assert "只读" in plan.now
    for key in ("Enter", "R", "K", "N"):
        assert key in plan.now
    assert plan.steps == ()


def test_steps_mode_is_one_line_pointing_at_apply():
    plan = G.plan_for(replace(FRESH, mode="steps"))
    assert plan.phase == G.PHASE_STEPS
    assert "Apply" in plan.now and "Annotate" in plan.now
    assert plan.steps == ()


def test_a_frame_with_no_image_says_so():
    plan = G.plan_for(replace(FRESH, has_image=False, view="oak1"))
    assert plan.phase == G.PHASE_NO_IMAGE
    assert "oak1" in plan.now and "PgDn" in plan.now


def test_a_missing_raw_drive_outranks_everything_but_the_mode():
    plan = G.plan_for(replace(FRESH, has_image=False,
                              raw_missing="F: 盘找不到"))
    assert plan.phase == G.PHASE_RAW_MISSING
    assert "F: 盘找不到" in plan.now


def test_holding_tab_says_to_let_go():
    plan = G.plan_for(replace(FRESH, flashing=True))
    assert plan.phase == G.PHASE_FLASH
    assert "Tab" in plan.now


@pytest.mark.parametrize("facts", [
    FRESH,
    replace(FRESH, editing="x"),
    replace(FRESH, mode="review"),
    replace(FRESH, has_image=False),
])
def test_the_plan_is_deterministic_and_hashable(facts):
    """The panel repaints only when the plan changes; equal facts, equal plans."""
    assert G.plan_for(facts) == G.plan_for(facts)
    hash(G.plan_for(facts))
