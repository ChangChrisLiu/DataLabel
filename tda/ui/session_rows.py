"""What the instance list and the canvas overlay show for one frame.

Both answer the same question -- in which order do these instances stack, and
what is visible of each -- and both have to answer it the way the *compiler*
did.  The stored z-order is only one of the compiler's inputs: a
``PairOverride`` can reverse a pair (spec 4.3 改层级) and bench parts are
composited apart from chassis parts (spec 3.3 step 5), so reading the order back
out of the record would show a stacking the pixels contradict -- and dragging a
row would look like it did nothing.
"""
from __future__ import annotations

import numpy as np

from tda.core.compiler import CompiledFrame
from tda.core.model import InstanceRec, Placement
from tda.core.states import FrameState

__all__ = ["instance_rows", "overlay_layers", "paint_order", "painted"]

IN_CHASSIS = Placement.IN_CHASSIS.value
ON_BENCH = Placement.ON_BENCH.value


def painted(compiled: CompiledFrame) -> list[str]:
    """The instances the compiler actually composited, bottom-up.

    The two placement groups are concatenated chassis-first: they never occlude
    each other, so any order between them is a presentation choice, and a part
    lying on the bench is in front of the machine it came out of.
    """
    return list(compiled.painted.get(IN_CHASSIS, [])) + list(
        compiled.painted.get(ON_BENCH, [])
    )


def paint_order(compiled: CompiledFrame) -> list[str]:
    """Every instance of the frame bottom-to-top; the unpainted ones on top.

    A bench box and a missing shape are in no layer group at all, so they have
    no painted position; they are listed above everything, which is where a
    shape drawn for them would go.
    """
    order = painted(compiled)
    return order + sorted(set(compiled.instances) - set(order))


def instance_rows(compiled: CompiledFrame, state: FrameState,
                  instances: dict[str, InstanceRec], hidden: set[str]) -> list[dict]:
    """``{"key","cls","state","placement","visibility","z","hidden"}``, top first.

    ``z`` counts from the bottom, so the top-most row carries the highest one and
    the list reads the way the layers are stacked on screen.
    """
    bottom_up = paint_order(compiled)
    rows = []
    for z, instance in enumerate(bottom_up):
        inst = compiled.instances[instance]
        rec = instances.get(instance)
        held = state.get(instance)
        rows.append({
            "key": instance,
            "cls": "" if rec is None else rec.cls,
            "state": "" if held is None else held.state,
            "placement": inst.placement,
            "visibility": inst.visibility,
            "z": z,
            "hidden": instance in hidden,
        })
    rows.reverse()
    return rows


def overlay_layers(compiled: CompiledFrame,
                   hidden: set[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Visible masks and paint order for the canvas overlay (spec 10.2).

    What :meth:`tda.ui.canvas.overlay.LabelOverlay.set_instances` wants:
    ``{instance: visible mask}`` and the bottom-up order to paint them in, with
    the instances the annotator has hidden left out entirely.
    """
    layers: dict[str, np.ndarray] = {}
    order: list[str] = []
    for instance in painted(compiled):
        if instance in hidden:
            continue
        visible = compiled.instances[instance].visible
        if visible is None:
            continue
        layers[instance] = visible
        order.append(instance)
    return layers, order
