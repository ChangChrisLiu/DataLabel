"""Layer ordering for the compiler: the z-order, its overrides and their cycles.

Split out of :mod:`tda.core.compiler` to keep both files small; :func:`above`
is re-exported there, which is where callers should import it from.

A *layer* is one ``(instance_key, part_name)`` pair. Two things order them:
the global :class:`~tda.core.model.ZOrderRec` of the ``(view, pose_segment)``,
which is a total order of the layers it lists, and the
:class:`~tda.core.model.PairOverride` rows, which are pairwise and win over it
(spec 3.3 step 5).

:func:`above` answers for one pair, which is all the truth table needs.
:func:`group_order` sorts a whole group for painting: it treats the overrides
as hard edges and runs a stable topological sort that always takes the
available layer with the smallest global index, so the global order survives
wherever the overrides leave it a choice. Contradictory overrides form a cycle
that cannot be satisfied at all; those are reported and the group falls back to
the plain global order.
"""
from __future__ import annotations

import heapq

from tda.core.model import PairOverride, ZOrderRec

__all__ = ["LayerKey", "above", "group_order"]

#: one layer of the composite: ``(instance_key, part_name)``
LayerKey = tuple[str, str]


def _global_index(layer: LayerKey, zorder: ZOrderRec) -> int | None:
    """Position of ``layer`` in the global order, ``None`` when unlisted."""
    for idx, row in enumerate(zorder.order):
        if tuple(row) == tuple(layer):
            return idx
    return None


def above(
    a: LayerKey, b: LayerKey, zorder: ZOrderRec, overrides: list[PairOverride]
) -> bool:
    """Is layer ``a`` above layer ``b``?

    A :class:`~tda.core.model.PairOverride` matches on **instance keys**
    regardless of the part, and wins over the global order; the first matching
    override decides (two overrides contradicting each other are reported by
    :func:`~tda.core.compiler.compile_frame` as a ``zorder_cycle`` problem).
    Otherwise the later position in ``zorder.order`` is the one above, and a
    layer that is not listed at all counts as being above every listed one --
    two unlisted layers are unordered, so this returns ``False`` for both
    directions.

    ``zorder`` and ``overrides`` must already be the ones of this
    ``(view, pose_segment)``. Group isolation (bench vs chassis) is *not*
    decided here: it is a property of the frame, applied by
    :func:`~tda.core.compiler.compile_frame`.
    """
    if tuple(a) == tuple(b):
        return False
    for ov in overrides:
        if ov.above == ov.below:
            continue
        if ov.above == a[0] and ov.below == b[0]:
            return True
        if ov.above == b[0] and ov.below == a[0]:
            return False
    ia, ib = _global_index(a, zorder), _global_index(b, zorder)
    if ia is None and ib is None:
        return False
    if ia is None:
        return True
    if ib is None:
        return False
    return ia > ib


def _find_cycles(inst_edges: dict[str, set[str]]) -> list[str]:
    """One ``zorder_cycle:<instances>`` problem per cyclic component.

    Edges point from the lower instance to the higher one, so a strongly
    connected component of more than one node is a set of overrides that cannot
    all hold at once. Each such component is reported once, its instances
    sorted, e.g. ``zorder_cycle:A,B``; a node that merely points *into* a cycle
    is not part of it and is not named.

    This is Tarjan's algorithm, iterative (an explicit work stack rather than
    recursion) so that neither a deep override chain nor a component reached
    from several roots can disturb the traversal.
    """
    order: dict[str, int] = {}  # discovery index
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    scc_stack: list[str] = []
    problems: list[str] = []
    counter = 0

    def push(node: str, work: list) -> None:
        nonlocal counter
        order[node] = low[node] = counter
        counter += 1
        scc_stack.append(node)
        on_stack.add(node)
        work.append((node, iter(sorted(inst_edges.get(node, ())))))

    nodes = set(inst_edges) | {n for targets in inst_edges.values() for n in targets}
    for root in sorted(nodes):
        if root in order:
            continue
        work: list = []
        push(root, work)
        while work:
            node, successors = work[-1]
            descended = False
            for nxt in successors:
                if nxt not in order:
                    push(nxt, work)
                    descended = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], order[nxt])
            if descended:
                continue
            work.pop()
            if low[node] == order[node]:  # root of a component
                component: list[str] = []
                while True:
                    top = scc_stack.pop()
                    on_stack.discard(top)
                    component.append(top)
                    if top == node:
                        break
                if len(component) > 1:
                    problems.append("zorder_cycle:" + ",".join(sorted(component)))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return sorted(problems)


def group_order(
    layers: list[LayerKey], zorder: ZOrderRec, overrides: list[PairOverride]
) -> tuple[list[LayerKey], list[str]]:
    """Order one group's layers bottom-to-top, plus the problems found.

    ``layers`` is in append order (sorted instance, then part order), which is
    what decides the relative position of layers missing from the global order:
    they all go above the listed ones, in that append order, and each is
    reported as ``zorder_missing:<instance>/<part>``.
    """
    problems: list[str] = []
    listed = {tuple(row): idx for idx, row in enumerate(zorder.order)}
    index: dict[LayerKey, int] = {}
    unlisted = len(listed)
    for layer in layers:
        if layer in listed:
            index[layer] = listed[layer]
        else:
            index[layer] = unlisted
            unlisted += 1
            problems.append(f"zorder_missing:{layer[0]}/{layer[1]}")
    fallback = sorted(layers, key=lambda lk: index[lk])

    by_instance: dict[str, list[LayerKey]] = {}
    for layer in layers:
        by_instance.setdefault(layer[0], []).append(layer)

    edges: set[tuple[LayerKey, LayerKey]] = set()
    inst_edges: dict[str, set[str]] = {}
    for ov in overrides:
        if ov.above == ov.below:
            continue
        highs, lows = by_instance.get(ov.above), by_instance.get(ov.below)
        if not highs or not lows:
            continue
        inst_edges.setdefault(ov.below, set()).add(ov.above)
        for low in lows:
            for high in highs:
                if low != high:
                    edges.add((low, high))
    if not edges:
        return fallback, problems

    adjacency: dict[LayerKey, set[LayerKey]] = {lk: set() for lk in layers}
    indegree: dict[LayerKey, int] = {lk: 0 for lk in layers}
    for low, high in sorted(edges):
        if high not in adjacency[low]:
            adjacency[low].add(high)
            indegree[high] += 1

    heap = [(index[lk], lk) for lk in layers if indegree[lk] == 0]
    heapq.heapify(heap)
    ordered: list[LayerKey] = []
    while heap:
        _, layer = heapq.heappop(heap)
        ordered.append(layer)
        for nxt in sorted(adjacency[layer]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(heap, (index[nxt], nxt))

    if len(ordered) != len(layers):  # contradictory overrides
        problems.extend(_find_cycles(inst_edges))
        return fallback, problems
    return ordered, problems
