"""最低代价冻结：把局面走到再也不能消除，并且途中 group 分之和最小。

.. code-block:: text

    D(S) = min_π Σ_t g(k_t)     （终局不再加清盘奖励）

    D(S) = 0                              若 S 已终局
    D(S) = min_a [ g(k_a) + D(T(S, a)) ]  否则

高分搜索想把团聚起来；这里想用尽量少的分数把同色邻接拆光。
两边以后共用 :mod:`popstar.topology` 的描述，但本模块先不改 ExactSolver。

冻结度（终局都是 0）：

* ``F0``：合法连通块个数
* ``F1``：这些块的格子数之和
* ``F2``：同色四邻接边的条数

第一版启发式是 ``(F(S) − F(S')) / g(k)``，越大越优先。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .board import (
    EMPTY,
    BoardState,
    Move,
    apply_move,
    get_legal_moves,
    is_terminal,
)
from .scoring import DEFAULT_SCORING, ScoringConfig
from .topology import aggregation_potential

FreezeKey = Tuple[float, float, float]


@dataclass(frozen=True)
class FreezeResult:
    """``score`` 是 ``D(S)`` 或 beam 找到的最小 group 分之和。不含终局奖励。"""

    score: float
    moves: Tuple[Move, ...]
    expanded: int
    proven: bool
    elapsed: float


def freeze_degrees(state: BoardState) -> FreezeKey:
    """返回 ``(F0, F1, F2)``。"""
    legal = 0
    covered = 0
    for move in get_legal_moves(state):
        legal += 1
        covered += move.size
    adjacent = 0
    height, width = state.height, state.width
    grid = state.grid
    for row in range(height):
        for col in range(width):
            color = grid[row][col]
            if color == EMPTY:
                continue
            if col + 1 < width and grid[row][col + 1] == color:
                adjacent += 1
            if row + 1 < height and grid[row + 1][col] == color:
                adjacent += 1
    return float(legal), float(covered), float(adjacent)


def freeze_heuristic(
    before: BoardState,
    after: BoardState,
    removed_size: int,
    scoring: Optional[ScoringConfig] = None,
    kind: str = "F1",
    phi_weight: float = 0.0,
) -> float:
    """``H = (λ1 ΔF + λ2 ΔΦ) / g(k)``。``phi_weight = 0`` 时只用冻结度下降。"""
    scoring = scoring or DEFAULT_SCORING
    gain = scoring.score_group(removed_size)
    if gain <= 0:
        return 0.0
    index = {"F0": 0, "F1": 1, "F2": 2}[kind]
    drop = freeze_degrees(before)[index] - freeze_degrees(after)[index]
    if phi_weight:
        drop += phi_weight * (
            aggregation_potential(before, scoring) - aggregation_potential(after, scoring)
        )
    return drop / gain


def exact_minimum(
    state: BoardState,
    scoring: Optional[ScoringConfig] = None,
    node_budget: Optional[int] = None,
) -> FreezeResult:
    """记忆化精确求 ``D(S)``。超出 ``node_budget`` 时返回已完成终局里的最小值，并标成未证明。"""
    import time

    scoring = scoring or DEFAULT_SCORING
    started = time.perf_counter()
    memo: Dict[bytes, Tuple[float, Optional[Tuple[int, int]]]] = {}
    expanded = 0
    proven = True

    def solve(current: BoardState) -> float:
        nonlocal expanded, proven
        key = current.packed_key
        cached = memo.get(key)
        if cached is not None:
            return cached[0]
        if is_terminal(current):
            memo[key] = (0.0, None)
            return 0.0
        if node_budget is not None and expanded >= node_budget:
            proven = False
            return float("inf")
        best = float("inf")
        best_cell: Optional[Tuple[int, int]] = None
        aborted = False
        for move in get_legal_moves(current):
            if node_budget is not None and expanded >= node_budget:
                aborted = True
                proven = False
                break
            expanded += 1
            child = apply_move(current, move, validate=False)
            tail = solve(child)
            if tail == float("inf"):
                aborted = True
                continue
            cost = scoring.score_group(move.size) + tail
            if cost < best:
                best = cost
                best_cell = move.representative_cell
        if aborted or best_cell is None:
            # 不缓存。漏掉的分支可能更便宜，这个数不能当成 D(S)。
            proven = False
            return float("inf")
        memo[key] = (best, best_cell)
        return best

    score = solve(state)
    moves = _replay(state, memo) if proven else ()
    return FreezeResult(
        score=score,
        moves=moves,
        expanded=expanded,
        proven=proven,
        elapsed=time.perf_counter() - started,
    )


def _replay(
    state: BoardState,
    memo: Dict[bytes, Tuple[float, Optional[Tuple[int, int]]]],
) -> Tuple[Move, ...]:
    path: List[Move] = []
    current = state
    seen = set()
    while not is_terminal(current):
        key = current.packed_key
        if key in seen:
            break
        seen.add(key)
        stored = memo.get(key)
        if stored is None or stored[1] is None:
            break
        cell = stored[1]
        move = next(
            (item for item in get_legal_moves(current) if item.representative_cell == cell),
            None,
        )
        if move is None:
            break
        path.append(move)
        current = apply_move(current, move, validate=False)
    return tuple(path)


def beam_minimum(
    state: BoardState,
    scoring: Optional[ScoringConfig] = None,
    beam_width: int = 64,
    kind: str = "F1",
) -> FreezeResult:
    """用冻结度做排序的 beam。保留 ``accumulated + F`` 较小的状态。结果是 best found。"""
    import time

    scoring = scoring or DEFAULT_SCORING
    started = time.perf_counter()
    index = {"F0": 0, "F1": 1, "F2": 2}[kind]
    frontier: List[Tuple[BoardState, float, Tuple[Move, ...]]] = [(state, 0.0, ())]
    best = float("inf")
    best_path: Tuple[Move, ...] = ()
    expanded = 0
    if is_terminal(state):
        return FreezeResult(0.0, (), 0, False, 0.0)

    while frontier:
        candidates: Dict[bytes, Tuple[BoardState, float, Tuple[Move, ...], float]] = {}
        for current, accumulated, path in frontier:
            if is_terminal(current):
                if accumulated < best:
                    best = accumulated
                    best_path = path
                continue
            parent_f = freeze_degrees(current)[index]
            for move in get_legal_moves(current):
                expanded += 1
                child = apply_move(current, move, validate=False)
                acc = accumulated + scoring.score_group(move.size)
                if is_terminal(child) and acc < best:
                    best = acc
                    best_path = path + (move,)
                child_f = freeze_degrees(child)[index]
                rank = acc + child_f
                # 同样代价下，冻结度下降更多的排在前面（rank 更小）
                rank -= max(0.0, parent_f - child_f) / scoring.score_group(move.size)
                key = child.packed_key
                previous = candidates.get(key)
                if previous is not None and previous[1] <= acc:
                    continue
                candidates[key] = (child, acc, path + (move,), rank)
        if not candidates:
            break
        ordered = sorted(candidates.values(), key=lambda item: item[3])
        frontier = [(item[0], item[1], item[2]) for item in ordered[:beam_width]]
    return FreezeResult(
        score=best if best < float("inf") else 0.0,
        moves=best_path,
        expanded=expanded,
        proven=False,
        elapsed=time.perf_counter() - started,
    )


__all__ = [
    "FreezeResult",
    "beam_minimum",
    "exact_minimum",
    "freeze_degrees",
    "freeze_heuristic",
]
