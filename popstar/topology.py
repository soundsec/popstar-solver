"""规范化棋盘的拓扑表示，以及三种互不混用的聚集量。

二维棋盘在重力、压列之后可以写成列词::

    S = (W_1, ..., W_m)

``W_j`` 是第 j 个非空列**自下而上**的颜色序列。空列不出现在这个元组里。
由此得到两个次序不变量，后面的拓扑论证都从这里出发：

* **column order invariant**：非空列的左右次序不变，只会整列消失。
* **within-column order invariant**：同一列里幸存方块的上下次序不变。

状态转移仍只有 :func:`popstar.board.apply_move`。这里不重写重力。

三个能量不要再混称「势能」：

* ``E_formed``：已经连成的团值 ``Σ g(s_i)``（``s_i < 2`` 记 0），就是搜索里的 available。
* ``E_latent``：还没兑现的聚团上界 ``Σ_c [g(n_c) − Σ_i g(s_{c,i})]``，即原来的 merge。
* ``ΔE_release(a)``：动作 a 在扣掉被消掉的那一团之后，坠落和压列**额外**形成的团值。

``Φ_agg`` 是启发式聚集势，用少数同色候选边上的 ``Δ_ij / (1 + b_ij)`` 求和。
它不是可采纳上界，不能拿去剪枝。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .board import EMPTY, BoardState, Move, color_counts, get_components
from .scoring import DEFAULT_SCORING, ScoringConfig

ColumnWord = Tuple[int, ...]
ColumnWords = Tuple[ColumnWord, ...]


def column_words(state: BoardState) -> ColumnWords:
    """非空列自下而上的颜色词。右侧补齐用的空列会被丢掉。"""
    height, width = state.height, state.width
    words: List[ColumnWord] = []
    for col in range(width):
        letters: List[int] = []
        for row in range(height - 1, -1, -1):
            value = state.grid[row][col]
            if value == EMPTY:
                break
            letters.append(value)
        if letters:
            words.append(tuple(letters))
    return tuple(words)


def _is_subsequence(small: Sequence[int], big: Sequence[int]) -> bool:
    index = 0
    for value in big:
        if index < len(small) and small[index] == value:
            index += 1
    return index == len(small)


def column_order_holds(before: ColumnWords, after: ColumnWords) -> bool:
    """``after`` 是否能由 ``before`` 经「列内删字母、删掉若干整列」得到。"""
    cursor = 0
    for word in after:
        matched = False
        while cursor < len(before):
            if _is_subsequence(word, before[cursor]):
                matched = True
                cursor += 1
                break
            cursor += 1
        if not matched:
            return False
    return True


def _g(scoring: ScoringConfig, size: int) -> float:
    if size < 2:
        return 0.0
    return float(scoring.score_group(size))


def formed_energy(state: BoardState, scoring: Optional[ScoringConfig] = None) -> float:
    """``E_formed(S) = Σ g(s_i)``。孤立块贡献 0。"""
    scoring = scoring or DEFAULT_SCORING
    total = 0.0
    for comp in get_components(state):
        total += _g(scoring, comp.size)
    return total


def latent_energy(state: BoardState, scoring: Optional[ScoringConfig] = None) -> float:
    """``E_latent(S) = Σ_c [g(n_c) − Σ_i g(s_{c,i})]``。"""
    scoring = scoring or DEFAULT_SCORING
    counts: Dict[int, int] = {}
    formed_by_color: Dict[int, float] = {}
    for comp in get_components(state):
        counts[comp.color] = counts.get(comp.color, 0) + comp.size
        formed_by_color[comp.color] = formed_by_color.get(comp.color, 0.0) + _g(scoring, comp.size)
    total = 0.0
    for color, count in counts.items():
        if count >= 2:
            total += _g(scoring, count) - formed_by_color.get(color, 0.0)
    return total


def release_from_formed(formed_before: float, formed_after: float, removed_score: float) -> float:
    """``E_formed(S') − (E_formed(S) − g(k_a))``。"""
    return formed_after - (formed_before - removed_score)


def aggregation_release(
    before: BoardState,
    after: BoardState,
    removed_size: int,
    scoring: Optional[ScoringConfig] = None,
) -> float:
    """动作实际释放的聚集价值。``removed_size`` 是被消掉的那一团的大小。"""
    scoring = scoring or DEFAULT_SCORING
    return release_from_formed(
        formed_energy(before, scoring),
        formed_energy(after, scoring),
        _g(scoring, removed_size),
    )


@dataclass(frozen=True)
class ClusterNode:
    """一个同色连通块，当作聚集图里的节点。"""

    index: int
    color: int
    size: int
    col_lo: int
    col_hi: int
    row_lo: int
    row_hi: int


def cluster_nodes(state: BoardState) -> Tuple[ClusterNode, ...]:
    nodes: List[ClusterNode] = []
    for index, comp in enumerate(get_components(state)):
        rows = [row for row, _col in comp.cells]
        cols = [col for _row, col in comp.cells]
        nodes.append(ClusterNode(
            index=index,
            color=comp.color,
            size=comp.size,
            col_lo=min(cols),
            col_hi=max(cols),
            row_lo=min(rows),
            row_hi=max(rows),
        ))
    return tuple(nodes)


def _columns_between(left: ClusterNode, right: ClusterNode) -> int:
    if left.col_hi < right.col_lo:
        return right.col_lo - left.col_hi - 1
    if right.col_hi < left.col_lo:
        return left.col_lo - right.col_hi - 1
    return 0


def _row_gap(a: ClusterNode, b: ClusterNode) -> int:
    if a.row_hi < b.row_lo:
        return b.row_lo - a.row_hi - 1
    if b.row_hi < a.row_lo:
        return a.row_lo - b.row_hi - 1
    return 0


def _columns_overlap(a: ClusterNode, b: ClusterNode) -> bool:
    return not (a.col_hi < b.col_lo or b.col_hi < a.col_lo)


def _blocker_cells(state: BoardState, a: ClusterNode, b: ClusterNode) -> int:
    """两团接触路径上的异色格。第一版只数夹在中间的整列，或竖直缝里的重叠列。"""
    color = a.color
    count = 0
    if a.col_hi < b.col_lo or b.col_hi < a.col_lo:
        lo = min(a.col_hi, b.col_hi) + 1
        hi = max(a.col_lo, b.col_lo)
        row_lo = min(a.row_lo, b.row_lo)
        row_hi = max(a.row_hi, b.row_hi)
        for col in range(lo, hi):
            for row in range(row_lo, row_hi + 1):
                value = state.grid[row][col]
                if value != EMPTY and value != color:
                    count += 1
        return count
    gap_lo = min(a.row_hi, b.row_hi) + 1
    gap_hi = max(a.row_lo, b.row_lo)
    col_lo = max(a.col_lo, b.col_lo)
    col_hi = min(a.col_hi, b.col_hi)
    for row in range(gap_lo, gap_hi):
        for col in range(col_lo, col_hi + 1):
            value = state.grid[row][col]
            if value != EMPTY and value != color:
                count += 1
    return count


def barrier_proxy(
    state: BoardState,
    a: ClusterNode,
    b: ClusterNode,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """``b_ij = α d_col + β d_height + γ m_blocker``。不是可证明的势垒。"""
    d_col = _columns_between(a, b)
    d_height = _row_gap(a, b)
    blockers = _blocker_cells(state, a, b)
    return alpha * d_col + beta * d_height + gamma * blockers


def _pair_gain(scoring: ScoringConfig, left: int, right: int) -> float:
    return _g(scoring, left + right) - _g(scoring, left) - _g(scoring, right)


def _candidate_partners(nodes: Sequence[ClusterNode]) -> List[Tuple[int, int]]:
    """每个团最多三条同色边：最近左、最近右、列重叠时最近的上下邻居。边不定向、不重复。"""
    by_color: Dict[int, List[ClusterNode]] = {}
    for node in nodes:
        by_color.setdefault(node.color, []).append(node)
    seen = set()
    edges: List[Tuple[int, int]] = []

    def add(i: int, j: int) -> None:
        if i == j:
            return
        key = (i, j) if i < j else (j, i)
        if key in seen:
            return
        seen.add(key)
        edges.append(key)

    for group in by_color.values():
        ordered = sorted(group, key=lambda node: (node.col_lo, node.row_lo, node.index))
        for node in ordered:
            best_left: Optional[ClusterNode] = None
            best_right: Optional[ClusterNode] = None
            best_vert: Optional[ClusterNode] = None
            vert_gap = 10**9
            for other in ordered:
                if other.index == node.index:
                    continue
                if other.col_hi < node.col_lo:
                    if best_left is None or other.col_hi > best_left.col_hi:
                        best_left = other
                elif other.col_lo > node.col_hi:
                    if best_right is None or other.col_lo < best_right.col_lo:
                        best_right = other
                elif _columns_overlap(node, other):
                    gap = _row_gap(node, other)
                    if gap < vert_gap:
                        vert_gap = gap
                        best_vert = other
            if best_left is not None:
                add(node.index, best_left.index)
            if best_right is not None:
                add(node.index, best_right.index)
            if best_vert is not None:
                add(node.index, best_vert.index)
    return edges


def aggregation_potential(
    state: BoardState,
    scoring: Optional[ScoringConfig] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """``Φ_agg(S)``。只加候选边，不做同色团的全连接。"""
    scoring = scoring or DEFAULT_SCORING
    nodes = cluster_nodes(state)
    if len(nodes) < 2:
        return 0.0
    by_index = {node.index: node for node in nodes}
    total = 0.0
    for i, j in _candidate_partners(nodes):
        a, b = by_index[i], by_index[j]
        gain = _pair_gain(scoring, a.size, b.size)
        if gain <= 0:
            continue
        barrier = barrier_proxy(state, a, b, alpha, beta, gamma)
        total += gain / (1.0 + barrier)
    return total


def dead_column_ceiling(state: BoardState, scoring: Optional[ScoringConfig] = None) -> float:
    """用「死色列永远清空不了」把同色块切成左右互不相通的几段。

    某种颜色全盘只剩 1 块时，它所在的列永远不会变成空列，左右次序又不能交换。
    另一种颜色若在这列里一格都没有，列两侧的块就永远拼不成一团。
    上界从 ``g(n)`` 收成各段 ``g(n_i)`` 之和。没有死色时与 ``Σ g(n_c)`` 相同。
    这是可证明的上界，不是 ``Φ_agg``。
    """
    scoring = scoring or DEFAULT_SCORING
    counts = color_counts(state)
    dead = {color for color, count in counts.items() if count == 1}
    columns: List[Dict[int, int]] = []
    for col in range(state.width):
        present: Dict[int, int] = {}
        for row in range(state.height):
            value = state.grid[row][col]
            if value == EMPTY:
                continue
            present[value] = present.get(value, 0) + 1
        if present:
            columns.append(present)

    total = 0.0
    for color, count in counts.items():
        if count < 2:
            continue
        segments = [0]
        for present in columns:
            permanent = any(present.get(color_id, 0) > 0 for color_id in dead)
            if permanent and present.get(color, 0) == 0:
                if segments[-1] != 0:
                    segments.append(0)
            else:
                segments[-1] += present.get(color, 0)
        for size in segments:
            if size >= 2:
                total += scoring.score_group(size)
    return total


def describe_move_release(
    state: BoardState,
    move: Move,
    scoring: Optional[ScoringConfig] = None,
) -> float:
    """走一步之后的 ``ΔE_release``。转移仍走 :func:`apply_move`。"""
    from .board import apply_move

    scoring = scoring or DEFAULT_SCORING
    child = apply_move(state, move, validate=False)
    return aggregation_release(state, child, move.size, scoring)


__all__ = [
    "ColumnWord",
    "ColumnWords",
    "ClusterNode",
    "aggregation_potential",
    "aggregation_release",
    "barrier_proxy",
    "cluster_nodes",
    "column_order_holds",
    "column_words",
    "dead_column_ceiling",
    "describe_move_release",
    "formed_energy",
    "latent_energy",
    "release_from_formed",
]
