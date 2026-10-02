"""完全穷举参考实现（ground truth）。

故意做成最朴素的形式：**无缓存、无剪枝、无排序**，纯递归枚举全部动作序列。
它存在的唯一目的是给优化后的求解器提供对拍基准（开发提纲第 28 节，见 ``SOURCES.md``）：
Memoized / B&B / 排序 / 颜色规范化后的最优分必须与它完全一致。

不要在正式求解中使用它——复杂度为动作序列数的阶乘级。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .board import BoardState, Move, apply_move, count_remaining, get_legal_moves, is_terminal
from .scoring import DEFAULT_SCORING, ScoringConfig


def solve_exhaustive(
    state: BoardState, scoring: Optional[ScoringConfig] = None
) -> Tuple[float, Tuple[Move, ...]]:
    """返回 ``(最优总分, 一条最优动作序列)``。

    动作序列中的 ``Move`` 是**当时状态下**的连通块对象，可依次 ``apply_move`` 复现。
    """
    scoring = scoring or DEFAULT_SCORING
    best_score = float("-inf")
    best_path: List[Move] = []
    visited_terminals = 0

    def recurse(current: BoardState, accumulated: float, path: List[Move]) -> None:
        nonlocal best_score, best_path, visited_terminals
        if is_terminal(current):
            visited_terminals += 1
            total = accumulated + scoring.score_terminal(count_remaining(current))
            if total > best_score:
                best_score = total
                best_path = list(path)
            return
        for move in get_legal_moves(current):
            child = apply_move(current, move)
            path.append(move)
            recurse(child, accumulated + scoring.score_group(move.size), path)
            path.pop()

    recurse(state, 0.0, [])
    return best_score, tuple(best_path)


def exhaustive_stats(
    state: BoardState, scoring: Optional[ScoringConfig] = None
) -> Tuple[float, Tuple[Move, ...], int]:
    """同 :func:`solve_exhaustive`，额外返回访问到的终局数量（复杂度参考）。"""
    scoring = scoring or DEFAULT_SCORING
    best_score = float("-inf")
    best_path: List[Move] = []
    visited_terminals = 0

    def recurse(current: BoardState, accumulated: float, path: List[Move]) -> None:
        nonlocal best_score, best_path, visited_terminals
        if is_terminal(current):
            visited_terminals += 1
            total = accumulated + scoring.score_terminal(count_remaining(current))
            if total > best_score:
                best_score = total
                best_path = list(path)
            return
        for move in get_legal_moves(current):
            child = apply_move(current, move)
            path.append(move)
            recurse(child, accumulated + scoring.score_group(move.size), path)
            path.pop()

    recurse(state, 0.0, [])
    return best_score, tuple(best_path), visited_terminals


__all__ = ["solve_exhaustive", "exhaustive_stats"]
