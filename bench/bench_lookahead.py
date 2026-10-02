"""深度上界能收掉多少 ``U - V``，以及把它放进精确搜索会不会少展开节点。

``U^(0)`` 是现有乐观上界。``U^(d)`` 先真实走 ``d`` 步，再套上界。
``V`` 是证完的最高分，含清盘奖励。默认搜索仍是 ``bound_depth = 0``。

用法::

    python -m bench.bench_lookahead
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from popstar.board import random_board
from popstar.solver import ExactSolver, lookahead_bound


ROOT_JOBS = (
    (6, 2, 4),
    (6, 3, 3),
    (6, 4, 2),
    (7, 2, 3),
    (7, 3, 2),
    (8, 2, 2),
)
DEPTHS = (0, 1, 2, 3)
SEARCH_JOBS = (
    (6, 2, 0),
    (6, 3, 0),
    (6, 4, 0),
    (7, 2, 0),
)


def _seed(size: int, colors: int, index: int) -> int:
    return 2000 + size * 50 + colors * 10 + index


def main() -> None:
    print(
        f"{'N':>2} {'M':>2} {'seed':>5} {'V':>7} "
        f"{'U0':>7} {'U1':>7} {'U2':>7} {'U3':>7} "
        f"{'s0':>6} {'s1':>6} {'s2':>6} {'s3':>6}",
        flush=True,
    )
    buckets: dict = {}
    for size, colors, count in ROOT_JOBS:
        buckets[(size, colors)] = []
        for index in range(count):
            seed = _seed(size, colors, index)
            state = random_board(size, size, colors, random.Random(seed))
            solved = ExactSolver(use_memo=True, use_bound=True, ordering="size").solve(state)
            if not solved.is_proven_optimal:
                print(f"{size:2d} {colors:2d} {seed:5d}  unproven", flush=True)
                continue
            bounds = [lookahead_bound(state, depth=depth) for depth in DEPTHS]
            slacks = [bound - solved.best_score for bound in bounds]
            buckets[(size, colors)].append(slacks)
            print(
                f"{size:2d} {colors:2d} {seed:5d} {solved.best_score:7.0f} "
                f"{bounds[0]:7.0f} {bounds[1]:7.0f} {bounds[2]:7.0f} {bounds[3]:7.0f} "
                f"{slacks[0]:6.0f} {slacks[1]:6.0f} {slacks[2]:6.0f} {slacks[3]:6.0f}",
                flush=True,
            )
    print()
    print(f"{'N':>2} {'M':>2} {'n':>3} {'s0':>8} {'s1':>8} {'s2':>8} {'s3':>8}")
    for (size, colors), rows in buckets.items():
        if not rows:
            continue
        means = [sum(row[i] for row in rows) / len(rows) for i in range(4)]
        print(
            f"{size:2d} {colors:2d} {len(rows):3d} "
            f"{means[0]:8.1f} {means[1]:8.1f} {means[2]:8.1f} {means[3]:8.1f}"
        )

    print()
    print(f"{'N':>2} {'M':>2} {'d':>2} {'nodes':>8} {'sec':>7} {'V':>7}")
    for size, colors, index in SEARCH_JOBS:
        seed = _seed(size, colors, index)
        state = random_board(size, size, colors, random.Random(seed))
        for depth in (0, 1, 2):
            solved = ExactSolver(
                use_memo=True,
                use_bound=True,
                ordering="size",
                bound_depth=depth,
            ).solve(state)
            flag = "" if solved.is_proven_optimal else " ?"
            print(
                f"{size:2d} {colors:2d} {depth:2d} {solved.states_expanded:8d} "
                f"{solved.elapsed_time:7.2f} {solved.best_score:7.0f}{flag}",
                flush=True,
            )


if __name__ == "__main__":
    main()
