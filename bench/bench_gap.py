"""6×6 及以上：把 group 分差和清盘奖励拆开，并看死色列上界收紧了多少。

每行：

* ``G*``：最高分那条路径的 group 分
* ``D``：冻结到终局的最小 group 分
* ``G*-D``：group 分本身的策略宽度
* ``B``：那条最高分路径的清盘奖励
* ``U``：忽略几何的 ``Σ g(n_c)``
* ``U_dead``：死色列把同色块切开之后的上界
* ``U-G*``：上界还剩多少空隙

用法::

    python -m bench.bench_gap
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from popstar.board import color_counts, random_board
from popstar.freeze import exact_minimum
from popstar.scoring import DEFAULT_SCORING
from popstar.solver import ExactSolver, group_score_ceiling
from popstar.topology import dead_column_ceiling


JOBS = (
    (6, 2, 4, 8.0),
    (6, 3, 3, 12.0),
    (6, 4, 2, 12.0),
    (7, 2, 3, 15.0),
    (7, 3, 2, 15.0),
    (8, 2, 2, 15.0),
)


def main() -> None:
    header = (
        f"{'N':>2} {'M':>2} {'seed':>5} {'ok':>3} "
        f"{'G*':>7} {'D':>7} {'G*-D':>7} {'B':>6} {'R':>3} "
        f"{'U':>7} {'Udead':>7} {'U-G*':>7} {'cut':>6} {'sec':>6}"
    )
    print(header)
    buckets: dict = {}
    for size, colors, count, limit in JOBS:
        key = (size, colors)
        buckets[key] = []
        for index in range(count):
            seed = 2000 + size * 50 + colors * 10 + index
            state = random_board(size, size, colors, random.Random(seed))
            solved = ExactSolver(
                use_memo=True, use_bound=True, ordering="size", time_limit=limit
            ).solve(state)
            frozen = exact_minimum(state, node_budget=400_000)
            counts = list(color_counts(state).values())
            upper = group_score_ceiling(counts, DEFAULT_SCORING.score_group)
            tight = dead_column_ceiling(state)
            group_gap = solved.group_score - frozen.score if frozen.proven else float("nan")
            row = {
                "proven": solved.is_proven_optimal and frozen.proven,
                "g": solved.group_score,
                "d": frozen.score,
                "gap": group_gap,
                "bonus": solved.terminal_bonus,
                "upper": upper,
                "tight": tight,
                "slack": upper - solved.group_score,
                "cut": upper - tight,
            }
            buckets[key].append(row)
            flag = "Y" if row["proven"] else "n"
            gap_text = f"{group_gap:7.0f}" if group_gap == group_gap else "    nan"
            print(
                f"{size:2d} {colors:2d} {seed:5d} {flag:>3} "
                f"{solved.group_score:7.0f} {frozen.score:7.0f} {gap_text} "
                f"{solved.terminal_bonus:6.0f} {solved.remaining_count:3d} "
                f"{upper:7.0f} {tight:7.0f} {upper - solved.group_score:7.0f} "
                f"{upper - tight:6.0f} {solved.elapsed_time + frozen.elapsed:6.1f}",
                flush=True,
            )
    print()
    print(f"{'N':>2} {'M':>2} {'n':>3} {'G*-D':>8} {'U-G*':>8} {'cut':>8} {'proven':>7}")
    for (size, colors), rows in buckets.items():
        done = [row for row in rows if row["proven"]]
        use = done or rows
        n = len(use)
        def avg(name: str) -> float:
            return sum(row[name] for row in use) / n
        print(
            f"{size:2d} {colors:2d} {n:3d} {avg('gap'):8.1f} {avg('slack'):8.1f} "
            f"{avg('cut'):8.1f} {len(done):3d}/{len(rows):<3d}"
        )


if __name__ == "__main__":
    main()
