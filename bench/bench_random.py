"""随机棋盘上的策略宽度。只在穷举还走得动的规模上跑。

对每个 (N, M) 记录：

* ``V_max``：穷举最高分，含清盘奖励
* ``V_min``：冻结代价 ``D(S)``，只累加 ``g(k)``
* ``U_0``：``Σ g(n_c) + B(0)``，不含几何
* ``Γ = V_max − V_min``
* ``η = V_max / U_0``

``U_topo`` 还没有可证明的版本，这里不报 τ。

用法::

    python -m bench.bench_random
    python -m bench.bench_random --quick
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from typing import List

from popstar.board import color_counts, random_board
from popstar.exhaustive import solve_exhaustive
from popstar.freeze import exact_minimum
from popstar.scoring import DEFAULT_SCORING
from popstar.solver import group_score_ceiling


def _jobs(quick: bool) -> List[tuple]:
    jobs = [
        (4, 2, 8),
        (4, 3, 6),
        (4, 4, 4),
        (5, 2, 4),
        (5, 3, 2),
        (6, 2, 2),
    ]
    if quick:
        jobs = [(4, 2, 3), (4, 3, 2), (5, 2, 1)]
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="随机棋盘的 Vmax / Vmin / U0")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(f"{'N':>3} {'M':>3} {'n':>3} {'Vmax':>8} {'Vmin':>8} {'U0':>8} {'Γ':>8} {'η':>6}")
    for size, colors, count in _jobs(args.quick):
        vmax, vmin, u0, gamma, eta = [], [], [], [], []
        for index in range(count):
            state = random_board(size, size, colors, random.Random(1000 * size + 10 * colors + index))
            maximum, _path = solve_exhaustive(state)
            frozen = exact_minimum(state)
            ceiling = group_score_ceiling(
                list(color_counts(state).values()), DEFAULT_SCORING.score_group
            ) + DEFAULT_SCORING.score_terminal(0)
            vmax.append(maximum)
            vmin.append(frozen.score)
            u0.append(ceiling)
            gamma.append(maximum - frozen.score)
            eta.append(maximum / ceiling if ceiling else 0.0)
        n = len(vmax)
        print(
            f"{size:3d} {colors:3d} {n:3d} "
            f"{sum(vmax)/n:8.1f} {sum(vmin)/n:8.1f} {sum(u0)/n:8.1f} "
            f"{sum(gamma)/n:8.1f} {sum(eta)/n:6.3f}"
        )


if __name__ == "__main__":
    main()
