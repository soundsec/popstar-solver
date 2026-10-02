"""拓扑量的三个对照实验。不改 beam 的默认权重。

1. ``ΔE_release`` 在夹心盘上是否分开「会砸出大团」和「什么也没发生」，
   以及把它临时加进 beam 排序后，小棋盘分数变不变。
2. ``E_formed``、``E_latent``、``Φ_agg`` 对小棋盘真实 ``V_max`` 的相关系数。
3. 同一批棋盘上 ``V_max``（含清盘奖励）和 ``D = V_min``（只计 group 分）差多远。

用法::

    python -m bench.bench_topology
    python -m bench.bench_topology --quick
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import time
from typing import Dict, List, Sequence, Tuple

from popstar.board import apply_move, create_board, get_legal_moves, random_board
from popstar.exhaustive import solve_exhaustive
from popstar.freeze import exact_minimum
from popstar.scoring import DEFAULT_SCORING
from popstar.solver import BeamSolver, group_score_ceiling
from popstar.topology import (
    aggregation_potential,
    aggregation_release,
    formed_energy,
    latent_energy,
)

SANDWICH = create_board([
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
])


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _samples(quick: bool) -> List[Tuple[int, int, int, int]]:
    """(边长, 色数, 种子, 重复数)。6×6 只留 2 色，避免穷举失控。"""
    plan = [
        (4, 2, 1, 6),
        (4, 3, 11, 6),
        (4, 4, 21, 4),
        (5, 2, 31, 4),
        (5, 3, 41, 2),
        (6, 2, 51, 2),
    ]
    if quick:
        plan = [(4, 2, 1, 3), (4, 3, 11, 2), (5, 2, 31, 1)]
    out = []
    for size, colors, seed, count in plan:
        for offset in range(count):
            out.append((size, colors, seed + offset, 1))
    return out


def experiment_release(quick: bool) -> None:
    print("=== 1. aggregation_release ===")
    blue = next(move for move in get_legal_moves(SANDWICH) if move.color == 1)
    red = next(move for move in get_legal_moves(SANDWICH) if move.color == 0)
    print(
        f"  夹心盘 消蓝 release={aggregation_release(SANDWICH, apply_move(SANDWICH, blue), blue.size):.0f}"
        f"  消红 release={aggregation_release(SANDWICH, apply_move(SANDWICH, red), red.size):.0f}"
    )
    print("  下面比较 beam 默认排序，和临时加上 release 权重（默认权重不变）。")
    width = 16 if quick else 32
    seeds = (1, 2) if quick else (1, 2, 3, 4)
    base_scores = []
    rel_scores = []
    for seed in seeds:
        state = random_board(5, 5, 3, random.Random(seed))
        base = BeamSolver(beam_width=width).solve(state)
        boosted = BeamSolver(beam_width=width, weight_release=1.0).solve(state)
        base_scores.append(base.best_score)
        rel_scores.append(boosted.best_score)
        print(
            f"  seed {seed}: 默认 {base.best_score:.0f} ({base.elapsed_time:.3f}s)"
            f"  +release {boosted.best_score:.0f} ({boosted.elapsed_time:.3f}s)"
        )
    print(
        f"  均值 默认 {sum(base_scores)/len(base_scores):.1f}"
        f"  +release {sum(rel_scores)/len(rel_scores):.1f}"
    )


def _evaluate(size: int, colors: int, seed: int) -> Dict[str, float]:
    state = random_board(size, colors, colors, random.Random(seed))
    started = time.perf_counter()
    maximum, _path = solve_exhaustive(state)
    frozen = exact_minimum(state)
    from popstar.board import color_counts
    counts = list(color_counts(state).values())
    u0 = group_score_ceiling(counts, DEFAULT_SCORING.score_group) + DEFAULT_SCORING.score_terminal(0)
    return {
        "vmax": maximum,
        "vmin": frozen.score,
        "formed": formed_energy(state),
        "latent": latent_energy(state),
        "phi": aggregation_potential(state),
        "singletons": float(sum(1 for n in counts if n == 1)),
        "u0": u0,
        "seconds": time.perf_counter() - started,
        "proven": 1.0 if frozen.proven else 0.0,
    }


def experiment_correlation_and_gap(quick: bool) -> None:
    print("\n=== 2–3. 小棋盘 ground truth：相关性与 Vmax / Vmin ===")
    print("  Vmax = 穷举最高分（含清盘奖励）。Vmin = D(S)，只累加 g(k)，不加奖励。")
    rows = []
    for size, colors, seed, _count in _samples(quick):
        row = _evaluate(size, colors, seed)
        row["size"] = size
        row["colors"] = colors
        rows.append(row)
        print(
            f"  {size}x{size}/{colors}色 seed {seed}: "
            f"Vmax={row['vmax']:.0f}  Vmin={row['vmin']:.0f}  "
            f"gap={row['vmax'] - row['vmin']:.0f}  "
            f"Φ={row['phi']:.1f}  {row['seconds']:.2f}s"
        )
    names = ("formed", "latent", "phi", "singletons")
    targets = [row["vmax"] for row in rows]
    print("  与 Vmax 的 Pearson：")
    for name in names:
        coef = _pearson([row[name] for row in rows], targets)
        print(f"    {name:12} {coef:+.3f}")
    gaps = [row["vmax"] - row["vmin"] for row in rows]
    print(
        f"  gap 均值 {sum(gaps)/len(gaps):.1f}"
        f"  最小 {min(gaps):.1f}  最大 {max(gaps):.1f}"
        f"  全部精确冻结 {all(row['proven'] for row in rows)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="拓扑量的三个对照实验")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    experiment_release(args.quick)
    experiment_correlation_and_gap(args.quick)


if __name__ == "__main__":
    main()
