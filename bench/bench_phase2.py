"""Phase 2 基准：正确性对拍 + 真实复杂度测量（开发提纲第 28 节，见 ``SOURCES.md``）。

用法::

    python -m bench.bench_phase2                 # 默认：对拍 + 5x5~8x8 + 10x10
    python -m bench.bench_phase2 --log runs/phase2.jsonl
    python -m bench.bench_phase2 --limit 10      # 缩短每个实例的时限

输出三段：

A. 与完全穷举对拍（3x3 / 4x4 / 5x5，2~4 色）——任何不一致都会直接报错退出。
B. 复杂度爬升（5x5 ~ 8x8，四色）——记录状态数、缓存命中、深度、耗时、峰值内存。
C. 10x10 四色——Fast(beam) 给出 best found，Exact 在时限内尽力证明。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import time
import tracemalloc
from typing import Any, Dict, List, Optional

from popstar.analysis import format_summary, summarize
from popstar.board import random_board
from popstar.exhaustive import solve_exhaustive
from popstar.records import RunLog, RunRecord
from popstar.solver import BeamSolver, ExactSolver

COLORS = 4
MEM_PROBE_NODES = 50_000
_probe_state = None  # 由 part_b 设置


def _fmt_row(label: str, solution, peak_mb: float) -> str:
    notes = solution.notes or {}
    return (
        f"{label:<26} score={solution.best_score:>8.0f} "
        f"proven={str(solution.is_proven_optimal):<5} "
        f"visit={solution.states_expanded:>8} hits={solution.cache_hits:>7} "
        f"pruned={notes.get('pruned_nodes', 0):>7} cache={solution.cache_size:>7} "
        f"depth={solution.max_depth:>3} "
        f"t={solution.elapsed_time:>6.2f}s mem={peak_mb:>5.1f}MB"
    )


def _measure(solver_call, label: str, log: Optional[RunLog], tag: str,
             mem_probe: Optional[Any] = None):
    """跑一次求解并打印。

    内存**不**在主跑次上用 tracemalloc 测量：tracemalloc 逐次跟踪分配，
    搜索越快、跟踪量越大，反而会成为瓶颈。改为在固定节点预算下单独探测。
    """
    solution = solver_call()
    peak_mb = 0.0
    if mem_probe is not None:
        peak_mb = _probe_memory(mem_probe)
    print(_fmt_row(label, solution, peak_mb))
    if log is not None:
        log.append(RunRecord.from_solution(solution, tag=tag))
    return solution


def _probe_memory(factory) -> float:
    """在固定节点预算下探测峰值内存（MB）。"""
    solver = factory()
    if hasattr(solver, "node_limit"):
        solver.node_limit = MEM_PROBE_NODES
        solver.time_limit = None
    tracemalloc.start()
    solver.solve(_probe_state)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


# ---------------------------------------------------------------------------
# A. 与穷举对拍
# ---------------------------------------------------------------------------


def part_a(count: int) -> None:
    print("\n" + "=" * 100)
    print("A. 与完全穷举对拍（任何不一致都会中止）")
    print("=" * 100)
    cases = (
        (3, 3, 2),
        (3, 3, 3),
        (3, 3, 4),
        (4, 4, 2),
        (4, 4, 3),
        (4, 4, 4),
        (5, 5, 2),
        (5, 5, 3),
    )
    total = 0
    for height, width, colors in cases:
        worst_ratio = 0.0
        for index in range(count):
            state = random_board(
                height, width, colors, random.Random(index * 7919 + height * 131 + colors)
            )
            truth, _ = solve_exhaustive(state)
            solution = ExactSolver(use_memo=True).solve(state)
            if abs(solution.best_score - truth) > 1e-9:
                raise AssertionError(
                    f"MISMATCH {height}x{width}/{colors} #{index}: "
                    f"solver={solution.best_score} exhaustive={truth}"
                )
            # 剪枝全开的版本也必须一致
            full = ExactSolver(
                use_memo=True, use_bound=True, ordering="heuristic", canonicalize=True
            ).solve(state)
            if abs(full.best_score - truth) > 1e-9:
                raise AssertionError(
                    f"MISMATCH(bound) {height}x{width}/{colors} #{index}: "
                    f"{full.best_score} != {truth}"
                )
            total += 1
            worst_ratio = max(worst_ratio, 1.0)
        print(f"  {height}x{width} / {colors} 色：{count} 例全部一致")
    print(f"  合计 {total} 例，全部与穷举一致 ✓")


# ---------------------------------------------------------------------------
# B. 复杂度爬升
# ---------------------------------------------------------------------------


def part_b(count: int, limit: float, log: Optional[RunLog]) -> None:
    print("\n" + "=" * 100)
    print(f"B. 复杂度爬升（四色，每实例时限 {limit:.0f}s）")
    print("=" * 100)
    configs = {
        "exact(memo)": lambda: ExactSolver(use_memo=True),
        "exact(memo+canon)": lambda: ExactSolver(use_memo=True, canonicalize=True),
        "exact(+bound)": lambda: ExactSolver(use_memo=True, use_bound=True),
        "fast(beam256)": lambda: BeamSolver(beam_width=256),
    }
    global _probe_state
    print(f"  （内存列 = 固定 {MEM_PROBE_NODES} 节点预算下的峰值，非全程峰值）")
    for size in (5, 6, 7, 8):
        state = random_board(size, size, COLORS, random.Random(size * 101))
        _probe_state = state
        print(f"\n  {size}x{size} 四色：")
        for name, factory in configs.items():
            solver = factory()
            if hasattr(solver, "time_limit"):
                solver.time_limit = limit
            _measure(
                lambda s=solver, st=state: s.solve(st),
                f"    {name}",
                log,
                tag=f"{size}x{size}",
                mem_probe=factory if hasattr(solver, "node_limit") else None,
            )


# ---------------------------------------------------------------------------
# C. 10x10
# ---------------------------------------------------------------------------


def part_c(limit: float, seed: int, log: Optional[RunLog]) -> None:
    print("\n" + "=" * 100)
    print(f"C. 10x10 四色（Exact 时限 {limit:.0f}s，先用 beam 取初始下界）")
    print("=" * 100)
    state = random_board(10, 10, COLORS, random.Random(seed))
    global _probe_state
    _probe_state = state

    beam = BeamSolver(beam_width=256).solve(state)
    print(_fmt_row("    fast(beam256)", beam, 0.0))
    if log is not None:
        log.append(RunRecord.from_solution(beam, tag="10x10"))

    # 第 25 节：Fast 的解作为 Exact 的初始下界
    path = [m.representative_cell for m in beam.move_sequence]
    exact = ExactSolver(use_memo=True, time_limit=limit)
    solution = exact.solve(state, lower_bound=beam.best_score, lower_bound_path=path)
    print(_fmt_row("    exact(beam-seeded)", solution, 0.0))
    if log is not None:
        log.append(RunRecord.from_solution(solution, tag="10x10"))
    print(
        f"\n  beam 得分 {beam.best_score:g}；exact 当前下界 {solution.best_score:g}"
        f"（proven={solution.is_proven_optimal}）"
    )
    print("  说明：10x10 状态空间远超当前上限，Fast 结果为 best found，尚不能宣称最优。")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Phase 2 求解器基准")
    parser.add_argument("--count", type=int, default=4, help="A 段每种规模的对拍实例数")
    parser.add_argument("--limit", type=float, default=20.0, help="每个实例的时限（秒）")
    parser.add_argument("--seed", type=int, default=2026, help="10x10 棋盘种子")
    parser.add_argument("--log", type=str, default=None, help="追加写入 JSONL 运行日志")
    parser.add_argument("--skip-a", action="store_true", help="跳过对拍段")
    args = parser.parse_args(argv)

    log = RunLog(args.log) if args.log else None

    if not args.skip_a:
        part_a(args.count)
    part_b(max(1, args.count // 2), args.limit, log)
    part_c(args.limit, args.seed, log)

    if log is not None:
        records = log.load()
        exact_runs = [r for r in records if r.mode == "exact"]
        print("\n" + "=" * 100)
        print(f"运行日志 {args.log}：共 {len(records)} 条（exact {len(exact_runs)} 条）")
        print("=" * 100)
        print(format_summary(summarize(records)))
        print("\n（group_score / terminal_bonus 已分开落盘，后续可直接重标定）")


if __name__ == "__main__":
    main()
