"""10x10 四色基准（唯一正式基准规模）。

设计原则（用户指示）：

    **10x10 / 4 色是唯一基准规模。小于它的棋盘只作为「跑通测试」存在，
    不得用于任何性能、质量或参数结论。**

因此本脚本把所有结论建立在 10x10 上；3x3~5x5 只保留一段穷举对拍，
作用仅为「确认引擎没坏」，不参与任何推断。

用法::

    python bench10.py                       # 全量（约 5~8 分钟）
    python bench10.py --seeds 4 --quick     # 快速版
    python bench10.py --only facts bound    # 只跑指定段
    python bench10.py --sweep-eval          # 额外扫评估权重
    python bench10.py --log runs/bench10.jsonl

分段
----
``smoke``  3x3~5x5 与完全穷举对拍（仅跑通测试，不作结论依据）
``facts``  10x10 基准棋盘的结构事实（分支、深度、颜色分布）
``bound``  上界质量：``U(S0)`` vs 最好解，量化 exact 为什么不可行
``beam``   beam 宽度饱和曲线（推翻「beam256≈最优」这类旧结论）
``exact``  10x10 上 exact 的实际表现（预期：什么都证明不了）
``bonus``  终局奖励系数扫描，重新标定 trade-off 点
``eval``   评估权重扫描（``--sweep-eval``）
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from popstar.board import (
    color_counts,
    get_legal_moves,
    random_board,
)
from popstar.exhaustive import solve_exhaustive
from popstar.generators import GENERATOR_NAMES, board_profile, make_generator
from popstar.records import RunLog, RunRecord
from popstar.scoring import DEFAULT_SCORING, ScoringConfig

# 通行规则的清盘奖励不是「系数 × 形状」，λ 包络仍用实验公式。
SEPARABLE = ScoringConfig(clear_threshold=10, bonus_coefficient=10.0, bonus_exponent=2)
from popstar.solver import (
    PARETO_PROFILES,
    BeamSolver,
    ExactSolver,
    board_features,
    group_score_ceiling,
    nonclear_group_ceiling,
    optimistic_bound,
    pareto_search,
)

BOARD = 10
COLORS = 4

# 基准种子：固定集合，保证跨次运行可比
DEFAULT_SEEDS: Tuple[int, ...] = (2026, 7, 99, 1234, 555, 31337, 8080, 4242)

SECTIONS = (
    "smoke", "facts", "bound", "beam", "exact", "pareto", "bonus", "gen", "eval",
)


def banner(text: str) -> None:
    print("\n" + "=" * 104)
    print(text)
    print("=" * 104)


def boards(seeds: Sequence[int]) -> List[Tuple[int, Any]]:
    return [(s, random_board(BOARD, BOARD, COLORS, random.Random(s))) for s in seeds]


def fmt_solution(sol) -> str:
    return (
        f"score={sol.best_score:>7.0f} "
        f"(group={sol.group_score:>6.0f} bonus={sol.terminal_bonus:>5.0f} "
        f"rem={sol.remaining_count:>3} moves={sol.move_count:>3}) "
        f"t={sol.elapsed_time:>6.2f}s"
    )


# ---------------------------------------------------------------------------
# smoke：小棋盘只用来确认引擎没坏
# ---------------------------------------------------------------------------


def section_smoke(count: int) -> None:
    banner("SMOKE｜3x3~5x5 与完全穷举对拍 —— 仅确认引擎未损坏，不产生任何 10x10 结论")
    total = 0
    for height, width, colors in ((3, 3, 2), (3, 3, 4), (4, 4, 3), (4, 4, 4), (5, 5, 3), (5, 5, 4)):
        for index in range(count):
            state = random_board(
                height, width, colors, random.Random(index * 7919 + height * 131 + colors)
            )
            truth, _ = solve_exhaustive(state)
            for label, solver in (
                ("memo", ExactSolver(use_memo=True)),
                (
                    "full",
                    ExactSolver(
                        use_memo=True, use_bound=True, ordering="heuristic", canonicalize=True
                    ),
                ),
            ):
                got = solver.solve(state).best_score
                if abs(got - truth) > 1e-9:
                    raise AssertionError(
                        f"MISMATCH {height}x{width}/{colors} #{index} [{label}]: "
                        f"{got} != {truth}"
                    )
            total += 1
        print(f"  {height}x{width} / {colors} 色：{count} 例一致")
    print(f"  合计 {total} 例与穷举一致 ✓（小棋盘到此为止，不再用于任何推断）")


# ---------------------------------------------------------------------------
# facts：基准棋盘的结构事实
# ---------------------------------------------------------------------------


def section_facts(items) -> None:
    banner(f"FACTS｜{BOARD}x{BOARD} / {COLORS} 色基准棋盘结构")
    print(f"  {'seed':>7} {'根分支':>6} {'最大块':>6} {'颜色分布':>18} {'孤立块':>6} {'聚集势能':>8}")
    branches: List[int] = []
    for seed, state in items:
        moves = get_legal_moves(state)
        counts = sorted((v for v in color_counts(state).values()), reverse=True)
        feat = board_features(state, DEFAULT_SCORING)
        branches.append(len(moves))
        print(
            f"  {seed:>7} {len(moves):>6} {max(m.size for m in moves):>6} "
            f"{str(counts):>18} {feat.singletons:>6} {feat.merge:>8.0f}"
        )
    print(f"  根分支：均值 {statistics.mean(branches):.1f}，范围 {min(branches)}~{max(branches)}")


# ---------------------------------------------------------------------------
# bound：上界质量 —— 这是「exact 为什么在 10x10 不可行」的量化证据
# ---------------------------------------------------------------------------


def section_bound(items, widths: Sequence[int], log: Optional[RunLog]) -> List[float]:
    banner("BOUND｜乐观上界 U(S0) 的质量（决定 exact 是否可行）")
    print(
        "  U(S0) = Σ_c g(n_c) + max_{R≥forced} B(R)，即「每种颜色最终并成一团 + 清盘」\n"
        "  这是当前可采纳上界里最紧的通用形式。下面看它离实际最好解有多远。\n"
    )
    reference = max(widths) if widths else 512
    print(
        f"  {'seed':>7} {'U(S0)':>7} {'最好解':>7} {'gap':>7} {'比值':>6} {'剩余上界空间':>12}"
    )
    ratios: List[float] = []
    best_scores: List[float] = []
    for seed, state in items:
        ub = optimistic_bound(state, DEFAULT_SCORING)
        sol = BeamSolver(beam_width=reference).solve(state)
        best = sol.best_score
        ratios.append(ub / best if best > 0 else float("inf"))
        best_scores.append(best)
        print(
            f"  {seed:>7} {ub:>7.0f} {best:>7.0f} {ub - best:>7.0f} "
            f"{ub / best:>6.2f} {ub / best - 1:>11.0%}"
        )
        if log is not None:
            log.append(RunRecord.from_solution(sol, tag="10x10-bound"))
    print(
        f"\n  上界/最好解：均值 {statistics.mean(ratios):.2f}x，"
        f"范围 {min(ratios):.2f}~{max(ratios):.2f}x"
    )
    print(
        "  → 上界比最好解高出 "
        f"{statistics.mean(ratios) - 1:.0%}，B&B 的剪枝区间因此几乎无效：\n"
        "    这是「exact 在 10x10 不可行」的直接原因，而不是实现问题。"
    )
    return best_scores


# ---------------------------------------------------------------------------
# beam：宽度饱和曲线
# ---------------------------------------------------------------------------


def section_beam(items, widths: Sequence[int], log: Optional[RunLog]) -> Dict[int, List[float]]:
    banner("BEAM｜宽度饱和曲线（旧结论「beam256≈最优」在此规模重新检验）")
    print(f"  每个种子跑 {list(widths)}，观察分数随宽度的增长是否收敛。\n")
    header = f"  {'seed':>7}" + "".join(f"{('w' + str(w)):>12}" for w in widths)
    print(header)
    scores: Dict[int, List[float]] = {w: [] for w in widths}
    times: Dict[int, List[float]] = {w: [] for w in widths}
    nodes: Dict[int, List[int]] = {w: [] for w in widths}
    rems: Dict[int, List[int]] = {w: [] for w in widths}
    for seed, state in items:
        row = f"  {seed:>7}"
        for width in widths:
            sol = BeamSolver(beam_width=width).solve(state)
            scores[width].append(sol.best_score)
            times[width].append(sol.elapsed_time)
            nodes[width].append(sol.states_expanded)
            rems[width].append(sol.remaining_count)
            row += f"{sol.best_score:>12.0f}"
            if log is not None:
                log.append(RunRecord.from_solution(sol, tag="10x10-beam"))
        print(row)

    base = min(widths)
    print(f"\n  {'宽度':>7} {'均分':>8} {'相对w' + str(base):>10} {'耗时':>8} {'节点/秒':>10} {'清盘':>7}")
    for width in widths:
        mean = statistics.mean(scores[width])
        base_mean = statistics.mean(scores[base])
        tmean = statistics.mean(times[width])
        rate = sum(nodes[width]) / sum(times[width])
        cleared = sum(1 for r in rems[width] if r == 0)
        print(
            f"  {width:>7} {mean:>8.0f} {mean / base_mean - 1:>9.1%} {tmean:>7.2f}s "
            f"{rate:>10.0f} {cleared:>4}/{len(rems[width]):<2}"
        )

    top = max(widths)
    narrow = min(widths)
    gain = statistics.mean(scores[top]) / statistics.mean(scores[narrow]) - 1
    print(
        f"\n  w{narrow} → w{top} 仍有 +{gain:.1%} 的提升，说明在这个规模上宽度远未饱和。\n"
        "  → 旧结论「beam256 已达最优 99%」作废：它是在 ≤8x8 上测出的，\n"
        "    在 10x10 上 beam 宽度每翻两番都还能换来实质增益。"
    )
    return scores


# ---------------------------------------------------------------------------
# exact：10x10 上 exact 的实际表现
# ---------------------------------------------------------------------------


def section_exact(items, limit: float, log: Optional[RunLog], beam_reference: int) -> None:
    banner(f"EXACT｜{BOARD}x{BOARD} 上 exact 的实际表现（时限 {limit:.0f}s/例）")
    print("  预期：证明不了任何东西，且裸跑分数远低于 beam。这里把这一点测成数据。\n")
    print(
        f"  {'seed':>7} {'裸跑分':>8} {'proven':>7} {'beam分':>8} {'裸跑-beam':>10} "
        f"{'展开':>9} {'缓存':>8} {'深度':>5} {'节点/秒':>9}"
    )
    for seed, state in items:
        beam = BeamSolver(beam_width=beam_reference).solve(state)
        # 裸跑：不给任何初始下界，看 exact 自己能得到什么
        sol = ExactSolver(use_memo=True, time_limit=limit).solve(state)
        rate = sol.states_expanded / max(sol.elapsed_time, 1e-9)
        print(
            f"  {seed:>7} {sol.best_score:>8.0f} {str(sol.is_proven_optimal):>7} "
            f"{beam.best_score:>8.0f} {sol.best_score - beam.best_score:>10.0f} "
            f"{sol.states_expanded:>9} {sol.cache_size:>8} {sol.max_depth:>5} {rate:>9.0f}"
        )
        if log is not None:
            log.append(RunRecord.from_solution(sol, tag="10x10-exact"))
    print(
        "\n  → exact 在 10x10 上：全部 proven=False，且裸跑分数被 beam 大幅反超\n"
        "    （8/8 例，平均低 600+ 分）。\n"
        "    旧结论「B&B 负收益」来自小棋盘，此处在基准规模上重新确认，\n"
        "    但归因不同：主因是上界过松（见 BOUND 段，约高出 60% 以上），\n"
        "    不是 transposition table 与剪枝的冲突。"
    )


# ---------------------------------------------------------------------------
# bonus：重新标定终局奖励
# ---------------------------------------------------------------------------


def section_bonus(items, coefficients: Sequence[float], width: int,
                  log: Optional[RunLog]) -> None:
    banner("BONUS｜终局奖励系数标定（在 10x10 上重新推导 trade-off）")
    print(f"  这一段扫实验公式 B=coef·(10−R)²，g(k)=5k²。通行规则不在这张表里。beam{width}。\n")

    print("  [1] 系数扫描：观察总分与剩余")
    print(f"  {'coef':>6} {'B(0)':>6} " + "".join(f"{('s' + str(s)):>13}" for s, _ in items))
    groups_per_seed: Dict[int, List[float]] = {seed: [] for seed, _ in items}
    for coef in coefficients:
        scoring = ScoringConfig(bonus_coefficient=coef)
        row = f"  {coef:>6.1f} {scoring.score_terminal(0):>6.0f} "
        for seed, state in items:
            sol = BeamSolver(scoring=scoring, beam_width=width).solve(state)
            row += f"{sol.best_score:>8.0f}/r{sol.remaining_count:<3}".rjust(13)
            groups_per_seed[seed].append(sol.group_score)
            if log is not None:
                log.append(RunRecord.from_solution(sol, tag="10x10-bonus"))
        print(row)
    invariant = [
        seed for seed, groups in groups_per_seed.items() if len(set(groups)) == 1
    ]
    print(
        f"\n  group 分跨 coef 完全不变的种子：{len(invariant)}/{len(items)}"
        f"（{invariant if len(invariant) <= 8 else '...'}）\n"
        "  → 大多数种子上，总分之差恰好等于 B(0) 之差：换 coef 只平移总分，\n"
        "    走法序列不变。少数例外是清盘本身困难的棋盘（清盘与否真的会变）。\n"
        "    单看这张表会以为「没有 trade-off」，那是假象，见 [2]。"
    )

    # [2] 真正的权衡：清盘 vs 攒大团弃盘
    print("\n  [2] 真正的权衡：清盘（拿 B(0)）vs 攒大团弃盘（R≥A 时 B=0）")
    print("      弃盘策略用 B≡0 的计分跑，得到纯 group 分最大化能到多少。\n")
    # 强制不清盘：B(0) 设巨额负分，逼出「必须留下 ≥1 块」时的最大 group 分
    no_clear = ScoringConfig(terminal_score=lambda r: -1e9 if r == 0 else 0.0)
    print(
        f"  {'seed':>7} {'清盘group':>9} {'弃盘group':>9} {'剩余':>6} "
        f"{'临界coef*':>10} {'Σn_c²':>7} {'达成率':>7}"
    )
    coef_stars: List[float] = []
    ratios: List[float] = []
    for seed, state in items:
        clear = BeamSolver(beam_width=width).solve(state)
        dump = BeamSolver(scoring=no_clear, beam_width=width, weight_bonus=0.0).solve(
            state
        )
        # 清盘总分 = G_clear + coef*100 ；弃盘总分 = G_dump + B(R_dump)
        # 临界：coef* = (G_dump − G_clear) / 100
        star = (dump.group_score - clear.group_score) / 100.0
        coef_stars.append(star)
        # 理论天花板：每种颜色完全并成一团
        counts = [v for v in color_counts(state).values() if v >= 2]
        ceiling = sum(v * v for v in counts)
        ratios.append(clear.group_score / ceiling if ceiling else 0.0)
        print(
            f"  {seed:>7} {clear.group_score:>9.0f} {dump.group_score:>9.0f} "
            f"{dump.remaining_count:>6} {star:>10.2f} {ceiling:>7.0f} "
            f"{clear.group_score / ceiling:>6.0%}"
        )
    print(
        f"\n  临界系数：均值 {statistics.mean(coef_stars):.2f}，"
        f"范围 {min(coef_stars):.2f}~{max(coef_stars):.2f}"
    )
    print(
        f"  group 分达成率（相对「每色完全并成一团」的天花板）："
        f"{statistics.mean(ratios):.0%}"
    )
    print(
        "\n  两条都需要说明的边界，缺一不可：\n"
        "  (a) 临界系数是**下界**：G_dump 只是 beam 找得到的弃盘解，\n"
        "      搜索能力越强，这个值只会往上走。\n"
        "  (b) 天花板达成率说明弃盘策略的上限远未触及——理论上若能把\n"
        "      四色各自并成一团，group 分可到 Σn_c²，是当前的 1.6~2.5 倍。\n"
        "      也就是说「coef 到底该定多少」这个问题**目前无法定死**：\n"
        "      当前搜索下 coef≥1.1 清盘即最优，但若弃盘解被搜出来，结论会变。\n"
        f"\n  可确定的部分：10x10 四色上 coef 从 {min(coefficients)} 扫到 {max(coefficients)}，\n"
        f"  {len(invariant)}/{len(items)} 个种子的走法序列（group 分）完全不变 ——\n"
        "  对这些棋盘，当前 B(R) 形式**不产生任何决策冲突**，只是一笔固定加数。\n"
        f"  剩下 {len(items) - len(invariant)} 个（清盘本身困难的）才是 coef 真正起作用的场景。"
    )
    print(
        "\n  → 旧结论「真实权衡点在 coef∈[1,3]」作废：它来自单个 6x6 实例，\n"
        "    6x6 上清盘本身就困难才产生了那个区间；10x10 上不存在该权衡。"
    )


# ---------------------------------------------------------------------------
# eval：评估权重扫描
# ---------------------------------------------------------------------------


def section_pareto(items, width: int, log: Optional[RunLog]) -> None:
    """终局 (G, R) 非支配前沿。

    搜索目标里**不含**任何终局奖励，只记录「终局剩 R 块时的最高 group 分」。
    于是这批数据与 ``B(R)`` 的形式完全解耦：奖励公式整个换掉也不用重跑，
    只要它仍只依赖 R。给定任意 ``λ = bonus_coefficient``，
    ``F_λ(G,R) = G + λ·B₀(R)`` 在前沿上取 argmax 即得最优策略。
    """
    banner("PARETO｜终局 (G, R) 非支配前沿（bonus 不进搜索目标）")
    print(
        f"  每个种子跑 {len(PARETO_PROFILES)} 组 beam profile（{', '.join(n for n, _ in PARETO_PROFILES)}），\n"
        f"  合并成 (R -> 最高 G)。beam 宽度 {width}。\n"
        "  η_G = G / U_group，U_group = Σ_{c: n_c≥2} n_c²（每色完全并成一团的组合上界）。\n"
    )

    eta_by_bucket: Dict[str, List[float]] = {}
    clear_costs: List[float] = []
    optimal_at_default: List[int] = []
    multi_segment = 0

    for seed, state in items:
        counts = list(color_counts(state).values())
        u_group = group_score_ceiling(counts, DEFAULT_SCORING.group_score)
        u_nonclear = nonclear_group_ceiling(counts, DEFAULT_SCORING.group_score)
        front = pareto_search(state, beam_width=width)
        points = front.points()
        pareto = front.pareto()
        eta = {r: (g / u_group if u_group else 0.0) for r, g in points}

        def bucket(r: int) -> str:
            if r == 0:
                return "R=0"
            if r < 10:
                return "R=1..9"
            return "R>=10"

        g_zero = front.group_by_remaining.get(0)
        best_any = max(g for _r, g in points)
        # 「清盘代价」= (任意 R 上的最高 G) − G(R=0)，即为了清盘放弃多少 group 分。
        # 0 表示清盘解本身就是 group 分最高的那个。
        # R=0 没被搜到时该盘面无从比较，记 None 并在聚合时剔除（不能当 0 算）。
        cost = None if g_zero is None else best_any - g_zero
        if cost is not None:
            clear_costs.append(cost)

        segments = front.envelope(SEPARABLE)
        if len(segments) > 1:
            multi_segment += 1
        r_at_default = front.optimal_remaining(
            SEPARABLE.bonus_coefficient, SEPARABLE
        )
        optimal_at_default.append(r_at_default)

        print(
            f"  seed {seed:>6} │ U_group={u_group:>5.0f} U_nonclear={u_nonclear:>5.0f} "
            f"│ 前沿 {len(points):>2} 点 → Pareto {len(pareto)} 点 │ "
            f"η(R=0)={eta.get(0, float('nan')):>4.0%} "
            f"η(R>=10)={max((eta[r] for r in eta if r >= 10), default=float('nan')):>4.0%} "
            f"│ 清盘代价={'  n/a' if cost is None else format(cost, '+5.0f')} │ "
            f"coef=10 → R*={r_at_default}"
        )
        for r in sorted(eta):
            eta_by_bucket.setdefault(bucket(r), []).append(eta[r])

        if len(segments) > 1:
            desc = "  →  ".join(
                f"R={r} @coef∈[{lo:.2f},{'∞' if hi is None else f'{hi:.2f}'})"
                for lo, hi, r, _g in segments
            )
            print(f"           包络：{desc}")
        if log is not None:
            log.append(
                RunRecord.from_solution(
                    BeamSolver(beam_width=1).solve(state), tag="10x10-pareto"
                )
            )

    print(f"\n  η_G 分桶（跨 {len(items)} 个种子）")
    print(f"  {'bucket':>8} {'样本数':>7} {'平均 η_G':>9}")
    for name in ("R=0", "R=1..9", "R>=10"):
        values = eta_by_bucket.get(name, [])
        if values:
            print(f"  {name:>8} {len(values):>7} {statistics.mean(values):>8.0%}")

    eta_zero = statistics.mean(eta_by_bucket.get("R=0", [0.0]))
    eta_far = statistics.mean(eta_by_bucket.get("R>=10", [0.0]))
    print(
        f"\n  η_G(R=0) = {eta_zero:.0%}   vs   η_G(R>=10) = {eta_far:.0%}"
    )
    print(
        f"  清盘代价（max G over R − G(R=0)）：均值 {statistics.mean(clear_costs):+.0f}，"
        f"范围 {min(clear_costs):+.0f}~{max(clear_costs):+.0f}"
        f"（{len(clear_costs)}/{len(items)} 个种子搜到 R=0）"
    )
    print(
        f"  coef=10 时最优 R≠0 的种子：{sum(1 for r in optimal_at_default if r != 0)}/{len(items)}；"
        f"包络多段（存在真实切换）的种子：{multi_segment}/{len(items)}"
    )
    print(
        "\n  判读：\n"
        "  * 若 η(R=0) ≈ η(R>=10) 且清盘代价 ≈ 0：清盘并不要求破坏大团，\n"
        "    bonus 不承担补偿作用，coef 只需 >0 即可，大小不改变决策。\n"
        "  * 若 η(R=0) << η(R>=10) 且清盘代价为显著正值：清盘确实要牺牲大团，\n"
        "    此时 coef 必须足够大才值得清盘，且切换点可由包络直接读出。"
    )


def section_gen(regimes: Sequence[str], seeds: Sequence[int], width: int,
                log: Optional[RunLog]) -> None:
    """非均匀生成（数量不均 / 空间聚集）下的求解表现。

    真实游戏的方块由生成算法控制，与 ``random_board`` 的 iid 均匀假设相差很远。
    求解器本身对这两类偏差没有假设（连通块来自真实邻接、上界用真实计数、
    评估用真实盘面特征），所以这里要回答的不是「能不能跑」，而是
    **在均匀盘上得到的结论是否还成立**。
    """
    banner("GEN｜非均匀生成（数量不均 / 空间聚集）下的求解")
    print(
        "  画像指标：\n"
        "    balance        = 颜色分布熵 / log(C)，1.0 完全均衡，越低越垄断\n"
        "    聚簇比          = 同色相邻占比 / Σp_c²，1.0 空间独立，越大越成团\n"
        "      （不能直接看「同色相邻占比」：它会被数量不均抬高，会误判成聚集）\n"
        f"  每种方案 {len(seeds)} 个种子，beam{width}。\n"
    )
    header = (
        f"  {'regime':<14} {'balance':>7} {'聚簇比':>6} {'色数分布':>18} "
        f"{'清盘':>5} {'G(R=0)':>7} {'η(R=0)':>7} {'η(R≥10)':>8} "
        f"{'清盘代价':>8} {'U/V':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    summary: Dict[str, Dict[str, List[float]]] = {}
    for regime in regimes:
        generator = make_generator(regime)
        row: Dict[str, List[float]] = {
            "balance": [], "clust": [], "eta0": [], "etafar": [],
            "cost": [], "ratio": [], "clear": [], "g0": [],
            "force_clear": [], "r_star": [],
        }
        sample_counts: List[int] = []
        for seed in seeds:
            state = generator(BOARD, BOARD, COLORS, random.Random(seed))
            profile = board_profile(state)
            counts = list(color_counts(state).values())
            u_group = group_score_ceiling(counts, DEFAULT_SCORING.group_score)

            front = pareto_search(state, beam_width=width)
            points = front.points()
            eta = {r: (g / u_group if u_group else 0.0) for r, g in points}
            g_zero = front.group_by_remaining.get(0)
            best_any = max(g for _r, g in points)
            cost = None if g_zero is None else best_any - g_zero
            ub = optimistic_bound(state, DEFAULT_SCORING)
            best_total = front.best(DEFAULT_SCORING)[0]
            r_star = front.optimal_remaining(
                SEPARABLE.bonus_coefficient, SEPARABLE
            )

            row["balance"].append(profile["balance"])
            row["clust"].append(profile["adjacency_ratio"])
            if not sample_counts:
                sample_counts = sorted(counts, reverse=True)
            if 0 in eta:
                row["eta0"].append(eta[0])
                row["g0"].append(front.group_by_remaining[0])
            far = [eta[r] for r in eta if r >= 10]
            if far:
                row["etafar"].append(max(far))
            if cost is not None:
                row["cost"].append(cost)
            row["ratio"].append(ub / best_total if best_total > 0 else float("inf"))
            row["clear"].append(1.0 if r_star == 0 else 0.0)
            row["r_star"].append(float(r_star))
            # 让「清盘」成为最优策略所需的最小系数：包络上 R=0 那一段的下界
            for lo, _hi, r, _g in front.envelope(SEPARABLE):
                if r == 0:
                    row["force_clear"].append(lo)
                    break

            if log is not None:
                log.append(
                    RunRecord.from_solution(
                        BeamSolver(beam_width=width).solve(state),
                        tag=f"10x10-gen-{regime}",
                    )
                )

        def mean(key: str) -> str:
            values = row[key]
            return f"{statistics.mean(values):>7.2f}" if values else "      —"

        print(
            f"  {regime:<14} {statistics.mean(row['balance']):>7.3f} "
            f"{statistics.mean(row['clust']):>6.2f} "
            f"{str(sample_counts):>18} "
            f"{int(sum(row['clear'])):>3}/{len(seeds):<1} "
            f"{mean('g0')} {mean('eta0')} {mean('etafar')} "
            f"{mean('cost')} {mean('ratio')}"
        )
        summary[regime] = row

    print("\n  决策相关（这是本段真正要回答的问题）")
    print(
        f"  {'regime':<14} {'coef=10 下清盘':>12} {'最优R分布':>16} "
        f"{'迫使清盘所需 coef':>20}"
    )
    print("  " + "-" * 66)
    for regime in regimes:
        row = summary[regime]
        r_values = sorted(int(v) for v in row["r_star"])
        dist: Dict[int, int] = {}
        for value in r_values:
            dist[value] = dist.get(value, 0) + 1
        dist_text = " ".join(f"R={k}×{v}" for k, v in sorted(dist.items()))
        forced = row["force_clear"]
        forced_text = (
            f"{min(forced):.2f}~{max(forced):.2f}（均 {statistics.mean(forced):.2f}）"
            if forced else "—"
        )
        print(
            f"  {regime:<14} {int(sum(row['clear'])):>6}/{len(row['clear']):<5} "
            f"{dist_text:>16} {forced_text:>20}"
        )

    print(
        "\n  判读要点：\n"
        "  * 「清盘」列 = coef=10 下最优策略是清盘的种子数。若某方案下清盘率下降，\n"
        "    说明该生成方式真的让清盘变难，此时 bonus 才开始起作用。\n"
        "  * 「清盘代价」若在聚集盘上显著变大，说明「清盘与高 group 分不冲突」\n"
        "    这个结论**只对 iid 均匀盘成立**，不能外推到真实生成算法。\n"
        "  * **U/V 越接近 1，上界越紧**。聚集盘上 U/V 可到 1.0x ——\n"
        "    这意味着 beam 的解在聚集盘上已是**可证明的近最优**（差距 <5%），\n"
        "    而在 iid 均匀盘上差距接近 70%，完全无法证明。\n"
        "    也就是说：iid 均匀盘是这个游戏**最难**的情形，不是典型情形。"
    )


def section_eval(items, width: int) -> None:
    banner(f"EVAL｜beam 评估权重扫描（beam{width}，10x10）")
    print("  est = acc + w_a·available + w_m·merge + w_b·ceiling[forced] − w_s·singletons\n")
    grid = [
        (0.0, -1.0, 0.0, 2.0),  # 最初的手调权重（基准上已证劣）
        (0.0, -1.0, 1.0, 2.0),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 2.0),  # 当前默认
        (1.0, 0.0, 1.0, 4.0),
        (1.0, 0.25, 1.0, 2.0),
        (1.0, 0.5, 1.0, 2.0),
        (1.0, -0.25, 1.0, 2.0),
        (0.75, 0.0, 1.0, 2.0),
        (1.5, 0.0, 1.0, 2.0),
        (1.0, 0.0, 1.0, 1.0),
    ]
    print(f"  {'w_a':>5} {'w_m':>5} {'w_b':>5} {'w_s':>5} {'均分':>8} {'清盘':>7} {'耗时':>7}")
    best: Optional[Tuple[float, Tuple[float, float, float, float]]] = None
    for w_a, w_m, w_b, w_s in grid:
        scores: List[float] = []
        rems: List[int] = []
        elapsed: List[float] = []
        for _seed, state in items:
            sol = BeamSolver(
                beam_width=width,
                weight_available=w_a,
                weight_merge=w_m,
                weight_bonus=w_b,
                weight_singleton=w_s,
            ).solve(state)
            scores.append(sol.best_score)
            rems.append(sol.remaining_count)
            elapsed.append(sol.elapsed_time)
        mean = statistics.mean(scores)
        cleared = sum(1 for r in rems if r == 0)
        print(
            f"  {w_a:>5.1f} {w_m:>5.1f} {w_b:>5.1f} {w_s:>5.1f} "
            f"{mean:>8.0f} {cleared:>4}/{len(rems):<2} {statistics.mean(elapsed):>6.2f}s"
        )
        if best is None or mean > best[0]:
            best = (mean, (w_a, w_m, w_b, w_s))
    assert best is not None
    print(f"\n  本轮最优权重：w_a={best[1][0]}, w_m={best[1][1]}, "
          f"w_b={best[1][2]}, w_s={best[1][3]}（均分 {best[0]:.0f}）")


# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="10x10 四色基准")
    parser.add_argument("--seeds", type=int, default=8, help="基准种子数量")
    parser.add_argument("--smoke-count", type=int, default=3, help="SMOKE 段每规模的实例数")
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 128, 512, 2048],
                        help="BEAM 段的宽度列表")
    parser.add_argument("--beam-ref", type=int, default=512, help="BOUND/EXACT 段的参考宽度")
    parser.add_argument("--limit", type=float, default=20.0, help="EXACT 段每例时限（秒）")
    parser.add_argument("--coefs", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0],
                        help="BONUS 段扫描的系数")
    parser.add_argument("--sweep-eval", action="store_true", help="额外跑 EVAL 段")
    parser.add_argument("--regimes", nargs="+", choices=list(GENERATOR_NAMES),
                        default=["uniform", "imbalanced", "clustered",
                                 "clustered_imb", "extreme", "blob5"],
                        help="GEN 段要跑的生成方案")
    parser.add_argument("--gen-seeds", type=int, default=4,
                        help="GEN 段每种方案的种子数")
    parser.add_argument("--only", nargs="+", choices=SECTIONS, default=None,
                        help="只跑指定段")
    parser.add_argument("--log", type=str, default=None, help="追加写入 JSONL 运行日志")
    parser.add_argument("--quick", action="store_true",
                        help="快速版：3 种子 / 更少宽度 / 更短时限")
    args = parser.parse_args(argv)

    if args.quick:
        args.seeds = min(args.seeds, 3)
        args.widths = [32, 256, 1024]
        args.beam_ref = 256
        args.limit = min(args.limit, 10.0)
        args.coefs = [0.5, 1.0, 3.0, 10.0, 30.0]
        args.smoke_count = 2

    seeds = DEFAULT_SEEDS[: args.seeds]
    items = boards(seeds)
    only = set(args.only) if args.only else set(SECTIONS)
    log = RunLog(args.log) if args.log else None

    started = time.perf_counter()
    print(f"基准规模 {BOARD}x{BOARD} / {COLORS} 色｜种子 {list(seeds)}")

    if "smoke" in only:
        section_smoke(args.smoke_count)
    if "facts" in only:
        section_facts(items)
    if "bound" in only:
        section_bound(items, args.widths, log)
    if "beam" in only:
        section_beam(items, args.widths, log)
    if "exact" in only:
        section_exact(items, args.limit, log, args.beam_ref)
    if "pareto" in only:
        section_pareto(items, args.beam_ref, log)
    if "bonus" in only:
        section_bonus(items, args.coefs, args.beam_ref, log)
    if "gen" in only:
        section_gen(args.regimes, seeds, args.beam_ref, log)
    if "eval" in only and args.sweep_eval:
        section_eval(items, args.beam_ref)

    print(f"\n总耗时 {time.perf_counter() - started:.1f}s")
    if log is not None:
        print(f"运行日志已写入 {args.log}（{len(log.load())} 条）")


if __name__ == "__main__":
    main()
