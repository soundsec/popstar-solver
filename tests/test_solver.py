"""Phase 2 基线验证：所有优化层都必须与完全穷举得到相同的最优分。

开发提纲第 28 节（见 ``SOURCES.md``）：Memoized Exact Solver 必须与完全穷举得到完全相同的最高分。
这里对 3x3 / 4x4 / 5x5（2~4 色）随机实例逐层对拍：

    exhaustive == memo == memo+bound == memo+bound+ordering == +canonicalize
"""

import random
import unittest

from typing import Any, Dict

from popstar.board import (
    EMPTY,
    apply_move,
    count_remaining,
    BoardState,
    create_board,
    get_components,
    get_legal_moves,
    is_canonical,
    is_terminal,
    random_board,
)
from popstar.exhaustive import solve_exhaustive
from popstar.game import Game
from popstar.scoring import DEFAULT_SCORING, ScoringConfig
from popstar.solver import (
    BeamSolver,
    ExactSolver,
    ParetoFront,
    board_features,
    canonical_grid,
    forced_residual,
    group_score_ceiling,
    merge_potential,
    nonclear_group_ceiling,
    lookahead_bound,
    optimistic_bound,
    optimistic_bound_with_ceiling,
    pareto_search,
    singleton_count,
    solve,
    terminal_bonus_ceiling,
    validate_scoring_for_bound,
)


def boards_for(seed_count, height, width, colors, base=0):
    out = []
    for i in range(seed_count):
        rng = random.Random(base + i)
        out.append(random_board(height, width, colors, rng))
    return out


def sequence_score(state, moves, scoring=None):
    """按动作序列实际走一遍，返回总分（可行性校验）。"""
    game = Game(state, scoring=scoring)
    for move in moves:
        game.play(move)
    return game.result(require_finished=True)


class TestExhaustiveVsMemo(unittest.TestCase):
    def test_3x3_all_colors(self):
        for colors in (2, 3, 4):
            for state in boards_for(8, 3, 3, colors, base=colors * 100):
                expected, path = solve_exhaustive(state)
                for solver in (
                    ExactSolver(use_memo=True),
                    ExactSolver(use_memo=False),
                ):
                    solution = solver.solve(state)
                    self.assertAlmostEqual(
                        solution.best_score,
                        expected,
                        msg=f"3x3/{colors} {state.grid}",
                    )
                    self.assertTrue(solution.is_proven_optimal)
                # 穷举给出的路径本身必须可行且等分
                self.assertAlmostEqual(sequence_score(state, path).total_score, expected)

    def test_4x4_all_colors(self):
        for colors in (2, 3, 4):
            for state in boards_for(6, 4, 4, colors, base=colors * 1000):
                expected, _ = solve_exhaustive(state)
                solution = ExactSolver(use_memo=True).solve(state)
                self.assertAlmostEqual(solution.best_score, expected)

    def test_5x5_two_and_three_colors(self):
        for colors in (2, 3):
            for state in boards_for(3, 5, 5, colors, base=colors * 5000):
                expected, _ = solve_exhaustive(state)
                solution = ExactSolver(use_memo=True).solve(state)
                self.assertAlmostEqual(solution.best_score, expected)

    def test_solution_sequence_is_feasible_and_optimal(self):
        for state in boards_for(6, 4, 4, 3, base=77):
            expected, _ = solve_exhaustive(state)
            solution = ExactSolver().solve(state)
            self.assertEqual(len(solution.move_sequence), solution.move_count)
            # 序列可行性：逐步 apply_move 必须合法
            cursor = state
            for move in solution.move_sequence:
                self.assertFalse(is_terminal(cursor))
                cursor = apply_move(cursor, move)
            self.assertTrue(is_terminal(cursor))
            self.assertEqual(cursor.grid, solution.final_board.grid)
            self.assertAlmostEqual(solution.best_score, expected)

    def test_two_parts_sum_to_total(self):
        for state in boards_for(5, 4, 4, 3, base=303):
            solution = ExactSolver().solve(state)
            self.assertAlmostEqual(
                solution.best_score,
                solution.group_score + solution.terminal_bonus,
            )


class TestFuzzBoundAgainstMemo(unittest.TestCase):
    """随机模糊对拍：剪枝 / 排序 / 规范化在任何实例上都不得改变最优分。

    这是本项目最重要的一条回归——B&B 与 transposition table 共存时容易静默丢解。
    """

    CASES = (
        (3, 3, 3, 25),
        (3, 3, 4, 20),
        (4, 4, 3, 25),
        (4, 4, 4, 15),
        (5, 5, 3, 12),
        (5, 5, 4, 10),
    )

    def _variants(self, state):
        return {
            "memo": ExactSolver(use_memo=True, use_bound=False),
            "memo+bound": ExactSolver(use_memo=True, use_bound=True),
            "bound only": ExactSolver(use_memo=False, use_bound=True),
            "full": ExactSolver(
                use_memo=True,
                use_bound=True,
                ordering="heuristic",
                canonicalize=True,
            ),
        }

    def test_all_variants_agree(self):
        for height, width, colors, count in self.CASES:
            for index in range(count):
                state = random_board(
                    height, width, colors, random.Random(index * 977 + height * 31 + colors)
                )
                variants = self._variants(state)
                reference = variants["memo"].solve(state).best_score
                for name, solver in variants.items():
                    self.assertAlmostEqual(
                        solver.solve(state).best_score,
                        reference,
                        msg=f"{name} mismatch on {height}x{width}/{colors} #{index}",
                    )

    def test_reported_score_matches_replayed_path(self):
        """报告分数必须来自一条真实可回放的路径（内部一致性）。"""
        for height, width, colors, count in self.CASES[:4]:
            for index in range(6):
                state = random_board(height, width, colors, random.Random(5000 + index))
                solution = ExactSolver(use_memo=True, use_bound=True).solve(state)
                total = sequence_score(state, solution.move_sequence).total_score
                self.assertAlmostEqual(solution.best_score, total)


class TestPruningAndOptimizations(unittest.TestCase):
    """逐层开关对拍：任何一层都不许改变最优分。"""

    def _configs(self):
        return [
            ("memo", ExactSolver(use_memo=True)),
            ("memo+bound", ExactSolver(use_memo=True, use_bound=True)),
            ("bound only", ExactSolver(use_memo=False, use_bound=True)),
            ("memo+bound+size", ExactSolver(use_memo=True, use_bound=True, ordering="size")),
            (
                "memo+bound+heuristic",
                ExactSolver(use_memo=True, use_bound=True, ordering="heuristic"),
            ),
            (
                "memo+bound+heuristic+canon",
                ExactSolver(
                    use_memo=True,
                    use_bound=True,
                    ordering="heuristic",
                    canonicalize=True,
                ),
            ),
        ]

    def test_all_configs_match_exhaustive_3x3(self):
        for colors in (2, 3, 4):
            for state in boards_for(5, 3, 3, colors, base=colors * 31):
                expected, _ = solve_exhaustive(state)
                for name, solver in self._configs():
                    solution = solver.solve(state)
                    self.assertAlmostEqual(
                        solution.best_score,
                        expected,
                        msg=f"{name} mismatch on {state.grid}",
                    )
                    self.assertTrue(solution.is_proven_optimal)

    def test_all_configs_match_exhaustive_4x4(self):
        for state in boards_for(4, 4, 4, 3, base=909):
            expected, _ = solve_exhaustive(state)
            for name, solver in self._configs():
                solution = solver.solve(state)
                self.assertAlmostEqual(
                    solution.best_score, expected, msg=f"{name} mismatch"
                )

    def test_ordering_does_not_change_result_5x5(self):
        for state in boards_for(3, 5, 5, 3, base=4242):
            expected, _ = solve_exhaustive(state)
            for ordering in ("none", "size", "heuristic"):
                solver = ExactSolver(use_memo=True, use_bound=True, ordering=ordering)
                self.assertAlmostEqual(solver.solve(state).best_score, expected)

    def test_canonicalization_preserves_value(self):
        for state in boards_for(6, 4, 4, 4, base=5151):
            plain = ExactSolver(use_memo=True).solve(state).best_score
            canon = ExactSolver(use_memo=True, canonicalize=True).solve(state).best_score
            self.assertAlmostEqual(plain, canon)

    def test_canonical_grid_is_bijective(self):
        grid = create_board([[3, 3, 7], [7, 1, 3]]).grid
        canon = canonical_grid(grid)
        self.assertEqual(canon, ((0, 0, 1), (1, 2, 0)))
        # 规范化后再规范化应保持不变
        self.assertEqual(canonical_grid(canon), canon)

    def test_determinism_across_runs(self):
        state = random_board(5, 5, 3, random.Random(11))
        first = ExactSolver(use_memo=True, use_bound=True, ordering="heuristic").solve(state)
        second = ExactSolver(use_memo=True, use_bound=True, ordering="heuristic").solve(state)
        self.assertAlmostEqual(first.best_score, second.best_score)
        self.assertEqual(
            [m.representative_cell for m in first.move_sequence],
            [m.representative_cell for m in second.move_sequence],
        )


class TestRefinedUpperBound(unittest.TestCase):
    """收紧版上界：死色（某颜色只剩 1 个方块）带来的残余下界。"""

    def test_forced_residual_counts_dead_colors(self):
        # 颜色 0 与 2 各只剩 1 个方块 -> 这两个方块永远无法被消除
        state = create_board([[0, 1, 1], [2, 1, 1]])
        self.assertEqual(forced_residual(state), 2)

    def test_forced_residual_zero_when_every_color_has_pairs(self):
        state = create_board([[0, 0, 1], [1, 1, 0]])
        self.assertEqual(forced_residual(state), 0)

    def test_terminal_bonus_ceiling_is_suffix_max(self):
        ceiling = terminal_bonus_ceiling(ScoringConfig(), 12)
        for r in range(13):
            expected = max(
                ScoringConfig().score_terminal(x) for x in range(r, 13)
            )
            self.assertAlmostEqual(ceiling[r], expected)

    def test_ceiling_works_for_non_monotone_bonus(self):
        # 故意构造非单调的 B：只在 R=3 给奖励
        weird = ScoringConfig(terminal_score=lambda r: 500 if r == 3 else 0)
        ceiling = terminal_bonus_ceiling(weird, 10)
        self.assertEqual(ceiling[0], 500)
        self.assertEqual(ceiling[3], 500)
        self.assertEqual(ceiling[4], 0)

    def test_refined_bound_never_underestimates(self):
        """关键安全性质：对可达状态逐一比对 U(S) >= V(S)。"""
        for height, width, colors, count in ((3, 3, 3, 8), (4, 4, 3, 5), (5, 5, 4, 3)):
            for index in range(count):
                state = random_board(
                    height, width, colors, random.Random(index * 331 + 7)
                )
                ceiling = terminal_bonus_ceiling(DEFAULT_SCORING, height * width)
                memo: Dict[Any, float] = {}

                def value_of(current: BoardState) -> float:
                    if is_terminal(current):
                        return DEFAULT_SCORING.score_terminal(count_remaining(current))
                    key = current.packed_key
                    if key in memo:
                        return memo[key]
                    best = max(
                        DEFAULT_SCORING.score_group(m.size)
                        + value_of(apply_move(current, m))
                        for m in get_legal_moves(current)
                    )
                    memo[key] = best
                    upper = optimistic_bound_with_ceiling(current, DEFAULT_SCORING, ceiling)
                    self.assertGreaterEqual(upper + 1e-9, best)
                    return best

                value_of(state)

    def test_refined_bound_is_tighter_than_naive(self):
        """有死色时必须严格更紧；无死色时两者相同。"""
        naive = ScoringConfig()
        with_dead = create_board([[0, 1, 1], [2, 1, 1], [3, 1, 1]])
        naive_value = (
            sum(naive.score_group(n) for n in (1, 6) if n >= 2)
            + naive.score_terminal(0)
        )
        refined = optimistic_bound(with_dead, naive)
        self.assertLess(refined, naive_value)

        without_dead = create_board([[0, 0, 1], [1, 1, 0]])
        counts = [3, 3]
        naive_value2 = sum(naive.score_group(n) for n in counts) + naive.score_terminal(0)
        self.assertAlmostEqual(optimistic_bound(without_dead, naive), naive_value2)


class TestLookaheadBound(unittest.TestCase):
    def test_depth_zero_matches_the_static_bound(self):
        state = create_board([[0, 0, 1], [1, 1, 0]])
        self.assertAlmostEqual(
            lookahead_bound(state, depth=0),
            optimistic_bound(state, DEFAULT_SCORING),
        )

    def test_terminal_is_the_bonus_at_every_depth(self):
        state = create_board([[0, 1], [1, 0]])
        bonus = DEFAULT_SCORING.score_terminal(4)
        for depth in range(3):
            self.assertAlmostEqual(lookahead_bound(state, depth=depth), bonus)

    def test_deeper_bound_stays_admissible_and_does_not_rise(self):
        state = create_board([
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ])
        exact = ExactSolver(use_memo=True, use_bound=True).solve(state).best_score
        bounds = [lookahead_bound(state, depth=depth) for depth in range(3)]
        self.assertLess(bounds[1], bounds[0] - 1)
        self.assertLessEqual(bounds[2], bounds[1] + 1e-9)
        self.assertGreaterEqual(bounds[2] + 1e-9, exact)

    def test_random_boards_stay_above_the_exact_score(self):
        for seed in range(4):
            state = random_board(4, 4, 3, random.Random(seed + 20))
            exact = ExactSolver(use_memo=True, use_bound=True).solve(state).best_score
            previous = float("inf")
            for depth in range(3):
                bound = lookahead_bound(state, depth=depth)
                self.assertGreaterEqual(previous + 1e-6, bound)
                self.assertGreaterEqual(bound + 1e-6, exact)
                previous = bound

    def test_bound_depth_does_not_change_the_proven_score(self):
        state = random_board(4, 4, 3, random.Random(8))
        plain = ExactSolver(use_memo=True, use_bound=True).solve(state)
        deeper = ExactSolver(use_memo=True, use_bound=True, bound_depth=2).solve(state)
        self.assertTrue(plain.is_proven_optimal and deeper.is_proven_optimal)
        self.assertAlmostEqual(plain.best_score, deeper.best_score)


class TestCompactState(unittest.TestCase):
    """紧凑状态键与内部快速构造路径的一致性。"""

    def test_packed_key_is_deterministic(self):
        a = create_board([[0, 1, EMPTY], [2, 0, 1]])
        b = create_board([[0, 1, EMPTY], [2, 0, 1]])
        self.assertEqual(a.packed_key, b.packed_key)
        self.assertEqual(len(a.packed_key), a.height * a.width)

    def test_packed_key_encodes_empty_as_zero(self):
        state = create_board([[EMPTY, 3], [EMPTY, 0]])
        self.assertEqual(tuple(state.packed_key), (0, 4, 0, 1))

    def test_different_boards_have_different_keys(self):
        a = create_board([[0, 1], [1, 0]])
        b = create_board([[1, 0], [0, 1]])
        self.assertNotEqual(a.packed_key, b.packed_key)

    def test_fast_constructor_matches_validated_constructor(self):
        """内部快速构造（跳过校验）的结果必须与公开构造完全一致。"""
        state = create_board([[0, 0, 1], [1, 2, 2]])
        move = get_legal_moves(state)[0]
        fast = apply_move(state, move)
        validated = create_board([list(row) for row in fast.grid])
        self.assertEqual(fast.grid, validated.grid)
        self.assertEqual(fast.packed_key, validated.packed_key)
        self.assertTrue(is_canonical(fast))
        # 快速构造出的状态仍具备完整能力
        self.assertEqual(len(get_components(fast)), len(get_components(validated)))
        self.assertEqual(get_legal_moves(fast), get_legal_moves(validated))

    def test_fast_constructor_states_are_usable_in_game(self):
        state = create_board([[0, 0, 1], [1, 2, 2]])
        game = Game(state)
        while not game.is_finished():
            game.play(game.legal_moves()[0])
        self.assertTrue(game.is_finished())
        self.assertGreater(game.accumulated_score, 0)


class TestUpperBound(unittest.TestCase):
    def test_bound_never_underestimates_optimum(self):
        """关键安全性质：U(S) >= V(S)，否则会剪掉最优解。"""
        for state in boards_for(8, 3, 3, 3, base=606):
            truth, _ = solve_exhaustive(state)
            bound = optimistic_bound(state, DEFAULT_SCORING)
            self.assertGreaterEqual(bound + 1e-9, truth)

    def test_bound_is_at_least_terminal_bonus(self):
        state = create_board([[0, 1], [1, 0]])
        self.assertGreaterEqual(optimistic_bound(state, ScoringConfig()), 0)

    def test_bound_validator_accepts_default(self):
        self.assertEqual(validate_scoring_for_bound(ScoringConfig(), max_blocks=64), [])

    def test_bound_validator_rejects_subadditive(self):
        import math

        bad = ScoringConfig(group_score=lambda k: math.sqrt(k))
        issues = validate_scoring_for_bound(bad, max_blocks=32)
        self.assertTrue(issues)
        with self.assertRaises(ValueError):
            ExactSolver(use_bound=True, scoring=bad)

    def test_additive_scoring_is_allowed(self):
        linear = ScoringConfig(group_score=lambda k: 10 * k)
        self.assertEqual(validate_scoring_for_bound(linear, max_blocks=48), [])
        state = boards_for(1, 3, 3, 3, base=7)[0]
        plain = ExactSolver(scoring=linear).solve(state).best_score
        bounded = ExactSolver(scoring=linear, use_bound=True).solve(state).best_score
        self.assertAlmostEqual(plain, bounded)


class TestLimits(unittest.TestCase):
    def test_node_limit_marks_not_proven(self):
        state = random_board(6, 6, 4, random.Random(2024))
        solution = ExactSolver(use_memo=True, node_limit=200).solve(state)
        self.assertFalse(solution.is_proven_optimal)
        self.assertGreater(solution.move_count, 0)
        self.assertTrue(solution.notes["aborted"])

    def test_time_limit_marks_not_proven(self):
        state = random_board(7, 7, 4, random.Random(2025))
        solution = ExactSolver(use_memo=True, time_limit=0.05).solve(state)
        self.assertFalse(solution.is_proven_optimal)


class TestBeamSolver(unittest.TestCase):
    def test_beam_is_feasible_but_not_claimed_optimal(self):
        for state in boards_for(4, 5, 5, 4, base=808):
            solution = BeamSolver(beam_width=8).solve(state)
            self.assertFalse(solution.is_proven_optimal)
            self.assertEqual(solution.search_mode, "fast")
            cursor = state
            for move in solution.move_sequence:
                cursor = apply_move(cursor, move)
            self.assertTrue(is_terminal(cursor))
            self.assertAlmostEqual(
                solution.best_score,
                solution.group_score + solution.terminal_bonus,
            )

    def test_beam_never_beats_exhaustive(self):
        for state in boards_for(5, 3, 3, 3, base=313):
            truth, _ = solve_exhaustive(state)
            solution = BeamSolver(beam_width=6).solve(state)
            self.assertLessEqual(solution.best_score, truth + 1e-6)

    def test_wider_beam_improves_or_equals(self):
        state = random_board(6, 6, 4, random.Random(99))
        narrow = BeamSolver(beam_width=2).solve(state).best_score
        wide = BeamSolver(beam_width=32).solve(state).best_score
        self.assertGreaterEqual(wide, narrow - 1e-9)

    def test_beam_result_seeds_exact_search(self):
        """第 25 节：Fast 结果作为下界，Exact 证明后不得更差。"""
        for state in boards_for(3, 4, 4, 3, base=1212):
            fast = BeamSolver(beam_width=8).solve(state)
            exact = solve(state, mode="exact", use_memo=True, use_bound=True, beam_width=8)
            truth, _ = solve_exhaustive(state)
            self.assertAlmostEqual(exact.best_score, truth)
            self.assertGreaterEqual(exact.best_score, fast.best_score - 1e-9)


class TestFeaturesTerminalFlag(unittest.TestCase):
    """``Features.terminal`` 必须与 ``is_terminal()`` 完全一致。

    beam 用它来决定「何时把终局计入 Pareto 前沿」，判错就会漏记或误记。
    """

    def test_agrees_with_is_terminal(self):
        rng = random.Random(909)
        for height, width, colors in ((4, 4, 3), (5, 6, 4), (6, 5, 2)):
            for _ in range(12):
                state = random_board(height, width, colors, rng)
                cursor = state
                for _step in range(height * width):
                    self.assertEqual(
                        board_features(cursor, DEFAULT_SCORING).terminal,
                        is_terminal(cursor),
                    )
                    moves = get_legal_moves(cursor)
                    if not moves:
                        break
                    cursor = apply_move(cursor, moves[rng.randrange(len(moves))])


class TestGroupScoreCeilings(unittest.TestCase):
    """group 分的两个组合上界。``nonclear`` 必须是「从最小活跃色里扣一块」。"""

    def test_group_ceiling_sums_squares_of_active_colors(self):
        # 死色（n=1）不构成可消除块，不计入
        self.assertEqual(group_score_ceiling([25, 25, 25, 25], lambda k: float(k * k)), 2500)
        self.assertEqual(group_score_ceiling([25, 25, 1], lambda k: float(k * k)), 1250)

    def test_nonclear_ceiling_is_strictly_smaller(self):
        g = lambda k: float(k * k)
        counts = [40, 24, 22, 14]
        full = group_score_ceiling(counts, g)
        nonclear = nonclear_group_ceiling(counts, g)
        self.assertLess(nonclear, full)
        # 从最小的活跃色（14）扣一块：39²... 不对，是 13²
        expected = 40 * 40 + 24 * 24 + 22 * 22 + 13 * 13
        self.assertEqual(nonclear, float(expected))
        self.assertEqual(full - nonclear, 2 * 14 - 1)  # 损失 = 2·n_c − 1

    def test_nonclear_picks_the_cheapest_color_to_lose(self):
        g = lambda k: float(k * k)
        # 从 n=3 扣（损失 5）优于从 n=10 扣（损失 19）
        counts = [10, 3]
        expected = 10 * 10 + 2 * 2
        self.assertEqual(nonclear_group_ceiling(counts, g), float(expected))

    def test_nonclear_when_losing_a_block_kills_the_group(self):
        g = lambda k: float(k * k)
        # n=2 的颜色扣掉一块后只剩 1，不构成可消除块 -> 贡献 0，损失 4（不是 3）
        self.assertEqual(nonclear_group_ceiling([2, 9], g), float(9 * 9))

    def test_no_active_color(self):
        g = lambda k: float(k * k)
        self.assertEqual(group_score_ceiling([1, 1], g), 0.0)
        self.assertEqual(nonclear_group_ceiling([1, 1], g), 0.0)


class TestParetoFront(unittest.TestCase):
    """Pareto 前沿的支配过滤与 λ 包络（切换点必须能解析求对）。"""

    def _front(self, mapping):
        return ParetoFront(initial_board=None, group_by_remaining=dict(mapping))

    def test_dominated_points_are_removed(self):
        # (R=5, G=100) 被 (R=0, G=120) 支配：剩更少、拿更多
        front = self._front({0: 120.0, 5: 100.0, 8: 150.0})
        self.assertEqual(front.pareto(), [(0, 120.0), (8, 150.0)])

    def test_equal_group_keeps_only_smaller_remaining(self):
        front = self._front({0: 100.0, 3: 100.0})
        self.assertEqual(front.pareto(), [(0, 100.0)])

    def test_envelope_switch_points_are_analytic(self):
        # 三条直线：1000+100λ、1400+25λ、1500+0λ
        # 手算包络：[0,4)→R=12；[4, 5.333)→R=5；[5.333,∞)→R=0
        separable = ScoringConfig(clear_threshold=10, bonus_exponent=2)
        front = self._front({0: 1000.0, 5: 1400.0, 12: 1500.0})
        segments = front.envelope(separable)
        self.assertEqual([seg[2] for seg in segments], [12, 5, 0])
        self.assertAlmostEqual(segments[0][1], 4.0)
        self.assertAlmostEqual(segments[1][1], 400.0 / 75.0)
        self.assertIsNone(segments[2][1])

        switches = front.switch_points(separable)
        self.assertEqual([(round(s[0], 4), s[1], s[2]) for s in switches],
                         [(4.0, 12, 5), (round(400.0 / 75.0, 4), 5, 0)])

    def test_switch_formula_matches_pairwise_definition(self):
        # λ* = (G₂ − G₁) / (B₀(R₁) − B₀(R₂))
        separable = ScoringConfig(clear_threshold=10, bonus_exponent=2)
        front = self._front({0: 1000.0, 12: 1500.0})
        lam = (1500.0 - 1000.0) / (separable.terminal_shape(0)
                                   - separable.terminal_shape(12))
        self.assertAlmostEqual(lam, 5.0)
        # 恰好在切换点两侧，最优策略应相反
        self.assertEqual(front.optimal_remaining(lam - 0.5, separable), 12)
        self.assertEqual(front.optimal_remaining(lam + 0.5, separable), 0)

    def test_best_respects_the_given_scoring(self):
        front = self._front({0: 1000.0, 12: 1500.0})
        total, remaining, group = front.best(DEFAULT_SCORING)
        self.assertEqual(remaining, 0)  # B(0)=2000 → 3000 > 1500
        self.assertAlmostEqual(total, 3000.0)
        self.assertAlmostEqual(group, 1000.0)


class TestParetoSearch(unittest.TestCase):
    """``pareto_search`` 产出的每个点都必须是**真实可回放**的走法。"""

    def test_every_frontier_point_replays_to_its_claimed_outcome(self):
        state = random_board(8, 8, 4, random.Random(2026))
        front = pareto_search(state, beam_width=64)
        self.assertTrue(front.group_by_remaining)
        for remaining, group in front.points():
            path = front.paths[remaining]
            game = Game(state, scoring=DEFAULT_SCORING)
            for cell in path:
                game.play_cell(cell)
            result = game.result(require_finished=False)
            self.assertTrue(is_terminal(game.state), f"R={remaining} 未走到终局")
            self.assertEqual(result.remaining_count, remaining)
            self.assertAlmostEqual(result.group_score, group, places=6)

    def test_merging_profiles_only_improves(self):
        """多 profile 合并后的每个 R 都不劣于单 profile。"""
        state = random_board(8, 8, 4, random.Random(7))
        merged = pareto_search(state, beam_width=64)
        single = BeamSolver(beam_width=64).solve_front(state)
        for remaining, (group, _path) in single.items():
            self.assertGreaterEqual(merged.group_by_remaining[remaining], group - 1e-9)

    def test_search_contains_the_scalar_best_solution(self):
        """前沿在默认计分下的最优解，不劣于单独跑一次 beam 的标量结果。"""
        for state in [random_board(8, 8, 4, random.Random(100 + i)) for i in range(3)]:
            front = pareto_search(state, beam_width=64)
            scalar = BeamSolver(beam_width=64).solve(state).best_score
            self.assertGreaterEqual(front.best(DEFAULT_SCORING)[0], scalar - 1e-9)


class TestPotentialHelpers(unittest.TestCase):
    def test_merge_potential_zero_when_each_color_is_one_cluster(self):
        # 颜色 0 与 1 各自恰好构成一个连通团 -> 两项均为 0
        state = create_board([[0, 0, 1], [0, 1, 1]])
        self.assertAlmostEqual(merge_potential(state, ScoringConfig()), 0.0)

    def test_merge_potential_counts_singletons_of_a_color(self):
        # 颜色 1 有一个孤立块：n_1=3，已成团只有 size=2。g=5k² 时 5·9 − 5·4 = 25
        state = create_board([[0, 0, 1], [1, 1, 2]])
        self.assertAlmostEqual(merge_potential(state, ScoringConfig()), 25.0)

    def test_merge_potential_positive_when_fragmented(self):
        state = create_board([[0, 1, 0], [1, 1, 1]])
        self.assertGreater(merge_potential(state, ScoringConfig()), 0)

    def test_singleton_count(self):
        state = create_board([[0, 1], [1, 0]])
        self.assertEqual(singleton_count(state), 4)


if __name__ == "__main__":
    unittest.main()
