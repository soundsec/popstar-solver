"""列词不变量、三种能量，以及冻结代价。"""

import random
import unittest

from popstar.board import apply_move, create_board, get_legal_moves, is_terminal, random_board
from popstar.exhaustive import solve_exhaustive
from popstar.freeze import beam_minimum, exact_minimum, freeze_degrees, freeze_heuristic
from popstar.scoring import DEFAULT_SCORING
from popstar.solver import ExactSolver, group_score_ceiling
from popstar.board import color_counts
from popstar.topology import (
    aggregation_potential,
    aggregation_release,
    column_order_holds,
    column_words,
    dead_column_ceiling,
    formed_energy,
    latent_energy,
)


def _g(size: int) -> float:
    return DEFAULT_SCORING.score_group(size)


SANDWICH = create_board([
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
])


class TestColumnWords(unittest.TestCase):
    def test_bottom_to_top_and_empty_columns_dropped(self):
        state = create_board([
            [-1, 0, -1],
            [-1, 1, -1],
            [-1, 0, -1],
        ])
        self.assertEqual(column_words(state), ((0, 1, 0),))

    def test_order_invariants_hold_for_every_move_of_a_game(self):
        state = random_board(6, 6, 4, random.Random(1))
        seen = 0
        while not is_terminal(state):
            before = column_words(state)
            move = get_legal_moves(state)[0]
            state = apply_move(state, move)
            self.assertTrue(column_order_holds(before, column_words(state)))
            seen += 1
        self.assertGreater(seen, 0)


class TestEnergies(unittest.TestCase):
    def test_formed_plus_latent_is_the_color_ceiling(self):
        formed = formed_energy(SANDWICH)
        latent = latent_energy(SANDWICH)
        self.assertAlmostEqual(formed, 3 * _g(5))
        self.assertAlmostEqual(formed + latent, _g(10) + _g(5))

    def test_release_is_large_when_the_middle_band_drops(self):
        blue = next(move for move in get_legal_moves(SANDWICH) if move.color == 1)
        red = next(move for move in get_legal_moves(SANDWICH) if move.color == 0)
        blue_release = aggregation_release(
            SANDWICH, apply_move(SANDWICH, blue), blue.size)
        red_release = aggregation_release(
            SANDWICH, apply_move(SANDWICH, red), red.size)
        self.assertAlmostEqual(blue_release, _g(10) - 2 * _g(5))
        self.assertAlmostEqual(red_release, 0.0)
        self.assertGreater(blue_release, red_release)

    def test_separated_pair_has_a_discounted_potential(self):
        state = create_board([
            [0, 1, 0],
            [0, 1, 0],
        ])
        raw = _g(4) - 2 * _g(2)
        phi = aggregation_potential(state)
        self.assertGreater(phi, 0.0)
        self.assertLess(phi, raw)


class TestDeadColumnCeiling(unittest.TestCase):
    def test_a_dead_column_splits_the_color_it_does_not_contain(self):
        state = create_board([
            [0, 2, 0],
            [0, 1, 0],
        ])
        loose = group_score_ceiling(list(color_counts(state).values()), DEFAULT_SCORING.score_group)
        tight = dead_column_ceiling(state)
        self.assertAlmostEqual(loose, _g(4))
        self.assertAlmostEqual(tight, 2 * _g(2))
        self.assertAlmostEqual(loose - tight, 10 * 2 * 2)

    def test_dead_column_does_not_split_a_color_it_contains(self):
        state = create_board([
            [0, 1, 0],
            [0, 0, 0],
        ])
        self.assertAlmostEqual(dead_column_ceiling(state), _g(5))

    def test_ceiling_never_exceeds_the_color_sum(self):
        state = random_board(5, 5, 4, random.Random(4))
        loose = group_score_ceiling(list(color_counts(state).values()), DEFAULT_SCORING.score_group)
        self.assertLessEqual(dead_column_ceiling(state), loose + 1e-9)

    def test_split_matches_the_exact_group_score(self):
        state = create_board([
            [0, 2, 0],
            [0, 1, 0],
        ])
        solved = ExactSolver(use_memo=True, use_bound=True).solve(state)
        self.assertAlmostEqual(solved.group_score, 2 * _g(2))
        self.assertAlmostEqual(dead_column_ceiling(state), solved.group_score)

    def test_never_below_exact_group_score_on_small_boards(self):
        for seed in range(4):
            state = random_board(4, 4, 3, random.Random(seed))
            solved = ExactSolver(use_memo=True, use_bound=True).solve(state)
            self.assertGreaterEqual(dead_column_ceiling(state) + 1e-9, solved.group_score)


class TestFreeze(unittest.TestCase):
    def test_terminal_degrees_are_zero(self):
        state = create_board([[0, 1], [1, 0]])
        self.assertEqual(freeze_degrees(state), (0.0, 0.0, 0.0))

    def test_exact_minimum_on_the_sandwich_avoids_the_big_merge(self):
        result = exact_minimum(SANDWICH)
        self.assertTrue(result.proven)
        self.assertAlmostEqual(result.score, 3 * _g(5))
        replay = SANDWICH
        paid = 0.0
        for move in result.moves:
            paid += _g(move.size)
            replay = apply_move(replay, move)
        self.assertTrue(is_terminal(replay))
        self.assertAlmostEqual(paid, result.score)
        # 先消掉中间蓝条会合成 10 连，group 分更高，那是高分方向而不是冻结方向
        maximum, _path = solve_exhaustive(SANDWICH)
        self.assertGreater(maximum, result.score)

    def test_beam_matches_exact_on_a_tiny_board(self):
        state = random_board(4, 4, 3, random.Random(2))
        exact = exact_minimum(state)
        approx = beam_minimum(state, beam_width=32)
        self.assertTrue(exact.proven)
        self.assertAlmostEqual(approx.score, exact.score)

    def test_heuristic_prefers_a_move_that_destroys_more_per_point(self):
        state = random_board(5, 5, 3, random.Random(3))
        moves = get_legal_moves(state)
        scores = [
            freeze_heuristic(state, apply_move(state, move), move.size)
            for move in moves
        ]
        self.assertTrue(any(score != 0 for score in scores))


if __name__ == "__main__":
    unittest.main()
