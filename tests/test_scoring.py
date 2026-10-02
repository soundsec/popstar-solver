"""计分模块：可配置、可替换、与游戏逻辑解耦。"""

import unittest

from popstar.game import Game
from popstar.scoring import (
    DEFAULT_SCORING,
    ScoringConfig,
    score_group,
    score_terminal,
)


class TestDefaultScoring(unittest.TestCase):
    def test_group_score_is_five_times_square(self):
        self.assertEqual(score_group(2), 20)
        self.assertEqual(score_group(5), 125)
        self.assertEqual(score_group(10), 500)

    def test_terminal_bonus_zero_at_or_above_threshold(self):
        for r in range(10, 40):
            self.assertEqual(score_terminal(r), 0)

    def test_terminal_bonus_increases_as_remaining_drops(self):
        values = [score_terminal(r) for r in range(10, -1, -1)]
        self.assertEqual(values[0], 0)
        for prev, cur in zip(values, values[1:]):
            self.assertGreater(cur, prev)

    def test_full_clear_is_maximum_bonus(self):
        bonuses = [score_terminal(r) for r in range(0, 30)]
        self.assertEqual(max(bonuses), score_terminal(0))
        self.assertEqual(score_terminal(0), 2000)
        self.assertEqual(score_terminal(9), 1955)
        self.assertEqual(score_terminal(1), 1995)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            score_group(1)
        with self.assertRaises(ValueError):
            score_terminal(-1)


class TestCustomScoring(unittest.TestCase):
    def test_custom_functions_replace_defaults(self):
        config = ScoringConfig(
            group_score=lambda k: 10 * k,
            terminal_score=lambda r: 5000 if r == 0 else 0,
        )
        self.assertEqual(config.score_group(3), 30)
        self.assertEqual(config.score_terminal(0), 5000)
        self.assertEqual(config.score_terminal(1), 0)
        # 默认配置不受影响
        self.assertEqual(DEFAULT_SCORING.score_group(3), 45)

    def test_scoring_change_does_not_touch_board_logic(self):
        """同一棋盘、同一动作序列下，换计分只改分数，不改最终棋盘。"""
        from popstar.board import create_board

        state = create_board([[0, 0, 1], [1, 1, 0], [2, 2, 0]])
        config_a = ScoringConfig(group_score=lambda k: k * k, bonus_coefficient=0.0)
        config_b = ScoringConfig(group_score=lambda k: 100 * k, bonus_coefficient=0.0)

        game_a = Game(state, scoring=config_a)
        game_b = Game(state, scoring=config_b)
        while not game_a.is_finished():
            move_a = game_a.legal_moves()[0]
            move_b = next(m for m in game_b.legal_moves() if m.cells == move_a.cells)
            game_a.play(move_a)
            game_b.play(move_b)

        self.assertEqual(game_a.state, game_b.state)
        self.assertNotEqual(game_a.accumulated_score, game_b.accumulated_score)

    def test_terminal_bonus_uses_config(self):
        from popstar.board import create_board

        state = create_board([[0, 1], [1, 0]])  # 直接终局
        game = Game(state, scoring=ScoringConfig(clear_threshold=4, bonus_coefficient=5))
        self.assertTrue(game.is_finished())
        result = game.result()
        self.assertEqual(result.remaining_count, 4)
        self.assertEqual(result.terminal_bonus, 0)  # R >= A -> 0

        game2 = Game(state, scoring=ScoringConfig(clear_threshold=5, bonus_coefficient=5))
        self.assertEqual(game2.result().terminal_bonus, 5 * 1 * 1)


if __name__ == "__main__":
    unittest.main()
