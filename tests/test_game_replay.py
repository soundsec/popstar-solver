"""游戏流程：动作记录、累计得分、终局汇总与严格回放。"""

import random
import unittest

from popstar.board import create_board, get_legal_moves, is_canonical, parse_board
from popstar.game import Game
from popstar.scoring import DEFAULT_SCORING, ScoringConfig


class TestGameFlow(unittest.TestCase):
    def setUp(self):
        self.state = parse_board(
            """
            A A B C
            D B B C
            D E F F
            G E H H
            """
        )

    def test_play_records_every_step(self):
        game = Game(self.state)
        move = game.legal_moves()[0]
        record = game.play(move)
        self.assertEqual(record.step, 1)
        self.assertEqual(record.removed_count, move.size)
        self.assertEqual(record.immediate_score, DEFAULT_SCORING.score_group(move.size))
        self.assertEqual(record.accumulated_score, DEFAULT_SCORING.score_group(move.size))
        self.assertNotEqual(record.board_after_move, self.state)

    def test_accumulated_score_is_sum_of_immediate(self):
        game = Game(self.state)
        total = 0.0
        while not game.is_finished():
            move = game.legal_moves()[0]
            total += DEFAULT_SCORING.score_group(move.size)
            game.play(move)
        self.assertAlmostEqual(game.accumulated_score, total)

    def test_play_cell_and_reject_singleton(self):
        game = Game(self.state)
        with self.assertRaises(ValueError):
            game.play_cell((3, 0))  # G 为孤立块，不可主动消除
        record = game.play_cell((0, 0))  # A-A 水平对
        self.assertEqual(record.removed_count, 2)

    def test_result_summary(self):
        game = Game(self.state)
        while not game.is_finished():
            game.play(game.legal_moves()[0])
        result = game.result()
        self.assertEqual(result.move_count, len(game.records))
        self.assertEqual(result.group_score, game.accumulated_score)
        self.assertAlmostEqual(
            result.total_score, result.group_score + result.terminal_bonus
        )
        self.assertEqual(result.final_board, game.state)
        self.assertEqual(result.initial_board, self.state)

    def test_result_before_finish_raises(self):
        game = Game(self.state)
        with self.assertRaises(ValueError):
            game.result()

    def test_play_after_finish_raises(self):
        game = Game(self.state)
        while not game.is_finished():
            game.play(game.legal_moves()[0])
        with self.assertRaises(ValueError):
            if game.legal_moves():
                game.play(game.legal_moves()[0])
            else:
                game.play_cell((0, 0))


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.state = parse_board(
            """
            A A B C
            D B B C
            D E F F
            G E H H
            """
        )
        self.game = Game(self.state)
        rng = random.Random(20260930)
        while not self.game.is_finished():
            moves = self.game.legal_moves()
            self.game.play(moves[rng.randrange(len(moves))])

    def test_replay_from_moves_reproduces_board_and_score(self):
        replayed = Game.replay(self.state, [m for m in self._played_moves()])
        self.assertEqual(replayed.state, self.game.state)
        self.assertAlmostEqual(replayed.accumulated_score, self.game.accumulated_score)
        self.assertEqual(replayed.records, self.game.records)

    def test_replay_records_strict_verification(self):
        replayed = Game.replay_records(self.state, self.game.records)
        self.assertEqual(replayed.state, self.game.state)
        self.assertEqual(
            replayed.result().total_score, self.game.result().total_score
        )

    def test_replay_records_detects_tampering(self):
        from popstar.game import MoveRecord

        tampered = list(self.game.records)
        last = tampered[-1]
        tampered[-1] = MoveRecord(
            step=last.step,
            color=last.color,
            representative_cell=last.representative_cell,
            removed_count=last.removed_count + 1,
            immediate_score=last.immediate_score,
            accumulated_score=last.accumulated_score,
            board_after_move=last.board_after_move,
        )
        with self.assertRaises(AssertionError):
            Game.replay_records(self.state, tampered)

    def test_determinism_same_sequence_same_result(self):
        moves = self._played_moves()
        first = Game.replay(self.state, moves)
        second = Game.replay(self.state, moves)
        self.assertEqual(first.state.grid, second.state.grid)
        self.assertEqual(first.accumulated_score, second.accumulated_score)

    def _played_moves(self):
        """从记录中还原动作序列（按代表格重建）。"""
        sequence = []
        state = self.state
        for record in self.game.records:
            for move in get_legal_moves(state):
                if move.representative_cell == record.representative_cell:
                    sequence.append(move)
                    break
            state = record.board_after_move
        return sequence


class TestStrictInvariant(unittest.TestCase):
    def test_every_intermediate_state_is_canonical(self):
        rng = random.Random(7)
        for _ in range(30):
            rows = [[rng.randrange(3) for _ in range(6)] for _ in range(6)]
            state = create_board(rows)
            game = Game(state)
            self.assertTrue(is_canonical(state))
            while not game.is_finished():
                moves = game.legal_moves()
                game.play(moves[rng.randrange(len(moves))])
                self.assertTrue(is_canonical(game.state))

    def test_scoring_config_propagates(self):
        config = ScoringConfig(clear_threshold=3, bonus_coefficient=1.0)
        game = Game(create_board([[0, 1], [1, 0]]), scoring=config)
        self.assertIs(game.scoring, config)
        self.assertEqual(game.result().remaining_count, 4)
        self.assertEqual(game.result().terminal_bonus, 0)


if __name__ == "__main__":
    unittest.main()
