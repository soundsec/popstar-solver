"""随机棋盘上的一致性回归：不变量、确定性、回放可复现。"""

import random
import unittest

from popstar.board import (
    apply_move,
    count_remaining,
    create_board,
    get_legal_moves,
    is_canonical,
    is_terminal,
    random_board,
)
from popstar.game import Game
from popstar.scoring import DEFAULT_SCORING, ScoringConfig


def play_random_game(seed, height, width, colors, config=None):
    rng = random.Random(seed)
    state = random_board(height, width, colors, rng)
    game = Game(state, scoring=config)
    while not game.is_finished():
        moves = game.legal_moves()
        game.play(moves[rng.randrange(len(moves))])
    return state, game


def moves_of(state, game):
    """从记录还原动作序列（按代表格在当时状态下重建 Move）。"""
    sequence = []
    cursor = state
    for record in game.records:
        for move in get_legal_moves(cursor):
            if move.representative_cell == record.representative_cell:
                sequence.append(move)
                break
        cursor = record.board_after_move
    return sequence


class TestRandomConsistency(unittest.TestCase):
    def test_no_illegal_transition_and_canonical_invariant(self):
        for seed in range(120):
            height = 3 + seed % 6
            width = 3 + (seed // 6) % 6
            colors = 2 + seed % 3
            state, game = play_random_game(seed, height, width, colors)
            self.assertTrue(is_canonical(state))
            previous_remaining = count_remaining(state)
            cursor = state
            for record in game.records:
                self.assertEqual(record.removed_count, previous_remaining - count_remaining(record.board_after_move))
                self.assertTrue(is_canonical(record.board_after_move))
                self.assertEqual(
                    (record.board_after_move.height, record.board_after_move.width),
                    (state.height, state.width),
                )
                previous_remaining = count_remaining(record.board_after_move)
                cursor = record.board_after_move
            self.assertEqual(cursor, game.state)
            self.assertTrue(is_terminal(game.state))

    def test_score_matches_independent_recomputation(self):
        for seed in range(60):
            _state, game = play_random_game(seed, 6, 6, 3)
            expected = sum(
                DEFAULT_SCORING.score_group(r.removed_count) for r in game.records
            )
            self.assertAlmostEqual(game.accumulated_score, expected)
            result = game.result()
            self.assertAlmostEqual(
                result.total_score,
                expected + DEFAULT_SCORING.score_terminal(result.remaining_count),
            )

    def test_replay_reproduces_every_random_game(self):
        for seed in range(60):
            state, game = play_random_game(seed, 7, 5, 4)
            replayed = Game.replay_records(state, game.records)
            self.assertEqual(replayed.state.grid, game.state.grid)
            self.assertAlmostEqual(
                replayed.accumulated_score, game.accumulated_score
            )

    def test_same_board_same_sequence_identical_result(self):
        for seed in range(40):
            state, game = play_random_game(seed, 6, 6, 3)
            moves_sequence = moves_of(state, game)
            first = Game.replay(state, moves_sequence)
            second = Game.replay(state, moves_sequence)
            self.assertEqual(first.state.grid, second.state.grid)
            self.assertEqual(first.accumulated_score, second.accumulated_score)
            self.assertEqual(first.state.grid, game.state.grid)

    def test_apply_move_matches_manual_pipeline(self):
        """独立地按 Remove -> Gravity -> Compression 复核 apply_move 的结果。"""
        for seed in range(40):
            rng = random.Random(1000 + seed)
            state = random_board(6, 6, 4, rng)
            moves = get_legal_moves(state)
            if not moves:
                continue
            move = moves[rng.randrange(len(moves))]
            after = apply_move(state, move)

            table = [list(row) for row in state.grid]
            for r, c in move.cells:
                table[r][c] = -1
            height, width = len(table), len(table[0])
            columns = []
            for c in range(width):
                stacked = [table[r][c] for r in range(height) if table[r][c] != -1]
                columns.append([-1] * (height - len(stacked)) + stacked)
            kept = [c for c in range(width) if any(v != -1 for v in columns[c])]
            expected = tuple(
                tuple(
                    [columns[c][r] for c in kept] + [-1] * (width - len(kept))
                )
                for r in range(height)
            )
            self.assertEqual(after.grid, expected)

    def test_terminal_states_are_stable(self):
        for seed in range(40):
            _state, game = play_random_game(seed, 5, 5, 2)
            self.assertEqual(get_legal_moves(game.state), ())
            with self.assertRaises(ValueError):
                Game(game.state).play_cell((0, 0))

    def test_scoring_config_replacement_changes_only_score(self):
        config = ScoringConfig(group_score=lambda k: 2 ** k, bonus_coefficient=1.0)
        for seed in range(20):
            state, game_default = play_random_game(seed, 6, 6, 3)
            # 按代表格序列用新配置回放：棋盘轨迹必须完全一致，只有分数改变
            replayed = Game.replay(
                state,
                moves_of(state, game_default),
                scoring=config,
            )
            self.assertEqual(replayed.state.grid, game_default.state.grid)
            expected = sum(config.score_group(r.removed_count) for r in replayed.records)
            self.assertAlmostEqual(replayed.accumulated_score, expected)
            self.assertEqual(
                [r.representative_cell for r in replayed.records],
                [r.representative_cell for r in game_default.records],
            )


if __name__ == "__main__":
    unittest.main()
