"""运行记录：两部分得分分别落盘、可序列化、可事后重算。"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from popstar.analysis import compare_configs, format_summary, rescore, summarize
from popstar.board import get_legal_moves, random_board
from popstar.game import Game
from popstar.records import RunLog, RunRecord, describe_scoring
from popstar.scoring import DEFAULT_SCORING, ScoringConfig


def play_random(seed, height=6, width=6, colors=3, scoring=None):
    rng = random.Random(seed)
    game = Game(random_board(height, width, colors, rng), scoring=scoring)
    while not game.is_finished():
        moves = game.legal_moves()
        game.play(moves[rng.randrange(len(moves))])
    return game


class TestRunRecord(unittest.TestCase):
    def test_two_score_parts_stored_separately(self):
        game = play_random(1)
        result = game.result()
        record = RunRecord.from_game(game, mode="exact", tag="unit")
        self.assertAlmostEqual(record.group_score, result.group_score)
        self.assertAlmostEqual(record.terminal_bonus, result.terminal_bonus)
        self.assertAlmostEqual(
            record.total_score, record.group_score + record.terminal_bonus
        )
        self.assertEqual(record.remaining_count, result.remaining_count)
        self.assertEqual(record.group_sizes, result.group_sizes())
        self.assertTrue(record.finished)

    def test_json_roundtrip(self):
        game = play_random(2)
        record = RunRecord.from_game(game, mode="fast", tag="roundtrip")
        restored = RunRecord.from_json(record.to_json())
        self.assertEqual(restored.to_dict(), record.to_dict())
        self.assertEqual(restored.initial_state().grid, game.initial_state.grid)
        self.assertEqual(restored.group_sizes, record.group_sizes)

    def test_json_is_serializable(self):
        record = RunRecord.from_game(play_random(3))
        json.loads(record.to_json())  # 不应抛出
        self.assertIsInstance(record.to_dict()["moves"], list)
        self.assertGreater(len(record.to_dict()["moves"]), 0)

    def test_scoring_descriptor(self):
        config = ScoringConfig(group_score=lambda k: 3.0 * k, bonus_coefficient=7.0)
        descriptor = describe_scoring(config)
        self.assertEqual(descriptor["clear_threshold"], 10)
        self.assertEqual(descriptor["bonus_coefficient"], 7.0)
        self.assertEqual(descriptor["terminal_score"], "default_formula")
        json.dumps(descriptor)  # 不含函数对象，可安全序列化

    def test_moves_capture_full_detail(self):
        record = RunRecord.from_game(play_random(4))
        first = record.to_dict()["moves"][0]
        for key in (
            "step",
            "color",
            "representative_cell",
            "removed_count",
            "immediate_score",
            "accumulated_score",
            "board_after_move",
        ):
            self.assertIn(key, first)
        self.assertEqual(first["step"], 1)

    def test_unfinished_game_records_zero_bonus(self):
        rng = random.Random(5)
        game = Game(random_board(6, 6, 3, rng))
        game.play(game.legal_moves()[0])
        record = RunRecord.from_game(game, require_finished=False)
        self.assertFalse(record.finished)
        self.assertEqual(record.terminal_bonus, 0)
        self.assertGreater(record.group_score, 0)


class TestRunLog(unittest.TestCase):
    def test_append_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs" / "phase1.jsonl"
            log = RunLog(str(path))
            self.assertEqual(len(log), 0)
            records = [RunRecord.from_game(play_random(s), tag=f"s{s}") for s in range(5)]
            log.append_many(records)
            self.assertEqual(len(log), 5)
            loaded = log.load()
            self.assertEqual(len(loaded), 5)
            self.assertEqual([r.tag for r in loaded], [f"s{s}" for s in range(5)])
            self.assertEqual(loaded[0].to_dict(), records[0].to_dict())

    def test_log_is_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.jsonl"
            log = RunLog(str(path))
            log.append(RunRecord.from_game(play_random(6)))
            log.append(RunRecord.from_game(play_random(7)))
            self.assertEqual(len(log), 2)
            self.assertEqual(len(path.read_text(encoding="utf-8").strip().splitlines()), 2)


class TestRescore(unittest.TestCase):
    def setUp(self):
        self.records = [
            RunRecord.from_game(play_random(seed), tag=f"r{seed}") for seed in range(12)
        ]

    def test_rescore_reproduces_original_under_default_config(self):
        for record in self.records:
            breakdown = rescore(record, DEFAULT_SCORING)
            self.assertAlmostEqual(breakdown["group_score"], record.group_score)
            self.assertAlmostEqual(breakdown["terminal_bonus"], record.terminal_bonus)
            self.assertAlmostEqual(breakdown["total_score"], record.total_score)

    def test_rescore_under_changed_group_function(self):
        config = ScoringConfig(group_score=lambda k: 10 * k, bonus_coefficient=10.0)
        for record in self.records:
            expected = sum(10 * k for k in record.group_sizes)
            self.assertAlmostEqual(rescore(record, config)["group_score"], expected)

    def test_rescore_under_changed_bonus(self):
        low = ScoringConfig(bonus_coefficient=1.0)
        high = ScoringConfig(bonus_coefficient=100.0)
        for record in self.records:
            self.assertLessEqual(
                rescore(record, low)["terminal_bonus"],
                rescore(record, high)["terminal_bonus"],
            )

    def test_rescore_is_equivalent_to_replaying_with_new_config(self):
        """重算结果必须等于用新配置真正重打一局的得分。"""
        config = ScoringConfig(group_score=lambda k: 5 * k, bonus_coefficient=3.0)
        for seed in range(6):
            rng = random.Random(seed)
            board = random_board(6, 6, 3, rng)
            game_default = Game(board)
            while not game_default.is_finished():
                game_default.play(game_default.legal_moves()[0])
            record = RunRecord.from_game(game_default)

            cursor = board
            moves = []
            for rec in game_default.records:
                for move in get_legal_moves(cursor):
                    if move.representative_cell == rec.representative_cell:
                        moves.append(move)
                        break
                cursor = rec.board_after_move
            replayed = Game.replay(board, moves, scoring=config)
            expected = replayed.result()
            breakdown = rescore(record, config)
            self.assertAlmostEqual(breakdown["group_score"], expected.group_score)
            self.assertAlmostEqual(breakdown["terminal_bonus"], expected.terminal_bonus)
            self.assertAlmostEqual(breakdown["total_score"], expected.total_score)


class TestSummarize(unittest.TestCase):
    def test_summary_reports_both_parts(self):
        records = [RunRecord.from_game(play_random(s)) for s in range(10)]
        summary = summarize(records)
        self.assertEqual(summary["runs"], 10)
        for key in ("group_score", "terminal_bonus", "total_score", "remaining_count"):
            self.assertIn(key, summary)
            self.assertEqual(summary[key]["count"], 10)
        share = summary["bonus_share"]
        self.assertIsNotNone(share["mean"])
        self.assertGreaterEqual(share["mean"], 0)
        self.assertLessEqual(share["max"], 1.0)
        self.assertEqual(sum(summary["remaining_histogram"].values()), 10)

    def test_summary_of_empty(self):
        summary = summarize([])
        self.assertEqual(summary["runs"], 0)
        self.assertIn("无数据", format_summary(summary))

    def test_compare_configs_ranking(self):
        records = [RunRecord.from_game(play_random(s)) for s in range(10)]
        report = compare_configs(
            records,
            {
                "low": ScoringConfig(bonus_coefficient=1.0),
                "high": ScoringConfig(bonus_coefficient=100.0),
            },
        )
        self.assertLess(
            report["low"]["bonus_share"]["mean"], report["high"]["bonus_share"]["mean"]
        )
        self.assertAlmostEqual(
            report["low"]["group_score"]["mean"], report["high"]["group_score"]["mean"]
        )

    def test_format_summary_is_renderable(self):
        records = [RunRecord.from_game(play_random(s)) for s in range(5)]
        text = format_summary(summarize(records))
        self.assertIn("group_score", text)
        self.assertIn("terminal_bonus", text)
        self.assertIn("奖励占比", text)


if __name__ == "__main__":
    unittest.main()
