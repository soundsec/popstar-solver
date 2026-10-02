"""受控棋盘生成器与盘面画像。

重点验证两件事：

1. 生成器确实能分别制造「数量不均」与「空间聚集」，且两者不互相污染；
2. 画像指标能正确区分这两种偏差——这是后续所有结论可信的前提。
"""

import random
import unittest

from popstar.board import EMPTY, color_counts, get_legal_moves
from popstar.generators import (
    GENERATOR_NAMES,
    adjacency_ratio,
    balance,
    board_profile,
    clustered_board,
    make_generator,
    same_color_adjacency,
    uniform_board,
    weighted_board,
)


class TestGeneratorsProduceValidBoards(unittest.TestCase):
    def test_uniform_matches_random_board(self):
        a = uniform_board(6, 6, 4, random.Random(11))
        from popstar.board import random_board

        b = random_board(6, 6, 4, random.Random(11))
        self.assertEqual(a.grid, b.grid)

    def test_weighted_board_respects_weights(self):
        # 大幅偏向 0 号色，样本量足够大时应接近 0.8
        state = weighted_board(20, 20, 2, random.Random(3), [0.8, 0.2])
        counts = color_counts(state)
        self.assertEqual(sum(counts.values()), 400)
        share = counts[0] / 400
        self.assertGreater(share, 0.7)
        self.assertLess(share, 0.9)

    def test_weighted_board_rejects_bad_weights(self):
        with self.assertRaises(ValueError):
            weighted_board(4, 4, 3, random.Random(1), [1.0, 1.0])
        with self.assertRaises(ValueError):
            weighted_board(4, 4, 2, random.Random(1), [0.0, 0.0])
        with self.assertRaises(ValueError):
            weighted_board(4, 4, 2, random.Random(1), [-1.0, 2.0])

    def test_clustered_board_keeps_every_color_present(self):
        # 中心抽样可能让某色缺席，生成器必须保证每色至少一个中心
        for seed in range(20):
            state = clustered_board(10, 10, 4, random.Random(seed), centers=6)
            self.assertEqual(len(color_counts(state)), 4, f"seed={seed}")

    def test_clustered_board_mix_bounds(self):
        with self.assertRaises(ValueError):
            clustered_board(5, 5, 3, random.Random(1), mix=1.5)

    def test_determinism(self):
        a = clustered_board(8, 8, 4, random.Random(77), centers=5, mix=0.1)
        b = clustered_board(8, 8, 4, random.Random(77), centers=5, mix=0.1)
        self.assertEqual(a.grid, b.grid)

    def test_generators_are_playable(self):
        for name in GENERATOR_NAMES:
            gen = make_generator(name)
            state = gen(10, 10, 4, random.Random(5))
            self.assertGreater(len(get_legal_moves(state)), 0, name)
            self.assertTrue(all(v >= 0 or v == EMPTY
                                for row in state.grid for v in row), name)

    def test_make_generator_unknown_name(self):
        with self.assertRaises(ValueError):
            make_generator("nope")


class TestProfileMetrics(unittest.TestCase):
    """画像指标必须能把「数量不均」与「空间聚集」分开。"""

    def test_balance_is_one_for_perfectly_even_counts(self):
        from popstar.board import create_board

        # 4 色各 4 块，且交错排布（交错只影响 adjacency，不影响 balance）
        state = create_board(
            [
                [0, 1, 0, 1],
                [2, 3, 2, 3],
                [0, 1, 0, 1],
                [2, 3, 2, 3],
            ]
        )
        self.assertAlmostEqual(balance(state), 1.0, places=6)

    def test_balance_drops_when_one_color_dominates(self):
        state = weighted_board(20, 20, 4, random.Random(9), [0.85, 0.05, 0.05, 0.05])
        self.assertLess(balance(state), 0.7)
        self.assertGreater(balance(state), 0.0)

    def test_adjacency_of_uniform_board_is_near_one_over_c(self):
        # iid 均匀下同色相邻概率 = Σ p_c² = 1/C；用大棋盘压住采样噪声
        state = uniform_board(40, 40, 4, random.Random(21))
        self.assertAlmostEqual(same_color_adjacency(state), 0.25, delta=0.05)

    def test_adjacency_ratio_near_one_for_uniform(self):
        state = uniform_board(40, 40, 4, random.Random(21))
        self.assertAlmostEqual(adjacency_ratio(state), 1.0, delta=0.2)

    def test_adjacency_ratio_detects_clustering(self):
        state = clustered_board(40, 40, 4, random.Random(21), centers=8, mix=0.0)
        self.assertGreater(adjacency_ratio(state), 2.0)

    def test_imbalance_alone_does_not_look_like_clustering(self):
        """关键：数量不均会把「同色相邻占比」抬高，但不能被误判成空间聚集。"""
        state = weighted_board(40, 40, 4, random.Random(21), [0.85, 0.05, 0.05, 0.05])
        # 裸指标确实被抬高了：Σp_c² = .7225+.0025*3 = 0.73
        self.assertGreater(same_color_adjacency(state), 0.6)
        # 但归一化后仍应 ≈1（空间上确实独立）
        self.assertAlmostEqual(adjacency_ratio(state), 1.0, delta=0.2)

    def test_profile_contents(self):
        state = uniform_board(10, 10, 4, random.Random(1))
        profile = board_profile(state)
        self.assertEqual(profile["remaining"], 100)
        self.assertEqual(profile["colors"], 4)
        self.assertAlmostEqual(profile["adjacency_expected"], 0.25)
        self.assertIn("balance", profile)
        self.assertIn("adjacency_ratio", profile)


if __name__ == "__main__":
    unittest.main()
