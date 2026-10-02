"""状态转移：Remove -> Vertical Gravity -> Horizontal Compression。"""

import random
import unittest

from popstar.board import (
    EMPTY,
    apply_gravity,
    apply_move,
    compress_columns,
    create_board,
    get_legal_moves,
    get_group,
    is_canonical,
    random_board,
)


def board(rows):
    return create_board(rows)


def move_at(state, cell):
    comp = get_group(state, cell)
    for m in get_legal_moves(state):
        if m.cells == comp.cells:
            return m
    raise AssertionError("no legal move for cell")


class TestGravityUnit(unittest.TestCase):
    def test_column_collapses_to_bottom(self):
        grid = ((0,), (EMPTY,), (1,), (EMPTY,), (2,))
        self.assertEqual(
            apply_gravity(grid), ((EMPTY,), (EMPTY,), (0,), (1,), (2,))
        )

    def test_relative_order_preserved(self):
        grid = ((3,), (EMPTY,), (1,), (3,), (2,))
        self.assertEqual(
            apply_gravity(grid), ((EMPTY,), (3,), (1,), (3,), (2,))
        )

    def test_already_settled_column_unchanged(self):
        grid = ((EMPTY,), (EMPTY,), (7,), (8,))
        self.assertEqual(apply_gravity(grid), grid)


class TestCompressionUnit(unittest.TestCase):
    def test_empty_column_removed(self):
        grid = ((0, EMPTY, 5), (1, EMPTY, 6))
        self.assertEqual(compress_columns(grid), ((0, 5, EMPTY), (1, 6, EMPTY)))

    def test_multiple_empty_columns_at_once(self):
        grid = ((0, EMPTY, 5, EMPTY), (1, EMPTY, 6, EMPTY))
        self.assertEqual(
            compress_columns(grid), ((0, 5, EMPTY, EMPTY), (1, 6, EMPTY, EMPTY))
        )

    def test_width_is_preserved(self):
        grid = ((EMPTY, EMPTY), (EMPTY, EMPTY))
        self.assertEqual(compress_columns(grid), grid)
        self.assertEqual(len(compress_columns(grid)[0]), 2)


class TestMoveIntegration(unittest.TestCase):
    def test_gravity_only_vertical(self):
        """移除中段连通块后，仅发生垂直下落，且列内相对顺序不变。

        第 0 列为 [5, 6, 6, 7]，移除中间的 6-6 后应为 [., ., 5, 7]。
        """
        state = board([[5, 0, 1], [6, 2, 1], [6, 3, 4], [7, 3, 4]])
        move = move_at(state, (1, 0))
        self.assertEqual(move.cells, ((1, 0), (2, 0)))
        after = apply_move(state, move)
        self.assertEqual(
            after.grid,
            (
                (EMPTY, 0, 1),
                (EMPTY, 2, 1),
                (5, 3, 4),
                (7, 3, 4),
            ),
        )
        self.assertTrue(is_canonical(after))

    def test_column_cleared_shifts_right_columns_left(self):
        state = board([[0, 1, 2], [0, 1, 3], [4, 1, 5]])
        move = move_at(state, (0, 1))  # 第 1 列整列同色，大小 3
        self.assertEqual(move.size, 3)
        after = apply_move(state, move)
        self.assertEqual(
            after.grid, ((0, 2, EMPTY), (0, 3, EMPTY), (4, 5, EMPTY))
        )
        self.assertTrue(is_canonical(after))

    def test_multiple_empty_columns_normalized_in_one_move(self):
        """一次消除同时清空第 1、3 列，右侧所有空列一次性归位。"""
        state = board([[0, 1, 2, 1], [0, 1, 1, 1], [4, 1, 2, 1]])
        move = move_at(state, (0, 1))
        self.assertEqual(move.size, 7)
        after = apply_move(state, move)
        self.assertEqual(
            after.grid,
            (
                (0, EMPTY, EMPTY, EMPTY),
                (0, 2, EMPTY, EMPTY),
                (4, 2, EMPTY, EMPTY),
            ),
        )
        self.assertTrue(is_canonical(after))

    def test_no_automatic_chain_reaction(self):
        """消除后新形成的同色连通块必须保留到下一步，不得自动连锁。"""
        # 第 1 列为 [2, 1, 1, 2]，移除中间的 1-1 后两个 2 变为相邻，
        # 它们必须保留在棋盘上，留待下一步由玩家主动消除。
        state = board([[0, 2, 5], [3, 1, 5], [3, 1, 6], [0, 2, 6]])
        move = move_at(state, (1, 1))
        self.assertEqual(move.cells, ((1, 1), (2, 1)))
        after = apply_move(state, move)
        self.assertEqual(
            after.grid,
            ((0, EMPTY, 5), (3, EMPTY, 5), (3, 2, 6), (0, 2, 6)),
        )
        pair = get_group(after, (2, 1))
        self.assertEqual(pair.cells, ((2, 1), (3, 1)))
        self.assertIn(pair.cells, [m.cells for m in get_legal_moves(after)])

    def test_diagonal_only_blocks_never_merge(self):
        state = board([[0, 1], [1, 0]])
        self.assertEqual(get_legal_moves(state), ())

    def test_full_cascade_keeps_canonical(self):
        state = board([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        # 棋盘全为孤立块 -> 无合法动作
        self.assertEqual(get_legal_moves(state), ())


class TestApplyMoveValidation(unittest.TestCase):
    def test_rejects_singleton(self):
        from popstar.board import Component, Move

        state = board([[0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            apply_move(state, Component(0, ((0, 0),)))

        with self.assertRaises(ValueError):
            apply_move(state, Move((0, 0), 0, ((0, 0),), 1))

    def test_rejects_color_mismatch(self):
        from popstar.board import Move

        state = board([[0, 0], [1, 1]])
        bad = Move((0, 0), 1, ((0, 0), (0, 1)), 2)
        with self.assertRaises(ValueError):
            apply_move(state, bad)

    def test_rejects_out_of_range(self):
        from popstar.board import Move

        state = board([[0, 0], [1, 1]])
        bad = Move((0, 0), 0, ((0, 0), (9, 9)), 2)
        with self.assertRaises(ValueError):
            apply_move(state, bad)

    def test_rejects_disconnected_cell_set(self):
        from popstar.board import Move

        # 三个同色块互不相连，不构成一个合法动作
        state = board([[0, 1, 0], [1, 1, 0]])
        bad = Move((0, 0), 0, ((0, 0), (0, 2), (1, 2)), 3)
        with self.assertRaises(ValueError):
            apply_move(state, bad)

    def test_component_input_accepted(self):
        from popstar.board import Component

        state = board([[0, 0], [1, 1]])
        after = apply_move(state, Component(0, ((0, 0), (0, 1))))
        self.assertEqual(after.grid, ((EMPTY, EMPTY), (1, 1)))


class TestUnvalidatedFastPath(unittest.TestCase):
    """``apply_move(..., validate=False)`` 只是省掉重复校验，转移管线必须一致。

    这是搜索热路径用到的快路径，因此必须对拍到「逐格相同」的强度。
    """

    def test_fast_path_matches_validated_path(self):
        rng = random.Random(4242)
        for _ in range(40):
            state = random_board(7, 7, 4, rng)
            cursor = state
            steps = 0
            while steps < 12:
                moves = get_legal_moves(cursor)
                if not moves:
                    break
                move = moves[rng.randrange(len(moves))]
                checked = apply_move(cursor, move)
                fast = apply_move(cursor, move, validate=False)
                self.assertEqual(checked.grid, fast.grid)
                self.assertEqual(checked.packed_key, fast.packed_key)
                self.assertTrue(is_canonical(fast))
                cursor = checked
                steps += 1

    def test_fast_path_still_uses_same_pipeline(self):
        """快路径不得绕过 Remove->Gravity->Compression 任一环节。"""
        state = board(
            [
                [0, 1, 2],
                [0, 1, 2],
                [3, 1, 0],
            ]
        )
        move = move_at(state, (0, 0))  # 竖着的两个 0
        fast = apply_move(state, move, validate=False)
        self.assertTrue(is_canonical(fast))
        # 手动按同一顺序重算一遍，作为独立参照
        from popstar.board import apply_gravity, compress_columns, remove_cells_raw

        manual = compress_columns(apply_gravity(remove_cells_raw(state.grid, move.cells)))
        self.assertEqual(fast.grid, manual)

    def test_validation_still_runs_by_default(self):
        from popstar.board import Move

        state = board([[0, 0], [1, 1]])
        bad = Move((0, 0), 1, ((0, 0), (0, 1)), 2)
        with self.assertRaises(ValueError):
            apply_move(state, bad)


if __name__ == "__main__":
    unittest.main()
