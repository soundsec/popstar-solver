"""连通块识别、合法动作与终局判定。"""

import unittest

from popstar.board import (
    EMPTY,
    BoardState,
    apply_move,
    can_remove,
    count_remaining,
    create_board,
    format_board,
    get_components,
    get_group,
    get_legal_moves,
    is_canonical,
    is_terminal,
    parse_board,
)


def board(rows):
    return create_board(rows)


class TestComponentDetection(unittest.TestCase):
    def test_single_block_is_not_removable(self):
        state = board([[0, 1], [1, 0]])
        self.assertEqual(len(get_components(state)), 4)
        self.assertEqual(get_legal_moves(state), ())
        self.assertFalse(can_remove(state, (0, 0)))
        self.assertTrue(is_terminal(state))

    def test_horizontal_pair(self):
        state = board([[0, 0, 1], [2, 2, 1]])
        comps = get_components(state, min_size=2)
        self.assertEqual(len(comps), 3)
        pair = get_group(state, (0, 0))
        self.assertEqual(pair.cells, ((0, 0), (0, 1)))
        self.assertTrue(can_remove(state, (0, 1)))

    def test_vertical_pair(self):
        state = board([[0, 1], [0, 1]])
        pair = get_group(state, (0, 0))
        self.assertEqual(pair.cells, ((0, 0), (1, 0)))
        self.assertEqual(pair.size, 2)

    def test_diagonal_is_not_connected(self):
        state = board([[0, 1], [1, 0]])
        self.assertEqual(get_group(state, (0, 0)).cells, ((0, 0),))
        self.assertEqual(get_group(state, (1, 1)).cells, ((1, 1),))
        self.assertEqual(len(get_components(state, min_size=2)), 0)

    def test_l_shape(self):
        state = board([[0, 0], [0, 1]])
        comp = get_group(state, (0, 0))
        self.assertEqual(comp.cells, ((0, 0), (0, 1), (1, 0)))
        self.assertEqual(comp.size, 3)

    def test_t_shape(self):
        state = board([[0, 0, 0], [1, 0, 1], [1, 0, 1]])
        comp = get_group(state, (0, 1))
        self.assertEqual(
            comp.cells, ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1))
        )
        self.assertEqual(comp.size, 5)

    def test_irregular_shape(self):
        state = board([[0, 0, 1], [1, 0, 1], [0, 0, 1]])
        comp = get_group(state, (1, 1))
        self.assertEqual(comp.cells, ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)))
        vertical = get_group(state, (0, 2))
        self.assertEqual(vertical.cells, ((0, 2), (1, 2), (2, 2)))

    def test_empty_and_out_of_range_lookup(self):
        state = board([[0, 0], [1, EMPTY]])
        self.assertIsNone(get_group(state, (1, 1)))
        self.assertIsNone(get_group(state, (5, 5)))
        self.assertFalse(can_remove(state, (1, 1)))

    def test_move_dedup_per_component(self):
        """同一连通块的每个坐标只产生一个动作。"""
        state = board([[0, 0, 0], [1, 2, 2]])
        moves = get_legal_moves(state)
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[0].representative_cell, (0, 0))
        self.assertEqual(moves[0].size, 3)
        indices = {m.component_index for m in moves}
        self.assertEqual(len(indices), len(moves))

    def test_move_rejects_size_one(self):
        from popstar.board import Move

        with self.assertRaises(ValueError):
            Move(representative_cell=(0, 0), color=0, cells=((0, 0),), size=1)


class TestTerminal(unittest.TestCase):
    def test_terminal_with_many_singletons(self):
        state = board([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        self.assertTrue(is_terminal(state))
        self.assertEqual(count_remaining(state), 9)

    def test_not_terminal(self):
        state = board([[0, 1], [1, 1]])
        self.assertFalse(is_terminal(state))


class TestBoardCreation(unittest.TestCase):
    def test_rectangular_requirement(self):
        with self.assertRaises(ValueError):
            create_board([[0, 1], [2]])

    def test_invalid_cell_value(self):
        with self.assertRaises(ValueError):
            create_board([[0, -2]])

    def test_parse_board(self):
        state = parse_board(
            """
            A B .
            A . B
            # 注释行
            C C B
            """
        )
        self.assertEqual(state.grid, ((0, 1, -1), (0, -1, 1), (2, 2, 1)))

    def test_parse_compact(self):
        state = parse_board("AAB\nBBA")
        self.assertEqual(state.grid, ((0, 0, 1), (1, 1, 0)))

    def test_format_roundtrip(self):
        state = parse_board("A B .\nA . B")
        self.assertIn(".", format_board(state))


class TestImmutability(unittest.TestCase):
    def test_apply_move_does_not_mutate_source(self):
        state = board([[0, 0], [1, 1]])
        move = get_legal_moves(state)[0]
        before = state.grid
        apply_move(state, move)
        self.assertEqual(state.grid, before)

    def test_board_state_is_hashable(self):
        state = board([[0, 0], [1, 1]])
        self.assertEqual(hash(state), hash(create_board([[0, 0], [1, 1]])))

    def test_canonical_initial_board(self):
        self.assertTrue(is_canonical(board([[0, 0], [1, 1]])))


if __name__ == "__main__":
    unittest.main()
