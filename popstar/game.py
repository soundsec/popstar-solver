"""一局游戏的推进、记录与回放（Phase 1）。

:class:`Game` 只负责：

* 调用 :func:`popstar.board.apply_move` 推进状态；
* 记录每一步的明细（颜色、代表格、移除数、即时得分、累计得分、动作后棋盘）；
* 在终局汇总 group_score / terminal_bonus / total_score / remaining。

它不实现任何搜索或决策逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .board import (
    BoardState,
    Cell,
    Component,
    Move,
    apply_move,
    assert_canonical,
    count_remaining,
    get_components,
    get_group,
    get_legal_moves,
    is_terminal,
)
from .scoring import DEFAULT_SCORING, ScoringConfig


@dataclass(frozen=True)
class MoveRecord:
    """一步动作的完整记录，可独立复现。"""

    step: int
    color: int
    representative_cell: Cell
    removed_count: int
    immediate_score: float
    accumulated_score: float
    board_after_move: BoardState


@dataclass(frozen=True)
class GameResult:
    """终局汇总。"""

    initial_board: BoardState
    final_board: BoardState
    group_score: float
    terminal_bonus: float
    total_score: float
    remaining_count: int
    move_count: int
    records: Tuple[MoveRecord, ...]

    def score_breakdown(self) -> Dict[str, float]:
        """拆分为 group score 与 terminal bonus 两部分（便于单独分析占比）。"""
        return {
            "group_score": self.group_score,
            "terminal_bonus": self.terminal_bonus,
            "total_score": self.total_score,
        }

    def group_sizes(self) -> Tuple[int, ...]:
        """逐步消除的方块数序列，可用于在任意 g(k) 下重算 group score。"""
        return tuple(record.removed_count for record in self.records)


class Game:
    """一局游戏的可变外壳（棋盘状态本身仍是不可变的）。"""

    def __init__(
        self,
        state: BoardState,
        scoring: Optional[ScoringConfig] = None,
        strict: bool = True,
    ) -> None:
        self.initial_state = state
        self.scoring = scoring or DEFAULT_SCORING
        self.strict = strict
        self.state = state
        self.accumulated_score = 0.0
        self.records: List[MoveRecord] = []

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def remaining_count(self) -> int:
        return count_remaining(self.state)

    def legal_moves(self) -> Tuple[Move, ...]:
        return get_legal_moves(self.state)

    def is_finished(self) -> bool:
        return is_terminal(self.state)

    # ------------------------------------------------------------------
    # 推进
    # ------------------------------------------------------------------

    def play(self, move: Move) -> MoveRecord:
        """执行一个合法动作并记录。"""
        if self.is_finished():
            raise ValueError("game is already finished")
        next_state = apply_move_checked(self.state, move, strict=self.strict)
        immediate = self.scoring.score_group(_move_size(move))
        self.accumulated_score += immediate
        record = MoveRecord(
            step=len(self.records) + 1,
            color=move.color,
            representative_cell=move.representative_cell,
            removed_count=_move_size(move),
            immediate_score=immediate,
            accumulated_score=self.accumulated_score,
            board_after_move=next_state,
        )
        self.state = next_state
        self.records.append(record)
        return record

    def play_cell(self, cell: Cell) -> MoveRecord:
        """按坐标消除：自动查询该坐标所属连通块并构造 Move。"""
        comp = get_group(self.state, cell)
        if comp is None:
            raise ValueError(f"cell {cell} is empty or out of range")
        if comp.size < 2:
            raise ValueError(
                f"cell {cell} belongs to a size-{comp.size} component; "
                "singletons cannot be removed"
            )
        return self.play(_component_to_move(self.state, comp))

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------

    def result(self, require_finished: bool = True) -> GameResult:
        """返回终局汇总。``require_finished=False`` 时允许中途查看。"""
        if require_finished and not self.is_finished():
            raise ValueError("game is not finished yet")
        remaining = count_remaining(self.state)
        bonus = self.scoring.score_terminal(remaining) if self.is_finished() else 0.0
        return GameResult(
            initial_board=self.initial_state,
            final_board=self.state,
            group_score=self.accumulated_score,
            terminal_bonus=bonus,
            total_score=self.accumulated_score + bonus,
            remaining_count=remaining,
            move_count=len(self.records),
            records=tuple(self.records),
        )

    # ------------------------------------------------------------------
    # 回放
    # ------------------------------------------------------------------

    @classmethod
    def replay(
        cls,
        initial_state: BoardState,
        moves: Sequence[Move],
        scoring: Optional[ScoringConfig] = None,
        strict: bool = True,
    ) -> "Game":
        """依据动作序列从初始棋盘完整重放一局（只使用代表格重建动作）。"""
        game = cls(initial_state, scoring=scoring, strict=strict)
        for move in moves:
            game.play_cell(move.representative_cell)
        return game

    @classmethod
    def replay_records(
        cls,
        initial_state: BoardState,
        records: Sequence[MoveRecord],
        scoring: Optional[ScoringConfig] = None,
        strict: bool = True,
    ) -> "Game":
        """依据历史记录（代表格序列）重放，并校验逐步棋盘与得分完全复现。"""
        game = cls(initial_state, scoring=scoring, strict=strict)
        for index, record in enumerate(records):
            new_record = game.play_cell(record.representative_cell)
            if new_record.board_after_move != record.board_after_move:
                raise AssertionError(
                    f"replay mismatch at step {index + 1}: board differs"
                )
            if abs(new_record.accumulated_score - record.accumulated_score) > 1e-9:
                raise AssertionError(
                    f"replay mismatch at step {index + 1}: score differs "
                    f"({new_record.accumulated_score} != {record.accumulated_score})"
                )
            if new_record.color != record.color:
                raise AssertionError(
                    f"replay mismatch at step {index + 1}: color differs"
                )
            if new_record.removed_count != record.removed_count:
                raise AssertionError(
                    f"replay mismatch at step {index + 1}: removed count differs"
                )
        return game


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _move_size(move: Move) -> int:
    return move.size if move.size else len(move.cells)


def _component_to_move(state: BoardState, comp: Component) -> Move:
    index = 0
    for current in get_components(state):
        if current.cells == comp.cells:
            break
        index += 1
    return Move(
        representative_cell=comp.representative_cell,
        color=comp.color,
        cells=comp.cells,
        size=comp.size,
        component_index=index,
    )


def apply_move_checked(
    state: BoardState, move: Move, strict: bool = True
) -> BoardState:
    """调用正式转移入口，并在 strict 模式下校验规范化不变量。"""
    next_state = apply_move(state, move)
    if strict:
        assert_canonical(next_state)
    return next_state


__all__ = [
    "Game",
    "GameResult",
    "MoveRecord",
]
