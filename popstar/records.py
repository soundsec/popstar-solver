"""一局对局的可序列化运行记录（Phase 1）。

设计目的
--------
Phase 2 求解器跑出真实最优解分布后，需要重新标定 ``g(k)`` 与 ``B(R)`` 的量级。
为了避免「换一次计分就要重跑一次搜索」，这里把**重算所需的全部原始量**落盘：

* ``group_sizes``：逐步消除的方块数序列 -> 可在任意 ``g`` 下重算 group score；
* ``remaining_count``：终局剩余 -> 可在任意 ``B`` 下重算 terminal bonus；
* ``group_score`` / ``terminal_bonus``：当时计分规则下的两部分得分，**分开存储**，
  不与 ``total_score`` 混在一起，便于直接观察两部分各自的分布。

日志为 JSONL（一行一条 JSON），追加写入，适合长期累积与后续批处理。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .board import BoardState
from .game import Game, GameResult
from .scoring import DEFAULT_SCORING, ScoringConfig

BoardGridJSON = List[List[int]]


def describe_scoring(config: ScoringConfig) -> Dict[str, Any]:
    """把计分配置压缩成可 JSON 化的描述（不含函数对象本身）。"""
    return {
        "group_score": getattr(config.group_score, "__name__", "custom"),
        "clear_threshold": config.clear_threshold,
        "bonus_coefficient": config.bonus_coefficient,
        "terminal_score": (
            getattr(config.terminal_score, "__name__", "custom")
            if config.terminal_score is not None
            else "default_formula"
        ),
    }


def _grid_to_json(state: BoardState) -> BoardGridJSON:
    return [list(row) for row in state.grid]


def _grid_from_json(grid: BoardGridJSON) -> BoardState:
    from .board import create_board

    return create_board(grid)


@dataclass(frozen=True)
class RunRecord:
    """一局对局的完整记录。

    ``group_score`` 与 ``terminal_bonus`` 始终分别保存，
    ``total_score`` 仅作为两者之和冗余存储，便于快速排序。
    """

    run_id: str
    timestamp: str
    mode: str
    tag: str
    height: int
    width: int
    initial_board: BoardGridJSON
    move_count: int
    group_score: float
    terminal_bonus: float
    total_score: float
    remaining_count: int
    group_sizes: Tuple[int, ...]
    scoring: Dict[str, Any]
    finished: bool
    is_proven_optimal: Optional[bool] = None
    elapsed_time: Optional[float] = None
    search_stats: Dict[str, Any] = field(default_factory=dict)
    moves: Tuple[Dict[str, Any], ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    def from_game(
        cls,
        game: Game,
        mode: str = "manual",
        tag: str = "",
        require_finished: bool = False,
        is_proven_optimal: Optional[bool] = None,
        elapsed_time: Optional[float] = None,
        search_stats: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> "RunRecord":
        result: GameResult = game.result(require_finished=require_finished)
        return cls(
            run_id=run_id or uuid.uuid4().hex[:12],
            timestamp=datetime.now().isoformat(timespec="seconds"),
            mode=mode,
            tag=tag,
            height=game.initial_state.height,
            width=game.initial_state.width,
            initial_board=_grid_to_json(game.initial_state),
            move_count=result.move_count,
            group_score=result.group_score,
            terminal_bonus=result.terminal_bonus,
            total_score=result.total_score,
            remaining_count=result.remaining_count,
            group_sizes=tuple(r.removed_count for r in game.records),
            scoring=describe_scoring(game.scoring),
            finished=game.is_finished(),
            is_proven_optimal=is_proven_optimal,
            elapsed_time=elapsed_time,
            search_stats=dict(search_stats or {}),
            moves=tuple(move_record_to_dict(r) for r in game.records),
            extra=dict(extra or {}),
        )

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "tag": self.tag,
            "height": self.height,
            "width": self.width,
            "initial_board": self.initial_board,
            "move_count": self.move_count,
            "group_score": self.group_score,
            "terminal_bonus": self.terminal_bonus,
            "total_score": self.total_score,
            "remaining_count": self.remaining_count,
            "group_sizes": list(self.group_sizes),
            "scoring": dict(self.scoring),
            "finished": self.finished,
            "is_proven_optimal": self.is_proven_optimal,
            "elapsed_time": self.elapsed_time,
            "search_stats": dict(self.search_stats),
            "moves": [dict(m) for m in self.moves],
            "extra": dict(self.extra),
        }
        return data

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        return cls(
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", ""),
            mode=data.get("mode", "unknown"),
            tag=data.get("tag", ""),
            height=data["height"],
            width=data["width"],
            initial_board=data["initial_board"],
            move_count=data["move_count"],
            group_score=data["group_score"],
            terminal_bonus=data["terminal_bonus"],
            total_score=data["total_score"],
            remaining_count=data["remaining_count"],
            group_sizes=tuple(data.get("group_sizes", ())),
            scoring=dict(data.get("scoring", {})),
            finished=data.get("finished", False),
            is_proven_optimal=data.get("is_proven_optimal"),
            elapsed_time=data.get("elapsed_time"),
            search_stats=dict(data.get("search_stats", {})),
            moves=tuple(dict(m) for m in data.get("moves", ())),
            extra=dict(data.get("extra", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "RunRecord":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_solution(
        cls,
        solution: Any,
        scoring: Optional[ScoringConfig] = None,
        tag: str = "",
        extra: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> "RunRecord":
        """从求解器的 :class:`~popstar.solver.Solution` 构造记录。

        与 :meth:`from_game` 一样把 ``group_score`` 与 ``terminal_bonus`` 分开存，
        并保留 ``group_sizes`` 以便后续换计分规则重算。
        """
        steps = tuple(solution.steps or ())
        stats = {
            "states_expanded": solution.states_expanded,
            "cache_hits": solution.cache_hits,
            "cache_size": solution.cache_size,
            "max_depth": solution.max_depth,
        }
        stats.update(
            {k: v for k, v in (solution.notes or {}).items() if isinstance(v, (int, float))}
        )
        return cls(
            run_id=run_id or uuid.uuid4().hex[:12],
            timestamp=datetime.now().isoformat(timespec="seconds"),
            mode=solution.search_mode,
            tag=tag,
            height=solution.initial_board.height,
            width=solution.initial_board.width,
            initial_board=_grid_to_json(solution.initial_board),
            move_count=solution.move_count,
            group_score=solution.group_score,
            terminal_bonus=solution.terminal_bonus,
            total_score=solution.best_score,
            remaining_count=solution.remaining_count,
            group_sizes=tuple(int(step["removed_count"]) for step in steps),
            scoring=describe_scoring(scoring or DEFAULT_SCORING),
            finished=True,
            is_proven_optimal=solution.is_proven_optimal,
            elapsed_time=solution.elapsed_time,
            search_stats=stats,
            moves=tuple(dict(step) for step in steps),
            extra=dict(extra or {}),
        )

    def initial_state(self) -> BoardState:
        """还原初始棋盘状态（便于回放复现）。"""
        return _grid_from_json(self.initial_board)

    def __str__(self) -> str:  # pragma: no cover - 调试便利
        return (
            f"RunRecord({self.mode}/{self.tag or '-'} moves={self.move_count} "
            f"group={self.group_score:g} bonus={self.terminal_bonus:g} "
            f"total={self.total_score:g} remaining={self.remaining_count})"
        )


def move_record_to_dict(record) -> Dict[str, Any]:
    return {
        "step": record.step,
        "color": record.color,
        "representative_cell": list(record.representative_cell),
        "removed_count": record.removed_count,
        "immediate_score": record.immediate_score,
        "accumulated_score": record.accumulated_score,
        "board_after_move": _grid_to_json(record.board_after_move),
    }


class RunLog:
    """JSONL 运行日志：追加写入、按行读取。"""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def append(self, record: RunRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")

    def append_many(self, records: Iterable[RunRecord]) -> None:
        for record in records:
            self.append(record)

    def load(self) -> List[RunRecord]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(RunRecord.from_json(line))
        return records

    def __len__(self) -> int:
        return len(self.load())


__all__ = [
    "RunRecord",
    "RunLog",
    "describe_scoring",
    "move_record_to_dict",
]
