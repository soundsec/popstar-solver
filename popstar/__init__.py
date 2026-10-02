"""PopStar / SameGame 确定性消除游戏核心（Phase 1）。

公开接口::

    from popstar.board import create_board, get_legal_moves, apply_move, is_terminal
    from popstar.scoring import ScoringConfig, score_group, score_terminal
    from popstar.game import Game
"""

from .board import (
    EMPTY,
    BoardState,
    Component,
    Move,
    apply_move,
    create_board,
    get_legal_moves,
    is_terminal,
    parse_board,
)
from .analysis import compare_configs, rescore, summarize
from .game import Game
from .records import RunLog, RunRecord
from .scoring import DEFAULT_SCORING, ScoringConfig, score_group, score_terminal

__version__ = "0.1.0"

__all__ = [
    "EMPTY",
    "BoardState",
    "Component",
    "Move",
    "Game",
    "ScoringConfig",
    "DEFAULT_SCORING",
    "RunRecord",
    "RunLog",
    "rescore",
    "summarize",
    "compare_configs",
    "create_board",
    "parse_board",
    "apply_move",
    "get_legal_moves",
    "is_terminal",
    "score_group",
    "score_terminal",
    "__version__",
]
