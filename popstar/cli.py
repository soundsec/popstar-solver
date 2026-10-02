"""命令行测试入口（Phase 1）。

用法::

    python main.py                          # 随机 10x10 四色棋盘，交互式
    python main.py --height 6 --width 6 --colors 3 --seed 7
    python main.py --file board.txt         # 从文本文件读取棋盘
    python main.py --auto                   # 自动演示：每步取最大连通块

交互模式下输入：

* ``<序号>``   选择列出的连通块
* ``r c``      直接按坐标选择（0-based，先行后列）
* ``q``        退出
"""

from __future__ import annotations

import argparse
import random
from typing import Optional

from .board import BoardState, Move, format_board, random_board
from .game import Game, MoveRecord
from .scoring import DEFAULT_SCORING, ScoringConfig


def _print_state(state: BoardState, title: str = "当前棋盘") -> None:
    print(f"\n{title} ({state.height}x{state.width}):")
    header = "   " + " ".join(f"{c % 10}" for c in range(state.width))
    print(header)
    for r, row in enumerate(state.grid):
        chars = []
        for value in row:
            if value == -1:
                chars.append(".")
            else:
                chars.append("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[value % 36])
        print(f"{r:2d} " + " ".join(chars))


def _print_moves(moves: tuple) -> None:
    print("\n可消除连通块:")
    if not moves:
        print("  （无，游戏结束）")
        return
    for index, move in enumerate(moves):
        cells_preview = " ".join(f"({r},{c})" for r, c in move.cells[:6])
        more = " ..." if move.size > 6 else ""
        print(
            f"  [{index:2d}] color={move.color} size={move.size:2d} "
            f"rep={move.representative_cell} cells={cells_preview}{more}"
        )


def _print_record(record: MoveRecord) -> None:
    print(
        f"  第 {record.step} 步: color={record.color} "
        f"rep={record.representative_cell} 消除 {record.removed_count} 个, "
        f"本步 {record.immediate_score:g} 分, 累计 {record.accumulated_score:g} 分"
    )


def _print_summary(game: Game, run_record=None) -> None:
    result = game.result(require_finished=False)
    print("\n" + "=" * 52)
    print("游戏结束" if game.is_finished() else "本局未走到终局")
    print(f"  步数            : {result.move_count}")
    print(f"  消除累计分      : {result.group_score:g}")
    print(f"  剩余方块        : {result.remaining_count}")
    print(f"  终局奖励        : {result.terminal_bonus:g}")
    print(f"  最终总分        : {result.total_score:g}")
    print(f"  计分配置        : {game.scoring}")
    if run_record is not None:
        print(f"  记录 ID         : {run_record.run_id}")
    print("=" * 52)


def _largest_move(game: Game) -> Optional[Move]:
    moves = game.legal_moves()
    if not moves:
        return None
    return max(moves, key=lambda m: (m.size, -m.representative_cell[0], -m.representative_cell[1]))


def run_interactive(game: Game) -> Game:
    while not game.is_finished():
        _print_state(game.state)
        print(f"累计得分: {game.accumulated_score:g}   剩余: {game.remaining_count}")
        moves = game.legal_moves()
        _print_moves(moves)
        raw = input("选择连通块（序号 / 'r c' / q 退出）: ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            return game
        if not raw:
            continue
        try:
            parts = raw.split()
            if len(parts) == 1:
                move = moves[int(parts[0])]
            elif len(parts) == 2:
                move = None
                cell = (int(parts[0]), int(parts[1]))
                for candidate in moves:
                    if cell in candidate.cells:
                        move = candidate
                        break
                if move is None:
                    print(f"坐标 {cell} 不可消除。")
                    continue
            else:
                print("输入格式错误。")
                continue
        except (ValueError, IndexError):
            print("输入无法解析，请重试。")
            continue
        _print_record(game.play(move))
    _print_state(game.state, "终局棋盘")
    return game


def run_auto(game: Game) -> Game:
    while not game.is_finished():
        move = _largest_move(game)
        if move is None:  # pragma: no cover - 由 is_finished 保证
            break
        _print_record(game.play(move))
    _print_state(game.state, "终局棋盘")
    return game


def _load_board(args: argparse.Namespace) -> BoardState:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as handle:
            from .board import parse_board

            return parse_board(handle.read())
    rng = random.Random(args.seed)
    return random_board(args.height, args.width, args.colors, rng)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PopStar / SameGame 基础引擎测试入口")
    parser.add_argument("--height", type=int, default=10, help="棋盘行数，默认 10")
    parser.add_argument("--width", type=int, default=10, help="棋盘列数，默认 10")
    parser.add_argument("--colors", type=int, default=4, help="颜色种类数，默认 4")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--file", type=str, default=None, help="棋盘文本文件")
    parser.add_argument("--auto", action="store_true", help="自动演示（每步取最大连通块）")
    parser.add_argument("--solve", choices=("none", "exact", "fast"), default="none",
                        help="用求解器直接求解而不进入交互（exact=精确，fast=beam 近似）")
    parser.add_argument("--beam-width", type=int, default=256, help="fast 模式的 beam 宽度")
    parser.add_argument("--time-limit", type=float, default=None, help="exact 模式时限（秒）")
    parser.add_argument("--log", type=str, default=None, help="把本局追加写入 JSONL 运行日志")
    parser.add_argument("--tag", type=str, default="", help="给本局打一个标签，便于后续分组统计")
    parser.add_argument("--threshold", type=int, default=None,
                        help="改用实验公式时的终局阈值 A（默认走通行规则）")
    parser.add_argument("--bonus", type=float, default=None,
                        help="改用实验公式 B=系数×(A−R)²（默认走通行规则）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    board = _load_board(args)
    if args.threshold is None and args.bonus is None:
        scoring = DEFAULT_SCORING
    else:
        scoring = ScoringConfig(
            clear_threshold=10 if args.threshold is None else args.threshold,
            bonus_coefficient=10.0 if args.bonus is None else args.bonus,
        )
    if args.solve != "none":
        from .records import RunLog, RunRecord
        from .solver import BeamSolver, ExactSolver, Solution

        if args.solve == "fast":
            solution: Solution = BeamSolver(
                scoring=scoring, beam_width=args.beam_width
            ).solve(board)
        else:
            solution = ExactSolver(
                scoring=scoring, use_memo=True, time_limit=args.time_limit
            ).solve(board)
        print(f"计分配置: {scoring}")
        print(solution.summary())
        print(
            f"  group_score={solution.group_score:g}  "
            f"terminal_bonus={solution.terminal_bonus:g}  "
            f"remaining={solution.remaining_count}"
        )
        if args.log:
            RunLog(args.log).append(RunRecord.from_solution(solution, scoring=scoring, tag=args.tag))
            print(f"已写入运行日志: {args.log}")
        return 0

    game = Game(board, scoring=scoring)
    print(f"计分配置: {scoring}")
    if args.auto:
        game = run_auto(game)
    else:
        game = run_interactive(game)

    run_record = None
    if args.log:
        from .records import RunLog, RunRecord

        run_record = RunRecord.from_game(
            game,
            mode="auto" if args.auto else "interactive",
            tag=args.tag,
            require_finished=False,
            extra={"seed": args.seed, "colors": args.colors},
        )
        RunLog(args.log).append(run_record)
        print(f"已写入运行日志: {args.log}")
    _print_summary(game, run_record)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
