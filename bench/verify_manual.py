"""手工构造棋盘的状态转移验证（Phase 1 人工检查用）。

运行::

    python -m bench.verify_manual

逐个展示：初始棋盘 -> 连通块清单 -> 选定动作 -> 转移后棋盘 -> 得分，
末尾附加一个 10x10 四色随机局的端到端回放校验。
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from popstar.analysis import format_summary, summarize
from popstar.board import (
    count_remaining,
    format_board,
    get_components,
    get_legal_moves,
    is_canonical,
    is_terminal,
    parse_board,
    random_board,
)
from popstar.game import Game
from popstar.records import RunLog, RunRecord


def show(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def inspect(state, note: str = "") -> None:
    print(f"\n{note}棋盘 ({state.height}x{state.width}), 剩余 {count_remaining(state)}:")
    print(format_board(state))
    comps = get_components(state)
    legal = get_legal_moves(state)
    print(f"连通块总数 {len(comps)}，其中可消除 {len(legal)}：")
    for comp in comps:
        flag = "可消除" if comp.size >= 2 else "孤立块"
        print(f"  color={comp.color} size={comp.size} {flag} rep={comp.representative_cell}")
    print(f"规范化检查: {is_canonical(state)}    终局: {is_terminal(state)}")


def case_diagonal() -> None:
    show("用例 1：仅对角接触的同色块不连通 -> 无合法动作，直接终局")
    state = parse_board("A B\nB A")
    inspect(state, "初始")
    assert get_legal_moves(state) == ()
    assert is_terminal(state)


def case_l_shape_and_gravity() -> None:
    show("用例 2：L 型连通块完整识别 + 中段消除只产生垂直下落")
    state = parse_board(
        """
        5 0 1
        6 2 1
        6 3 4
        7 3 4
        """
    )
    inspect(state, "初始")
    game = Game(state)
    move = next(m for m in game.legal_moves() if m.representative_cell == (1, 0))
    print(f"\n选定动作: color={move.color} size={move.size} cells={move.cells}")
    record = game.play(move)
    inspect(record.board_after_move, "消除后")
    print(f"本步得分 {record.immediate_score:g}，累计 {record.accumulated_score:g}")
    print(
        "第 0 列原字符为 [5, 6, 6, 7]，移除中间的 6-6 后应为 [., ., 5, 7]："
        "上方两块整体落到底部且相对顺序不变（仅垂直下落，无水平位移）。"
    )


def case_two_empty_columns() -> None:
    show("用例 3：一次消除清空两列 -> 多个空列一次性规范化到最右侧")
    state = parse_board(
        """
        0 1 2 1
        0 1 1 1
        4 1 2 1
        """
    )
    inspect(state, "初始")
    game = Game(state)
    move = next(m for m in game.legal_moves() if m.representative_cell == (0, 1))
    print(f"\n选定动作: color={move.color} size={move.size} cells={move.cells}")
    record = game.play(move)
    inspect(record.board_after_move, "消除后")
    assert is_canonical(record.board_after_move)


def case_no_auto_chain() -> None:
    show("用例 4：消除后新形成的同色块不得自动连锁，必须留到下一步")
    state = parse_board(
        """
        A C E
        B D E
        B D F
        A C F
        """
    )
    inspect(state, "初始")
    game = Game(state)
    move = next(m for m in game.legal_moves() if m.representative_cell == (1, 1))
    print(f"\n选定动作（第 1 列的 D-D）: size={move.size} cells={move.cells}")
    record = game.play(move)
    inspect(record.board_after_move, "消除后")
    print("注意：第 1 列的两个 C 已相邻，但仍在棋盘上，未被自动消除。")


def case_full_game_10x10() -> RunRecord:
    show("用例 5：10x10 四色随机局端到端 + 回放校验")
    rng = random.Random(20260930)
    state = random_board(10, 10, 4, rng)
    game = Game(state)
    print("初始棋盘：")
    print(format_board(state))
    while not game.is_finished():
        moves = game.legal_moves()
        game.play(moves[rng.randrange(len(moves))])
    result = game.result()
    print("\n终局棋盘：")
    print(format_board(result.final_board))
    print("得分拆分（两部分分别记录）：")
    print(f"  group_score    = {result.group_score:g}   （{result.group_sizes()}）")
    print(f"  terminal_bonus = {result.terminal_bonus:g}   （剩余 {result.remaining_count}）")
    print(f"  total_score    = {result.total_score:g}")
    replayed = Game.replay_records(state, game.records)
    assert replayed.state.grid == game.state.grid
    assert abs(replayed.result().total_score - result.total_score) < 1e-9
    print("回放校验：最终棋盘与总分完全一致 ✓")
    return RunRecord.from_game(game, mode="random", tag="manual-case5")


def case_batch_records(num_games: int = 40) -> list:
    """批量随机对局，验证 group score / terminal bonus 两部分被分别落盘。"""
    show(f"用例 6：{num_games} 局随机对局的两部分得分记录与事后重算")
    records = []
    for seed in range(num_games):
        rng = random.Random(90_000 + seed)
        height = 6 + seed % 5
        width = 6 + (seed // 5) % 5
        state = random_board(height, width, 4, rng)
        game = Game(state)
        while not game.is_finished():
            moves = game.legal_moves()
            game.play(moves[rng.randrange(len(moves))])
        records.append(
            RunRecord.from_game(game, mode="random", tag=f"batch-{height}x{width}")
        )

    summary = summarize(records)
    print(format_summary(summary))
    print("\n同一批对局在不同 bonus_coefficient 下的重算结果：")
    from popstar.analysis import compare_configs
    from popstar.scoring import ScoringConfig

    report = compare_configs(
        records,
        {
            "实验公式 coef=10": ScoringConfig(bonus_coefficient=10.0),
            "coef=50": ScoringConfig(bonus_coefficient=50.0),
            "coef=200": ScoringConfig(bonus_coefficient=200.0),
        },
    )
    for name, stats in report.items():
        share = stats["bonus_share"]
        print(
            f"  {name:<20} group_mean={stats['group_score']['mean']:8.2f} "
            f"bonus_mean={stats['terminal_bonus']['mean']:9.2f} "
            f"奖励占比 mean={share['mean']:.3f} max={share['max']:.3f}"
        )
    print("\n（无需重跑对局，仅凭 group_sizes + remaining 即可重算）")
    return records


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="手工棋盘状态转移验证")
    parser.add_argument("--log", type=str, default=None, help="把对局追加写入 JSONL 日志")
    parser.add_argument("--games", type=int, default=40, help="用例 6 的随机对局数")
    args = parser.parse_args(argv)

    case_diagonal()
    case_l_shape_and_gravity()
    case_two_empty_columns()
    case_no_auto_chain()
    records = [case_full_game_10x10()]
    records.extend(case_batch_records(args.games))

    if args.log:
        RunLog(args.log).append_many(records)
        print(f"\n已写入 {len(records)} 条记录到 {args.log}")
    print("\n全部手工用例通过。")


if __name__ == "__main__":
    main(sys.argv[1:])
