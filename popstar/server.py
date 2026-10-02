"""本地可视化测试台：为 popstar 求解器提供一个可交互的 UI。

规则实现**始终**留在 Python 侧（``popstar.board.apply_move``）。前端只负责
渲染与交互，任何「点一下消掉一块」都要回到服务端裁决，避免 JS 里出现第二份
规则实现。

用法::

    python -m popstar.server            # http://127.0.0.1:8765
    python -m popstar.server --port 9000

只监听 127.0.0.1，且静态文件限制在 ``ui/`` 目录内（防路径穿越）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from popstar.board import (
    apply_move,
    color_counts,
    create_board,
    get_legal_moves,
    is_terminal,
    random_board,
)
from popstar.game import Game
from popstar.generators import GENERATOR_NAMES, board_profile, make_generator
import popstar.scoring as scoring_rules
from popstar.scoring import ScoringConfig
from popstar.solver import (
    BeamSolver,
    ExactSolver,
    board_features,
    group_score_ceiling,
    nonclear_group_ceiling,
    optimistic_bound,
    pareto_search,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_DIR = os.path.join(ROOT, "ui")

MAX_BOARD_CELLS = 400  # 防止手滑提交一个巨大棋盘把服务拖死


class PopstarHandler(BaseHTTPRequestHandler):
    server_version = "popstar-ui"

    # -- 基础 ------------------------------------------------------------

    def _send(self, code: int, payload: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, code: int, data: Any) -> None:
        self._send(code, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

    def _send_error_json(self, code: int, message: str) -> None:
        self._send_json(code, {"error": message})

    def log_message(self, fmt: str, *args: Any) -> None:
        # 默认会把每个请求打到 stderr，这里压掉静态资源的噪音
        path = args[0] if args else ""
        if isinstance(path, str) and path.startswith("/api/"):
            super().log_message(fmt, *args)

    # -- 路由 ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server 的固定命名
        route = urlparse(self.path).path
        if route == "/api/generators":
            self._send_json(200, {"generators": list(GENERATOR_NAMES)})
            return
        if route == "/api/scoring":
            self._send_json(200, self.api_scoring())
            return
        if route.startswith("/api/board/"):
            # /api/board/<id>：取回一张已导入的棋盘
            key = route.rsplit("/", 1)[-1]
            slot = BOARD_SLOTS.get(key)
            if slot is None:
                self._send_error_json(404, f"unknown board id: {key}")
            else:
                self._send_json(200, dict(slot, id=key))
            return
        if route.startswith("/api/"):
            self._send_error_json(404, f"unknown endpoint: {route}")
            return
        self._serve_static(route)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._send_error_json(400, f"invalid JSON body: {exc}")
            return
        try:
            if route == "/api/board":
                self._send_json(200, self.api_board(body))
            elif route == "/api/solve":
                self._send_json(200, self.api_solve(body))
            elif route == "/api/pareto":
                self._send_json(200, self.api_pareto(body))
            elif route == "/api/apply":
                self._send_json(200, self.api_apply(body))
            elif route == "/api/replay":
                self._send_json(200, self.api_replay(body))
            elif route == "/api/import":
                self._send_json(200, {"id": new_board_slot(
                    body["grid"], body.get("meta")
                ), "moves": len(get_legal_moves(create_board(body["grid"])))})
            else:
                self._send_error_json(404, f"unknown endpoint: {route}")
        except Exception as exc:  # 规则层抛出的 ValueError 也要能回给前端
            self._send_error_json(400, f"{type(exc).__name__}: {exc}")

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    def _serve_static(self, route: str) -> None:
        if route == "/":
            route = "/index.html"
        # 只允许 ui/ 目录内的文件
        safe = os.path.normpath(route.lstrip("/")).replace("\\", "/")
        if safe.startswith("..") or safe.startswith("/"):
            self._send_error_json(403, "forbidden")
            return
        path = os.path.join(UI_DIR, safe)
        if not os.path.isfile(path):
            self._send_error_json(404, "not found")
            return
        ctype = "text/html; charset=utf-8"
        if path.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        elif path.endswith(".js"):
            ctype = "application/javascript; charset=utf-8"
        with open(path, "rb") as handle:
            self._send(200, handle.read(), ctype)

    # -- API -------------------------------------------------------------

    @staticmethod
    def _state(body: Dict[str, Any]):
        grid = body.get("grid")
        if not grid or not isinstance(grid, list):
            raise ValueError("missing grid")
        height, width = len(grid), len(grid[0])
        if height * width > MAX_BOARD_CELLS:
            raise ValueError(f"board too large (>{MAX_BOARD_CELLS} cells)")
        return create_board(grid)

    @staticmethod
    def api_scoring() -> Dict[str, Any]:
        """把 ``popstar/scoring.py`` 里的当前计分告诉页面，用来填默认值。"""
        scoring = PopstarHandler._scoring({})
        return {
            "group_coefficient": scoring_rules.GROUP_COEFFICIENT,
            "group_exponent": scoring_rules.GROUP_EXPONENT,
            "group_custom": scoring_rules.GROUP_GAIN_FN is not None,
            "clear_threshold": scoring_rules.CLEAR_THRESHOLD,
            "bonus_coefficient": scoring_rules.CLEAR_COEFFICIENT,
            "clear_exponent": scoring_rules.CLEAR_EXPONENT,
            "clear_custom": scoring_rules.CLEAR_REWARD_FN is not None,
            "samples": {str(k): scoring.score_group(k) for k in (2, 3, 5, 10)},
            "clear_at_zero": scoring.score_terminal(0),
        }

    @staticmethod
    def _scoring(body: Dict[str, Any]) -> ScoringConfig:
        """计分以 ``popstar/scoring.py`` 为准；请求里若带了阈值或系数，只覆盖这两项。"""
        threshold = body.get("clear_threshold", scoring_rules.CLEAR_THRESHOLD)
        coefficient = body.get("bonus_coefficient", scoring_rules.CLEAR_COEFFICIENT)
        return ScoringConfig(
            group_score=scoring_rules.group_gain,
            clear_threshold=int(threshold),
            bonus_coefficient=float(coefficient),
            bonus_exponent=scoring_rules.CLEAR_EXPONENT,
            terminal_score=scoring_rules.CLEAR_REWARD_FN,
        )

    @staticmethod
    def _frames(state, path_cells: List[Any], scoring: ScoringConfig) -> List[Any]:
        """把动作序列逐步回放，产出每一帧的棋盘（供前端做动画）。"""
        frames = [[list(row) for row in state.grid]]
        cursor = state
        for cell in path_cells:
            move = None
            for candidate in get_legal_moves(cursor):
                if candidate.representative_cell == tuple(cell):
                    move = candidate
                    break
            if move is None:
                raise ValueError(f"no legal move at {cell}")
            cursor = apply_move(cursor, move)
            frames.append([list(row) for row in cursor.grid])
        return frames

    def api_board(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """生成棋盘，或（当给出 ``grid`` 时）直接画像一块外部导入的棋盘。

        图片识别 / 手改棋盘都走这条路径：服务端只负责校验与画像，
        识别本身在浏览器 canvas 里做（免去 Pillow / numpy 依赖）。
        """
        grid = body.get("grid")
        seed = body.get("seed")  # 两个分支都要定义，返回体里会用到
        if grid:
            state = self._state(body)
            generator_name = "imported"
        else:
            height = int(body.get("height", 10))
            width = int(body.get("width", 10))
            colors = int(body.get("colors", 4))
            # 生成路径不走 _state，尺寸上限要在这里自己把住
            if height * width > MAX_BOARD_CELLS:
                raise ValueError(f"board too large (>{MAX_BOARD_CELLS} cells)")
            rng = random.Random(
                seed if seed is not None else random.randrange(1 << 30)
            )
            generator_name = body.get("generator") or "uniform"
            state = make_generator(generator_name)(height, width, colors, rng)

        scoring = self._scoring(body)
        counts = list(color_counts(state).values())
        profile = board_profile(state)
        return {
            "grid": [list(row) for row in state.grid],
            "seed": seed,
            "generator": generator_name,
            "moves": len(get_legal_moves(state)),
            # 盘面画像：数量均衡度 / 空间聚集比
            "profile": {
                "balance": profile["balance"],
                "adjacency": profile["adjacency"],
                "adjacency_ratio": profile["adjacency_ratio"],
                "counts": profile["counts"],
            },
            "u_group": group_score_ceiling(counts, scoring.group_score),
            "u_nonclear": nonclear_group_ceiling(counts, scoring.group_score),
            "upper_bound": optimistic_bound(state, scoring),
        }

    def api_solve(self, body: Dict[str, Any]) -> Dict[str, Any]:
        state = self._state(body)
        scoring = self._scoring(body)
        mode = body.get("mode", "fast")
        beam_width = int(body.get("beam_width", 256))
        time_limit = body.get("time_limit")
        time_limit = float(time_limit) if time_limit else None

        if mode == "exact":
            solver = ExactSolver(scoring=scoring, use_memo=True, time_limit=time_limit)
            solution = solver.solve(state)
        elif mode == "fast":
            solution = BeamSolver(
                scoring=scoring, beam_width=beam_width, time_limit=time_limit
            ).solve(state)
        else:
            raise ValueError(f"unknown mode: {mode}")

        path = [list(move.representative_cell) for move in solution.move_sequence]
        return {
            "mode": mode,
            "total": solution.best_score,
            "group": solution.group_score,
            "bonus": solution.terminal_bonus,
            "remaining": solution.remaining_count,
            "move_count": solution.move_count,
            "proven": solution.is_proven_optimal,
            "elapsed": solution.elapsed_time,
            "stats": {
                "states_expanded": solution.states_expanded,
                "cache_hits": solution.cache_hits,
                "cache_size": solution.cache_size,
                "max_depth": solution.max_depth,
                "beam_width": beam_width,
            },
            "moves": [
                {
                    "cell": list(move.representative_cell),
                    "color": move.color,
                    "size": move.size,
                    "score": scoring.score_group(move.size),
                }
                for move in solution.move_sequence
            ],
            "path": path,
            "frames": self._frames(state, path, scoring),
        }

    def api_pareto(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """终局 (G, R) 前沿。**不含**终局奖励，因此与 bonus_coefficient 无关。"""
        state = self._state(body)
        scoring = self._scoring(body)
        beam_width = int(body.get("beam_width", 256))
        time_limit = body.get("time_limit")
        time_limit = float(time_limit) if time_limit else None

        front = pareto_search(
            state, scoring=scoring, beam_width=beam_width, time_limit=time_limit
        )
        counts = list(color_counts(state).values())
        u_group = group_score_ceiling(counts, scoring.group_score)
        u_nonclear = nonclear_group_ceiling(counts, scoring.group_score)

        points = [
            {
                "r": r,
                "g": g,
                "eta": (g / u_group) if u_group else 0.0,
                "shape": scoring.terminal_shape(r),
                "bonus": scoring.score_terminal(r),
            }
            for r, g in front.points()
        ]
        pareto_ids = {(r, g) for r, g in front.pareto()}
        for point in points:
            point["pareto"] = (point["r"], point["g"]) in pareto_ids

        try:
            segments = [
                {"lo": lo, "hi": hi, "r": r, "g": g}
                for lo, hi, r, g in front.envelope(scoring)
            ]
            switches = [
                {"lam": lam, "from": r1, "to": r2}
                for lam, r1, r2 in front.switch_points(scoring)
            ]
        except ValueError:
            # 自定义清盘奖励不是「系数 × 形状」，前沿包络无法按 λ 切开
            segments = []
            switches = []
        return {
            "u_group": u_group,
            "u_nonclear": u_nonclear,
            "points": points,
            "segments": segments,
            "switches": switches,
            "elapsed": front.elapsed_time,
            "expanded": front.states_expanded,
            "profiles": list(front.profiles),
            "paths": {
                str(r): [list(cell) for cell in path]
                for r, path in front.paths.items()
            },
        }

    def api_replay(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """把一条代表格序列完整回放，产出逐帧棋盘与两项得分。

        前端用它来「载入前沿上某一点对应的走法」——回放仍然走服务端的
        ``apply_move``，所以载入的解一定合法。
        """
        state = self._state(body)
        scoring = self._scoring(body)
        path = body.get("path") or []
        game = Game(state, scoring=scoring)
        for cell in path:
            game.play_cell(tuple(cell))
        result = game.result(require_finished=False)
        return {
            "total": result.total_score,
            "group": result.group_score,
            "bonus": result.terminal_bonus,
            "remaining": result.remaining_count,
            "move_count": result.move_count,
            "frames": self._frames(state, path, scoring),
        }

    def api_apply(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """手动走一步：点中连通块里的任意一格，都消除整块。

        搜索用的代表格只是该块的一个坐标，不能拿来限制玩家能点哪一格。
        """
        state = self._state(body)
        scoring = self._scoring(body)
        cell = body.get("cell")
        if not cell:
            raise ValueError("missing cell")
        cell = (int(cell[0]), int(cell[1]))
        game = Game(state, scoring=scoring)
        try:
            record = game.play_cell(cell)
        except ValueError as exc:
            raise ValueError(f"cell {cell} is not a legal move") from exc
        result = game.result(require_finished=False)
        return {
            "grid": [list(row) for row in game.state.grid],
            "size": record.removed_count,
            "score": record.immediate_score,
            "group": result.group_score,
            "total": result.total_score,
            "remaining": result.remaining_count,
            "finished": is_terminal(game.state),
            "moves_left": len(get_legal_moves(game.state)),
            "features": dict(
                forced=board_features(game.state, scoring).forced,
            ),
        }


def new_board_slot(grid: List[List[int]], meta: Optional[Dict[str, Any]] = None) -> str:
    """登记一张「从图片识别出来 / 手工导入」的棋盘，返回可跨页面引用的 id。

    识别在浏览器里做，但要让结果从 scan 页面流转到求解页 / 游玩页，
    需要一个同源的中转。这里用进程内字典（本工具是本地单人使用，无需持久化）。
    """
    state = create_board(grid)
    slot = BOARD_SLOTS  # 模块级字典
    key = f"b{len(slot) + 1}-{random.randrange(1 << 30):x}"
    slot[key] = {
        "grid": [list(row) for row in state.grid],
        "height": state.height,
        "width": state.width,
        "moves": len(get_legal_moves(state)),
        "meta": meta or {},
    }
    # 只保留最近若干张，避免长时间使用无限增长
    while len(slot) > 32:
        slot.pop(next(iter(slot)))
    return key


# 进程内「导入棋盘」暂存区
BOARD_SLOTS: Dict[str, Dict[str, Any]] = {}


def build_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), PopstarHandler)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="popstar 可视化测试台")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args(argv)

    if not os.path.isdir(UI_DIR):
        raise SystemExit(f"missing ui directory: {UI_DIR}")

    server = ThreadingHTTPServer((args.host, args.port), PopstarHandler)
    print(f"popstar 测试台已启动： http://{args.host}:{args.port}")
    print("（Ctrl+C 退出；规则计算全部在服务端，前端只做渲染）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
