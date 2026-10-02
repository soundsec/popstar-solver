"""UI 服务的回归测试。

这里覆盖两类**已经真实发生过**的故障：

1. ``api_*`` 方法被写成 ``@staticmethod`` 却在方法体里用 ``self._state``，
   结果是 ``NameError`` → HTTP 400 → 前端拿到空棋盘 → 表现为「点击没反应」，
   而界面上没有任何提示。因此对每个端点都用**前端实际发送的请求形状**打一遍。
2. JS 里 ``$('某个id')`` 引用了 HTML 中不存在的 id —— 一处就会让整个脚本
   在解析事件监听时抛 TypeError 中止，所有按钮全部失效。
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import unittest
from http.client import HTTPConnection
from typing import Any, Dict, List

from popstar import server as app_server
from popstar.board import create_board, get_group, get_legal_moves

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_DIR = os.path.join(ROOT, "ui")

SCORING = {"clear_threshold": 10, "bonus_coefficient": 10}
TIMEOUT = 180


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestEndpoints(unittest.TestCase):
    """每个端点都用前端真实发送的字段打一遍，要求 200 且字段齐全。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _free_port()
        cls.server = app_server.build_server("127.0.0.1", cls.port)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path: str, body: Dict[str, Any]):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=TIMEOUT)
        try:
            conn.request("POST", path, json.dumps(body).encode(),
                         {"Content-Type": "application/json"})
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read().decode())
        finally:
            conn.close()

    def get(self, path: str):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=TIMEOUT)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read().decode())
        finally:
            conn.close()

    def test_scoring_endpoint_and_server_knobs(self):
        """计分默认值来自 popstar/scoring.py，改常数或换函数后服务端跟着变。"""
        import popstar.scoring as scoring

        status, data = self.get("/api/scoring")
        self.assertEqual(status, 200, data)
        self.assertEqual(data["samples"]["2"], scoring.group_gain(2))
        self.assertEqual(data["clear_threshold"], scoring.CLEAR_THRESHOLD)
        self.assertEqual(data["clear_at_zero"], scoring.commercial_clear_reward(0))

        old_coeff = scoring.GROUP_COEFFICIENT
        old_clear = scoring.CLEAR_REWARD_FN
        scoring.GROUP_COEFFICIENT = 5
        scoring.CLEAR_REWARD_FN = lambda remaining: 123.0 if remaining == 0 else 0.0
        try:
            cfg = app_server.PopstarHandler._scoring({"clear_threshold": 3, "bonus_coefficient": 9})
            self.assertEqual(cfg.score_group(4), 5 * 16)
            self.assertEqual(cfg.clear_threshold, 3)
            # 自定义清盘函数接管后，请求里的系数不再参与
            self.assertEqual(cfg.score_terminal(0), 123.0)
            self.assertEqual(cfg.score_terminal(2), 0.0)
        finally:
            scoring.GROUP_COEFFICIENT = old_coeff
            scoring.CLEAR_REWARD_FN = old_clear

    def test_generators_endpoint(self):
        status, data = self.get("/api/generators")
        self.assertEqual(status, 200)
        self.assertIn("uniform", data["generators"])

    def test_board_endpoint_with_every_generator(self):
        """生成器下拉里的每一档都要能真的生成出可玩的棋盘。"""
        status, listing = self.get("/api/generators")
        self.assertEqual(status, 200)
        for name in listing["generators"]:
            with self.subTest(generator=name):
                status, data = self.post("/api/board", {
                    "height": 10, "width": 10, "colors": 4,
                    "seed": 7, "generator": name, **SCORING,
                })
                self.assertEqual(status, 200, f"{name}: {data}")
                self.assertEqual(len(data["grid"]), 10)
                self.assertGreater(data["moves"], 0)
                self.assertIn("balance", data["profile"])
                self.assertIn("adjacency_ratio", data["profile"])
                self.assertGreater(data["upper_bound"], 0)

    def test_apply_endpoint_click_shape(self):
        """点击格子：前端提交被点中的那一格，由服务端找出整块并消除。"""
        _, board = self.post("/api/board", {
            "height": 10, "width": 10, "colors": 4, "seed": 99,
            "generator": "clustered", **SCORING,
        })
        state = create_board(board["grid"])
        move = get_legal_moves(state)[0]
        status, data = self.post("/api/apply", {
            "grid": board["grid"],
            "cell": list(move.representative_cell),
            **SCORING,
        })
        self.assertEqual(status, 200, data)
        self.assertEqual(data["size"], move.size)
        self.assertEqual(data["remaining"], 100 - move.size)

    def test_apply_accepts_every_cell_in_the_group(self):
        """同一连通块里点任意一格，结果必须相同。"""
        _, board = self.post("/api/board", {
            "height": 8, "width": 8, "colors": 4, "seed": 5, **SCORING})
        state = create_board(board["grid"])
        move = next(m for m in get_legal_moves(state) if m.size >= 3)
        grids = []
        for cell in move.cells:
            status, data = self.post("/api/apply", {
                "grid": board["grid"], "cell": list(cell), **SCORING})
            self.assertEqual(status, 200, data)
            self.assertEqual(data["size"], move.size)
            grids.append(tuple(tuple(row) for row in data["grid"]))
        self.assertEqual(len(set(grids)), 1)

    def test_apply_rejects_illegal_cell(self):
        _, board = self.post("/api/board", {
            "height": 8, "width": 8, "colors": 4, "seed": 5, **SCORING})
        state = create_board(board["grid"])
        bad = next(
            (r, c) for r in range(8) for c in range(8)
            if (comp := get_group(state, (r, c))) is not None and comp.size < 2
        )
        status, data = self.post("/api/apply", {
            "grid": board["grid"], "cell": [bad[0], bad[1]], **SCORING})
        self.assertEqual(status, 400)
        self.assertIn("not a legal move", data["error"])

    def test_solve_fast_and_exact(self):
        _, board = self.post("/api/board", {
            "height": 8, "width": 8, "colors": 4, "seed": 11, **SCORING})
        for mode in ("fast", "exact"):
            with self.subTest(mode=mode):
                status, data = self.post("/api/solve", {
                    "grid": board["grid"], "mode": mode,
                    "beam_width": 32, "time_limit": 5, **SCORING,
                })
                self.assertEqual(status, 200, data)
                self.assertGreaterEqual(data["total"], 0)
                self.assertEqual(len(data["frames"]), data["move_count"] + 1)

    def test_replay_matches_solve(self):
        _, board = self.post("/api/board", {
            "height": 8, "width": 8, "colors": 4, "seed": 13, **SCORING})
        _, solved = self.post("/api/solve", {
            "grid": board["grid"], "mode": "fast",
            "beam_width": 32, "time_limit": 5, **SCORING})
        status, data = self.post("/api/replay", {
            "grid": board["grid"], "path": solved["path"], **SCORING})
        self.assertEqual(status, 200, data)
        self.assertAlmostEqual(data["total"], solved["total"], places=6)

    def test_pareto_endpoint(self):
        _, board = self.post("/api/board", {
            "height": 8, "width": 8, "colors": 4, "seed": 17, **SCORING})
        status, data = self.post("/api/pareto", {
            "grid": board["grid"], "beam_width": 32, **SCORING})
        self.assertEqual(status, 200, data)
        self.assertTrue(data["points"])
        # 每个前沿点都必须带得回一条走法
        for point in data["points"]:
            self.assertIn(str(point["r"]), data["paths"])

    def test_import_roundtrip(self):
        _, board = self.post("/api/board", {
            "height": 10, "width": 10, "colors": 4, "seed": 23, **SCORING})
        status, data = self.post("/api/import", {"grid": board["grid"]})
        self.assertEqual(status, 200, data)
        status2, fetched = self.get("/api/board/" + data["id"])
        self.assertEqual(status2, 200)
        self.assertEqual(fetched["grid"], board["grid"])

    def test_unknown_board_id(self):
        status, data = self.get("/api/board/does-not-exist")
        self.assertEqual(status, 404)
        self.assertIn("error", data)

    def test_oversized_board_is_rejected(self):
        status, _ = self.post("/api/board", {
            "height": 60, "width": 60, "colors": 4, **SCORING})
        self.assertEqual(status, 400)


class TestFrontendDomContract(unittest.TestCase):
    """JS 引用的每个 DOM id 都必须存在于对应 HTML。

    少一个 id，``addEventListener`` 就会在 null 上抛错，整个脚本中止，
    页面上所有交互同时失效 —— 而且没有任何可见提示。
    """

    # 游玩 / 分析 / 图片识别已合并进 index.html，不再有独立页面
    PAGES = {
        "index.html": ["app.js", "vision.js"],
    }
    # 运行时由 JS 动态创建、不属于静态 HTML 的 id
    DYNAMIC = {"tightness", "fatal"}

    def _html_ids(self, page: str) -> set:
        with open(os.path.join(UI_DIR, page), encoding="utf-8") as fh:
            return set(re.findall(r'id="([^"]+)"', fh.read()))

    def _js_ids(self, script: str) -> set:
        with open(os.path.join(UI_DIR, script), encoding="utf-8") as fh:
            text = fh.read()
        return set(re.findall(r"\$\('([^']+)'\)", text)) | set(
            re.findall(r"getElementById\('([^']+)'\)", text))

    def test_all_referenced_ids_exist(self):
        for page, scripts in self.PAGES.items():
            html_ids = self._html_ids(page)
            for script in scripts:
                missing = sorted(self._js_ids(script) - self.DYNAMIC - html_ids)
                with self.subTest(page=page, script=script):
                    self.assertEqual(missing, [])

    def test_common_js_is_included(self):
        """common.js 提供错误呈现，每个页面都必须先加载它。"""
        for page in self.PAGES:
            with open(os.path.join(UI_DIR, page), encoding="utf-8") as fh:
                html = fh.read()
            scripts = re.findall(r'<script src="([^"]+)"', html)
            with self.subTest(page=page):
                self.assertIn("common.js", scripts)
                self.assertEqual(scripts.index("common.js"), 0)

    def test_script_files_exist(self):
        for page in self.PAGES:
            with open(os.path.join(UI_DIR, page), encoding="utf-8") as fh:
                for src in re.findall(r'<script src="([^"]+)"', fh.read()):
                    with self.subTest(page=page, src=src):
                        self.assertTrue(os.path.isfile(os.path.join(UI_DIR, src)))


if __name__ == "__main__":
    unittest.main()
