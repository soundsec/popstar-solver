"""图片识别（``ui/vision.js``）的回归测试。

识别正确性不能只靠肉眼看，所以用**已知真值的合成图片**来跑：
用 Python 拼出一张「像游戏截图」的图（含背景、格子间隙、中心星星装饰、
以及部分空格），交给 JS 识别，再比对还原出的棋盘是否等于真值。

通过 Node 执行 ``ui/vision.js``（该文件不依赖 DOM，同时支持 CommonJS 导出）。
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
import unittest
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VISION = os.path.join(ROOT, "ui", "vision.js")

# 四种差异明显的颜色（对应游戏里常见的红/蓝/绿/黄）
COLORS = [(230, 72, 77), (62, 99, 221), (70, 167, 88), (240, 180, 41)]
BACKGROUND = (247, 245, 240)
STAR = (255, 255, 255)  # 每格中心的装饰（模拟星星图案）

NODE_CANDIDATES = [
    shutil.which("node"),
    r"C:\Users\11321\.workbuddy\binaries\node\versions\22.22.2-5\node.exe",
    r"C:\Program Files\nodejs\node.exe",
]


def find_node() -> str:
    for candidate in NODE_CANDIDATES:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise unittest.SkipTest("未找到 node，跳过图片识别回归测试")


def render_board_image(
    grid: List[List[int]],
    cell: int = 24,
    gap: int = 2,
    margin: int = 30,
    decorate: bool = True,
    star_radius: float = 0,
) -> Dict[str, Any]:
    """按游戏截图的外观合成一张 RGBA 图片。返回 ``{data,width,height}`` 的 JSON。"""
    rows, cols = len(grid), len(grid[0])
    board_w = cols * cell + (cols - 1) * gap
    board_h = rows * cell + (rows - 1) * gap
    width = board_w + margin * 2
    height = board_h + margin * 2

    data = [0] * (width * height * 4)

    def put(x: int, y: int, rgb: Tuple[int, int, int]) -> None:
        i = (y * width + x) * 4
        data[i] = rgb[0]
        data[i + 1] = rgb[1]
        data[i + 2] = rgb[2]
        data[i + 3] = 255

    for y in range(height):
        for x in range(width):
            put(x, y, BACKGROUND)

    for r in range(rows):
        for c in range(cols):
            value = grid[r][c]
            x0 = margin + c * (cell + gap)
            y0 = margin + r * (cell + gap)
            if value < 0:
                continue  # 空格：露出背景
            color = COLORS[value % len(COLORS)]
            for y in range(y0, y0 + cell):
                for x in range(x0, x0 + cell):
                    put(x, y, color)
            if decorate and star_radius > 0:
                # 大圆星星：盖住旧算法取样的中央 60%，环带取样必须还能认对颜色
                cx0 = x0 + cell / 2
                cy0 = y0 + cell / 2
                r2 = star_radius * star_radius
                for y in range(y0, y0 + cell):
                    for x in range(x0, x0 + cell):
                        if (x + 0.5 - cx0) ** 2 + (y + 0.5 - cy0) ** 2 <= r2:
                            put(x, y, STAR)
            elif decorate:
                # 中心一小块白色装饰 —— 用来验证中位采样的抗干扰能力
                for y in range(y0 + cell // 2 - 2, y0 + cell // 2 + 2):
                    for x in range(x0 + cell // 2 - 2, x0 + cell // 2 + 2):
                        put(x, y, STAR)

    return {"data": data, "width": width, "height": height}


def run_vision(rows: int, cols: int, k: int, empty_threshold: int,
               image: Dict[str, Any]) -> Dict[str, Any]:
    """把合成图片交给 JS 识别，返回 ``{rect, grid, clusters, empties}``。

    图片数据走临时文件传递：直接内联进 ``node -e`` 的命令行会超过
    Windows 的命令行长度上限。
    """
    script = """
const fs = require('fs');
const V = require(process.argv[1]);
const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const img = {data: Uint8ClampedArray.from(payload.data),
             width: payload.width, height: payload.height};
const rect = V.detectBoardRect(img, 55, {
  rows: Number(process.argv[3]), cols: Number(process.argv[4]),
});
if (!rect) { console.log(JSON.stringify({error: 'no rect'})); process.exit(0); }
const out = V.extractGrid(img, rect, {
  rows: Number(process.argv[3]), cols: Number(process.argv[4]),
  k: Number(process.argv[5]), emptyThreshold: Number(process.argv[6]),
});
console.log(JSON.stringify({
  rect, grid: out.grid, empties: out.empties,
  clusters: out.clusters.map(c => c.centroid),
}));
"""
    handle, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(image, fh)
        proc = subprocess.run(
            [find_node(), "-e", script, VISION, path,
             str(rows), str(cols), str(k), str(empty_threshold)],
            capture_output=True, text=True, timeout=180,
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)
    if proc.returncode != 0:
        raise AssertionError(f"node failed: {proc.stderr[:400]}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if "error" in result:
        raise AssertionError(f"识别失败：{result['error']}")
    return result


def grids_equivalent(a: List[List[int]], b: List[List[int]]) -> bool:
    """颜色只是相等性标签，编号可以整体重排，因此比对「划分」而非编号。"""
    mapping: Dict[int, int] = {}
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            if va < 0 or vb < 0:
                if va != vb:
                    return False
                continue
            if mapping.get(va, vb) != vb:
                return False
            mapping[va] = vb
    # 反向也要单射，避免两种真值颜色被并成一种
    return len(set(mapping.values())) == len(mapping)


class TestVisionRecognition(unittest.TestCase):
    def test_recovers_a_random_full_board(self):
        rng = random.Random(4242)
        grid = [[rng.randrange(4) for _ in range(10)] for _ in range(10)]
        image = render_board_image(grid)
        out = run_vision(10, 10, 4, 60, image)
        self.assertTrue(
            grids_equivalent(grid, out["grid"]),
            f"识别结果与真值不一致\n真值 {grid}\n识别 {out['grid']}",
        )

    def test_recovers_board_with_empty_cells(self):
        """残局截图：部分格子已空，应被识别为 EMPTY(-1)。"""
        rng = random.Random(77)
        grid = [[rng.randrange(4) for _ in range(10)] for _ in range(10)]
        holes = [(0, 0), (0, 9), (3, 4), (9, 0), (9, 9), (5, 5)]
        for r, c in holes:
            grid[r][c] = -1
        image = render_board_image(grid)
        out = run_vision(10, 10, 4, 60, image)
        self.assertEqual(
            sorted(map(tuple, out["empties"])), sorted(holes),
            "空格识别错误",
        )
        self.assertTrue(grids_equivalent(grid, out["grid"]))

    def test_center_decoration_does_not_break_sampling(self):
        """去掉装饰 vs 保留装饰，识别结果应完全一致（验证中位采样抗干扰）。"""
        rng = random.Random(9)
        grid = [[rng.randrange(4) for _ in range(8)] for _ in range(8)]
        plain = run_vision(8, 8, 4, 60, render_board_image(grid, decorate=False))
        fancy = run_vision(8, 8, 4, 60, render_board_image(grid, decorate=True))
        self.assertEqual(plain["grid"], fancy["grid"])

    def test_merge_to_k_respects_requested_color_count(self):
        """请求 3 种颜色时不得返回 4 簇。"""
        rng = random.Random(31337)
        grid = [[rng.randrange(4) for _ in range(10)] for _ in range(10)]
        out = run_vision(10, 10, 3, 60, render_board_image(grid))
        self.assertLessEqual(len(out["clusters"]), 3)

    def test_detect_rect_covers_the_board(self):
        grid = [[0] * 10 for _ in range(10)]
        image = render_board_image(grid, margin=30)
        out = run_vision(10, 10, 4, 60, image)
        rect = out["rect"]
        # 格子间距 26px × 10，拟合出的棋盘宽约 260
        self.assertGreaterEqual(rect["w"], 250)
        self.assertLessEqual(rect["w"], 270)

    def test_large_center_star_does_not_recolor(self):
        """星星大到盖住格子中央时，仍要取到方块本体的颜色。"""
        rng = random.Random(9)
        grid = [[rng.randrange(4) for _ in range(8)] for _ in range(8)]
        image = render_board_image(grid, decorate=True, star_radius=0.30 * 24)
        out = run_vision(8, 8, 4, 60, image)
        self.assertTrue(
            grids_equivalent(grid, out["grid"]),
            f"大星星把颜色认错了\n真值 {grid}\n识别 {out['grid']}",
        )

    def test_empty_rows_at_top_stay_on_the_lattice(self):
        """顶部整行已空时，不能把剩下的方块摊成 10 行。"""
        rng = random.Random(11)
        grid = [[rng.randrange(4) for _ in range(10)] for _ in range(10)]
        for c in range(10):
            grid[0][c] = -1
            grid[1][c] = -1
        grid[4][3] = -1
        grid[6][8] = -1
        out = run_vision(10, 10, 4, 60, render_board_image(grid))
        self.assertTrue(
            grids_equivalent(grid, out["grid"]),
            f"顶部空行没有对齐\n真值 {grid}\n识别 {out['grid']}",
        )


# 识别测试/ 里的五张实机截图。字母只是颜色标签，比对时允许整体重排。
# 2.jpg 是残局：最上面一整行已经空了，棋盘仍是 10×10。
SCREENSHOT_BOARDS = {
    "1.jpg": """
        BRBBYPRGYR
        BGGYGPRGYP
        YBBRBRGRYG
        BRPRRBBYBB
        GRBBGPPBPG
        PGRYPBGGRP
        PPBBPRBBRY
        RGBGGBYBPB
        BBRPBPGYRG
        YYBBRRYYRP
    """,
    "2.jpg": """
        ..........
        ...G......
        ..PR......
        ..GY..B.P.
        P.YY.YBBR.
        B.YR.BRRRP
        B.GYGBBPGR
        P.PPBPGBBG
        RPPRPYGRYR
        PRRBPRPBRR
    """,
    "3.jpg": """
        BBPGPYBBYY
        YYGRPGGRBY
        PRPYGGBPBB
        BRRPPPPBPB
        BPYYGBRYRY
        PBYRBBBPRP
        BBGYGPGPGR
        YYPPGGYYBG
        RYPRPYGRYR
        PRRBPRPBRR
    """,
    "4.jpg": """
        RGPPPGGPPP
        RRGBGYGGGP
        BYGGPYPYRY
        GBRBPPYYRG
        PBRBGYPRGB
        BBGYYRRGPG
        BPBRGBGRBG
        RPYGGBRYRP
        GBGBYYPPPG
        YBBPRYRPYB
    """,
    "5.jpg": """
        RPBGPRYRBR
        RPBBBRYGBB
        PPPGRRGYRY
        PRGPRRRBGG
        PGRBGGPPGG
        BPGBBPYRYR
        BRPGYBPBRY
        BGYPGPYYRB
        YGGPGPBRPG
        YPPRRGGPYY
    """,
}

_LETTER = {'.': -1, 'R': 0, 'Y': 1, 'G': 2, 'B': 3, 'P': 4}


def parse_board(text: str) -> List[List[int]]:
    rows = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return [[_LETTER[ch] for ch in row] for row in rows]


def run_vision_file(path: str, rows: int, cols: int, k: int, empty_threshold: int) -> Dict[str, Any]:
    """解码一张实机截图交给 JS。像素走裸 RGBA 文件，避免 JSON 把命令和内存撑爆。"""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise unittest.SkipTest(f"需要 Pillow 和 numpy 才能解码实机截图：{exc}")

    image = Image.open(path).convert("RGBA")
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    script = """
const fs = require('fs');
const V = require(process.argv[1]);
const buf = fs.readFileSync(process.argv[2]);
const data = new Uint8ClampedArray(buf.length);
data.set(buf);
const img = {data, width: Number(process.argv[3]), height: Number(process.argv[4])};
const rows = Number(process.argv[5]), cols = Number(process.argv[6]);
const rect = V.detectBoardRect(img, 55, {rows, cols});
if (!rect) { console.log(JSON.stringify({error: 'no rect'})); process.exit(0); }
const out = V.extractGrid(img, rect, {
  rows, cols, k: Number(process.argv[7]), emptyThreshold: Number(process.argv[8]),
});
console.log(JSON.stringify({
  rect, grid: out.grid, empties: out.empties,
  clusters: out.clusters.length,
}));
"""
    handle, raw_path = tempfile.mkstemp(suffix=".raw")
    try:
        with os.fdopen(handle, "wb") as fh:
            fh.write(arr.tobytes())
        proc = subprocess.run(
            [find_node(), "-e", script, VISION, raw_path,
             str(width), str(height), str(rows), str(cols), str(k), str(empty_threshold)],
            capture_output=True, text=True, timeout=180,
        )
    finally:
        if os.path.exists(raw_path):
            os.unlink(raw_path)
    if proc.returncode != 0:
        raise AssertionError(f"node failed: {proc.stderr[:400]}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if "error" in result:
        raise AssertionError(f"识别失败：{result['error']}")
    return result


class TestScreenshotRecognition(unittest.TestCase):
    """``识别测试/`` 里的五张手机截图。自动框选必须躲开界面，并还原 5 色棋盘。"""

    def test_screenshots_match_labeled_boards(self):
        folder = os.path.join(ROOT, "识别测试")
        if not os.path.isdir(folder):
            self.skipTest("没有识别测试目录")
        for name, text in SCREENSHOT_BOARDS.items():
            path = os.path.join(folder, name)
            truth = parse_board(text)
            out = run_vision_file(path, 10, 10, 5, 60)
            rect = out["rect"]
            self.assertGreater(rect["y"], 400, f"{name} 把顶栏框进来了：{rect}")
            self.assertLess(rect["h"], 1600, f"{name} 框太高，可能包含了广告：{rect}")
            self.assertGreater(rect["w"], 900, f"{name} 棋盘宽度不对：{rect}")
            self.assertTrue(
                grids_equivalent(truth, out["grid"]),
                f"{name} 与标注不一致\n标注 {truth}\n识别 {out['grid']}\n区域 {rect}",
            )
            self.assertEqual(out["clusters"], 5, f"{name} 颜色数不是 5")


if __name__ == "__main__":
    unittest.main()
