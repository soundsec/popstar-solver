"""受控棋盘生成器与盘面画像（真实游戏的生成算法通常既非均匀、也非空间无关）。

为什么需要这个模块
------------------
此前全部结论都建立在 ``random_board``（每格独立均匀取色）上。真实游戏里
方块由生成算法控制，典型偏差有两类：

* **数量不均**：某些颜色占比远高于其它（权重可控）。
* **分布不均**：同色方块在空间上成团（Voronoi / 分块生长 + 打散比例可控）。

求解器本身对这两类偏差**没有任何假设**——连通块来自真实邻接、上界用真实
计数、评估用真实盘面特征——所以「能不能求解」的答案是肯定的；
真正需要实测的是**结论是否还成立**：可清盘性、η_G、上界松紧、以及
「清盘与高 group 分不冲突」这个判断。

本模块同时给出两个可解释的画像指标，用来把「不均匀」量化：

* ``balance``：颜色分布熵 / log(C)。1.0 = 完全均衡，→0 = 单色垄断。
* ``adjacency``：相邻同色对占比。iid 均匀下期望为 ``1/C``（4 色 ≈ 0.25），
  明显高于它即说明空间聚集。
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .board import BoardState, EMPTY, color_counts, random_board

# ---------------------------------------------------------------------------
# 生成器
# ---------------------------------------------------------------------------


def uniform_board(height: int, width: int, colors: int, rng) -> BoardState:
    """基线：每格独立均匀取色（等价于 :func:`popstar.board.random_board`）。"""
    return random_board(height, width, colors, rng)


def weighted_board(
    height: int, width: int, colors: int, rng, weights: Sequence[float]
) -> BoardState:
    """**数量不均**、空间上仍独立（无聚集）的盘面。

    ``weights`` 为各颜色的相对权重，不必归一化。
    每格独立按权重取色，因此同色方块仍是空间散点。
    """
    if len(weights) != colors:
        raise ValueError(f"expected {colors} weights, got {len(weights)}")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive number")

    # 累积分布 + 线性查找：10x10 规模下比 bisect 更快也更好读
    cumulative: List[float] = []
    running = 0.0
    for weight in weights:
        running += weight / total
        cumulative.append(running)
    cumulative[-1] = 1.0

    grid: List[Tuple[int, ...]] = []
    for _row in range(height):
        row: List[int] = []
        for _col in range(width):
            draw = rng.random()
            for index, edge in enumerate(cumulative):
                if draw <= edge:
                    row.append(index)
                    break
        grid.append(tuple(row))
    return BoardState(tuple(grid))


def clustered_board(
    height: int,
    width: int,
    colors: int,
    rng,
    centers: Optional[int] = None,
    mix: float = 0.0,
    weights: Optional[Sequence[float]] = None,
) -> BoardState:
    """**空间聚集**的盘面：先撒若干种子中心，每格取最近中心的颜色。

    这模拟真实生成里「同色成片出现」的情形——同色方块不再是空间散点，
    初始就有大连通块。

    Parameters
    ----------
    centers:
        种子中心个数。越少则色块越大越连片；``None`` 时取 ``colors * 2``。
    mix:
        打散比例：``mix`` 比例的格子改为随机取色，用来削弱聚集强度。
        ``mix=0`` 是纯分块，``mix=1`` 退化为 iid。
    weights:
        中心颜色的抽样权重，可同时制造数量不均。
    """
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be in [0, 1]")
    if centers is None:
        centers = max(1, colors * 2)

    # 先给每种颜色各分配一个中心，避免抽样时整色缺席（否则盘面颜色数会漂）
    guaranteed = list(range(colors))
    rng.shuffle(guaranteed)
    extra = max(0, centers - colors)

    def draw_weighted() -> int:
        draw = rng.random()
        for index, edge in enumerate(cumulative):
            if draw <= edge:
                return index
        return colors - 1

    if weights is None:
        center_colors = list(guaranteed) + [
            rng.randrange(colors) for _ in range(extra)
        ]
    else:
        if len(weights) != colors:
            raise ValueError(f"expected {colors} weights, got {len(weights)}")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        cumulative = []
        running = 0.0
        for weight in weights:
            running += weight / total
            cumulative.append(running)
        cumulative[-1] = 1.0
        center_colors = list(guaranteed) + [draw_weighted() for _ in range(extra)]

    spots = [
        (rng.randrange(height), rng.randrange(width))
        for _ in range(len(center_colors))
    ]

    grid: List[Tuple[int, ...]] = []
    for row in range(height):
        cells: List[int] = []
        for col in range(width):
            if mix > 0.0 and rng.random() < mix:
                cells.append(rng.randrange(colors))
                continue
            best_index, best_dist = 0, None
            for index, (sr, sc) in enumerate(spots):
                dist = (sr - row) ** 2 + (sc - col) ** 2
                if best_dist is None or dist < best_dist:
                    best_index, best_dist = index, dist
            cells.append(center_colors[best_index])
        grid.append(tuple(cells))
    return BoardState(tuple(grid))


# ---------------------------------------------------------------------------
# 盘面画像
# ---------------------------------------------------------------------------


def balance(state: BoardState) -> float:
    """颜色分布均衡度 = 归一化熵 ``H / log(C)``。

    1.0 表示各颜色数量相等；数值越低表示数量越集中。
    """
    counts = [n for n in color_counts(state).values() if n > 0]
    if len(counts) <= 1:
        return 0.0
    total = float(sum(counts))
    entropy = -sum((n / total) * math.log(n / total) for n in counts)
    return entropy / math.log(len(counts))


def same_color_adjacency(state: BoardState) -> float:
    """相邻同色对占比。

    注意它**同时**受数量不均与空间聚集影响：即便空间完全独立，
    期望也是 ``Σ_c p_c²``（颜色越集中该值越大），而不是 ``1/C``。
    因此要单独度量「空间聚集」必须看 :func:`adjacency_ratio`。
    """
    grid = state.grid
    height, width = len(grid), len(grid[0])
    same = 0
    pairs = 0
    for r in range(height):
        for c in range(width):
            value = grid[r][c]
            if value == EMPTY:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr >= height or nc >= width:
                    continue
                other = grid[nr][nc]
                if other == EMPTY:
                    continue
                pairs += 1
                if other == value:
                    same += 1
    return (same / pairs) if pairs else 0.0


def adjacency_ratio(state: BoardState) -> float:
    """空间聚集比 = ``同色相邻占比 / Σ_c p_c²``。

    分子是实测，分母是「空间完全独立时」的期望值。因此：

    * ≈ 1.0：给定该颜色配比下，空间分布与独立采样无异；
    * 明显大于 1.0：同色成团（真实生成算法里最常见的情况）。

    这把「数量不均」与「分布不均」两个因素彻底分开——只看
    :func:`same_color_adjacency` 会把数量不均误读成空间聚集。
    """
    counts = [n for n in color_counts(state).values() if n > 0]
    if not counts:
        return 0.0
    total = float(sum(counts))
    baseline = sum((n / total) ** 2 for n in counts)
    if baseline <= 0:
        return 0.0
    return same_color_adjacency(state) / baseline


def board_profile(state: BoardState) -> Dict[str, float]:
    """盘面画像：数量均衡度、空间聚集度、颜色计数。"""
    counts = sorted(
        (n for n in color_counts(state).values() if n > 0), reverse=True
    )
    return {
        "colors": float(len(counts)),
        "counts": counts,
        "balance": balance(state),
        "adjacency": same_color_adjacency(state),
        "adjacency_expected": 1.0 / len(counts) if counts else 0.0,
        "adjacency_ratio": adjacency_ratio(state),
        "remaining": float(sum(counts)),
    }


# ---------------------------------------------------------------------------
# 预置生成方案（基准脚本直接引用，保证跨次运行可比）
# ---------------------------------------------------------------------------


def make_generator(name: str):
    """按名字取生成器；返回 ``(height, width, colors, rng) -> BoardState``。

    已定义：

    ================  ==================================================
    ``uniform``       每格 iid 均匀（此前全部结论的基线）
    ``imbalanced``    数量不均 [.52,.24,.16,.08]，空间仍散点
    ``clustered``     空间聚集（8 中心，5% 打散），数量大致均衡
    ``clustered_imb`` 聚集 + 数量不均
    ``extreme``       极端不均 [.70,.15,.10,.05]，空间散点
    ``blob5``         5 色强聚集（6 中心，纯分块）
    ================  ==================================================
    """
    table = {
        "uniform": lambda h, w, c, rng: uniform_board(h, w, c, rng),
        "imbalanced": lambda h, w, c, rng: weighted_board(
            h, w, c, rng, [0.52, 0.24, 0.16, 0.08]
        ),
        "clustered": lambda h, w, c, rng: clustered_board(
            h, w, c, rng, centers=8, mix=0.05
        ),
        "clustered_imb": lambda h, w, c, rng: clustered_board(
            h, w, c, rng, centers=8, mix=0.05, weights=[0.52, 0.24, 0.16, 0.08]
        ),
        "extreme": lambda h, w, c, rng: weighted_board(
            h, w, c, rng, [0.70, 0.15, 0.10, 0.05]
        ),
        "blob5": lambda h, w, c, rng: clustered_board(h, w, c, rng, centers=6, mix=0.0),
    }
    if name not in table:
        raise ValueError(
            f"unknown generator: {name!r}; available: {sorted(table)}"
        )
    return table[name]


GENERATOR_NAMES: Tuple[str, ...] = (
    "uniform",
    "imbalanced",
    "clustered",
    "clustered_imb",
    "extreme",
    "blob5",
)


__all__ = [
    "uniform_board",
    "weighted_board",
    "clustered_board",
    "balance",
    "same_color_adjacency",
    "board_profile",
    "make_generator",
    "GENERATOR_NAMES",
]
