"""核心棋盘引擎（Phase 1）

本模块是全局唯一的规则实现处。任何上层模块（求解器、UI、测试）都只能通过
``apply_move`` 推进状态，禁止自行实现 gravity / compression。

坐标约定
--------
``(row, col)``，row = 0 为最上方一行，row 增大方向为重力方向。
空格统一用 ``EMPTY = -1``，颜色为 ``0 .. M-1`` 的整数 ID，仅用于判等。

一次合法动作的状态转移顺序固定为::

    Remove -> Vertical Gravity -> Horizontal Compression

转移结果天然满足规范化条件（见 :func:`is_canonical`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from operator import itemgetter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# ---------------------------------------------------------------------------
# 基础类型与常量
# ---------------------------------------------------------------------------

EMPTY = -1

Cell = Tuple[int, int]
BoardGrid = Tuple[Tuple[int, ...], ...]

_NEIGHBOR_OFFSETS: Tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

# 棋盘字符解析：以下字符表示空格
_EMPTY_TOKENS = {".", "-", "_", "*"}

# 连通块缓存容量。搜索时每个状态一个条目，是主要的内存占用项，可调。
COMPONENTS_CACHE_SIZE: Optional[int] = 100_000


@dataclass(frozen=True, slots=True)
class Component:
    """一个同色四邻域连通块。

    ``cells`` 始终按 ``(row, col)`` 字典序排序，因此 ``cells[0]`` 是该连通块
    的稳定代表格，可用于 Move 表示与回放。

    使用 ``slots`` 去掉实例 ``__dict__``：搜索时会大量创建 Component，
    每个实例节省上百字节。
    """

    color: int
    cells: Tuple[Cell, ...]

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def representative_cell(self) -> Cell:
        return self.cells[0]

    def __contains__(self, cell: Cell) -> bool:
        return cell in self.cells


@dataclass(frozen=True)
class Move:
    """一个合法动作的显式表示。

    同一个连通块内的不同坐标属于同一个动作，因此搜索时不得按坐标枚举，
    必须按连通块枚举（``component_index`` 唯一确定一个动作）。
    """

    representative_cell: Cell
    color: int
    cells: Tuple[Cell, ...]
    size: int
    component_index: int = -1

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError(f"illegal move: component size must be >= 2, got {self.size}")


# ---------------------------------------------------------------------------
# 棋盘状态
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardState:
    """逻辑上不可变的棋盘状态。

    只保存棋盘本身，不保存累计分数 / 历史动作。累计分数由搜索节点或
    :mod:`popstar.game` 持有，这样状态缓存 ``V(S)`` 不因到达路径不同而重复。
    """

    grid: BoardGrid

    def __post_init__(self) -> None:
        if not self.grid:
            raise ValueError("board must contain at least one row")
        width = len(self.grid[0])
        if width == 0:
            raise ValueError("board must contain at least one column")
        for row in self.grid:
            if len(row) != width:
                raise ValueError("all rows must have the same length")
            for value in row:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"cell value must be int, got {value!r}")
                if value < EMPTY:
                    raise ValueError(f"cell value must be >= {EMPTY} (EMPTY), got {value}")
        object.__setattr__(self, "packed", None)

    @property
    def packed_key(self) -> bytes:
        """棋盘的紧凑字节键（每格 1 字节），首次访问时惰性计算并缓存。"""
        cached = self.packed  # type: ignore[attr-defined]
        if cached is None:
            cached = pack_grid(self.grid)
            object.__setattr__(self, "packed", cached)
        return cached

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0])

    def cell(self, row: int, col: int) -> int:
        return self.grid[row][col]

    def __str__(self) -> str:  # pragma: no cover - 调试便利
        return format_board(self)


def _make_state(grid: BoardGrid) -> BoardState:
    """内部快速构造：跳过逐格合法性校验。

    仅用于本模块内部由合法状态变换出的结果（gravity / compression 只搬运
    原有取值，不会产生非法值）。公开构造仍走 :func:`create_board` 的完整校验。
    搜索热路径每步都要构造十余个棋盘，这条路径省下的开销相当可观。
    """
    state = BoardState.__new__(BoardState)
    object.__setattr__(state, "grid", grid)
    object.__setattr__(state, "packed", None)
    return state


# ---------------------------------------------------------------------------
# 构造与解析
# ---------------------------------------------------------------------------


def create_board(rows: Sequence[Sequence[int]]) -> BoardState:
    """从二维矩阵创建棋盘状态。"""
    grid = tuple(tuple(int(value) for value in row) for row in rows)
    return BoardState(grid)


def parse_board(text: str) -> BoardState:
    """从文本解析棋盘。

    ``.`` / ``-`` / ``_`` / ``*`` 表示空格，其余字符按**首次出现顺序**映射为
    颜色 ID 0,1,2,... 行内可用空白分隔（``A B .``）也可紧凑书写（``AB.``）。
    ``#`` 开头的行为注释。
    """
    mapping: Dict[str, int] = {}
    rows: List[List[int]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split() if len(line.split()) > 1 else list(line)
        row: List[int] = []
        for token in tokens:
            if token in _EMPTY_TOKENS:
                row.append(EMPTY)
                continue
            if token not in mapping:
                mapping[token] = len(mapping)
            row.append(mapping[token])
        rows.append(row)
    if not rows:
        raise ValueError("empty board text")
    return create_board(rows)


def random_board(height: int, width: int, num_colors: int, rng) -> BoardState:
    """随机生成棋盘，``rng`` 需提供 ``randrange``（如 ``random.Random``）。"""
    grid = tuple(
        tuple(rng.randrange(num_colors) for _ in range(width)) for _ in range(height)
    )
    return BoardState(grid)


_COLOR_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def format_board(state: BoardState) -> str:
    """将棋盘渲染为可读文本，空格为 ``.``。"""
    lines = []
    for row in state.grid:
        chars = []
        for value in row:
            if value == EMPTY:
                chars.append(".")
            elif 0 <= value < len(_COLOR_CHARS):
                chars.append(_COLOR_CHARS[value])
            else:
                chars.append("?")
        lines.append(" ".join(chars))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 连通块
# ---------------------------------------------------------------------------


def pack_grid(grid: BoardGrid) -> bytes:
    """把棋盘压成紧凑字节串（每格 1 字节，``EMPTY -> 0, color -> color+1``）。

    用作字典 / lru_cache 的键：相比嵌套元组，占用约降至 1/10 且哈希更快。
    """
    # 列表推导喂给 bytes() 比生成器表达式略快（bytes 对 list 有专门的快路径）
    return bytes([value + 1 for row in grid for value in row])


def set_components_cache_size(maxsize: Optional[int]) -> None:
    """调整连通块缓存容量（搜索时可据此控制内存上限）。"""
    global COMPONENTS_CACHE_SIZE
    COMPONENTS_CACHE_SIZE = maxsize
    wrapped = globals()["_components_cached"].__wrapped__
    globals()["_components_cached"] = lru_cache(maxsize=maxsize)(wrapped)


@lru_cache(maxsize=8)
def _neighbor_table(height: int, width: int) -> Tuple[Tuple[int, ...], ...]:
    """预计算扁平下标的四邻域邻接表（按棋盘尺寸缓存）。

    把 ``(r, c)`` 四元组栈换成整数下标栈 + 预计算邻接表，避免 BFS 内层循环里
    反复做 ``r*width+c``、边界判断和元组分配。实测这是 10x10 搜索的首要热点：
    原实现每次 cache miss 约 117us，改写后降到约 30us。
    """
    table: List[Tuple[int, ...]] = []
    for r in range(height):
        for c in range(width):
            neighbours: List[int] = []
            if r > 0:
                neighbours.append((r - 1) * width + c)
            if r + 1 < height:
                neighbours.append((r + 1) * width + c)
            if c > 0:
                neighbours.append(r * width + c - 1)
            if c + 1 < width:
                neighbours.append(r * width + c + 1)
            table.append(tuple(neighbours))
    return tuple(table)


@lru_cache(maxsize=COMPONENTS_CACHE_SIZE)
def _components_cached(key: bytes, height: int, width: int) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
    """计算全部同色连通块（含大小为 1 的孤立块）的**紧凑表示**。

    键是 :func:`pack_grid` 的结果，值是 ``((color, flat_cells), ...)``，
    其中 ``flat_cells = (r0, c0, r1, c1, ...)``。之所以不直接缓存
    ``Component`` 对象：每个状态会生成十余个 Component、每个 Component 又持有
    若干坐标二元组，实测每条目高达 6.8KB（7x7），是整个搜索的内存瓶颈。
    """
    # 直接以 packed 字节比色（EMPTY -> 0，颜色 -> color+1），省去一次 list 物化
    size = height * width
    neighbours = _neighbor_table(height, width)
    visited = bytearray(size)
    components: List[Tuple[int, Tuple[int, ...]]] = []

    for start in range(size):
        if visited[start]:
            continue
        marker = key[start]
        if marker == 0:  # EMPTY
            continue
        visited[start] = 1
        stack = [start]
        idxs: List[int] = []
        while stack:
            index = stack.pop()
            idxs.append(index)
            for neighbour in neighbours[index]:
                if not visited[neighbour] and key[neighbour] == marker:
                    visited[neighbour] = 1
                    stack.append(neighbour)
        # 扁平下标序 == (row, col) 字典序（index = row*width + col）
        idxs.sort()
        flat: List[int] = []
        for index in idxs:
            row, col = divmod(index, width)
            flat.append(row)
            flat.append(col)
        components.append((marker - 1, tuple(flat)))
    return tuple(components)


def _raw_components(state: BoardState) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
    return _components_cached(state.packed_key, state.height, state.width)


def _unpack(packed: Tuple[int, Tuple[int, ...]]) -> Component:
    color, flat = packed
    return Component(color, tuple(zip(flat[0::2], flat[1::2])))


def get_components(state: BoardState, min_size: int = 1) -> Tuple[Component, ...]:
    """返回当前全部同色连通块。

    ``min_size`` 默认 1 表示返回所有连通块；设为 2 则只返回可消除的块。
    返回顺序为确定性的扫描顺序（先行后列，按代表格排序）。
    """
    packed = _raw_components(state)
    components = tuple(_unpack(item) for item in packed)
    if min_size <= 1:
        return components
    return tuple(comp for comp in components if comp.size >= min_size)


def get_group(state: BoardState, cell: Cell) -> Optional[Component]:
    """返回指定坐标所属的完整连通块；坐标越界或为空格时返回 ``None``。"""
    row, col = cell
    if not (0 <= row < state.height and 0 <= col < state.width):
        return None
    if state.grid[row][col] == EMPTY:
        return None
    for packed in _raw_components(state):
        flat = packed[1]
        for index in range(0, len(flat), 2):
            if flat[index] == row and flat[index + 1] == col:
                return _unpack(packed)
    return None  # pragma: no cover - 理论上不可达


def can_remove(state: BoardState, cell: Cell) -> bool:
    """指定坐标是否可消除（所属连通块大小 >= 2）。"""
    comp = get_group(state, cell)
    return comp is not None and comp.size >= 2


def _legal_moves(packed: Tuple[Tuple[int, Tuple[int, ...]], ...]) -> Tuple[Move, ...]:
    """从紧凑表示直接构造 Move，避免为每个连通块分配 Component 对象。"""
    moves: List[Move] = []
    for index, (color, flat) in enumerate(packed):
        size = len(flat) // 2
        if size < 2:
            continue
        cells = tuple(zip(flat[0::2], flat[1::2]))
        moves.append(Move(cells[0], color, cells, size, index))
    return tuple(moves)


def get_legal_moves(state: BoardState) -> Tuple[Move, ...]:
    """返回当前全部合法动作，每个连通块恰好对应一个动作。"""
    return _legal_moves(_raw_components(state))


# ---------------------------------------------------------------------------
# 状态转移的三个原子步骤
# ---------------------------------------------------------------------------


def remove_cells_raw(grid: BoardGrid, cells: Iterable[Cell]) -> BoardGrid:
    """Remove：把给定坐标置空。

    这是 ``apply_move`` 的内部构件。**不**单独对外产生规范化状态。
    """
    table = [[value for value in row] for row in grid]
    for row, col in cells:
        table[row][col] = EMPTY
    return tuple(tuple(row) for row in table)


def apply_gravity(grid: BoardGrid) -> BoardGrid:
    """Vertical Gravity：逐列下落，保持列内相对顺序，空格全部上浮到顶部。"""
    if not grid:  # pragma: no cover - BoardState 已保证非空
        return grid
    height = len(grid)
    columns: List[Tuple[int, ...]] = []
    # zip(*grid) 在 C 层完成转置，比逐格 grid[r][c] 索引快约 1/3
    for column in zip(*grid):
        stacked = [value for value in column if value != EMPTY]
        columns.append((EMPTY,) * (height - len(stacked)) + tuple(stacked))
    return tuple(zip(*columns))


def compress_columns(grid: BoardGrid) -> BoardGrid:
    """Horizontal Compression：删除空列，其右侧非空列整体左移。"""
    if not grid:  # pragma: no cover - BoardState 已保证非空
        return grid
    height, width = len(grid), len(grid[0])
    kept: List[int] = []
    for c in range(width):
        # 自底向上扫：重力后的非空列在第一行判断就能命中；空列才会扫满
        for r in range(height - 1, -1, -1):
            if grid[r][c] != EMPTY:
                kept.append(c)
                break
    if len(kept) == width:
        return grid
    tail = (EMPTY,) * (width - len(kept))
    if not kept:  # 全部为空列
        return tuple(tail for _ in range(height))
    if len(kept) == 1:  # itemgetter 单索引返回标量，不是元组
        only = kept[0]
        return tuple((row[only],) + tail for row in grid)
    pick = itemgetter(*kept)
    return tuple(pick(row) + tail for row in grid)


def apply_gravity_to_state(state: BoardState) -> BoardState:
    """对状态单独施加垂直重力（测试 / 调试用，非正常流程入口）。"""
    return BoardState(apply_gravity(state.grid))


def compress_state(state: BoardState) -> BoardState:
    """对状态单独施加空列左移（测试 / 调试用，非正常流程入口）。"""
    return BoardState(compress_columns(state.grid))


# ---------------------------------------------------------------------------
# 唯一正式状态转移入口
# ---------------------------------------------------------------------------


def apply_move(
    state: BoardState, move: Union[Move, Component], validate: bool = True
) -> BoardState:
    """执行一次合法消除，返回规范化后的新棋盘状态。

    这是全局唯一正式状态转移入口：Remove -> Vertical Gravity -> Compression。
    原状态不被修改。

    Parameters
    ----------
    validate:
        默认 ``True``，走完整的 :func:`_validate_move`（颜色、越界、大小、
        代表格归属、四连通性）。搜索热路径可传 ``False`` 跳过**重复校验**——
        前提是 ``move`` 必须来自 ``get_legal_moves(state)``。

        注意这**不是**另一条转移管线：``validate=False`` 只是省掉一次
        「重新推导本就已知的连通块」的防御性检查，
        Remove -> Gravity -> Compression 三步与默认路径完全一致。
        :func:`popstar.tests` 中有对拍用例保证两者输出逐格相同。
    """
    cells = _validate_move(state, move) if validate else move.cells
    grid = remove_cells_raw(state.grid, cells)
    grid = apply_gravity(grid)
    grid = compress_columns(grid)
    return _make_state(grid)


def _validate_move(state: BoardState, move: Union[Move, Component]) -> Tuple[Cell, ...]:
    """校验动作并返回要移除的坐标集合。"""
    if isinstance(move, Component):
        comp = move
    else:
        if move.cells:
            comp = Component(move.color, tuple(move.cells))
        else:
            comp = get_group(state, move.representative_cell)
            if comp is None:
                raise ValueError(
                    f"move representative cell {move.representative_cell} is empty or out of range"
                )
    if comp.size < 2:
        raise ValueError(f"illegal move: component size must be >= 2, got {comp.size}")
    for row, col in comp.cells:
        if not (0 <= row < state.height and 0 <= col < state.width):
            raise ValueError(f"cell {(row, col)} out of board range")
        if state.grid[row][col] != comp.color:
            raise ValueError(
                f"cell {(row, col)} has color {state.grid[row][col]}, "
                f"expected {comp.color}"
            )
    if isinstance(move, Move) and move.representative_cell not in comp.cells:
        raise ValueError(
            f"representative cell {move.representative_cell} not inside component"
        )
    _assert_connected(state, comp)
    return comp.cells


def _assert_connected(state: BoardState, comp: Component) -> None:
    """防御性校验：待移除的格子必须是同一个四邻域连通块。"""
    cell_set = set(comp.cells)
    if len(cell_set) != len(comp.cells):
        raise ValueError("component contains duplicate cells")
    start = comp.cells[0]
    seen: Set[Cell] = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBOR_OFFSETS:
            neighbour = (r + dr, c + dc)
            if neighbour in cell_set and neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    if len(seen) != len(cell_set):
        raise ValueError("cells of a move must form one 4-connected component")


# ---------------------------------------------------------------------------
# 查询：剩余数量 / 终局 / 规范化校验
# ---------------------------------------------------------------------------


def count_remaining(state: BoardState) -> int:
    """当前剩余方块数量。"""
    return sum(1 for row in state.grid for value in row if value != EMPTY)


def is_terminal(state: BoardState) -> bool:
    """是否终局：不存在任何大小 >= 2 的同色连通块。"""
    for _color, flat in _raw_components(state):
        if len(flat) >= 4:  # flat 中每两数表示一个格子
            return False
    return True


def is_canonical(state: BoardState) -> bool:
    """检查状态是否满足规范化条件（debug assertion 用）。"""
    height, width = state.height, state.width
    for c in range(width):
        seen_block = False
        for r in range(height):
            if state.grid[r][c] != EMPTY:
                seen_block = True
            elif seen_block:
                # 方块下方出现空格：违反垂直重力（列内必须为「空格在上、方块在下」）
                return False
    seen_empty_col = False
    for c in range(width):
        col_empty = all(state.grid[r][c] == EMPTY for r in range(height))
        if col_empty:
            seen_empty_col = True
        elif seen_empty_col:
            # 空列右侧出现非空列：违反水平压缩
            return False
    return True


def assert_canonical(state: BoardState) -> None:
    """不满足规范化条件时抛出 ``AssertionError``。"""
    if not is_canonical(state):
        raise AssertionError(f"non-canonical board state:\n{format_board(state)}")


# ---------------------------------------------------------------------------
# 统计辅助（Phase 2 求解器会用到，Phase 1 仅提供）
# ---------------------------------------------------------------------------


def color_counts(state: BoardState) -> Dict[int, int]:
    """每种颜色的剩余方块数量（只统计仍存在的颜色）。"""
    counts: Dict[int, int] = {}
    for row in state.grid:
        for value in row:
            if value != EMPTY:
                counts[value] = counts.get(value, 0) + 1
    return counts


def equal_boards(a: BoardState, b: BoardState) -> bool:
    """棋盘逐格相等（尺寸与内容）。"""
    return a.grid == b.grid


# 显式导出，便于 ``from popstar.board import *`` 之外的批量引用
__all__ = [
    "EMPTY",
    "BoardState",
    "Component",
    "Move",
    "create_board",
    "parse_board",
    "random_board",
    "format_board",
    "get_components",
    "get_group",
    "can_remove",
    "get_legal_moves",
    "apply_move",
    "apply_gravity",
    "apply_gravity_to_state",
    "compress_columns",
    "compress_state",
    "remove_cells_raw",
    "count_remaining",
    "is_terminal",
    "is_canonical",
    "assert_canonical",
    "color_counts",
    "equal_boards",
]
