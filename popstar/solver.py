"""求解器（Phase 2 基线）。

层次由简到繁，每一层都能被单独关闭并对拍验证：

1. ``use_memo``       Exact DFS + Memoization（第 14 节基线）
2. ``use_bound``      Branch-and-Bound，乐观上界 ``U(S)=Σ_c g(n_c)+B(0)``（第 17 节）
3. ``ordering``       move ordering（第 18 节），只改顺序，不删分支
4. ``canonicalize``   颜色规范化后作为哈希键（第 16 节），不改变回放中的颜色

以及近似模式 ``BeamSolver``（第 24 节），结果标记为 ``best found``，
可作为 Exact 搜索的初始下界（第 25 节）。

注释里的「第 N 节」是本仓库开发提纲的编号，不是外部论文的章节。
算法本身是常规搜索（记忆化 DFS、分支定界、beam search），在这里实现，
没有移植别人的 SameGame 求解器。出处见仓库根目录 ``SOURCES.md``。

所有模式都只能通过 :func:`popstar.board.apply_move` 推进状态。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .board import (
    EMPTY,
    BoardGrid,
    BoardState,
    Move,
    _raw_components,
    apply_move,
    color_counts,
    count_remaining,
    get_components,
    get_legal_moves,
    is_terminal,
)
from .game import Game
from .records import move_record_to_dict
from .scoring import DEFAULT_SCORING, ScoringConfig

# ---------------------------------------------------------------------------
# 上界与势函数
# ---------------------------------------------------------------------------


def forced_residual(state: BoardState) -> int:
    """终局剩余数的**可证明下界**。

    若某颜色当前只剩 1 个方块，它再也找不到同色伙伴，因此在任何后续走法中
    都不可能属于一个大小 ≥ 2 的连通块 —— 该方块永远无法被消除。
    把所有这类「死色」的方块数相加即得下界。
    """
    counts = color_counts(state)
    return sum(1 for count in counts.values() if count == 1)


def terminal_bonus_ceiling(scoring: ScoringConfig, max_blocks: int) -> List[float]:
    """预计算 ``ceil[r] = max_{R ≥ r} B(R)``（后缀最大值）。

    用它可以把「终局必然至少剩 r 个方块」这一信息安全地折进上界，
    且不依赖 ``B`` 单调 —— 对任意自定义 ``B`` 都成立。
    """
    values = [scoring.score_terminal(r) for r in range(max_blocks + 1)]
    ceiling = [0.0] * (max_blocks + 2)
    running = float("-inf")
    for r in range(max_blocks, -1, -1):
        running = max(running, values[r])
        ceiling[r] = running
    return ceiling


def optimistic_bound(state: BoardState, scoring: ScoringConfig) -> float:
    """第 17 节乐观上界的**收紧版**（仍然绝不低于真实 ``V(S)``）。

    .. code-block:: text

        U(S) = Σ_{c: n_c ≥ 2} g(n_c) + max_{R ≥ forced(S)} B(R)

    相比原始形式的收紧点：终局不再无条件假设全清，而是用
    :func:`forced_residual` 给出的可证明残余下界限制奖励项。
    死色越多，上界越低；当 ``forced ≥ A`` 时奖励项直接归零。

    有效性前提仍是 ``g`` 超可加且单调不减（见 :func:`validate_scoring_for_bound`）。
    """
    return optimistic_bound_with_ceiling(state, scoring, None)


def optimistic_bound_with_ceiling(
    state: BoardState,
    scoring: ScoringConfig,
    ceiling: Optional[List[float]],
) -> float:
    """同 :func:`optimistic_bound`，可传入预计算的 ``ceiling`` 后缀表以省去重复求值。"""
    counts = color_counts(state)
    group_part = 0.0
    forced = 0
    for count in counts.values():
        if count >= 2:
            group_part += scoring.score_group(count)
        elif count == 1:
            forced += 1
    if ceiling is not None:
        bonus_part = ceiling[min(forced, len(ceiling) - 1)]
    else:
        bonus_part = max(
            scoring.score_terminal(r)
            for r in range(forced, count_remaining(state) + 1)
        )
    return group_part + bonus_part


def lookahead_bound(
    state: BoardState,
    scoring: Optional[ScoringConfig] = None,
    depth: int = 0,
    ceiling: Optional[List[float]] = None,
) -> float:
    """深度 ``d`` 的可采纳上界 ``U^(d)``。

    终局就是 ``B(R)``。``depth == 0`` 的非终局就是 :func:`optimistic_bound`。
    再深一层先走一步真实转移，再套更浅的上界::

        U^(d+1)(S) = max_a [ g(k_a) + U^(d)(T(S, a)) ]

    ``g`` 超可加且单调不减时，层数加深不会把上界抬高，也绝不会低于 ``V(S)``。
    这道上界可以剪枝。它不是 ``Φ_agg``。默认搜索仍用 ``depth == 0``。
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    scoring = scoring or DEFAULT_SCORING
    if ceiling is None and depth > 0:
        ceiling = terminal_bonus_ceiling(scoring, state.height * state.width)
    memo: Dict[Tuple[bytes, int], float] = {}

    def value(current: BoardState, remaining_depth: int) -> float:
        if is_terminal(current):
            return scoring.score_terminal(count_remaining(current))
        if remaining_depth <= 0:
            return optimistic_bound_with_ceiling(current, scoring, ceiling)
        key = (current.packed_key, remaining_depth)
        cached = memo.get(key)
        if cached is not None:
            return cached
        best = float("-inf")
        for move in get_legal_moves(current):
            child = apply_move(current, move, validate=False)
            total = scoring.score_group(move.size) + value(child, remaining_depth - 1)
            if total > best:
                best = total
        memo[key] = best
        return best

    return value(state, depth)


def validate_scoring_for_bound(
    scoring: ScoringConfig, max_blocks: int = 128
) -> List[str]:
    """检查计分函数是否满足上界所需条件，返回问题列表（空列表表示可用）。"""
    issues: List[str] = []
    g = scoring.group_score
    # 单调不减
    for k in range(2, max_blocks):
        if g(k + 1) < g(k):
            issues.append(f"g is not non-decreasing at k={k}")
            break
    # 超可加性
    for a in range(2, max_blocks):
        for b in range(2, max_blocks - a):
            if g(a) + g(b) > g(a + b) + 1e-9:
                issues.append(f"g is not superadditive: g({a})+g({b}) > g({a+b})")
                return issues
    # B(0) 必须是最大值
    bonus_max = max(scoring.score_terminal(r) for r in range(0, max_blocks))
    if scoring.score_terminal(0) < bonus_max - 1e-9:
        issues.append("B(0) is not the maximum terminal bonus")
    return issues


@dataclass(frozen=True, slots=True)
class Features:
    """一个局面的一次性诊断/评估量（见 :func:`board_features`）。"""

    color_counts: Dict[int, int]  # 各颜色剩余方块数
    per_color: Dict[int, float]  # 各颜色「当前已成团」的得分 Σ_i g(s_i)
    singletons: int  # 大小为 1 的同色连通块个数
    remaining: int  # 剩余方块总数
    available: float  # Σ_{s_i ≥ 2} g(s_i)：现在就抓得到的分数
    merge: float  # 聚集势能 Σ_c [g(n_c) − Σ_i g(s_{c,i})]
    forced: int  # 死色块数（该颜色只剩 1 个），即终局剩余的可证明下界
    terminal: bool  # 是否终局（不存在大小 ≥ 2 的同色连通块）


def make_score_table(scoring: ScoringConfig, max_size: int) -> List[float]:
    """预计算 ``g(k)`` 查表（``k = 0..max_size``）。

    热路径上每个连通块都要取一次 ``g(k)``，10x10 一局能到两百万次。
    查表把它降为一次列表索引，同时保持 ``g`` 可由 ``ScoringConfig`` 任意替换。
    """
    table = [0.0] * (max_size + 1)
    for k in range(2, max_size + 1):
        table[k] = scoring.score_group(k)
    return table


def board_features(
    state: BoardState,
    scoring: ScoringConfig,
    packed: Optional[Sequence[Tuple[int, Tuple[int, ...]]]] = None,
    g_table: Optional[List[float]] = None,
) -> "Features":
    """一次遍历同时算出搜索评估需要的全部量。

    原先 :func:`merge_potential` 与 :func:`singleton_count` 各遍历一次连通块，
    :func:`color_counts` 还要再全盘扫描一遍。10x10 上每个子状态都要评估，
    这三趟占了相当比例的耗时，这里合并成一趟。

    ``g_table`` 为 :func:`make_score_table` 的结果；省略时按棋盘尺寸现算
    （调用方若能复用查表请务必传入）。
    """
    comps = _raw_components(state) if packed is None else packed
    if g_table is None:
        g_table = make_score_table(scoring, state.height * state.width)
    counts: Dict[int, int] = {}
    per_color: Dict[int, float] = {}
    singletons = 0
    remaining = 0
    available = 0.0
    groups = 0
    for color, flat in comps:
        size = len(flat) // 2
        remaining += size
        counts[color] = counts.get(color, 0) + size
        if size >= 2:
            groups += 1
            value = g_table[size]
            per_color[color] = per_color.get(color, 0.0) + value
            available += value
        else:
            singletons += 1
    merge = 0.0
    for color, count in counts.items():
        if count >= 2:
            merge += g_table[count] - per_color.get(color, 0.0)
    forced = sum(1 for count in counts.values() if count == 1)
    return Features(
        color_counts=counts,
        per_color=per_color,
        singletons=singletons,
        remaining=remaining,
        available=available,
        merge=merge,
        forced=forced,
        terminal=groups == 0,
    )


def group_score_ceiling(counts: Sequence[int], group_score) -> float:
    """终局 group 分的组合上界 ``U_group = Σ_{c: n_c ≥ 2} g(n_c)``。

    含义是「每种颜色最终全部汇成一团」。这是完全忽略几何约束的松弛，
    把所有本不可能合并的同色块都当成可以合并。
    """
    total = 0.0
    for count in counts:
        if count >= 2:
            total += float(group_score(count))
    return total


def nonclear_group_ceiling(counts: Sequence[int], group_score) -> float:
    """**非清盘**时的 group 分上界（用户修正版，比 ``U_group`` 严格更小）。

    不清盘意味着终局至少留下一个「本来可以被消掉」的方块，于是必然要从某个
    活跃颜色里扣掉一块::

        U_nonclear = max_{c: n_c ≥ 2} [ Σ_{d≠c, n_d≥2} g(n_d) + ĝ(n_c − 1) ]

    其中 ``ĝ(n) = g(n)`` 当 ``n ≥ 2``，否则 0（只剩 1 块时不构成可消除块）。
    取 max 是因为损失 ``g(n_c) − ĝ(n_c−1)`` 随 ``n_c`` 增大，
    所以最优做法是**从最小的活跃颜色里扣**，而不是随便挑一个。

    注意若棋盘存在死色（``n_c == 1``），那些方块本来就是强制残余，
    不计入任何一种情况的 group 分，也不构成「额外扣减」的来源。
    """
    active = [n for n in counts if n >= 2]
    if not active:
        return 0.0
    base = sum(float(group_score(n)) for n in active)
    best = float("-inf")
    for n in active:
        reduced = float(group_score(n - 1)) if n - 1 >= 2 else 0.0
        best = max(best, base - float(group_score(n)) + reduced)
    return best


def merge_potential(state: BoardState, scoring: ScoringConfig) -> float:
    """第 20 节的聚集势能 ``E_merge(S) = Σ_c [g(n_c) − Σ_i g(s_{c,i})]``。

    含义是当前因碎片化而尚未释放的理论得分价值，仅用于排序 / 诊断，
    **不能**当作精确上界使用。

    这个量在 :mod:`popstar.topology` 里叫做 ``E_latent``。
    已经兑现的团值是 ``E_formed``，一步坠落额外形成的团值是 ``ΔE_release``。
    三者不要再混称势能。
    """
    return board_features(state, scoring).merge


def formed_energy(state: BoardState, scoring: Optional[ScoringConfig] = None) -> float:
    """调用 :func:`popstar.topology.formed_energy`。搜索代码不要在这里加新权重。"""
    from .topology import formed_energy as _formed_energy
    return _formed_energy(state, scoring)


def latent_energy(state: BoardState, scoring: Optional[ScoringConfig] = None) -> float:
    """调用 :func:`popstar.topology.latent_energy`。"""
    from .topology import latent_energy as _latent_energy
    return _latent_energy(state, scoring)


def aggregation_release(
    before: BoardState,
    after: BoardState,
    removed_size: int,
    scoring: Optional[ScoringConfig] = None,
) -> float:
    """调用 :func:`popstar.topology.aggregation_release`。"""
    from .topology import aggregation_release as _aggregation_release
    return _aggregation_release(before, after, removed_size, scoring)


def cluster_sizes(state: BoardState) -> Tuple[Tuple[int, int], ...]:
    """返回 ``((cluster_size, color), ...)``，直接读紧凑表示，不构造 Component。"""
    return tuple(
        (len(flat) // 2, color) for color, flat in _raw_components(state)
    )


def singleton_count(state: BoardState) -> int:
    """危险孤立块数量（大小为 1 的同色连通块个数）。"""
    return sum(1 for _color, flat in _raw_components(state) if len(flat) < 4)


@lru_cache(maxsize=100_000)
def canonical_grid(grid: BoardGrid) -> BoardGrid:
    """第 16 节的颜色规范化：按行优先扫描的首次出现次序重新编号颜色。

    颜色重编号是双射，连通块结构一一对应，因此 ``V(S)`` 不变。
    返回的棋盘仅用于哈希，不改变实际回放中的颜色映射。
    """
    mapping: Dict[int, int] = {}
    rows: List[Tuple[int, ...]] = []
    for row in grid:
        new_row: List[int] = []
        for value in row:
            if value == EMPTY:
                new_row.append(EMPTY)
                continue
            if value not in mapping:
                mapping[value] = len(mapping)
            new_row.append(mapping[value])
        rows.append(tuple(new_row))
    return tuple(rows)


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------


@dataclass
class Solution:
    """第 26 节要求的输出字段。"""

    initial_board: BoardState
    best_score: float
    group_score: float
    terminal_bonus: float
    remaining_count: int
    move_count: int
    move_sequence: Tuple[Move, ...]
    final_board: BoardState
    search_mode: str
    is_proven_optimal: bool
    states_expanded: int
    cache_hits: int
    cache_size: int
    max_depth: int
    elapsed_time: float
    steps: Tuple[Dict[str, Any], ...] = ()
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_board": [list(row) for row in self.initial_board.grid],
            "best_score": self.best_score,
            "group_score": self.group_score,
            "terminal_bonus": self.terminal_bonus,
            "remaining_count": self.remaining_count,
            "move_count": self.move_count,
            "best_move_sequence": [
                [
                    list(move.representative_cell),
                    move.color,
                    move.size,
                ]
                for move in self.move_sequence
            ],
            "final_board": [list(row) for row in self.final_board.grid],
            "search_mode": self.search_mode,
            "is_proven_optimal": self.is_proven_optimal,
            "states_expanded": self.states_expanded,
            "cache_hits": self.cache_hits,
            "cache_size": self.cache_size,
            "max_depth": self.max_depth,
            "elapsed_time": self.elapsed_time,
            "steps": [dict(step) for step in self.steps],
            "notes": dict(self.notes),
        }

    def summary(self) -> str:
        label = "已证明最优" if self.is_proven_optimal else "best found（未证明最优）"
        return (
            f"[{self.search_mode}] score={self.best_score:g} "
            f"(group={self.group_score:g} + bonus={self.terminal_bonus:g}) "
            f"moves={self.move_count} remaining={self.remaining_count} | {label} | "
            f"expanded={self.states_expanded} hits={self.cache_hits} "
            f"depth={self.max_depth} time={self.elapsed_time:.3f}s"
        )


class _SearchAborted(Exception):
    """节点 / 时间上限触发，向上层抛出以快速退出。"""


# ---------------------------------------------------------------------------
# Exact Solver
# ---------------------------------------------------------------------------


@dataclass
class ExactSolver:
    """精确求解器：Exact DFS + 可选 Memoization / B&B / 排序 / 颜色规范化。

    Parameters
    ----------
    scoring:
        计分配置。
    use_memo / use_bound / ordering / canonicalize:
        四个可独立开关的优化，便于逐层对拍。
    ordering:
        ``"none"`` 扫描顺序；``"size"`` 大连通块优先；``"heuristic"`` 按第 23 节的
        ``H(a)=g(|a|)+η[Φ(S')−Φ(S)]`` 排序（Φ 只含聚团项与孤块惩罚项）。
    node_limit / time_limit:
        上限控制；触发后结果仍可用，但 ``is_proven_optimal=False``。
    """

    scoring: ScoringConfig = DEFAULT_SCORING
    use_memo: bool = True
    use_bound: bool = False
    use_upper_cache: bool = True
    ordering: str = "none"
    canonicalize: bool = False
    node_limit: Optional[int] = None
    time_limit: Optional[float] = None
    # 0：每个节点用乐观上界。正数改用 U^(d)，默认搜索不打开。
    bound_depth: int = 0
    eta: float = 1.0
    lambda_cluster: float = 1.0
    lambda_isolated: float = 2.0

    def __post_init__(self) -> None:
        if self.ordering not in {"none", "size", "heuristic"}:
            raise ValueError(f"unknown ordering: {self.ordering}")
        if self.bound_depth < 0:
            raise ValueError(f"bound_depth must be >= 0, got {self.bound_depth}")
        if self.use_bound:
            issues = validate_scoring_for_bound(self.scoring)
            if issues:
                raise ValueError(
                    "optimistic bound is unsafe with this scoring: " + "; ".join(issues)
                )

    # ------------------------------------------------------------------

    def solve(
        self,
        state: BoardState,
        lower_bound: Optional[float] = None,
        lower_bound_path: Optional[Sequence[Any]] = None,
    ) -> Solution:
        """求解，返回 :class:`Solution`。

        ``lower_bound`` 可传入 Fast 模式得到的已知可行分数作为初始剪枝阈值；
        若同时给出 ``lower_bound_path``（代表格序列），则在未被超越时作为最终答案。
        """
        start = time.perf_counter()
        self._t0 = start
        self._cache: Dict[bytes, Tuple[float, Any]] = {}
        self._upper: Dict[Any, float] = {}
        self._upper_hits = 0
        self._pruned = 0
        self._ceiling = terminal_bonus_ceiling(
            self.scoring, state.height * state.width
        )
        self._best_score = float("-inf")
        self._best_path: List[Any] = []
        self._threshold = float("-inf")
        self._states = 0
        self._hits = 0
        self._max_depth = 0
        self._aborted = False

        if lower_bound is not None:
            self._threshold = float(lower_bound)
            if lower_bound_path is not None:
                self._best_score = float(lower_bound)
                self._best_path = list(lower_bound_path)

        root_value = float("-inf")
        try:
            root_value, _, _ = self._search(state, 0.0, 0, [])
        except _SearchAborted:
            self._aborted = True

        # 缓存命中会跳过末端节点，因此仅靠「走到过的最优终局」可能低于 V(S0)。
        # 这里再沿缓存中的最优动作重建一条主变例（PV），取两者中更好的一条。
        pv_path = self._extract_pv(state)
        if pv_path and (not self._best_path or len(pv_path) > 0):
            pv_score = _replay_score(state, pv_path, self.scoring)
            if pv_score > self._best_score:
                self._best_score = pv_score
                self._best_path = pv_path

        if (
            not self._aborted
            and self._best_path
            and root_value > float("-inf")
            and root_value > self._best_score + 1e-6
        ):
            # DP 认为存在更好的解，却无法重建出对应路径 —— 属于内部不一致，必须暴露
            raise AssertionError(
                f"best reconstructable {self._best_score} < DP value {root_value}; "
                "principal variation reconstruction failed"
            )

        elapsed = time.perf_counter() - start

        # 用实际回放重建动作序列、逐步明细与两部分得分
        game = Game(state, scoring=self.scoring)
        for cell in self._best_path:
            game.play_cell(cell)
        result = game.result(require_finished=False)

        # 一致性校验：搜索得到的分数必须等于回放分数
        if game.is_finished() and self._best_path:
            if abs(result.total_score - self._best_score) > 1e-6:
                raise AssertionError(
                    f"solver score {self._best_score} != replay score {result.total_score}"
                )

        return Solution(
            initial_board=state,
            best_score=result.total_score if self._best_path else float("-inf"),
            group_score=result.group_score,
            terminal_bonus=result.terminal_bonus,
            remaining_count=result.remaining_count,
            move_count=result.move_count,
            move_sequence=tuple(_records_to_moves(state, game)) if game.records else (),
            final_board=game.state,
            search_mode="exact",
            is_proven_optimal=not self._aborted,
            states_expanded=self._states,
            cache_hits=self._hits,
            cache_size=len(self._cache),
            max_depth=self._max_depth,
            elapsed_time=elapsed,
            steps=tuple(move_record_to_dict(r) for r in game.records),
            notes={
                "use_memo": self.use_memo,
                "use_bound": self.use_bound,
                "ordering": self.ordering,
                "canonicalize": self.canonicalize,
                "seeded_lower_bound": lower_bound,
                "upper_hits": self._upper_hits,
                "upper_size": len(self._upper),
                "pruned_nodes": self._pruned,
                "aborted": self._aborted,
            },
        )

    # ------------------------------------------------------------------

    def _key(self, state: BoardState) -> bytes:
        """紧凑的哈希键：每个格子 1 字节（``EMPTY -> 0, color -> color+1``）。

        相比 ``tuple(tuple(row) ...)``，内存占用约降至 1/10，且哈希更快。
        """
        grid = canonical_grid(state.grid) if self.canonicalize else state.grid
        if self.canonicalize:
            return bytes(value + 1 for row in grid for value in row)
        return state.packed_key

    def _bound(self, state: BoardState) -> float:
        if self.bound_depth > 0:
            return lookahead_bound(
                state, self.scoring, self.bound_depth, self._ceiling
            )
        return optimistic_bound_with_ceiling(state, self.scoring, self._ceiling)

    def _check_limits(self) -> None:
        if self.node_limit is not None and self._states >= self.node_limit:
            raise _SearchAborted("node limit")
        if self.time_limit is not None and time.perf_counter() - self._t0 > self.time_limit:
            raise _SearchAborted("time limit")

    def _search(
        self,
        state: BoardState,
        accumulated: float,
        depth: int,
        path: List[Any],
    ) -> Tuple[float, bool, float]:
        """返回 ``(值, 是否为精确值, 上界)``。

        值是「从该状态起还能获得的分数」的下界；只有当**所有未被完整搜索的
        分支的上界都不超过已找到的最佳分支**时，该值才等于 V(S) 并写入缓存。
        这条判据比「任一子树被剪枝即不精确」宽松得多，是让 B&B 与
        transposition table 真正共存的关键。
        """
        self._states += 1
        if depth > self._max_depth:
            self._max_depth = depth
        self._check_limits()

        if is_terminal(state):
            bonus = self.scoring.score_terminal(count_remaining(state))
            self._offer(accumulated + bonus, path)
            return bonus, True, bonus

        key = self._key(state)
        if self.use_memo and key in self._cache:
            self._hits += 1
            cached_value = self._cache[key][0]
            # 缓存命中意味着不再下降到终局，因此这里必须用「当前累计 + 缓存值」
            # 即时补记一条候选解，否则最优路径可能被整体跳过。
            if accumulated + cached_value > self._best_score + 1e-9:
                sub_path = self._extract_pv(state)
                if sub_path is not None:
                    self._offer(accumulated + cached_value, path + sub_path)
            return cached_value, True, cached_value

        bound = self._bound(state) if self.use_bound else None
        if self.use_bound:
            # 先查已记录的上界：命中则无需重新展开（等价于标准 TT 的 UPPER 标志）
            known_upper = self._upper.get(key) if self.use_upper_cache else None
            if known_upper is not None and accumulated + known_upper <= self._threshold:
                self._upper_hits += 1
                self._pruned += 1
                return float("-inf"), False, known_upper
            if accumulated + bound <= self._threshold:
                self._record_upper(key, bound)
                self._pruned += 1
                return float("-inf"), False, bound

        children = self._ordered_children(state)

        best_future = float("-inf")
        best_move: Any = None
        dominated_upper = float("-inf")  # 未完整搜索的分支中最大的上界
        for move, child in children:
            immediate = self.scoring.score_group(move.size)
            path.append(move.representative_cell)
            value, child_exact, child_upper = self._search(
                child, accumulated + immediate, depth + 1, path
            )
            path.pop()
            if not child_exact:
                dominated_upper = max(dominated_upper, immediate + child_upper)
            if value > float("-inf"):
                total_future = immediate + value
                if total_future > best_future:
                    best_future = total_future
                    best_move = move.representative_cell

        if best_move is None:
            # 全部分支都被剪枝
            if self.use_bound:
                self._record_upper(key, bound)
            return float("-inf"), False, bound if bound is not None else float("inf")

        # 被剪枝/未完整搜索的分支不可能超过 best_future 时，该值即为精确值
        exact = best_future >= dominated_upper
        upper_estimate = max(best_future, dominated_upper)
        if self.use_memo and exact:
            self._cache[key] = (best_future, best_move)
        if self.use_bound:
            self._record_upper(key, upper_estimate)
        return best_future, exact, upper_estimate

    def _record_upper(self, key: Any, value: Optional[float]) -> None:
        if value is None:
            return
        previous = self._upper.get(key)
        self._upper[key] = value if previous is None else min(previous, value)

    def _extract_pv(self, state: BoardState) -> Optional[List[Any]]:
        """沿缓存中的最优动作重建主变例；任一段缺失缓存则返回 ``None``。"""
        if not self._cache:
            return None
        path: List[Any] = []
        cursor = state
        seen = set()
        while not is_terminal(cursor):
            entry = self._cache.get(self._key(cursor))
            if entry is None:
                return None
            rep = entry[1]
            move = None
            for candidate in get_legal_moves(cursor):
                if candidate.representative_cell == rep:
                    move = candidate
                    break
            if move is None:
                return None
            key = self._key(cursor)
            if key in seen:  # 防御：理论上状态空间无环
                return None
            seen.add(key)
            path.append(rep)
            cursor = apply_move(cursor, move)
        return path

    def _offer(self, total: float, path: List[Any]) -> None:
        if total > self._best_score:
            self._best_score = total
            self._best_path = list(path)
        if total > self._threshold:
            self._threshold = total

    def _ordered_children(self, state: BoardState) -> List[Tuple[Move, BoardState]]:
        moves = get_legal_moves(state)
        # move 直接来自 get_legal_moves，跳过重复的合法性校验
        children = [(move, apply_move(state, move, validate=False)) for move in moves]
        if self.ordering == "none":
            return children
        if self.ordering == "size":
            return sorted(children, key=lambda item: (-item[0].size, item[0].representative_cell))
        return sorted(children, key=lambda item: -_heuristic_key(item, self))


def _heuristic_key(item: Tuple[Move, BoardState], solver: "ExactSolver") -> float:
    """第 23 节：``H(a) = g(|a|) + η[Φ(S') − Φ(S)]``，这里省去与 a 无关的 Φ(S)。"""
    move, child = item
    cluster_value = 0.0
    isolated = 0
    for size, _color in cluster_sizes(child):
        if size >= 2:
            cluster_value += solver.scoring.score_group(size)
        else:
            isolated += 1
    phi = solver.lambda_cluster * cluster_value - solver.lambda_isolated * isolated
    return solver.scoring.score_group(move.size) + solver.eta * phi


def _replay_score(state: BoardState, path: Sequence[Any], scoring: ScoringConfig) -> float:
    """把代表格序列实际走一遍，返回总分（用于候选方案比较）。"""
    game = Game(state, scoring=scoring)
    for cell in path:
        game.play_cell(cell)
    return game.result(require_finished=False).total_score


def _records_to_moves(state: BoardState, game: Game) -> List[Move]:
    """从回放记录还原每一步的 Move 对象（含完整 cells）。"""
    moves: List[Move] = []
    cursor = state
    for record in game.records:
        for move in get_legal_moves(cursor):
            if move.representative_cell == record.representative_cell:
                moves.append(move)
                break
        cursor = record.board_after_move
    return moves


# ---------------------------------------------------------------------------
# Terminal Pareto Frontier：(G, R) 非支配前沿
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParetoFront:
    """终局的 ``(group_score G, remaining R)`` 前沿。

    搜索目标里**不含任何终局奖励**：只记录「终局剩 ``R`` 块时能拿到的最高
    ``G``」。于是这批数据与 ``B(R)`` 的具体形式完全解耦 —— 以后把奖励公式
    整个换掉，只要它仍只依赖 ``R``，前沿就不用重跑。

    给定任意线性可分离的 ``F_λ(G, R) = G + λ·B₀(R)``，直接在前沿上取
    ``argmax`` 即得该系数下的最优策略，切换点可解析求出（见 :meth:`envelope`）。
    """

    initial_board: BoardState
    group_by_remaining: Dict[int, float]  # R -> 该 R 下找到的最高 group 分
    paths: Dict[int, Tuple[Any, ...]] = field(default_factory=dict)
    states_expanded: int = 0
    elapsed_time: float = 0.0
    profiles: Tuple[str, ...] = ()

    # -- 基础视图 ---------------------------------------------------------

    def points(self) -> List[Tuple[int, float]]:
        """全部观测点 ``(R, G)``，按 R 升序。"""
        return sorted(self.group_by_remaining.items())

    def pareto(self) -> List[Tuple[int, float]]:
        """Pareto 非支配点。

        ``(G₂, R₂)`` 被支配 ⟺ 存在 ``(G₁, R₁)`` 使 ``R₁ ≤ R₂`` 且 ``G₁ ≥ G₂``
        且至少一个严格 —— 即「剩得更少却拿得不少于它」。
        """
        pts = self.points()
        out: List[Tuple[int, float]] = []
        for r2, g2 in pts:
            dominated = False
            for r1, g1 in pts:
                if r1 <= r2 and g1 >= g2 and (r1, g1) != (r2, g2):
                    dominated = True
                    break
            if not dominated:
                out.append((r2, g2))
        return out

    # -- 给定奖励公式后的最优策略 ----------------------------------------

    def best(self, scoring: ScoringConfig) -> Tuple[float, int, float]:
        """在该计分下最优的一条，返回 ``(总分, 剩余数, group 分)``。"""
        best_total = float("-inf")
        best = (float("-inf"), -1, float("-inf"))
        for remaining, group in self.points():
            total = group + scoring.score_terminal(remaining)
            if total > best_total:
                best_total = total
                best = (total, remaining, group)
        return best

    def optimal_remaining(self, coefficient: float, scoring: ScoringConfig) -> int:
        """给定 ``bonus_coefficient``，最优策略对应的终局剩余数。"""
        if scoring.terminal_score is not None:
            raise ValueError(
                "optimal_remaining() needs the coefficient-separable default bonus"
            )
        return self.best(replace(scoring, bonus_coefficient=coefficient))[1]

    def envelope(
        self, scoring: ScoringConfig, lam_max: float = 200.0
    ) -> List[Tuple[float, Optional[float], int, float]]:
        """``λ = bonus_coefficient`` 的分段最优结构。

        返回 ``[(λ_lo, λ_hi, R, G), ...]``：在该 ``λ`` 区间内，前沿上
        ``F_λ = G + λ·B₀(R)`` 的最优策略是「终局剩 ``R``、group 分 ``G``」。

        原理：每个前沿点对应一条直线 ``G + λ·B₀(R)``，取上包络。
        相邻两条直线的交点即用户给出的切换公式::

            λ* = (G₂ − G₁) / (B₀(R₁) − B₀(R₂))

        ``B₀`` 相同的前沿点（例如所有 ``R ≥ A`` 的点，其 ``B₀ = 0``）
        只有 ``G`` 最大者可能出现在包络上，其余恒被支配。
        """
        shape = {}
        for remaining in self.group_by_remaining:
            value = scoring.terminal_shape(remaining)
            if value is None:
                raise ValueError(
                    "envelope() requires a linear-in-coefficient terminal bonus; "
                    "custom terminal_score is not separable"
                )
            shape[remaining] = value

        # B₀ 相同的点上只保留 G 最大的
        lines: List[Tuple[int, float, float]] = []  # (R, G, B₀)
        for remaining, group in self.points():
            b0 = shape[remaining]
            replaced = False
            for index, (line_r, line_g, line_b) in enumerate(lines):
                if line_b == b0:
                    if group > line_g:
                        lines[index] = (remaining, group, b0)
                    replaced = True
                    break
            if not replaced:
                lines.append((remaining, group, b0))

        # 候选切换点：所有两两交点
        breaks = {0.0}
        for i, (_r1, g1, b1) in enumerate(lines):
            for _r2, g2, b2 in lines[i + 1 :]:
                if abs(b1 - b2) < 1e-12:
                    continue
                lam = (g2 - g1) / (b1 - b2)
                if 0.0 < lam < lam_max:
                    breaks.add(lam)
        breaks.add(lam_max)
        ordered = sorted(breaks)

        def argmax_at(lam: float) -> Tuple[int, float]:
            best_r, best_g = -1, float("-inf")
            best_key = float("-inf")
            for r, g, b0 in lines:
                key = g + lam * b0
                # 同分时偏好剩余更少（清盘）的结果，便于给出稳定裁决
                if key > best_key + 1e-9 or (
                    abs(key - best_key) <= 1e-9 and (best_r < 0 or r < best_r)
                ):
                    best_key, best_r, best_g = key, r, g
            return best_r, best_g

        segments: List[Tuple[float, Optional[float], int, float]] = []
        for index in range(len(ordered) - 1):
            lo, hi = ordered[index], ordered[index + 1]
            r, g = argmax_at((lo + hi) / 2.0)  # 区间内任取一点即可，中点最稳
            if segments and segments[-1][2] == r:
                prev = segments[-1]
                segments[-1] = (prev[0], hi, r, g)
            else:
                segments.append((lo, None if hi >= lam_max else hi, r, g))
        return segments

    def switch_points(
        self, scoring: ScoringConfig, lam_max: float = 200.0
    ) -> List[Tuple[float, int, int]]:
        """策略切换点 ``(λ*, R_from, R_to)``，即相邻包络段的交界。"""
        segments = self.envelope(scoring, lam_max=lam_max)
        out: List[Tuple[float, int, int]] = []
        for index in range(1, len(segments)):
            lam = segments[index][0]
            out.append((lam, segments[index - 1][2], segments[index][2]))
        return out


# 不再落后的验证口径：默认 profile 集合
#
# 不同 profile 让 beam 在「多清 vs 多留」上产生分歧，合并后前沿才铺得开。
# 单跑一个 profile 只能得到前沿的一小段 —— 这是这套做法最容易踩的坑。
PARETO_PROFILES: Tuple[Tuple[str, Dict[str, float]], ...] = (
    ("clear", dict(weight_available=1.0, weight_merge=0.0, weight_bonus=1.0, weight_singleton=2.0)),
    ("neutral", dict(weight_available=1.0, weight_merge=0.0, weight_bonus=0.0, weight_singleton=0.0)),
    ("merge+", dict(weight_available=1.0, weight_merge=0.5, weight_bonus=0.0, weight_singleton=0.0)),
    ("merge++", dict(weight_available=1.0, weight_merge=1.0, weight_bonus=0.0, weight_singleton=0.0)),
)


def pareto_search(
    state: BoardState,
    scoring: ScoringConfig = DEFAULT_SCORING,
    beam_width: int = 256,
    profiles: Sequence[Tuple[str, Dict[str, float]]] = PARETO_PROFILES,
    time_limit: Optional[float] = None,
) -> ParetoFront:
    """跑多组 beam profile，合并出 ``(R -> 最高 G)`` 前沿。

    每个 profile 都是一次独立的合法 beam 搜索，所以合并结果里的每个点
    都仍然是一条**真实可行**的走法（可回放验证），只是不再保证最优。

    ``time_limit`` 是**每个 profile** 的时限，不是总时限。
    """
    started = time.perf_counter()
    group_by_remaining: Dict[int, float] = {}
    paths: Dict[int, Tuple[Any, ...]] = {}
    expanded = 0

    for name, overrides in profiles:
        solver = BeamSolver(scoring=scoring, beam_width=beam_width, **overrides)
        if time_limit is not None:
            solver.time_limit = time_limit
        run = solver._run(state)
        expanded += run.expanded
        for remaining, group in run.group_by_remaining.items():
            current = group_by_remaining.get(remaining)
            if current is None or group > current:
                group_by_remaining[remaining] = group
                paths[remaining] = tuple(
                    _path_from_parents(run.parents, run.node_by_remaining[remaining])
                )

    return ParetoFront(
        initial_board=state,
        group_by_remaining=group_by_remaining,
        paths=paths,
        states_expanded=expanded,
        elapsed_time=time.perf_counter() - started,
        profiles=tuple(name for name, _ in profiles),
    )


# ---------------------------------------------------------------------------
# Fast Solver（Beam Search，近似）
# ---------------------------------------------------------------------------


@dataclass
class BeamSolver:
    """第 24 节的 Beam Search：每层只保留评分最高的 K 个候选。

    结果是 ``best found``，``is_proven_optimal`` 恒为 ``False``。
    其解可作为 ExactSolver 的初始下界（第 25 节）。

    候选排序用的是**对最终总分的乐观估计**（不是可采纳上界，因此只能排序、
    不能剪枝）::

        est(S) = accumulated
               + w_available * available(S)
               + w_merge     * merge(S)
               + w_bonus     * ceiling[forced(S)]
               - w_singleton * singletons(S)

    * ``available`` = ``Σ_{s_i ≥ 2} g(s_i)``：现在就抓得到的分数
    * ``merge``     = 聚集势能，尚未因碎片化释放的部分
    * ``ceiling[forced]`` = 死色决定的终局奖励上限（清盘可得 ``B(0)``）

    权重默认值由 10x10 四色基准扫参得到（``python -m bench.bench10 --sweep-eval``）。
    最初的手调权重是 ``(0, -1, 0, 2)``，即在「惩罚碎片化」上做优化；
    基准实测它明显劣于「奖励已成团」的 ``(1, 0, 1, 2)``：
    同样 beam256、同样耗时，10x10 均分 1745 → 2270（+30%），
    group 分 812 → 1270（+56%）。原因是对碎片化取负会把搜索推向
    「尽快消掉小块」，反而牺牲了攒大团的机会。
    """

    scoring: ScoringConfig = DEFAULT_SCORING
    beam_width: int = 256
    weight_available: float = 1.0
    weight_merge: float = 0.0
    weight_bonus: float = 1.0
    weight_singleton: float = 2.0
    # 默认 0：不把 ΔE_release 加进排序。实验要开时再设正权重。
    weight_release: float = 0.0
    dedupe: bool = True
    time_limit: Optional[float] = None

    def __post_init__(self) -> None:
        self._ceiling = terminal_bonus_ceiling(self.scoring, 1024)

    def _run(self, state: BoardState) -> "_BeamRun":
        """跑一次 beam，返回内部结果（含每个终局剩余数上的最佳 group 分）。

        与 :meth:`solve` 的关键区别：**每个被生成的终局都会被记录**，
        不论它有没有在 beam 截断里活下来。原先只有存活到下一层的终局
        才计入最佳解，等于白白丢掉大量可行终局。
        """
        start = time.perf_counter()
        ceiling = self._ceiling
        scoring = self.scoring
        g_table = make_score_table(scoring, state.height * state.width)
        w_a, w_m, w_b, w_s = (
            self.weight_available,
            self.weight_merge,
            self.weight_bonus,
            self.weight_singleton,
        )
        w_r = self.weight_release
        release_from_formed = None
        if w_r:
            from .topology import release_from_formed as release_from_formed

        # 路径用父指针链表保存（parent_id, cell），避免每层为上万个候选复制路径列表
        parents: List[Tuple[int, Any]] = [(-1, None)]
        # 终局账本：R -> (该 R 下的最高 group 分, 对应节点 id)
        group_by_remaining: Dict[int, float] = {}
        node_by_remaining: Dict[int, int] = {}
        frontier: List[Tuple[BoardState, float, int]] = [(state, 0.0, 0)]
        expanded = 0
        depth = 0
        timed_out = False

        def record(remaining: int, group: float, node_id: int) -> None:
            current = group_by_remaining.get(remaining)
            if current is None or group > current:
                group_by_remaining[remaining] = group
                node_by_remaining[remaining] = node_id

        # 根本身就是终局的退化情形
        if is_terminal(state):
            record(count_remaining(state), 0.0, 0)

        while frontier:
            depth += 1
            candidates: Dict[bytes, Tuple[BoardState, float, int, float]] = {}
            for current, accumulated, node_id in frontier:
                if is_terminal(current):
                    record(count_remaining(current), accumulated, node_id)
                    continue
                parent_formed = (
                    board_features(current, scoring, g_table=g_table).available
                    if w_r else 0.0
                )
                for move in get_legal_moves(current):
                    expanded += 1
                    # move 直接来自 get_legal_moves，跳过重复的合法性校验
                    child = apply_move(current, move, validate=False)
                    acc = accumulated + g_table[move.size]
                    features = board_features(child, scoring, g_table=g_table)
                    rank = (
                        acc
                        + w_a * features.available
                        + w_m * features.merge
                        + w_b * ceiling[min(features.forced, len(ceiling) - 1)]
                        - w_s * features.singletons
                    )
                    if w_r:
                        rank += w_r * release_from_formed(
                            parent_formed, features.available, g_table[move.size]
                        )
                    if features.terminal:
                        # 终局在被生成的当下就记账，不等 beam 截断
                        record(features.remaining, acc, len(parents))
                    key = child.packed_key
                    if self.dedupe and key in candidates:
                        # 同一状态的更高累计分是严格支配的（后续完全相同）
                        if candidates[key][1] >= acc:
                            continue
                    parents.append((node_id, move.representative_cell))
                    candidates[key] = (child, acc, len(parents) - 1, rank)
            if not candidates:
                break
            if len(candidates) > self.beam_width:
                frontier = [
                    (item[0], item[1], item[2])
                    for item in sorted(candidates.values(), key=lambda item: -item[3])[
                        : self.beam_width
                    ]
                ]
            else:
                frontier = [(item[0], item[1], item[2]) for item in candidates.values()]
            if (
                self.time_limit is not None
                and time.perf_counter() - start > self.time_limit
            ):
                timed_out = True
                break

        # 总分最优的一条（含终局奖励）—— 现在基于完整终局账本挑，不再漏解
        best_score = float("-inf")
        best_id = -1
        for remaining, group in group_by_remaining.items():
            total = group + scoring.score_terminal(remaining)
            if total > best_score:
                best_score = total
                best_id = node_by_remaining[remaining]

        return _BeamRun(
            parents=parents,
            group_by_remaining=group_by_remaining,
            node_by_remaining=node_by_remaining,
            best_score=best_score,
            best_id=best_id,
            expanded=expanded,
            depth=depth,
            elapsed=time.perf_counter() - start,
            timed_out=timed_out,
        )

    def solve_front(self, state: BoardState) -> Dict[int, Tuple[float, List[Any]]]:
        """返回 ``{剩余数 R: (该 R 下最高 group 分, 代表格序列)}``。"""
        run = self._run(state)
        return {
            remaining: (group, _path_from_parents(run.parents, run.node_by_remaining[remaining]))
            for remaining, group in run.group_by_remaining.items()
        }

    def solve(self, state: BoardState) -> Solution:
        """返回总分最高的一条终局计划（``best found``，``is_proven_optimal`` 恒假）。"""
        run = self._run(state)
        best_path = _path_from_parents(run.parents, run.best_id)
        game = Game(state, scoring=self.scoring)
        for cell in best_path:
            game.play_cell(cell)
        result = game.result(require_finished=False)

        return Solution(
            initial_board=state,
            best_score=result.total_score,
            group_score=result.group_score,
            terminal_bonus=result.terminal_bonus,
            remaining_count=result.remaining_count,
            move_count=result.move_count,
            move_sequence=tuple(_records_to_moves(state, game)),
            final_board=game.state,
            search_mode="fast",
            is_proven_optimal=False,
            states_expanded=run.expanded,
            cache_hits=0,
            cache_size=0,
            max_depth=run.depth,
            elapsed_time=run.elapsed,
            steps=tuple(move_record_to_dict(r) for r in game.records),
            notes={
                "beam_width": self.beam_width,
                "weight_available": self.weight_available,
                "weight_merge": self.weight_merge,
                "weight_bonus": self.weight_bonus,
                "weight_singleton": self.weight_singleton,
                "terminal_states_seen": len(run.group_by_remaining),
                "timed_out": run.timed_out,
            },
        )


@dataclass
class _BeamRun:
    """:meth:`BeamSolver._run` 的内部结果。"""

    parents: List[Tuple[int, Any]]
    group_by_remaining: Dict[int, float]
    node_by_remaining: Dict[int, int]
    best_score: float
    best_id: int
    expanded: int
    depth: int
    elapsed: float
    timed_out: bool

def solve(
    state: BoardState,
    mode: str = "exact",
    scoring: Optional[ScoringConfig] = None,
    **kwargs: Any,
) -> Solution:
    """统一入口：``mode="exact"`` 走 ExactSolver，``mode="fast"`` 走 BeamSolver。"""
    scoring = scoring or DEFAULT_SCORING
    if mode == "exact":
        beam_width = kwargs.pop("beam_width", None)
        solver = ExactSolver(scoring=scoring, **kwargs)
        if beam_width:
            # 第 25 节：先用 Fast 拿到下界，再交给 Exact 证明
            fast = BeamSolver(scoring=scoring, beam_width=beam_width).solve(state)
            return solver.solve(
                state, lower_bound=fast.best_score, lower_bound_path=_path_of(fast)
            )
        return solver.solve(state)
    if mode == "fast":
        return BeamSolver(scoring=scoring, **kwargs).solve(state)
    raise ValueError(f"unknown mode: {mode}")


def _path_of(solution: Solution) -> List[Any]:
    return [move.representative_cell for move in solution.move_sequence]


def _path_from_parents(parents: List[Tuple[int, Any]], node_id: int) -> List[Any]:
    """沿父指针链表回溯出代表格序列。"""
    path: List[Any] = []
    cursor = node_id
    while cursor > 0:
        parent, cell = parents[cursor]
        path.append(cell)
        cursor = parent
    path.reverse()
    return path


__all__ = [
    "Solution",
    "ExactSolver",
    "BeamSolver",
    "ParetoFront",
    "PARETO_PROFILES",
    "pareto_search",
    "group_score_ceiling",
    "nonclear_group_ceiling",
    "solve",
    "lookahead_bound",
    "optimistic_bound",
    "validate_scoring_for_bound",
    "merge_potential",
    "singleton_count",
    "canonical_grid",
    "board_features",
    "make_score_table",
]
