"""计分系统（与游戏逻辑完全解耦）。

总分::

    F = sum_t g(k_t) + B(R)

* ``g(k)``：一次消除 k 个方块的即时得分。通行规则是 ``5 k²``。
* ``B(R)``：终局还剩 R 块时的奖励。通行规则是剩余少于 10 块时 ``2000 − 5 R``，否则 0。

  .. code-block:: text

      B(R) = 0                 , R >= 10
      B(R) = 2000 - 5 R        , 0 <= R < 10

``5 k²`` 仍是凸的，搜索用的上界条件不用改。求解器只调用
:meth:`ScoringConfig.score_group` 和 :meth:`ScoringConfig.score_terminal`。

想换规则时，改本文件顶部的常数，或把 ``GROUP_GAIN_FN`` / ``CLEAR_REWARD_FN``
设成自己的函数。本地页面会在启动时读这些默认值。出处见 ``SOURCES.md``。

下面的 ``bonus_coefficient`` / ``bonus_exponent`` 只服务「系数 × (A−R)²」
这种可按 λ 切开的实验公式。不传入 ``terminal_score`` 时才会用到。
通行规则走 ``CLEAR_REWARD_FN``，不走这条实验公式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

ScoreFn = Callable[[int], float]

# ---------------------------------------------------------------------------
# 服务端计分（直接改这里）
# ---------------------------------------------------------------------------
#
# 多连通增益 g(k)：消除 k 个同色方块。
# 通行规则是 5 k²。若设置 GROUP_GAIN_FN，则完全替换该公式。
GROUP_COEFFICIENT = 5.0
GROUP_EXPONENT = 2
GROUP_GAIN_FN: Optional[ScoreFn] = None

# 清盘奖励 B(R)：终局还剩 R 块。
# 通行规则：R < 10 时 2000 − 5R，否则 0。清盘是 2000。
CLEAR_THRESHOLD = 10
CLEAR_BONUS_AT_ZERO = 2000.0
CLEAR_BONUS_PER_TILE = 5.0

# 实验用的「系数 × (A−R)²」，不是通行规则。
CLEAR_COEFFICIENT = 10.0
CLEAR_EXPONENT = 2


def commercial_clear_reward(remaining: int) -> float:
    """通行的清盘奖励：剩余少于 10 块时 ``2000 − 5R``，否则 0。"""
    if remaining >= CLEAR_THRESHOLD:
        return 0.0
    return float(CLEAR_BONUS_AT_ZERO - CLEAR_BONUS_PER_TILE * remaining)


# 设成 None 则退回下面的实验公式。默认就是通行规则。
CLEAR_REWARD_FN: Optional[ScoreFn] = commercial_clear_reward


def group_gain(size: int) -> float:
    """当前服务端的多连通增益 ``g(k)``。"""
    if GROUP_GAIN_FN is not None:
        return float(GROUP_GAIN_FN(size))
    return float(GROUP_COEFFICIENT * (size ** GROUP_EXPONENT))


def clear_reward(remaining: int, threshold: int, coefficient: float, exponent: float) -> float:
    """阈值公式下的清盘奖励。``CLEAR_REWARD_FN`` 由调用方优先使用。"""
    if remaining >= threshold:
        return 0.0
    gap = threshold - remaining
    return float(coefficient * (gap ** exponent))


def default_group_score(size: int) -> float:
    """默认 ``g(k)``，走上面的服务端配置。"""
    return group_gain(size)


@dataclass(frozen=True)
class ScoringConfig:
    """计分配置。

    Parameters
    ----------
    group_score:
        ``g(k)``，输入本次消除的方块数，返回即时得分。
    clear_threshold:
        终局奖励阈值 ``A``。``R >= A`` 时奖励为 0。
    bonus_coefficient:
        默认终局奖励公式的系数（仅在使用默认 ``terminal_score`` 时生效）。
    bonus_exponent:
        默认终局奖励里 ``(A-R)`` 的指数。保持为 2 时，奖励对系数线性可分离。
    terminal_score:
        可选的 ``B(R)`` 自定义函数；提供时覆盖默认公式。
    """

    group_score: ScoreFn = default_group_score
    clear_threshold: int = CLEAR_THRESHOLD
    bonus_coefficient: float = CLEAR_COEFFICIENT
    bonus_exponent: float = CLEAR_EXPONENT
    terminal_score: Optional[ScoreFn] = None

    def score_group(self, size: int) -> float:
        """``g(k)``"""
        if size < 2:
            raise ValueError(f"group size must be >= 2, got {size}")
        return float(self.group_score(size))

    def score_terminal(self, remaining: int) -> float:
        """``B(R)``"""
        if remaining < 0:
            raise ValueError(f"remaining must be >= 0, got {remaining}")
        if self.terminal_score is not None:
            return float(self.terminal_score(remaining))
        return self._default_terminal_bonus(remaining)

    def terminal_shape(self, remaining: int) -> Optional[float]:
        """``B₀(R)``：``coefficient = 1`` 时的终局奖励形状。

        默认公式下 ``B(R) = bonus_coefficient · B₀(R)``，于是
        ``F(G, R) = G + λ·B₀(R)`` 对 ``λ`` 是一条直线 —— 这正是
        Pareto 前沿上做「策略切换点」分析所需要的可分离形式。

        自定义 ``terminal_score`` 无法保证这种线性可分离性，此时返回 ``None``。
        """
        if self.terminal_score is not None:
            return None
        if remaining >= self.clear_threshold:
            return 0.0
        gap = self.clear_threshold - remaining
        return float(gap ** self.bonus_exponent)

    def _default_terminal_bonus(self, remaining: int) -> float:
        return clear_reward(
            remaining, self.clear_threshold, self.bonus_coefficient, self.bonus_exponent
        )

    def __str__(self) -> str:  # pragma: no cover - 调试便利
        mode = "custom" if self.terminal_score is not None else "default"
        return (
            f"ScoringConfig(A={self.clear_threshold}, "
            f"bonus_coefficient={self.bonus_coefficient}, terminal={mode})"
        )


DEFAULT_SCORING = ScoringConfig(terminal_score=commercial_clear_reward)


def score_group(size: int, config: Optional[ScoringConfig] = None) -> float:
    """模块级 ``g(k)``，默认使用 :data:`DEFAULT_SCORING`。"""
    return (config or DEFAULT_SCORING).score_group(size)


def score_terminal(remaining: int, config: Optional[ScoringConfig] = None) -> float:
    """模块级 ``B(R)``，默认使用 :data:`DEFAULT_SCORING`。"""
    return (config or DEFAULT_SCORING).score_terminal(remaining)


__all__ = [
    "ScoringConfig",
    "ScoreFn",
    "DEFAULT_SCORING",
    "default_group_score",
    "score_group",
    "score_terminal",
]
