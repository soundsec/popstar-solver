"""对已记录对局的事后重算与分布统计（面向 Phase 2 的计分标定）。

由于 :class:`~popstar.records.RunRecord` 保存了 ``group_sizes`` 与
``remaining_count``，更换计分规则**不需要重跑求解**，直接重算即可：

.. code-block:: python

    from popstar.records import RunLog
    from popstar.analysis import rescore, summarize, compare_configs
    from popstar.scoring import ScoringConfig

    runs = RunLog("runs/phase2.jsonl").load()
    print(summarize(runs))
    print(compare_configs(runs, {"A10x10": ScoringConfig(bonus_coefficient=10),
                                 "A10x50": ScoringConfig(bonus_coefficient=50)}))
"""

from __future__ import annotations

import statistics
from typing import Dict, Iterable, List, Optional, Sequence

from .records import RunRecord
from .scoring import ScoringConfig

ScoreBreakdown = Dict[str, float]


def rescore(record: RunRecord, config: ScoringConfig) -> ScoreBreakdown:
    """用给定计分配置重算一条记录，返回拆分后的三部分。

    未完成对局的 terminal bonus 记为 0（与 :class:`~popstar.game.GameResult` 一致）。
    """
    group = sum(config.score_group(k) for k in record.group_sizes)
    bonus = config.score_terminal(record.remaining_count) if record.finished else 0.0
    return {
        "group_score": float(group),
        "terminal_bonus": float(bonus),
        "total_score": float(group + bonus),
        "remaining_count": float(record.remaining_count),
    }


def rescore_many(
    records: Iterable[RunRecord], config: ScoringConfig
) -> List[ScoreBreakdown]:
    return [rescore(record, config) for record in records]


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(records: Sequence[RunRecord]) -> Dict[str, object]:
    """汇总一批记录的两部分得分分布。

    返回字段
    --------
    ``group_score`` / ``terminal_bonus`` / ``total_score`` / ``remaining_count``：
        count / mean / median / min / max
    ``bonus_share``：
        terminal_bonus / total_score 的均值与最大值，用于判断终局奖励是否
        在数值上支配消除得分（标定 ``bonus_coefficient`` 的核心指标）。
    ``remaining_histogram``：
        终局剩余数的分布直方图。
    """
    group = [r.group_score for r in records]
    bonus = [r.terminal_bonus for r in records]
    total = [r.total_score for r in records]
    remaining = [float(r.remaining_count) for r in records]

    shares = [
        b / t for b, t in zip(bonus, total) if t > 0
    ]

    histogram: Dict[int, int] = {}
    for value in remaining:
        key = int(value)
        histogram[key] = histogram.get(key, 0) + 1

    return {
        "runs": len(records),
        "finished": sum(1 for r in records if r.finished),
        "group_score": _stats(group),
        "terminal_bonus": _stats(bonus),
        "total_score": _stats(total),
        "remaining_count": _stats(remaining),
        "bonus_share": {
            "mean": statistics.fmean(shares) if shares else None,
            "max": max(shares) if shares else None,
        },
        "remaining_histogram": dict(sorted(histogram.items())),
    }


def compare_configs(
    records: Sequence[RunRecord], configs: Dict[str, ScoringConfig]
) -> Dict[str, Dict[str, object]]:
    """在**同一批对局**上比较多个计分配置，便于挑选量级。

    返回 ``{配置名: {"group_score":..., "terminal_bonus":..., "total_score":...,
    "bonus_share": {...}}}``。
    """
    report: Dict[str, Dict[str, object]] = {}
    for name, config in configs.items():
        breakdowns = rescore_many(records, config)
        group = [b["group_score"] for b in breakdowns]
        bonus = [b["terminal_bonus"] for b in breakdowns]
        total = [b["total_score"] for b in breakdowns]
        shares = [b / t for b, t in zip(bonus, total) if t > 0]
        report[name] = {
            "group_score": _stats(group),
            "terminal_bonus": _stats(bonus),
            "total_score": _stats(total),
            "bonus_share": {
                "mean": statistics.fmean(shares) if shares else None,
                "max": max(shares) if shares else None,
            },
        }
    return report


def format_summary(summary: Dict[str, object]) -> str:
    """把 :func:`summarize` 的结果渲染为可读文本。"""
    lines = [
        f"对局数 {summary['runs']}（已完成 {summary['finished']}）",
    ]
    for key in ("group_score", "terminal_bonus", "total_score", "remaining_count"):
        stats = summary[key]  # type: ignore[index]
        if not stats or stats.get("count") == 0:
            lines.append(f"{key:<16}: 无数据")
            continue
        lines.append(
            f"{key:<16}: mean={stats['mean']:.2f} median={stats['median']:.2f} "
            f"min={stats['min']:.2f} max={stats['max']:.2f}"
        )
    share = summary["bonus_share"]  # type: ignore[index]
    if share["mean"] is not None:
        lines.append(
            f"奖励占比        : mean={share['mean']:.3f} max={share['max']:.3f}"
        )
    histogram = summary["remaining_histogram"]  # type: ignore[index]
    if histogram:
        rendered = " ".join(f"{k}:{v}" for k, v in sorted(histogram.items()))
        lines.append(f"剩余分布        : {rendered}")
    return "\n".join(lines)


__all__ = [
    "rescore",
    "rescore_many",
    "summarize",
    "compare_configs",
    "format_summary",
]
