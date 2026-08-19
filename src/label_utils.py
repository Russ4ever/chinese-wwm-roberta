"""标签构造共享的无副作用分类函数。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trimmed_mean(values, fraction: float = 0.1) -> float:
    """去掉两端固定比例后取均值；样本不足时退回普通均值。"""
    if not 0 <= fraction < 0.5:
        raise ValueError("fraction 必须满足 0 <= fraction < 0.5")
    array = np.sort(np.asarray(values, dtype=float))
    if array.size == 0:
        return np.nan
    trim = int(round(array.size * fraction))
    if trim == 0 or array.size - 2 * trim <= 0:
        return float(np.mean(array))
    return float(np.mean(array[trim:-trim]))


def classify_direction(value, flat_threshold: float) -> str | None:
    if flat_threshold < 0:
        raise ValueError("flat_threshold 不得为负")
    if pd.isna(value):
        return None
    if abs(value) <= flat_threshold:
        return "flat"
    return "up" if value > 0 else "down"


def classify_five(
    value,
    flat_threshold: float,
    strong_threshold: float,
) -> str | None:
    if flat_threshold < 0 or strong_threshold < flat_threshold:
        raise ValueError("阈值必须满足 0 <= flat_threshold <= strong_threshold")
    if pd.isna(value):
        return None
    magnitude = abs(value)
    if magnitude <= flat_threshold:
        return "flat"
    if magnitude < strong_threshold:
        return "up" if value > 0 else "down"
    return "strong_up" if value > 0 else "strong_down"


def profit_state(median, minimum, maximum, near_zero_threshold: float) -> str:
    if near_zero_threshold < 0:
        raise ValueError("near_zero_threshold 不得为负")
    if minimum < -near_zero_threshold and maximum > near_zero_threshold:
        return "mixed_sign"
    if abs(median) <= near_zero_threshold:
        return "near_zero"
    return "profit" if median > 0 else "loss"


def transition_class(a, b, state_now, state_future, five_class) -> str | None:
    """根据当前/未来盈利状态与正盈利修正类别生成迁移标签。"""
    if pd.isna(a) or pd.isna(b):
        return None
    if state_now == "profit" and state_future == "profit":
        mapping = {
            "flat": "profit_flat",
            "up": "profit_up",
            "down": "profit_down",
            "strong_up": "profit_strong_up",
            "strong_down": "profit_strong_down",
        }
        return mapping.get(five_class)
    if state_now == "loss" and state_future == "loss":
        return "loss_narrowing" if b > a else ("loss_widening" if b < a else "loss_unchanged")
    if state_now == "profit" and state_future == "loss":
        return "turn_to_loss"
    if state_now in ("loss", "near_zero", "mixed_sign") and state_future == "profit":
        return "turn_to_profit"
    if "mixed_sign" in (state_now, state_future):
        return "mixed_sign_transition"
    if "near_zero" in (state_now, state_future):
        return "near_zero_transition"
    return "other"
