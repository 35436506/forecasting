"""Train/Test Split, Walk-Forward Validation, kiểm tra quá khớp.

Khớp Lecture Notes Chương 7 — CẢNH BÁO quan trọng: luôn chia dữ liệu chuỗi
thời gian THEO THỨ TỰ THỜI GIAN, không bao giờ dùng random split (xem mục
"Cạm bẫy thường gặp: Random Split cho chuỗi thời gian").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainTestSplitResult:
    train: np.ndarray
    test: np.ndarray
    n_train: int
    n_test: int


def chronological_split(y: np.ndarray, n_test: int) -> TrainTestSplitResult:
    """Chia THEO THỨ TỰ THỜI GIAN — Train là đoạn đầu, Test là đoạn cuối.
    KHÔNG BAO GIỜ dùng xáo trộn ngẫu nhiên (sklearn.train_test_split) cho
    chuỗi thời gian — xem cảnh báo look-ahead bias trong Lecture Notes.
    """
    n = len(y)
    if n_test < 1 or n_test >= n:
        raise ValueError(f"n_test phải nằm trong khoảng [1, {n-1}].")
    return TrainTestSplitResult(train=y[:-n_test], test=y[-n_test:], n_train=n - n_test, n_test=n_test)


@dataclass
class WalkForwardRound:
    origin: int
    train_size: int
    forecast: np.ndarray
    actual: np.ndarray
    mape: float


def walk_forward_validation(
    y: np.ndarray,
    fit_forecast_fn,
    origins: list[int],
    horizon: int = 2,
) -> list[WalkForwardRound]:
    """Rolling-origin: với mỗi origin, Train = y[:origin], dự báo horizon kỳ
    tiếp theo, so với thực tế. fit_forecast_fn(y_train, horizon) -> np.ndarray.

    Đáng tin cậy hơn một lần chia Train/Test duy nhất vì đánh giá qua NHIỀU
    "lát cắt" thời gian khác nhau thay vì phụ thuộc vào đúng một đoạn dữ liệu.
    """
    from src.metrics import calculate_metrics

    n = len(y)
    results = []
    for origin in origins:
        if origin + horizon > n:
            raise ValueError(f"origin={origin} + horizon={horizon} vượt quá độ dài chuỗi ({n}).")
        y_train = y[:origin]
        y_actual = y[origin:origin + horizon]
        forecast = fit_forecast_fn(y_train, horizon)
        m = calculate_metrics(y_actual, forecast)
        results.append(WalkForwardRound(
            origin=origin, train_size=len(y_train), forecast=np.asarray(forecast),
            actual=y_actual, mape=m.mape,
        ))
    return results


def adjusted_r_squared(r_squared: float, n_obs: int, n_params: int) -> float:
    """R²_adj = 1 - (1-R²)(n-1)/(n-k-1). ÂM là dấu hiệu quá khớp kinh điển."""
    denominator = n_obs - n_params - 1
    if denominator <= 0:
        raise ValueError(
            f"Số tham số ({n_params}) quá lớn so với số quan sát ({n_obs}) — không tính được Adjusted R²."
        )
    return 1 - (1 - r_squared) * (n_obs - 1) / denominator


def overfitting_check(r_squared: float, n_obs: int, n_params: int) -> dict[str, float | str | bool]:
    """Gói kiểm tra quá khớp: Adjusted R², tỷ lệ quan sát/tham số, cảnh báo."""
    adj_r2 = adjusted_r_squared(r_squared, n_obs, n_params)
    ratio = n_obs / n_params if n_params > 0 else float("inf")

    if adj_r2 < 0:
        verdict = "SỤP ĐỔ — Adjusted R² âm, mô hình không đáng tin dù MAPE có thể thấp."
    elif ratio < 10:
        verdict = f"CẢNH BÁO — chỉ {ratio:.1f} quan sát/tham số (khuyến nghị ≥10–15)."
    else:
        verdict = "ỔN — Adjusted R² dương và tỷ lệ quan sát/tham số hợp lý."

    return {
        "adjusted_r_squared": adj_r2, "observations_per_param": ratio,
        "n_obs": n_obs, "n_params": n_params, "verdict": verdict, "is_risky": adj_r2 < 0 or ratio < 10,
    }
