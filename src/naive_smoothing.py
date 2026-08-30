"""NF1, NF2 và họ Exponential Smoothing: SES, Holt, Holt-Winters.

Khớp Lecture Notes Chương 2 (Naive) và Chương 3 (Exponential Smoothing).
Mỗi hàm trả về DataFrame có cột fitted (trong mẫu) và, khi có future_steps,
thêm bảng dự báo ra tương lai.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SmoothingResult:
    method_name: str
    fitted: np.ndarray
    params: dict[str, float]
    forecast: np.ndarray | None = None


def naive_nf1(y: np.ndarray) -> SmoothingResult:
    """NF1: F(t+1) = Y(t) — mốc chuẩn tối thiểu, không có tham số."""
    fitted = np.array([np.nan] + list(y[:-1]))
    return SmoothingResult(method_name="NF1 (Naive đơn giản)", fitted=fitted, params={})


def naive_nf2(y: np.ndarray, season_length: int = 12) -> SmoothingResult:
    """NF2: dự báo bằng giá trị cùng kỳ năm trước, điều chỉnh theo xu hướng gần đây.
    Công thức đơn giản hoá dùng trong khóa: F(t) = Y(t-s) + [Y(t-1) - Y(t-1-s)].
    """
    n = len(y)
    fitted = np.full(n, np.nan)
    for t in range(season_length, n):
        seasonal_base = y[t - season_length]
        recent_trend = y[t - 1] - y[t - 1 - season_length] if t - 1 - season_length >= 0 else 0.0
        fitted[t] = seasonal_base + recent_trend
    return SmoothingResult(method_name="NF2 (Naive mùa vụ)", fitted=fitted, params={"season_length": season_length})


def moving_average(y: np.ndarray, k: int = 3, future_steps: int = 0) -> SmoothingResult:
    """Moving Average bậc k — F(t) = trung bình cộng ĐƠN GIẢN của k quan sát gần nhất
    (t-1, t-2, ..., t-k). Không có tham số làm mịn — k phải xác định bằng thử nghiệm
    thủ công (khác Exponential Smoothing, nơi alpha được Solver/statsmodels tự tối ưu).
    """
    if k < 1:
        raise ValueError("k (bậc Moving Average) phải >= 1.")
    n = len(y)
    if n <= k:
        raise ValueError(f"Cần tối thiểu {k + 1} quan sát cho Moving Average bậc {k}.")

    fitted = np.full(n, np.nan)
    for t in range(k, n):
        fitted[t] = np.mean(y[t - k:t])

    forecast = None
    if future_steps > 0:
        # Du bao ngoai mau: dung k quan sat GAN NHAT (bao gom du bao truoc do neu can)
        extended = list(y[-k:])
        forecast = []
        for _ in range(future_steps):
            next_val = float(np.mean(extended[-k:]))
            forecast.append(next_val)
            extended.append(next_val)
        forecast = np.array(forecast)

    return SmoothingResult(method_name=f"MA (k={k})", fitted=fitted, params={"k": k}, forecast=forecast)


def weighted_moving_average(
    y: np.ndarray, weights: list[float] | None = None, future_steps: int = 0,
) -> SmoothingResult:
    """Weighted Moving Average — F(t) = tổng có trọng số của k quan sát gần nhất,
    trọng số CHUẨN HOÁ về tổng = 1. Mặc định trọng số 1:2:3 (k=3) — quan sát GẦN
    NHẤT nhận trọng số LỚN NHẤT, khớp benchmark đã kiểm chứng của khóa học.
    """
    if weights is None:
        weights = [1, 2, 3]
    weights = np.asarray(weights, dtype=float)
    if len(weights) < 1 or np.any(weights < 0) or weights.sum() == 0:
        raise ValueError("Trọng số phải có ít nhất 1 phần tử, không âm, tổng khác 0.")

    k = len(weights)
    w_norm = weights / weights.sum()  # chuan hoa ve tong = 1
    n = len(y)
    if n <= k:
        raise ValueError(f"Cần tối thiểu {k + 1} quan sát cho WMA bậc {k}.")

    fitted = np.full(n, np.nan)
    for t in range(k, n):
        # w_norm[-1] la trong so LON NHAT, ung voi quan sat GAN t NHAT (t-1)
        window = y[t - k:t]
        fitted[t] = float(np.dot(window, w_norm))

    forecast = None
    if future_steps > 0:
        extended = list(y[-k:])
        forecast = []
        for _ in range(future_steps):
            next_val = float(np.dot(extended[-k:], w_norm))
            forecast.append(next_val)
            extended.append(next_val)
        forecast = np.array(forecast)

    weights_label = ":".join(str(int(w)) if w == int(w) else str(w) for w in weights)
    return SmoothingResult(
        method_name=f"WMA (k={k}, trọng số {weights_label})", fitted=fitted,
        params={"k": k, "weights": weights.tolist()}, forecast=forecast,
    )


def double_moving_average(y: np.ndarray, k: int = 3, future_steps: int = 0) -> SmoothingResult:
    """Double Moving Average — mở rộng MA cho dữ liệu CÓ XU HƯỚNG bằng cách tính
    thêm một lớp trung bình trượt thứ hai (MA của chính MA), dùng độ chênh lệch
    giữa hai lớp để ước lượng độ dốc xu hướng. Là tiền thân trực tiếp của Holt —
    cùng nguyên lý tách mức nền + xu hướng, chỉ khác trọng số (đều nhau ở đây,
    giảm dần cấp số nhân ở Holt).

        a_t = 2*MA'_t - MA''_t          (mức nền)
        b_t = (2/(k-1)) * (MA'_t - MA''_t)   (độ dốc xu hướng)
        F(t+1) = a_t + b_t
    """
    if k < 2:
        raise ValueError("k (bậc Double Moving Average) phải >= 2.")
    n = len(y)
    if n < 2 * k:
        raise ValueError(f"Cần tối thiểu {2 * k} quan sát cho Double Moving Average bậc {k}.")

    ma1 = np.full(n, np.nan)
    for t in range(k - 1, n):
        ma1[t] = np.mean(y[t - k + 1:t + 1])

    ma2 = np.full(n, np.nan)
    for t in range(k - 1, n):
        window = ma1[t - k + 1:t + 1]
        if not np.any(np.isnan(window)):
            ma2[t] = np.mean(window)

    a = 2 * ma1 - ma2
    b = (2 / (k - 1)) * (ma1 - ma2)

    fitted = np.full(n, np.nan)
    for t in range(n - 1):
        if not np.isnan(a[t]):
            fitted[t + 1] = a[t] + b[t]

    forecast = None
    if future_steps > 0 and not np.isnan(a[-1]):
        # Ngoai suy tuyen tinh bang muc nen + doc do cuoi cung da uoc luong
        forecast = np.array([a[-1] + b[-1] * (h + 1) for h in range(future_steps)])

    return SmoothingResult(
        method_name=f"Double-MA (k={k})", fitted=fitted, params={"k": k}, forecast=forecast,
    )


def fit_ses(y: np.ndarray, future_steps: int = 0) -> SmoothingResult:
    """Simple Exponential Smoothing — dùng statsmodels, alpha tối ưu tự động."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    if len(y) < 4:
        raise ValueError("Cần tối thiểu 4 quan sát để fit SES.")

    model = SimpleExpSmoothing(y, initialization_method="estimated").fit()
    forecast = model.forecast(future_steps) if future_steps > 0 else None
    return SmoothingResult(
        method_name="SES",
        fitted=np.asarray(model.fittedvalues),
        params={"alpha": float(model.params["smoothing_level"])},
        forecast=np.asarray(forecast) if forecast is not None else None,
    )


def fit_holt(y: np.ndarray, damped: bool = True, future_steps: int = 0) -> SmoothingResult:
    """Holt (Double Exponential Smoothing) — có xu hướng, không mùa vụ."""
    from statsmodels.tsa.holtwinters import Holt

    if len(y) < 4:
        raise ValueError("Cần tối thiểu 4 quan sát để fit Holt.")

    model = Holt(y, damped_trend=damped, initialization_method="estimated").fit()
    forecast = model.forecast(future_steps) if future_steps > 0 else None
    return SmoothingResult(
        method_name="Holt" + (" (damped)" if damped else ""),
        fitted=np.asarray(model.fittedvalues),
        params={
            "alpha": float(model.params["smoothing_level"]),
            "beta": float(model.params["smoothing_trend"]),
        },
        forecast=np.asarray(forecast) if forecast is not None else None,
    )


def fit_holt_winters(
    y: np.ndarray,
    season_length: int = 12,
    trend: str = "add",
    seasonal: str = "mul",
    future_steps: int = 0,
) -> SmoothingResult:
    """Holt-Winters — xu hướng + mùa vụ. trend/seasonal nhận 'add' hoặc 'mul'."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    min_required = season_length * 2
    if len(y) < min_required:
        raise ValueError(
            f"Holt-Winters cần tối thiểu {min_required} quan sát (2 chu kỳ mùa vụ {season_length} kỳ), "
            f"chuỗi hiện chỉ có {len(y)}."
        )

    model = ExponentialSmoothing(
        y, seasonal_periods=season_length, trend=trend, seasonal=seasonal,
        initialization_method="estimated",
    ).fit()
    forecast = model.forecast(future_steps) if future_steps > 0 else None
    return SmoothingResult(
        method_name=f"Holt-Winters ({trend}+{seasonal})",
        fitted=np.asarray(model.fittedvalues),
        params={
            "alpha": float(model.params["smoothing_level"]),
            "beta": float(model.params["smoothing_trend"]),
            "gamma": float(model.params["smoothing_seasonal"]),
        },
        forecast=np.asarray(forecast) if forecast is not None else None,
    )
