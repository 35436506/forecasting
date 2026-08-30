"""Sai số dự báo: ME, MAE, MSE, RMSE, MPE, MAPE, WMAPE.

Toàn bộ công thức khớp chính xác với Lecture Notes Chương 2 của khóa học
Forecasting Cho Chuỗi Cung Ứng — dùng chung một module để mọi trang trong
app tính sai số theo ĐÚNG MỘT chuẩn, tránh sai lệch giữa các trang.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AccuracyMetrics:
    """Gói đầy đủ các chỉ số sai số cho một cặp (thực tế, dự báo)."""

    me: float
    mae: float
    mse: float
    rmse: float
    mpe: float
    mape: float
    wmape: float
    n_obs: int

    def as_dict(self) -> dict[str, float]:
        return {
            "ME (Forecast Bias)": round(self.me, 2),
            "MAE": round(self.mae, 2),
            "MSE": round(self.mse, 2),
            "RMSE": round(self.rmse, 2),
            "MPE (%)": round(self.mpe, 2),
            "MAPE (%)": round(self.mape, 2),
            "WMAPE (%)": round(self.wmape, 2),
            "Số kỳ đối chiếu": self.n_obs,
        }


def calculate_metrics(actual: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> AccuracyMetrics:
    """Tính đầy đủ 7 chỉ số sai số — bỏ qua các kỳ actual = 0 khi tính MPE/MAPE
    (tránh chia cho 0, xem Lecture Notes Chương 2 mục "Bẫy MAPE khi có kỳ=0").
    WMAPE vẫn tính được cả khi actual có giá trị 0, vì gộp tổng trước khi chia.
    """
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)

    if len(actual_arr) != len(forecast_arr):
        raise ValueError("Chuỗi thực tế và chuỗi dự báo phải có cùng độ dài.")

    valid_mask = ~np.isnan(actual_arr) & ~np.isnan(forecast_arr)
    actual_arr = actual_arr[valid_mask]
    forecast_arr = forecast_arr[valid_mask]

    if len(actual_arr) == 0:
        raise ValueError("Không có cặp (thực tế, dự báo) hợp lệ nào để tính sai số.")

    error = actual_arr - forecast_arr
    abs_error = np.abs(error)

    me = float(np.mean(error))
    mae = float(np.mean(abs_error))
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))

    nonzero_mask = actual_arr != 0
    if nonzero_mask.sum() == 0:
        mpe = float("nan")
        mape = float("nan")
    else:
        pct_error = error[nonzero_mask] / actual_arr[nonzero_mask] * 100
        mpe = float(np.mean(pct_error))
        mape = float(np.mean(np.abs(pct_error)))

    sum_actual = np.sum(np.abs(actual_arr))
    wmape = float(np.sum(abs_error) / sum_actual * 100) if sum_actual != 0 else float("nan")

    return AccuracyMetrics(
        me=me, mae=mae, mse=mse, rmse=rmse, mpe=mpe, mape=mape, wmape=wmape,
        n_obs=len(actual_arr),
    )


def tracking_signal_series(actual: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> pd.Series:
    """Tracking Signal luỹ kế qua từng kỳ — TS = (tổng lỗi luỹ kế) / MAD luỹ kế.
    Ngưỡng cảnh báo thông dụng: |TS| > 4 (xem Lecture Notes Chương 6).
    """
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    error = actual_arr - forecast_arr

    cum_error = np.cumsum(error)
    cum_abs_error = np.cumsum(np.abs(error))
    periods = np.arange(1, len(error) + 1)
    mad = cum_abs_error / periods

    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.where(mad != 0, cum_error / mad, 0.0)

    return pd.Series(ts, name="tracking_signal")
