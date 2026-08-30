"""Phát hiện outlier (IQR, Z-score) và xử lý dữ liệu thiếu bằng nội suy.

Khớp Lecture Notes Chương 8 — luôn TRỰC QUAN HÓA trước, ĐỊNH LƯỢNG bằng
IQR/Z-score, ĐIỀU TRA nguyên nhân trước khi sửa (lỗi dữ liệu hay sự kiện
thật), rồi mới chọn cách xử lý phù hợp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OutlierReport:
    method: str
    lower_bound: float
    upper_bound: float
    outlier_indices: list[int]
    outlier_values: list[float]


def detect_outliers_iqr(y: np.ndarray, multiplier: float = 1.5) -> OutlierReport:
    """Ngưỡng = [Q1 - k*IQR, Q3 + k*IQR], mặc định k=1.5 (quy ước Tukey)."""
    y_arr = np.asarray(y, dtype=float)
    valid = y_arr[~np.isnan(y_arr)]
    if len(valid) < 4:
        raise ValueError("Cần tối thiểu 4 quan sát hợp lệ để tính IQR.")

    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr

    mask = (y_arr < lower) | (y_arr > upper)
    indices = np.where(mask)[0].tolist()
    return OutlierReport(
        method="IQR", lower_bound=float(lower), upper_bound=float(upper),
        outlier_indices=indices, outlier_values=[float(y_arr[i]) for i in indices],
    )


def detect_outliers_zscore(y: np.ndarray, threshold: float = 3.0) -> OutlierReport:
    """|z| > threshold (mặc định 3) được xem là outlier."""
    y_arr = np.asarray(y, dtype=float)
    valid = y_arr[~np.isnan(y_arr)]
    if len(valid) < 4:
        raise ValueError("Cần tối thiểu 4 quan sát hợp lệ để tính Z-score.")

    mean_y, std_y = valid.mean(), valid.std(ddof=0)
    if std_y == 0:
        return OutlierReport(method="Z-score", lower_bound=mean_y, upper_bound=mean_y,
                              outlier_indices=[], outlier_values=[])

    z_scores = (y_arr - mean_y) / std_y
    mask = np.abs(z_scores) > threshold
    indices = np.where(mask)[0].tolist()
    return OutlierReport(
        method="Z-score", lower_bound=float(mean_y - threshold * std_y),
        upper_bound=float(mean_y + threshold * std_y),
        outlier_indices=indices, outlier_values=[float(y_arr[i]) for i in indices],
    )


def interpolate_missing(y: np.ndarray, method: str = "linear") -> np.ndarray:
    """Nội suy các giá trị NaN. method: 'linear' hoặc 'seasonal' (cùng kỳ
    trung bình các năm lân cận, cần dữ liệu dạng chuỗi tháng)."""
    if method not in ("linear", "seasonal"):
        raise ValueError("method phải là 'linear' hoặc 'seasonal'.")

    series = pd.Series(y, dtype=float)
    if method == "linear":
        return series.interpolate(method="linear", limit_direction="both").to_numpy()

    # seasonal: dien gia tri thieu bang trung binh cung ky (cach nhau 12) lan can
    result = series.copy()
    n = len(series)
    for i in series[series.isna()].index:
        same_season = [series[j] for j in range(i % 12, n, 12) if j != i and not pd.isna(series[j])]
        result[i] = np.mean(same_season) if same_season else series.mean()
    return result.to_numpy()


def apply_manual_outlier_fix(y: np.ndarray, indices_to_fix: list[int], method: str = "linear") -> np.ndarray:
    """Đặt các vị trí chỉ định thành NaN rồi nội suy lại — dùng sau khi người
    dùng xác nhận một điểm là LỖI DỮ LIỆU (không phải sự kiện thật)."""
    y_copy = np.asarray(y, dtype=float).copy()
    y_copy[indices_to_fix] = np.nan
    return interpolate_missing(y_copy, method=method)
