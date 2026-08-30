"""ACF/PACF cho nhận diện p, q và kiểm tra phần dư.

Khớp Lecture Notes Chương 1 (ACF phát hiện mùa vụ) và Chương 4 (đọc "chữ ký"
AR/MA để gợi ý p, q).
"""
from __future__ import annotations

import numpy as np


def compute_acf_pacf(y: np.ndarray, n_lags: int | None = None) -> dict[str, np.ndarray]:
    """Trả về ACF, PACF và ngưỡng tin cậy 95% xấp xỉ ±1.96/√n."""
    from statsmodels.tsa.stattools import acf, pacf

    y_arr = np.asarray(y, dtype=float)
    y_arr = y_arr[~np.isnan(y_arr)]
    n = len(y_arr)
    if n < 8:
        raise ValueError("Cần tối thiểu 8 quan sát để vẽ ACF/PACF có ý nghĩa.")

    max_lag = min(24, n // 2)
    if n_lags is None:
        n_lags = max_lag
    n_lags = min(n_lags, max_lag)

    acf_values = acf(y_arr, nlags=n_lags, fft=False)
    pacf_values = pacf(y_arr, nlags=n_lags, method="ywm")
    conf_bound = 1.96 / np.sqrt(n)

    return {
        "lags": np.arange(n_lags + 1), "acf": acf_values, "pacf": pacf_values,
        "conf_bound": conf_bound, "n_obs": n,
    }
