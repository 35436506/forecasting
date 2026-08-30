"""Regression: Trend tuyến tính/bậc hai, biến giả mùa vụ, biến Promo.

Khớp Lecture Notes Chương 5 — bao gồm mở rộng Quadratic Trend (Ŷt =
b0+b1t+b2t²) cho sản phẩm mới có tốc độ tăng trưởng không đều.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegressionFitResult:
    model_name: str
    fitted: np.ndarray
    r_squared: float
    adj_r_squared: float
    f_pvalue: float
    n_params: int
    n_obs: int
    coefficients: dict[str, float]
    forecast: np.ndarray | None = None


def _build_month_dummies(month_numbers: np.ndarray) -> pd.DataFrame:
    """Tạo 11 biến giả D2..D12 — tháng 1 làm baseline."""
    dummy_df = pd.DataFrame({"month": month_numbers.astype(int)})
    dummies = pd.get_dummies(dummy_df["month"], prefix="D", drop_first=False)
    if "D_1" in dummies.columns:
        dummies = dummies.drop(columns=["D_1"])
    return dummies.astype(float)


def fit_linear_trend(y: np.ndarray, future_steps: int = 0) -> RegressionFitResult:
    """Ŷt = b0 + b1*t — trend tuyến tính đơn giản."""
    import statsmodels.api as sm

    n = len(y)
    t = np.arange(1, n + 1)
    X = sm.add_constant(t)
    model = sm.OLS(y, X).fit()

    forecast = None
    if future_steps > 0:
        t_future = np.arange(n + 1, n + future_steps + 1)
        X_future = sm.add_constant(t_future, has_constant="add")
        forecast = np.asarray(model.predict(X_future))

    return RegressionFitResult(
        model_name="Linear Trend", fitted=np.asarray(model.fittedvalues),
        r_squared=float(model.rsquared), adj_r_squared=float(model.rsquared_adj),
        f_pvalue=float(model.f_pvalue), n_params=2, n_obs=n,
        coefficients={"b0": float(model.params[0]), "b1": float(model.params[1])},
        forecast=forecast,
    )


def fit_quadratic_trend(y: np.ndarray, future_steps: int = 0) -> RegressionFitResult:
    """Ŷt = b0 + b1*t + b2*t² — bắt được độ cong tăng tốc/chững lại.
    Hệ số b2 ÂM: tăng trưởng đang chững lại. b2 DƯƠNG: tăng trưởng đang tăng tốc.
    """
    import statsmodels.api as sm

    n = len(y)
    if n < 6:
        raise ValueError("Cần tối thiểu 6 quan sát để fit Quadratic Trend một cách đáng tin cậy.")

    t = np.arange(1, n + 1)
    X = sm.add_constant(np.column_stack([t, t**2]))
    model = sm.OLS(y, X).fit()

    forecast = None
    if future_steps > 0:
        t_future = np.arange(n + 1, n + future_steps + 1)
        X_future = sm.add_constant(np.column_stack([t_future, t_future**2]), has_constant="add")
        forecast = np.asarray(model.predict(X_future))

    return RegressionFitResult(
        model_name="Quadratic Trend", fitted=np.asarray(model.fittedvalues),
        r_squared=float(model.rsquared), adj_r_squared=float(model.rsquared_adj),
        f_pvalue=float(model.f_pvalue), n_params=3, n_obs=n,
        coefficients={"b0": float(model.params[0]), "b1": float(model.params[1]), "b2": float(model.params[2])},
        forecast=forecast,
    )


def fit_seasonal_regression(
    y: np.ndarray,
    month_numbers: np.ndarray,
    promo_flags: np.ndarray | None = None,
    use_quadratic: bool = False,
    future_month_numbers: np.ndarray | None = None,
    future_promo_flags: np.ndarray | None = None,
) -> RegressionFitResult:
    """Ŷt = b0 + b1*t (+ b2*t²) + Σ bi*Di (+ b_promo*Promo).

    Đây là mô hình ĐẦY ĐỦ của Chương 5 — trend + biến giả mùa vụ + Promo,
    tuỳ chọn thêm bậc hai (Quadratic) cho sản phẩm mới có độ cong tăng trưởng.
    """
    import statsmodels.api as sm

    n = len(y)
    if n < 15:
        raise ValueError("Cần tối thiểu 15 quan sát để fit hồi quy đầy đủ 11 biến giả tháng một cách đáng tin cậy.")

    t = np.arange(1, n + 1)
    trend_cols = [t, t**2] if use_quadratic else [t]
    dummies = _build_month_dummies(month_numbers)

    X_parts = [np.column_stack(trend_cols), dummies.values]
    col_names = (["t", "t2"] if use_quadratic else ["t"]) + list(dummies.columns)

    if promo_flags is not None:
        X_parts.append(np.asarray(promo_flags, dtype=float).reshape(-1, 1))
        col_names.append("promo")

    X = np.column_stack(X_parts)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    n_params = X.shape[1]
    if n <= n_params + 2:
        raise ValueError(
            f"Số tham số ({n_params}) quá lớn so với số quan sát ({n}) — rủi ro quá khớp cao. "
            "Xem Lecture Notes Chương 7 (Adjusted R² âm)."
        )

    forecast = None
    if future_month_numbers is not None:
        n_future = len(future_month_numbers)
        t_future = np.arange(n + 1, n + n_future + 1)
        trend_future = [t_future, t_future**2] if use_quadratic else [t_future]
        future_dummies = _build_month_dummies(future_month_numbers)
        for col in dummies.columns:
            if col not in future_dummies.columns:
                future_dummies[col] = 0.0
        future_dummies = future_dummies[dummies.columns]

        X_future_parts = [np.column_stack(trend_future), future_dummies.values]
        if promo_flags is not None:
            promo_future = future_promo_flags if future_promo_flags is not None else np.zeros(n_future)
            X_future_parts.append(np.asarray(promo_future, dtype=float).reshape(-1, 1))

        X_future = np.column_stack(X_future_parts)
        X_future = sm.add_constant(X_future, has_constant="add")
        forecast = np.asarray(model.predict(X_future))

    coefficients = dict(zip(["const"] + col_names, model.params))

    model_name = "Seasonal Regression"
    if use_quadratic:
        model_name += " + Quadratic"
    if promo_flags is not None:
        model_name += " + Promo"

    return RegressionFitResult(
        model_name=model_name, fitted=np.asarray(model.fittedvalues),
        r_squared=float(model.rsquared), adj_r_squared=float(model.rsquared_adj),
        f_pvalue=float(model.f_pvalue), n_params=n_params, n_obs=n,
        coefficients={k: float(v) for k, v in coefficients.items()},
        forecast=forecast,
    )
