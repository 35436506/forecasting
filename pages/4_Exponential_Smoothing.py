import streamlit as st
import numpy as np

from src.ui_helpers import select_series, show_series_preview
from src.naive_smoothing import fit_ses, fit_holt, fit_holt_winters
from src.metrics import calculate_metrics
from src.plotting import line_chart, forecast_chart_with_ci
from src.app_state import register_method_mape

st.title("📉 Exponential Smoothing: SES · Holt · Holt-Winters")
st.caption("Chương 3 — càng nhiều tham số KHÔNG đồng nghĩa càng tốt; phải khớp đúng cấu trúc dữ liệu.")

result = select_series("expsmooth")
if result is None:
    st.stop()

df, time_col, value_col = result
show_series_preview(df)
y = df[value_col].values
n = len(y)

method = st.selectbox("Chọn phương pháp", ["SES", "Holt", "Holt-Winters"])
future_steps = st.slider("Số kỳ dự báo tương lai", 0, 12, 6)

fitted = None
forecast = None
params = {}

try:
    if method == "SES":
        r = fit_ses(y, future_steps=future_steps)
        fitted, forecast, params = r.fitted, r.forecast, r.params

    elif method == "Holt":
        damped = st.checkbox("Dùng damped trend (khuyến nghị)", value=True)
        r = fit_holt(y, damped=damped, future_steps=future_steps)
        fitted, forecast, params = r.fitted, r.forecast, r.params

    else:  # Holt-Winters
        col1, col2, col3 = st.columns(3)
        with col1:
            season_length = st.number_input("Chu kỳ mùa vụ", min_value=2, max_value=24, value=12)
        with col2:
            trend_type = st.selectbox("Xu hướng", ["add", "mul"], format_func=lambda x: "Cộng" if x == "add" else "Nhân")
        with col3:
            seasonal_type = st.selectbox("Mùa vụ", ["add", "mul"], format_func=lambda x: "Cộng" if x == "add" else "Nhân")
        r = fit_holt_winters(y, season_length=season_length, trend=trend_type, seasonal=seasonal_type,
                              future_steps=future_steps)
        fitted, forecast, params = r.fitted, r.forecast, r.params

except ValueError as error:
    st.error(str(error))
    st.stop()

valid_mask = ~np.isnan(fitted)
m = calculate_metrics(y[valid_mask], fitted[valid_mask])
register_method_mape(method, m.mape, value_col)

col1, col2, col3 = st.columns(3)
col1.metric("MAPE", f"{m.mape:.2f}%")
col2.metric("MSE", f"{m.mse:,.0f}")
col3.metric("Tham số", ", ".join(f"{k}={v:.4f}" for k, v in params.items()) or "—")

if forecast is not None and future_steps > 0:
    future_index = np.arange(n, n + future_steps)
    fig = forecast_chart_with_ci(
        df[time_col], y, future_index, forecast, title=f"{method} — Dự báo {future_steps} kỳ tới",
        xaxis_title="Thời gian", yaxis_title=value_col,
    )
else:
    fig = line_chart(df[time_col], {value_col: y, method: fitted}, title=f"{method} — Khớp trong mẫu",
                      xaxis_title="Thời gian", yaxis_title=value_col)
st.plotly_chart(fig, width="stretch")

if any(v < 0.05 for v in params.values() if isinstance(v, float)):
    st.info(
        "💡 Tham số làm mịn gần 0 KHÔNG phải lỗi — nghĩa là mô hình đang 'tin vào mức nền' hơn là "
        "chạy theo quan sát mới, thường xảy ra khi SKU gần như thuần nhiễu ngẫu nhiên (xem Lecture "
        "Notes Chương 3)."
    )
