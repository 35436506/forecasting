import numpy as np
import pandas as pd
import streamlit as st

from src.ui_helpers import select_series, show_series_preview
from src.data_utils import month_numbers_from_dates
from src.regression_modeling import fit_linear_trend, fit_quadratic_trend, fit_seasonal_regression
from src.validation import overfitting_check
from src.metrics import calculate_metrics
from src.plotting import line_chart
from src.app_state import register_method_mape

st.title("📐 Regression: Trend · Mùa vụ · Promo")
st.caption("Chương 5 — biết trước NGUYÊN NHÂN luôn thắng đoán mẫu hình, nhưng cần biết trước giá trị "
           "tương lai của biến giải thích.")

result = select_series("regression")
if result is None:
    st.stop()

df, time_col, value_col = result
show_series_preview(df)
y = df[value_col].values
n = len(y)
month_numbers = month_numbers_from_dates(df[time_col]).values

model_type = st.radio(
    "Chọn mô hình", ["Linear Trend", "Quadratic Trend", "Seasonal Regression đầy đủ"], horizontal=True,
)

if model_type == "Linear Trend":
    try:
        r = fit_linear_trend(y)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    fitted = r.fitted

elif model_type == "Quadratic Trend":
    st.caption(
        "Hệ số b2 ÂM: tăng trưởng đang CHỮNG LẠI (thường gặp ở sản phẩm mới tiến gần bão hoà). "
        "b2 DƯƠNG: tăng trưởng đang TĂNG TỐC."
    )
    try:
        r = fit_quadratic_trend(y)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    fitted = r.fitted
    st.metric("Hệ số b2 (độ cong)", f"{r.coefficients['b2']:.4f}")

else:
    use_promo = st.checkbox("Thêm biến Promo (đánh dấu tháng khuyến mãi)")
    promo_flags = None
    if use_promo:
        st.caption("Chọn các kỳ (theo chỉ số 1-based) có khuyến mãi — cách nhau bởi dấu phẩy, ví dụ: 14,34")
        promo_input = st.text_input("Các kỳ khuyến mãi", value="")
        promo_flags = np.zeros(n)
        if promo_input.strip():
            try:
                promo_periods = [int(x.strip()) for x in promo_input.split(",") if x.strip()]
                for p in promo_periods:
                    if 1 <= p <= n:
                        promo_flags[p - 1] = 1
            except ValueError:
                st.warning("Định dạng danh sách kỳ khuyến mãi không hợp lệ — bỏ qua.")

    use_quad = st.checkbox("Thêm bậc hai (Quadratic) cho xu hướng")

    try:
        r = fit_seasonal_regression(y, month_numbers, promo_flags=promo_flags, use_quadratic=use_quad)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    fitted = r.fitted

m = calculate_metrics(y, fitted)
register_method_mape(r.model_name, m.mape, value_col)

col1, col2, col3, col4 = st.columns(4)
col1.metric("R²", f"{r.r_squared:.4f}")
col2.metric("Adjusted R²", f"{r.adj_r_squared:.4f}")
col3.metric("F p-value", f"{r.f_pvalue:.2e}")
col4.metric("MAPE", f"{m.mape:.2f}%")

fig = line_chart(df[time_col], {value_col: y, r.model_name: fitted}, title=r.model_name,
                  xaxis_title="Thời gian", yaxis_title=value_col)
st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Kiểm tra quá khớp")
overfit = overfitting_check(r.r_squared, r.n_obs, r.n_params)
col1, col2 = st.columns(2)
col1.metric("Quan sát / Tham số", f"{overfit['observations_per_param']:.1f}")
col2.metric("Adjusted R²", f"{overfit['adjusted_r_squared']:.4f}")

if overfit["is_risky"]:
    st.error(f"⚠️ {overfit['verdict']}")
else:
    st.success(f"✓ {overfit['verdict']}")

with st.expander("Xem toàn bộ hệ số hồi quy"):
    st.dataframe(pd.DataFrame({"Biến": list(r.coefficients.keys()), "Hệ số": list(r.coefficients.values())}),
                 width="stretch")
