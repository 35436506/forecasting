import numpy as np
import streamlit as st

from src.ui_helpers import select_series, show_series_preview
from src.arima_modeling import adf_test, apply_differencing, fit_sarima, aic_grid_search, ljung_box_test
from src.diagnostics import compute_acf_pacf
from src.metrics import calculate_metrics
from src.plotting import line_chart, acf_pacf_charts, heatmap_chart, forecast_chart_with_ci
from src.app_state import register_method_mape

st.title("🌀 ARIMA / SARIMA Lab")
st.caption(
    "Chương 4 — quy trình Box-Jenkins đầy đủ: sai phân → ACF/PACF → grid search AIC → "
    "chẩn đoán Ljung-Box → dự báo."
)

result = select_series("arima")
if result is None:
    st.stop()

df, time_col, value_col = result
show_series_preview(df)
y = df[value_col].values
n = len(y)

# ---------- 1. Kiểm tra tính dừng ----------
st.divider()
st.subheader("1. Kiểm tra tính dừng (ADF)")

diff_order = st.radio("Bậc sai phân để quan sát", [0, 1, 2], horizontal=True,
                       format_func=lambda x: {0: "Chuỗi gốc", 1: "Sai phân bậc 1", 2: "Sai phân bậc 2"}[x])

y_diff = apply_differencing(y, diff_order)
x_diff = df[time_col].values[diff_order:]

fig = line_chart(x_diff, {f"Sai phân bậc {diff_order}": y_diff}, title="Chuỗi sau sai phân",
                  xaxis_title="Thời gian", yaxis_title=value_col)
st.plotly_chart(fig, width="stretch")

try:
    adf_result = adf_test(y_diff)
    col1, col2, col3 = st.columns(3)
    col1.metric("ADF statistic", f"{adf_result['adf_statistic']:.3f}")
    col2.metric("p-value", f"{adf_result['p_value']:.4f}")
    col3.metric("Kết luận", "DỪNG ✓" if adf_result["is_stationary_5pct"] else "CHƯA dừng")
    if not adf_result["is_stationary_5pct"]:
        st.warning("p-value ≥ 0.05 — chưa đủ bằng chứng bác bỏ giả thuyết không dừng. Thử tăng bậc sai phân.")
except ValueError as error:
    st.warning(str(error))

# ---------- 2. ACF/PACF ----------
st.divider()
st.subheader("2. ACF / PACF trên chuỗi đã sai phân")
st.caption("PACF cắt đột ngột → gợi ý bậc AR (p). ACF cắt đột ngột → gợi ý bậc MA (q).")

try:
    diag = compute_acf_pacf(y_diff, n_lags=min(20, len(y_diff) // 2))
    fig_acf, fig_pacf = acf_pacf_charts(diag["lags"], diag["acf"], diag["pacf"], diag["conf_bound"])
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_acf, width="stretch")
    col2.plotly_chart(fig_pacf, width="stretch")
except ValueError as error:
    st.warning(str(error))

# ---------- 3. Grid search AIC ----------
st.divider()
st.subheader("3. Bản đồ nhiệt AIC — quét toàn bộ lưới (p,q)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    p_max = st.number_input("p tối đa", min_value=1, max_value=5, value=3)
with col2:
    q_max = st.number_input("q tối đa", min_value=1, max_value=5, value=3)
with col3:
    d_fixed = st.number_input("d (cố định)", min_value=0, max_value=2, value=max(diff_order, 1))
with col4:
    use_seasonal = st.checkbox("Có mùa vụ (SARIMA)", value=False)

seasonal_order = (0, 0, 0, 0)
if use_seasonal:
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        P = st.number_input("P", min_value=0, max_value=2, value=0)
    with col6:
        D = st.number_input("D", min_value=0, max_value=1, value=1)
    with col7:
        Q = st.number_input("Q", min_value=0, max_value=2, value=1)
    with col8:
        s = st.number_input("Chu kỳ mùa vụ s", min_value=2, max_value=24, value=12)
    seasonal_order = (P, D, Q, s)

if st.button("🔍 Chạy Grid Search AIC", type="primary"):
    with st.spinner("Đang quét lưới AIC — có thể mất vài giây..."):
        try:
            grid, best_idx = aic_grid_search(y, p_max=p_max, q_max=q_max, d=d_fixed,
                                              seasonal_order=seasonal_order)
            st.session_state["arima_grid"] = grid
            st.session_state["arima_best_idx"] = best_idx
            st.session_state["arima_d"] = d_fixed
            st.session_state["arima_seasonal_order"] = seasonal_order
        except ValueError as error:
            st.error(str(error))

if "arima_grid" in st.session_state:
    grid = st.session_state["arima_grid"]
    best_idx = st.session_state["arima_best_idx"]
    fig_heat = heatmap_chart(grid, best_idx, title="Lưới AIC (đỏ = thấp nhất)")
    st.plotly_chart(fig_heat, width="stretch")
    best_p, best_q = best_idx
    st.success(f"Order tốt nhất trong lưới: (p={best_p}, d={st.session_state['arima_d']}, q={best_q}), "
               f"AIC={grid[best_idx]:.2f}")
    st.caption(
        "⚠️ Kết quả phụ thuộc TRỰC TIẾP vào p_max/q_max bạn chọn — một lưới quá hẹp có thể bỏ lỡ "
        "mô hình tốt hơn. Luôn đối chiếu tỷ lệ quan sát/tham số trước khi chọn mô hình phức tạp "
        "hơn chỉ vì AIC thấp hơn một chút (xem Lecture Notes Chương 4 và Chương 7)."
    )

# ---------- 4. Fit mô hình đã chọn ----------
st.divider()
st.subheader("4. Chọn order cuối cùng, huấn luyện và dự báo")

col1, col2, col3 = st.columns(3)
default_p = st.session_state.get("arima_best_idx", (0, 0))[0]
default_q = st.session_state.get("arima_best_idx", (0, 0))[1]
with col1:
    final_p = st.number_input("p", min_value=0, max_value=5, value=int(default_p), key="final_p")
with col2:
    final_d = st.number_input("d", min_value=0, max_value=2, value=int(d_fixed), key="final_d")
with col3:
    final_q = st.number_input("q", min_value=0, max_value=5, value=int(default_q), key="final_q")

future_steps = st.slider("Số kỳ dự báo tương lai", 1, 12, 6)

if st.button("🚀 Huấn luyện SARIMA và dự báo", type="primary"):
    try:
        fit_result = fit_sarima(
            y, order=(final_p, final_d, final_q), seasonal_order=seasonal_order,
            future_steps=future_steps,
        )
        st.session_state["arima_fit_result"] = fit_result
    except ValueError as error:
        st.error(str(error))

if "arima_fit_result" in st.session_state:
    fit_result = st.session_state["arima_fit_result"]
    valid_mask = ~np.isnan(fit_result.fitted)

    m = calculate_metrics(y[valid_mask], fit_result.fitted[valid_mask])
    method_label = f"SARIMA{fit_result.order}{fit_result.seasonal_order if use_seasonal else ''}"
    register_method_mape(method_label, m.mape, value_col)

    col1, col2, col3 = st.columns(3)
    col1.metric("AIC", f"{fit_result.aic:.2f}")
    col2.metric("MAPE (trong mẫu)", f"{m.mape:.2f}%")
    col3.metric("MSE", f"{m.mse:,.0f}")

    future_index = np.arange(n, n + future_steps)
    fig_fc = forecast_chart_with_ci(
        df[time_col], y, future_index, fit_result.forecast_mean,
        fit_result.forecast_ci_lower, fit_result.forecast_ci_upper,
        title=f"Dự báo {method_label}", xaxis_title="Thời gian", yaxis_title=value_col,
    )
    st.plotly_chart(fig_fc, width="stretch")

    st.markdown("**Chẩn đoán phần dư — ACF + kiểm định Ljung-Box**")
    resid = fit_result.residuals
    try:
        diag_resid = compute_acf_pacf(resid[~np.isnan(resid)], n_lags=min(20, len(resid) // 2))
        fig_acf_r, _ = acf_pacf_charts(diag_resid["lags"], diag_resid["acf"], diag_resid["pacf"],
                                        diag_resid["conf_bound"])
        fig_acf_r.update_layout(title="ACF phần dư")
        st.plotly_chart(fig_acf_r, width="stretch")
    except ValueError as error:
        st.caption(str(error))

    try:
        lb = ljung_box_test(resid, lags=[6, 12] if len(resid) > 24 else None)
        st.dataframe(lb, width="stretch")
        all_pass = (lb["lb_pvalue"] > 0.05).all()
        if all_pass:
            st.success("Toàn bộ p-value > 0.05 — không đủ bằng chứng bác bỏ H0 — phần dư chấp nhận được.")
        else:
            st.warning("Có lag với p-value ≤ 0.05 — phần dư vẫn còn tự tương quan, cân nhắc order khác.")
    except ValueError as error:
        st.caption(str(error))
