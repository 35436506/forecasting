import numpy as np
import streamlit as st

from src.ui_helpers import select_series, show_series_preview
from src.naive_smoothing import fit_holt_winters
from src.metrics import calculate_metrics, tracking_signal_series
from src.inventory import safety_stock, reorder_point, holding_cost_per_period
from src.croston import croston_family, suggest_variant
from src.plotting import line_chart, tracking_signal_chart
from src.app_state import register_method_mape

st.title("📦 Vận hành: Safety Stock · ROP · Tracking Signal · Croston")
st.caption("Chương 6 — biến forecast thành quyết định tồn kho cụ thể.")

tab1, tab2, tab3 = st.tabs(["Safety Stock & ROP", "Tracking Signal", "Croston / SBA / TSB"])

with tab1:
    result = select_series("safety_stock")
    if result is not None:
        df, time_col, value_col = result
        show_series_preview(df)
        y = df[value_col].values

        st.caption("Dùng MSE từ Holt-Winters (nếu chuỗi đủ dài) làm ước lượng sai số dự báo.")
        mse_source = st.radio("Nguồn MSE", ["Tự nhập", "Tính từ Holt-Winters"], horizontal=True)

        if mse_source == "Tính từ Holt-Winters":
            season_length = st.number_input("Chu kỳ mùa vụ", min_value=2, max_value=24, value=12)
            try:
                r = fit_holt_winters(y, season_length=season_length)
                m = calculate_metrics(y, r.fitted)
                mse_value = m.mse
                st.info(f"MSE từ Holt-Winters: {mse_value:,.1f}")
            except ValueError as error:
                st.warning(str(error))
                mse_value = float(np.var(y))
        else:
            mse_value = st.number_input("MSE", min_value=0.0, value=float(np.var(y)), step=100.0)

        col1, col2, col3 = st.columns(3)
        with col1:
            lead_time = st.number_input("Lead time (số kỳ)", min_value=0.1, value=1.0, step=0.5)
        with col2:
            service_level = st.selectbox("Mức phục vụ", [80, 90, 95, 99], index=2)
        with col3:
            avg_demand = st.number_input("Nhu cầu trung bình/kỳ", min_value=0.0, value=float(np.mean(y)))

        try:
            rop_result = reorder_point(avg_demand, lead_time, mse_value, service_level)
        except ValueError as error:
            st.error(str(error))
            st.stop()

        col1, col2, col3 = st.columns(3)
        col1.metric("z", f"{rop_result.z:.3f}")
        col2.metric("Safety Stock", f"{rop_result.safety_stock:,.1f}")
        col3.metric("Reorder Point (ROP)", f"{rop_result.rop:,.1f}")

        st.divider()
        st.markdown("**Quy đổi chi phí vốn tồn kho**")
        col1, col2 = st.columns(2)
        with col1:
            unit_cost = st.number_input("Giá vốn/đơn vị (VNĐ)", min_value=0.0, value=350000.0, step=10000.0)
        with col2:
            holding_rate = st.number_input("Tỷ lệ chi phí giữ hàng (%/kỳ)", min_value=0.0, value=2.0, step=0.5)

        cost = holding_cost_per_period(rop_result.safety_stock, unit_cost, holding_rate)
        st.metric("Chi phí vốn Safety Stock mỗi kỳ", f"{cost:,.0f} VNĐ")

with tab2:
    result2 = select_series("tracking_signal")
    if result2 is not None:
        df2, time_col2, value_col2 = result2
        y2 = df2[value_col2].values
        method_ts = st.selectbox("Phương pháp forecast để tính Tracking Signal", ["NF1", "Holt-Winters"])

        from src.naive_smoothing import naive_nf1

        if method_ts == "NF1":
            r = naive_nf1(y2)
            fitted = r.fitted
        else:
            season_length2 = st.number_input("Chu kỳ mùa vụ", min_value=2, max_value=24, value=12, key="ts_season")
            try:
                r = fit_holt_winters(y2, season_length=season_length2)
                fitted = r.fitted
            except ValueError as error:
                st.warning(str(error))
                fitted = None

        if fitted is not None:
            valid = ~np.isnan(fitted)
            ts = tracking_signal_series(y2[valid], fitted[valid])
            fig = tracking_signal_chart(df2[time_col2].values[valid], ts.values)
            st.plotly_chart(fig, width="stretch")
            if (ts.abs() > 4).any():
                st.warning("⚠️ Có kỳ Tracking Signal vượt ngưỡng ±4 — dấu hiệu mô hình thiên lệch hệ thống, "
                           "nên xem lại phương pháp dự báo.")
            else:
                st.success("Tracking Signal nằm trong ngưỡng ±4 suốt chuỗi — chưa có dấu hiệu thiên lệch hệ thống.")

with tab3:
    result3 = select_series("croston", help_text="Dùng cho SKU có nhiều kỳ = 0 (ví dụ Phụ tùng thay thế).")
    if result3 is not None:
        df3, time_col3, value_col3 = result3
        y3 = df3[value_col3].values
        pct_zero = float(np.mean(y3 == 0) * 100)
        st.metric("% kỳ = 0", f"{pct_zero:.1f}%")
        if pct_zero < 20:
            st.info("SKU này có ít kỳ = 0 — Croston/SBA/TSB có thể chưa cần thiết, các phương pháp thường "
                    "(SES/Holt-Winters) vẫn hoạt động tốt.")

        col1, col2 = st.columns(2)
        with col1:
            alpha = st.slider("alpha", 0.05, 0.5, 0.2, 0.05)
        with col2:
            beta = st.slider("beta (chỉ TSB dùng)", 0.05, 0.5, 0.1, 0.05)

        try:
            suggestion = suggest_variant(y3)
            st.info(f"💡 Gợi ý: **{suggestion['suggestion']}** — {suggestion.get('reason', '')}")
        except Exception:
            pass

        series_dict = {value_col3: y3}
        for variant in ["croston", "sba", "tsb"]:
            try:
                r = croston_family(y3, alpha=alpha, beta=beta, variant=variant)
                series_dict[variant.upper()] = r.forecast
                valid = ~np.isnan(r.forecast)
                if valid.sum() > 0:
                    m = calculate_metrics(y3[valid], r.forecast[valid])
                    register_method_mape(variant.upper(), m.mape, value_col3)
            except ValueError as error:
                st.warning(f"{variant.upper()}: {error}")

        fig = line_chart(df3[time_col3], series_dict, title="Croston / SBA / TSB",
                          xaxis_title="Thời gian", yaxis_title=value_col3)
        st.plotly_chart(fig, width="stretch")
