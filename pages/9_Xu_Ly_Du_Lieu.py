import numpy as np
import streamlit as st

from src.ui_helpers import select_series, show_series_preview
from src.outliers import detect_outliers_iqr, detect_outliers_zscore, interpolate_missing, apply_manual_outlier_fix
from src.naive_smoothing import fit_holt_winters
from src.metrics import calculate_metrics
from src.plotting import line_chart

st.title("🧹 Outlier & Dữ liệu thiếu")
st.caption("Chương 8 — mô hình giỏi nhất cũng vô dụng nếu dữ liệu đầu vào sai.")

result = select_series("outliers")
if result is None:
    st.stop()

df, time_col, value_col = result
show_series_preview(df)
y = df[value_col].values.copy()

st.divider()
st.subheader("1. Phát hiện Outlier")

col1, col2 = st.columns(2)
with col1:
    iqr_multiplier = st.slider("Hệ số IQR", 1.0, 3.0, 1.5, 0.1)
with col2:
    z_threshold = st.slider("Ngưỡng Z-score", 2.0, 4.0, 3.0, 0.5)

try:
    iqr_report = detect_outliers_iqr(y, multiplier=iqr_multiplier)
    zscore_report = detect_outliers_zscore(y, threshold=z_threshold)
except ValueError as error:
    st.error(str(error))
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.metric("Outlier theo IQR", len(iqr_report.outlier_indices))
    if iqr_report.outlier_indices:
        st.write({f"Kỳ {i+1}": v for i, v in zip(iqr_report.outlier_indices, iqr_report.outlier_values)})
with col2:
    st.metric("Outlier theo Z-score", len(zscore_report.outlier_indices))
    if zscore_report.outlier_indices:
        st.write({f"Kỳ {i+1}": v for i, v in zip(zscore_report.outlier_indices, zscore_report.outlier_values)})

fig = line_chart(df[time_col], {value_col: y}, title="Chuỗi gốc", xaxis_title="Thời gian", yaxis_title=value_col)
fig.add_hline(y=iqr_report.upper_bound, line_dash="dash", line_color="#C0392B", annotation_text="Ngưỡng IQR trên")
fig.add_hline(y=iqr_report.lower_bound, line_dash="dash", line_color="#C0392B", annotation_text="Ngưỡng IQR dưới")
st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("2. Xử lý")

combined_outliers = sorted(set(iqr_report.outlier_indices) | set(zscore_report.outlier_indices))
if combined_outliers:
    st.warning(
        f"Phát hiện {len(combined_outliers)} điểm nghi ngờ. **Trước khi sửa, hãy hỏi**: có sự kiện "
        "kinh doanh nào giải thích được giá trị này không (khuyến mãi, thiên tai...)? Nếu CÓ — giữ "
        "nguyên, đánh dấu làm biến giải thích. Nếu KHÔNG — nhiều khả năng là lỗi nhập liệu, nên sửa."
    )
    fix_indices = st.multiselect(
        "Chọn các kỳ XÁC NHẬN là lỗi dữ liệu để sửa (nội suy lại)",
        options=combined_outliers, format_func=lambda i: f"Kỳ {i+1} (giá trị={y[i]:.0f})",
    )
else:
    fix_indices = []
    st.success("Không phát hiện outlier rõ rệt theo cả hai phương pháp.")

has_missing = np.isnan(y).any()
if has_missing:
    st.info(f"Chuỗi có {np.isnan(y).sum()} kỳ dữ liệu thiếu (NaN).")

interp_method = st.radio("Phương pháp nội suy", ["linear", "seasonal"], horizontal=True,
                          format_func=lambda x: "Tuyến tính" if x == "linear" else "Theo mùa")

if fix_indices or has_missing:
    if st.button("🔧 Áp dụng làm sạch", type="primary"):
        y_clean = apply_manual_outlier_fix(y, fix_indices, method=interp_method) if fix_indices else y
        y_clean = interpolate_missing(y_clean, method=interp_method)
        st.session_state["cleaned_series"] = y_clean

if "cleaned_series" in st.session_state:
    y_clean = st.session_state["cleaned_series"]
    fig2 = line_chart(df[time_col], {f"{value_col} (bẩn)": y, f"{value_col} (đã làm sạch)": y_clean},
                       title="Trước và sau khi làm sạch", xaxis_title="Thời gian", yaxis_title=value_col)
    st.plotly_chart(fig2, width="stretch")

    st.divider()
    st.subheader("3. Tác động lên độ chính xác (Holt-Winters)")
    season_length = st.number_input("Chu kỳ mùa vụ", min_value=2, max_value=24, value=12)
    try:
        r_dirty = fit_holt_winters(y, season_length=season_length)
        r_clean = fit_holt_winters(y_clean, season_length=season_length)
        m_dirty = calculate_metrics(y[~np.isnan(r_dirty.fitted)], r_dirty.fitted[~np.isnan(r_dirty.fitted)])
        m_clean = calculate_metrics(y_clean[~np.isnan(r_clean.fitted)], r_clean.fitted[~np.isnan(r_clean.fitted)])

        col1, col2 = st.columns(2)
        col1.metric("MAPE — dữ liệu bẩn", f"{m_dirty.mape:.2f}%")
        col2.metric("MAPE — đã làm sạch", f"{m_clean.mape:.2f}%", delta=f"{m_clean.mape - m_dirty.mape:+.2f}%",
                    delta_color="inverse")
    except ValueError as error:
        st.caption(str(error))
