import streamlit as st
import numpy as np

from src.ui_helpers import select_series, show_series_preview
from src.pegels import classify_demand_shape
from src.plotting import line_chart

st.title("🔍 Phân loại hình dạng nhu cầu")
st.caption("Chương 1 — trước khi chọn phương pháp, luôn xác định CV, %kỳ=0, SLOPE trước.")

result = select_series("shape", help_text="Chọn một SKU mẫu hoặc tải CSV riêng để phân loại.")
if result is None:
    st.stop()

df, time_col, value_col = result
show_series_preview(df)

fig = line_chart(df[time_col], {value_col: df[value_col].values}, title=f"Time plot — {value_col}",
                  xaxis_title="Thời gian", yaxis_title="Sản lượng")
st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Kết quả phân loại")

try:
    shape_result = classify_demand_shape(df[value_col].values)
except ValueError as error:
    st.error(str(error))
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("CV (hệ số biến thiên)", f"{shape_result.cv:.3f}")
col2.metric("% kỳ = 0", f"{shape_result.pct_zero:.1f}%")
col3.metric("SLOPE (xu hướng/kỳ)", f"{shape_result.slope:.2f}")

st.success(f"**Hình dạng: {shape_result.shape_label}**")
st.markdown(f"**Gợi ý Pegels:** {shape_result.pegels_hint}")
st.info(shape_result.explanation)

with st.expander("Công thức tính (Lecture Notes Chương 1)"):
    st.latex(r"CV = \dfrac{\sigma}{\mu}")
    st.latex(r"\%\text{kỳ}=0 = \dfrac{n(Y_t=0)}{n} \times 100\%")
    st.latex(r"\text{SLOPE} = \dfrac{\sum(t-\bar t)(Y_t-\bar Y)}{\sum(t-\bar t)^2}")
    st.caption(
        "App dùng thêm quy tắc thực hành: xu hướng lũy kế (SLOPE × n / mean) ≥20% được xem là "
        '"Mùa vụ-Xu hướng"; %kỳ=0 tập trung ở đầu chuỗi (không rải rác) được xem là "Vòng đời '
        'ngắn" thay vì "Rời rạc" — xem Lecture Notes Chương 1 để biết chi tiết.'
    )
