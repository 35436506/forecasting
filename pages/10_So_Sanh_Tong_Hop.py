import streamlit as st

from src.app_state import get_method_registry
from src.plotting import method_comparison_bar
from src.data_utils import SAMPLE_SKU_INFO

st.title("🏆 So sánh toàn bộ phương pháp")
st.caption(
    "Bảng này TỰ ĐỘNG ghi nhận MAPE mỗi khi bạn chạy một phương pháp ở các trang trước — "
    "ghé qua từng trang, quay lại đây để xem bảng xếp hạng."
)

sku_names = list(SAMPLE_SKU_INFO.keys()) + ["(SKU tự tải lên)"]
sku_choice = st.selectbox(
    "Chọn SKU để xem bảng so sánh", options=sku_names,
    format_func=lambda k: SAMPLE_SKU_INFO.get(k, {}).get("label", k),
)

registry = get_method_registry(sku_choice)

if not registry:
    st.info(
        "Chưa có kết quả nào được ghi nhận cho SKU này. Hãy sang các trang **Baseline**, "
        "**Exponential Smoothing**, **ARIMA Lab**, **Regression**, hoặc **Croston** — chọn đúng SKU "
        "này và chạy thử ít nhất một phương pháp, rồi quay lại đây."
    )
    st.stop()

st.divider()
st.subheader(f"Bảng xếp hạng MAPE — {SAMPLE_SKU_INFO.get(sku_choice, {}).get('label', sku_choice)}")

methods = list(registry.keys())
mapes = list(registry.values())

fig = method_comparison_bar(methods, mapes)
st.plotly_chart(fig, width="stretch")

sorted_pairs = sorted(zip(methods, mapes), key=lambda x: x[1])
st.dataframe(
    {"Phương pháp": [p[0] for p in sorted_pairs], "MAPE (%)": [round(p[1], 2) for p in sorted_pairs]},
    width="stretch",
)

best_method, best_mape = sorted_pairs[0]
st.success(f"🥇 Tốt nhất hiện tại: **{best_method}** với MAPE {best_mape:.2f}%")

st.info(
    "⚠️ Nhắc lại nguyên tắc cốt lõi của khóa học: **không có phương pháp nào luôn thắng** — "
    "chỉ có phương pháp được CHỨNG MINH tốt hơn bằng chỉ số khách quan, trên đúng dữ liệu của bạn. "
    "Trước khi chốt phương pháp cho công việc thật, luôn kiểm tra thêm bằng Train/Test Split và "
    "Adjusted R² (trang **Train/Test · Walk-Forward**) để chắc chắn kết quả không đến từ quá khớp."
)

if st.button("🗑️ Xóa toàn bộ kết quả đã lưu"):
    st.session_state.pop("method_comparison_registry", None)
    st.rerun()
