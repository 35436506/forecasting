import streamlit as st
import numpy as np

from src.ui_helpers import select_series, show_series_preview
from src.naive_smoothing import naive_nf1, naive_nf2, moving_average, weighted_moving_average, double_moving_average
from src.metrics import calculate_metrics
from src.plotting import line_chart
from src.app_state import register_method_mape

st.title("📏 Baseline: NF1, NF2, MA, WMA, Double-MA & 7 chỉ số sai số")
st.caption(
    "Chương 2–3 — các phương pháp KHÔNG cần tối ưu tham số (khác SES/Holt/Holt-Winters ở trang sau). "
    "Mọi phương pháp phức tạp hơn PHẢI đánh bại các mốc chuẩn này mới đáng dùng."
)

result = select_series("baseline")
if result is None:
    st.stop()

df, time_col, value_col = result
show_series_preview(df)
y = df[value_col].values

col1, col2 = st.columns(2)
with col1:
    season_length = st.slider("Chu kỳ mùa vụ dùng cho NF2 (số kỳ)", min_value=2, max_value=12, value=12)
with col2:
    k_window = st.slider("Bậc k dùng cho MA và WMA (số kỳ nhìn lại)", min_value=2, max_value=6, value=3)

st.caption(
    f"WMA mặc định dùng trọng số tăng dần {'·'.join(str(i) for i in range(1, k_window + 1))} "
    "(kỳ GẦN NHẤT có trọng số LỚN NHẤT) — khớp cách dạy trong Lecture Notes."
)

results = {}

nf1_result = naive_nf1(y)
results["NF1"] = nf1_result

try:
    results["NF2"] = naive_nf2(y, season_length=season_length)
except Exception:
    results["NF2"] = None

try:
    results["MA"] = moving_average(y, k=k_window)
except ValueError as error:
    st.warning(f"MA: {error}")
    results["MA"] = None

try:
    results["WMA"] = weighted_moving_average(y, weights=list(range(1, k_window + 1)))
except ValueError as error:
    st.warning(f"WMA: {error}")
    results["WMA"] = None

try:
    results["Double-MA"] = double_moving_average(y, k=k_window)
except ValueError as error:
    st.warning(f"Double-MA: {error}")
    results["Double-MA"] = None

series_to_plot = {value_col: y}
metrics_by_method = {}
for label, r in results.items():
    if r is None:
        continue
    valid_mask = ~np.isnan(r.fitted)
    if valid_mask.sum() == 0:
        continue
    m = calculate_metrics(y[valid_mask], r.fitted[valid_mask])
    metrics_by_method[label] = (r, m)
    register_method_mape(r.method_name, m.mape, value_col)
    series_to_plot[r.method_name] = r.fitted

fig = line_chart(df[time_col], series_to_plot, title="So sánh NF1 · NF2 · MA · WMA · Double-MA",
                  xaxis_title="Thời gian", yaxis_title=value_col)
st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Bảng 7 chỉ số sai số — từng phương pháp")

cols = st.columns(len(metrics_by_method)) if metrics_by_method else []
for col, (label, (r, m)) in zip(cols, metrics_by_method.items()):
    with col:
        st.markdown(f"**{r.method_name}**")
        st.table(m.as_dict())

if not metrics_by_method:
    st.info("Chuỗi chưa đủ dài để tính bất kỳ phương pháp nào ở trên.")

st.divider()
st.subheader("Bảng xếp hạng nhanh (MAPE)")
if metrics_by_method:
    ranking = sorted(((r.method_name, m.mape) for r, m in metrics_by_method.values()), key=lambda x: x[1])
    st.table({"Phương pháp": [x[0] for x in ranking], "MAPE (%)": [round(x[1], 2) for x in ranking]})

with st.expander("Ý nghĩa từng chỉ số (Lecture Notes Chương 2)"):
    st.markdown(
        """
        - **ME (Forecast Bias)**: trung bình sai số CÓ DẤU — phát hiện thiên lệch hệ thống,
          nhưng dễ "nói dối" vì dương/âm triệt tiêu nhau.
        - **MAE**: trung bình trị tuyệt đối sai số — cùng đơn vị dữ liệu gốc.
        - **MSE / RMSE**: phạt nặng sai số lớn — MSE là đầu vào trực tiếp cho Safety Stock.
        - **MPE / MAPE**: sai số phần trăm — MAPE là chỉ số phổ biến nhất, nhưng "vỡ" khi có
          kỳ giá trị 0.
        - **WMAPE**: tổng sai số tuyệt đối chia tổng doanh số — ưu tiên khi các kỳ có quy mô
          doanh số chênh lệch lớn (xem Chương 2, mục WMAPE).
        """
    )

with st.expander("Phân biệt MA, WMA và Double-MA"):
    st.markdown(
        f"""
        - **MA (Moving Average) bậc k**: trung bình cộng ĐƠN GIẢN của k quan sát gần nhất —
          mọi kỳ trong cửa sổ có trọng số BẰNG NHAU (1/k).
        - **WMA (Weighted Moving Average) bậc k**: trung bình có TRỌNG SỐ — kỳ CÀNG GẦN hiện
          tại càng được coi trọng hơn. Với k={k_window}, trọng số dùng ở đây là
          {'·'.join(str(i) for i in range(1, k_window + 1))} (đã chuẩn hoá về tổng = 1).
        - **Double-MA (Double Moving Average) bậc k**: dành cho dữ liệu CÓ XU HƯỚNG — tính
          thêm một lớp MA của chính MA, dùng độ chênh lệch giữa hai lớp để ước lượng độ dốc.
          Là tiền thân trực tiếp của Holt (trang sau) — cùng nguyên lý tách mức nền + xu
          hướng, chỉ khác trọng số (đều nhau ở đây, giảm dần cấp số nhân ở Holt).
        - **Khác biệt với SES/Holt/Holt-Winters** (trang sau): cả ba phương pháp trên đều
          KHÔNG có tham số cần tối ưu bằng Solver — bậc k và trọng số phải chọn bằng thử
          nghiệm thủ công, khác với alpha/beta/gamma được statsmodels tự tìm.
        - **Bài học quan trọng**: thêm đúng thành phần (như Double-MA cho Áo khoác mùa đông —
          SKU CÓ xu hướng thật) cải thiện đáng kể độ chính xác; thêm sai thành phần (như
          Double-MA cho Bếp từ đôi — SKU ổn định, không có xu hướng) lại làm mô hình XẤU ĐI vì
          tạo ra "xu hướng giả".
        """
    )
