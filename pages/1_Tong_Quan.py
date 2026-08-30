import streamlit as st

st.title("🧭 Tổng quan quy trình dự báo")
st.caption("Ôn nhanh khung tư duy của cả khóa học trước khi thực hành trên các trang tiếp theo.")

st.markdown(
    """
    ### Quy trình 6 bước (Lecture Notes, Chương 10)

    | Bước | Nội dung | Trang tương ứng trong app |
    |---|---|---|
    | 1 | Đọc hình dạng nhu cầu — CV, %kỳ=0, SLOPE, Ma trận Pegels | **Phân loại hình dạng nhu cầu** |
    | 2 | Dựng baseline Naive (NF1, NF2) | **Baseline: NF1, NF2** |
    | 3 | Thử các phương pháp phù hợp (SES/Holt/HW, ARIMA, Regression) | **SES·Holt·HW**, **ARIMA Lab**, **Regression** |
    | 4 | Kiểm tra quá khớp (Adjusted R², Train/Test Split) | **Train/Test · Walk-Forward** |
    | 5 | Chuyển thành quyết định vận hành (Safety Stock, Tracking Signal) | **Safety Stock · ROP · Croston** |
    | 6 | Trình bày kết quả theo kim tự tháp ngược | *(xem Lecture Notes Chương 9)* |
    """
)

st.divider()
st.subheader("Ma trận Pegels — 9 mẫu hình Xu hướng × Mùa vụ")
st.caption(
    "Đường nét đứt xám là BASELINE (mức nền/xu hướng), vẽ trước. Đường đen là dữ liệu thật "
    "sau khi cộng mùa vụ, chồng lên trên. So sánh trực tiếp ô B2 với B3 để thấy hiệu ứng "
    '"loa kèn" của mùa vụ NHÂN khi có xu hướng.'
)

try:
    st.image("assets/pegels_3x3_grid.png", width="stretch")
except Exception:
    st.warning("Không tìm thấy ảnh minh họa — xem file `assets/pegels_3x3_grid.png`.")

with st.expander("Cách đọc lưới 9 ô"):
    st.markdown(
        """
        - **Theo HÀNG**: xu hướng thay đổi ra sao khi giữ nguyên mùa vụ — hàng A dao động
          quanh một mức cố định; hàng B là đường thẳng dốc lên; hàng C là đường cong dốc dần
          (giống lãi kép).
        - **Theo CỘT**: mùa vụ thay đổi ra sao khi giữ nguyên xu hướng — cột 1 không có sóng
          lặp lại; cột 2 có sóng biên độ CỐ ĐỊNH; cột 3 có sóng biên độ PHÌNH RA theo xu hướng.
        - **Câu hỏi thường gặp**: tại sao ô A3 (không xu hướng, mùa vụ nhân) các mùa lại BẰNG
          NHAU? Vì mùa vụ nhân = % × mức nền TẠI THỜI ĐIỂM ĐÓ — ở hàng A mức nền không đổi,
          nên biên độ % của nó cũng không đổi. Hình "loa kèn" chỉ xuất hiện khi mùa vụ NHÂN
          kết hợp VỚI xu hướng (hàng B, C).
        """
    )

st.divider()
st.subheader("Bảng chọn phương pháp nhanh")
st.table(
    {
        "Hình dạng nhu cầu": ["Ổn định", "Mùa vụ – Xu hướng", "Rời rạc", "Vòng đời ngắn"],
        "Ô Pegels": ["(N,N)", "(A,A)/(A,M)/(M,A)/(M,M)", "Ngoài ma trận", "Ngoài ma trận"],
        "Phương pháp đề xuất": ["SES", "Holt-Winters", "Croston/SBA/TSB", "Holt hoặc Quadratic Trend"],
    }
)

st.info(
    "💡 Toàn bộ công thức trong app này khớp 1-1 với Lecture Notes và Python Instructions của "
    "khóa học *Forecasting Cho Chuỗi Cung Ứng* — dùng app để thực hành trực quan, rồi đối chiếu "
    "lại với file Excel/Python đã làm trên lớp."
)
