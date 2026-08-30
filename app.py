import streamlit as st

st.set_page_config(
    page_title="Forecasting Lab — GDMN",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Phân tích nhanh": [
        st.Page("pages/0_Phan_Tich_Nhanh.py", title="Phân tích nhanh (dữ liệu tự do)", icon="⚡", default=True),
    ],
    "Bắt đầu": [
        st.Page("pages/1_Tong_Quan.py", title="Tổng quan & Ma trận Pegels", icon="🧭"),
        st.Page("pages/2_Phan_Loai_Du_Lieu.py", title="Phân loại hình dạng nhu cầu", icon="🔍"),
    ],
    "Mô hình hóa": [
        st.Page("pages/3_Baseline_Naive.py", title="Baseline: NF1, NF2 & 7 chỉ số", icon="📏"),
        st.Page("pages/4_Exponential_Smoothing.py", title="SES · Holt · Holt-Winters", icon="📉"),
        st.Page("pages/5_ARIMA_Lab.py", title="ARIMA / SARIMA Lab", icon="🌀"),
        st.Page("pages/6_Regression.py", title="Regression (Trend · Mùa vụ · Promo)", icon="📐"),
    ],
    "Vận hành & Kiểm định": [
        st.Page("pages/7_Van_Hanh.py", title="Safety Stock · ROP · Croston", icon="📦"),
        st.Page("pages/8_Kiem_Tra_Mo_Hinh.py", title="Train/Test · Walk-Forward · Quá khớp", icon="🧪"),
        st.Page("pages/9_Xu_Ly_Du_Lieu.py", title="Outlier & Dữ liệu thiếu", icon="🧹"),
    ],
    "Tổng hợp": [
        st.Page("pages/10_So_Sanh_Tong_Hop.py", title="So sánh toàn bộ phương pháp", icon="🏆"),
    ],
}

navigation = st.navigation(pages)

with st.sidebar:
    st.title("📈 Forecasting Lab")
    st.caption("Đồng hành cùng khóa học Forecasting Cho Chuỗi Cung Ứng — GDMN")
    st.divider()
    st.markdown(
        """
        **Cách dùng nhanh:**

        1. Có sẵn file dữ liệu riêng (Excel/CSV bất kỳ)? Vào ngay
           *Phân tích nhanh* — app tự dò cột và chạy toàn bộ quy trình dự
           báo cho bạn, kèm 7 bộ dữ liệu mẫu để thử ngay.
        2. Muốn học theo đúng trình tự khóa học? Bắt đầu ở trang *Tổng quan*
           để ôn Ma trận Pegels.
        3. Vào *Phân loại hình dạng nhu cầu* để biết SKU của bạn hợp phương
           pháp nào.
        4. Thử các trang mô hình hóa tương ứng — mỗi trang đều dùng CÙNG
           công thức đã học trong Lecture Notes.
        5. Ghé *So sánh toàn bộ phương pháp* để xem bảng xếp hạng MAPE
           (bao gồm cả kết quả từ trang *Phân tích nhanh*).

        Các trang theo khóa học dùng chung bộ dữ liệu mẫu **ERP_4SKU**,
        hoặc bạn có thể tải CSV riêng lên bất kỳ trang nào. Trang *Phân
        tích nhanh* nhận thêm cả file Excel nhiều sheet.
        """
    )
    st.divider()
    st.caption("Dữ liệu và công thức khớp 1-1 với Lecture Notes & Python Instructions của khóa học.")

navigation.run()
