# Forecasting Lab — GDMN

Ứng dụng Streamlit đồng hành cùng khóa học **Forecasting Cho Chuỗi Cung Ứng**
(GDMN) — thực hành trực quan toàn bộ 7 phương pháp dự báo và các công cụ vận
hành đã học, dùng đúng bộ dữ liệu mẫu **ERP_4SKU** của khóa học. Ngoài ra,
trang **⚡ Phân tích nhanh** cho phép tải lên MỘT FILE Excel/CSV bất kỳ (không
theo khuôn ERP_4SKU) và tự động chạy toàn bộ quy trình dự báo — phù hợp khi
bạn muốn thử nhanh với dữ liệu thật của riêng mình.

Khác với bản "ARIMA Demo Lab" trước đây (chỉ có một trang, chỉ có ARIMA), app
này mở rộng thành 11 trang bao phủ toàn bộ giáo trình — Naive, Exponential
Smoothing, ARIMA/SARIMA, Regression, Safety Stock/Croston, Train/Test
Validation, xử lý dữ liệu bẩn, và một trang phân tích tự động cho dữ liệu tự
do — đồng thời cải tiến phần ARIMA với bản đồ nhiệt AIC và kiểm định
Ljung-Box (tham khảo tài liệu *"Mô hình ARIMA trong bài toán dự báo chuỗi
thời gian"*, AI VIETNAM 2026).

## Cách chạy

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Cấu trúc project

```text
.
├── app.py                          # Điều hướng đa trang
├── pages/
│   ├── 0_Phan_Tich_Nhanh.py        # ⚡ Phân tích tự động cho file Excel/CSV bất kỳ
│   ├── 1_Tong_Quan.py              # Ma trận Pegels, quy trình 6 bước
│   ├── 2_Phan_Loai_Du_Lieu.py      # CV, %kỳ=0, SLOPE
│   ├── 3_Baseline_Naive.py         # NF1, NF2, MA, WMA, Double-MA — 7 chỉ số sai số (kể cả WMAPE)
│   ├── 4_Exponential_Smoothing.py  # SES, Holt, Holt-Winters
│   ├── 5_ARIMA_Lab.py              # ADF, ACF/PACF, heatmap AIC, Ljung-Box
│   ├── 6_Regression.py             # Trend/Quadratic/Dummy/Promo
│   ├── 7_Van_Hanh.py               # Safety Stock, ROP, Tracking Signal, Croston/SBA/TSB
│   ├── 8_Kiem_Tra_Mo_Hinh.py       # Train/Test Split, Walk-Forward Validation
│   ├── 9_Xu_Ly_Du_Lieu.py          # Outlier IQR/Z-score, nội suy dữ liệu thiếu
│   └── 10_So_Sanh_Tong_Hop.py      # Bảng xếp hạng MAPE mọi phương pháp (kể cả từ trang Phân tích nhanh)
├── src/
│   ├── app_state.py                # Session state + registry so sánh phương pháp
│   ├── data_utils.py                # Nạp dữ liệu mẫu ERP_4SKU / CSV riêng (khuôn cố định)
│   ├── quick_data_utils.py          # Nạp dữ liệu TỰ DO: tự dò header/sheet/cột ngày cho trang Phân tích nhanh
│   ├── quick_sample_registry.py     # Danh mục 7 bộ dữ liệu mẫu của trang Phân tích nhanh
│   ├── ui_helpers.py                # Component chọn dữ liệu dùng chung (khuôn ERP_4SKU)
│   ├── metrics.py                   # ME, MAE, MSE, RMSE, MPE, MAPE, WMAPE
│   ├── pegels.py                    # Phân loại hình dạng nhu cầu
│   ├── naive_smoothing.py           # NF1, NF2, MA, WMA, Double-MA, SES, Holt, Holt-Winters
│   ├── arima_modeling.py            # ADF, SARIMA, grid search AIC, tìm nhanh SARIMA, Ljung-Box
│   ├── regression_modeling.py       # Linear/Quadratic/Seasonal Regression
│   ├── croston.py                   # Croston, SBA, TSB
│   ├── inventory.py                 # Safety Stock, ROP, chi phí giữ hàng
│   ├── validation.py                # Train/Test, Walk-Forward, Adjusted R²
│   ├── outliers.py                  # IQR, Z-score, nội suy
│   ├── diagnostics.py               # ACF/PACF
│   └── plotting.py                  # Các hàm vẽ Plotly dùng chung (kể cả decomposition/seasonal/backtest)
├── data/
│   ├── ERP_4SKU_sample.csv          # 4 SKU, 36 tháng — cùng bộ dữ liệu Lecture Notes dùng
│   └── quick_samples/               # 7 bộ dữ liệu mẫu độc lập cho trang Phân tích nhanh
├── assets/
│   └── pegels_3x3_grid.png
└── requirements.txt
```

## Nguyên tắc thiết kế

- **Mọi công thức khớp 1-1 với Lecture Notes và Python Instructions** của
  khóa học — dùng app để thực hành trực quan, rồi đối chiếu lại với Excel/
  Python đã làm trên lớp. Toàn bộ số liệu trong `src/` đã được kiểm chứng
  khớp với các con số MAPE/AIC/R² đã công bố trong Lecture Notes.
- **Tách biệt logic (`src/`) và giao diện (`pages/`)** — mỗi hàm trong `src/`
  có thể test độc lập bằng `python3 -c "from src.xxx import ...`, không cần
  chạy Streamlit.
- **Luôn chia dữ liệu THEO THỨ TỰ THỜI GIAN** — không có nơi nào trong app
  dùng `train_test_split()` ngẫu nhiên cho chuỗi thời gian (xem trang
  *Train/Test · Walk-Forward*).
- **Bảng So Sánh Tổng Hợp** tự động ghi nhận MAPE của mọi phương pháp đã
  chạy trong phiên làm việc — không cần copy tay từng con số, kể cả kết quả
  từ trang *Phân tích nhanh*.
- **Trang Phân tích nhanh dùng LẠI đúng các hàm trong `src/`** (Holt-Winters
  từ `naive_smoothing.py`, SARIMA từ `arima_modeling.py`, chỉ số sai số từ
  `metrics.py`...) — chỉ khác ở lớp nạp dữ liệu (`quick_data_utils.py`) và
  cách trình bày kết quả tổng hợp, để mọi con số luôn nhất quán với phần còn
  lại của app.

## Dữ liệu đầu vào tự tải lên

**Ở các trang theo khóa học (1–10):** CSV cần tối thiểu:

- một cột thời gian convert được sang `datetime`
- một cột giá trị convert được sang số

App sẽ tự động chuẩn hoá (convert datetime, sort tăng dần, ép kiểu số) ở mọi
trang thông qua `src/data_utils.py`.

**Ở trang ⚡ Phân tích nhanh:** chấp nhận cả Excel (`.xlsx`/`.xls`, kể cả
nhiều sheet — app tự chấm điểm và đề xuất sheet có dữ liệu chuỗi thời gian
phù hợp nhất, tự bỏ qua các sheet "hướng dẫn") và CSV, tự dò dòng header
(không cần nằm ở dòng đầu tiên) và tự nhận diện cột thời gian ngay cả với
định dạng lạ như `"2014 JAN"`. Nếu chưa có file sẵn, trang này đi kèm 7 bộ dữ
liệu mẫu (`data/quick_samples/`) minh hoạ các tình huống khác nhau: mùa vụ
cộng/nhân, có biến giải thích, dữ liệu bị xáo trộn bởi COVID, dữ liệu ngày,
và file Excel có sheet hướng dẫn cần tự động bỏ qua.

