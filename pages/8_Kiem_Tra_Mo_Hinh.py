import numpy as np
import streamlit as st

from src.ui_helpers import select_series, show_series_preview
from src.validation import chronological_split, walk_forward_validation
from src.naive_smoothing import fit_holt_winters
from src.metrics import calculate_metrics
from src.plotting import line_chart

st.title("🧪 Train/Test Split · Walk-Forward Validation")
st.caption("Chương 7 — con số đẹp trong mẫu chưa chắc đáng tin; luôn kiểm tra bằng dữ liệu CHƯA TỪNG thấy.")

st.error(
    "🚫 **KHÔNG BAO GIỜ** dùng hàm chia ngẫu nhiên (`train_test_split()`, `sample()`) cho chuỗi thời "
    "gian — luôn chia THEO THỨ TỰ THỜI GIAN. Xem Lecture Notes Chương 7, mục \"Cạm bẫy thường gặp\"."
)

result = select_series("traintest")
if result is None:
    st.stop()

df, time_col, value_col = result
show_series_preview(df)
y = df[value_col].values
n = len(y)

tab1, tab2 = st.tabs(["Một lần chia Train/Test", "Walk-Forward Validation (nhiều vòng)"])

with tab1:
    n_test = st.slider("Số kỳ dùng làm Test (đoạn CUỐI chuỗi)", 1, max(1, n // 3), min(6, max(1, n // 6)))
    season_length = st.number_input("Chu kỳ mùa vụ cho Holt-Winters", min_value=2, max_value=24, value=12)

    try:
        split = chronological_split(y, n_test=n_test)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    try:
        r = fit_holt_winters(split.train, season_length=season_length, future_steps=n_test)
        if r.forecast is None:
            raise ValueError("Không tạo được dự báo.")

        m_train = calculate_metrics(split.train[~np.isnan(r.fitted)], r.fitted[~np.isnan(r.fitted)])
        m_test = calculate_metrics(split.test, r.forecast)

        col1, col2 = st.columns(2)
        col1.metric("MAPE trong mẫu (Train)", f"{m_train.mape:.2f}%")
        col2.metric("MAPE ngoài mẫu (Test)", f"{m_test.mape:.2f}%")

        gap = m_test.mape - m_train.mape
        if gap > 5:
            st.warning(f"⚠️ Chênh lệch {gap:+.1f} điểm — mô hình có thể đang quá khớp vào giai đoạn Train, "
                       "hoặc cấu trúc dữ liệu thay đổi giữa hai giai đoạn.")
        else:
            st.success(f"✓ Chênh lệch chỉ {gap:+.1f} điểm — mô hình khá ổn định khi kiểm tra ngoài mẫu.")

        full_index = np.arange(n)
        train_index = full_index[:-n_test]
        test_index = full_index[-n_test:]
        fig = line_chart(
            df[time_col],
            {value_col: y, "Dự báo (Test)": np.concatenate([np.full(len(split.train), np.nan), r.forecast])},
            title="Train/Test Split", xaxis_title="Thời gian", yaxis_title=value_col,
        )
        st.plotly_chart(fig, width="stretch")

    except ValueError as error:
        st.error(str(error))

with tab2:
    st.caption("Rolling-origin: mở rộng cửa sổ Train dần, dự báo một đoạn ngắn, lặp lại qua nhiều điểm.")

    season_length_wf = st.number_input("Chu kỳ mùa vụ", min_value=2, max_value=24, value=12, key="wf_season")
    horizon = st.slider("Số kỳ dự báo mỗi vòng", 1, 6, 2)
    n_rounds = st.slider("Số vòng", 2, 6, 4)

    min_origin = season_length_wf * 2
    if n - min_origin < horizon * n_rounds:
        st.warning("Chuỗi không đủ dài để chạy đủ số vòng yêu cầu với chu kỳ mùa vụ này — giảm số vòng "
                   "hoặc horizon.")
    else:
        step = max(1, (n - min_origin - horizon) // max(1, n_rounds - 1))
        origins = [min_origin + i * step for i in range(n_rounds)]
        origins = [o for o in origins if o + horizon <= n]

        def _fit_forecast(y_train, h):
            r = fit_holt_winters(y_train, season_length=season_length_wf, future_steps=h)
            return r.forecast

        if st.button("▶️ Chạy Walk-Forward Validation", type="primary"):
            try:
                rounds = walk_forward_validation(y, _fit_forecast, origins, horizon)
                mapes = [rnd.mape for rnd in rounds]

                col1, col2 = st.columns(2)
                col1.metric("MAPE trung bình qua các vòng", f"{np.mean(mapes):.2f}%")
                col2.metric("Độ lệch chuẩn MAPE", f"{np.std(mapes):.2f}%")

                st.dataframe(
                    {"Vòng": list(range(1, len(rounds) + 1)),
                     "Origin (kỳ)": [rnd.origin for rnd in rounds],
                     "MAPE (%)": [round(rnd.mape, 2) for rnd in rounds]},
                    width="stretch",
                )
                st.info(
                    "💡 So sánh con số này với MAPE của một lần chia Train/Test duy nhất ở tab bên cạnh — "
                    "nếu chênh lệch đáng kể, nghĩa là một lần chia đơn lẻ có thể đã 'may mắn' hoặc "
                    "'không may mắn' (xem Lecture Notes Chương 7)."
                )
            except ValueError as error:
                st.error(str(error))
