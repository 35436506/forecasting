"""Thành phần UI dùng chung: chọn dữ liệu (mẫu 4 SKU hoặc CSV riêng), chọn cột.

Mọi trang trong app gọi `select_series()` để lấy về một chuỗi thời gian đã
chuẩn hoá (DataFrame 2 cột: thời gian, giá trị) theo đúng MỘT quy trình,
tránh lặp lại logic chọn dữ liệu ở 10 trang khác nhau.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from src.data_utils import (
    SAMPLE_SKU_INFO, load_sample_dataset, load_csv_data, prepare_time_series,
)


def select_series(key_prefix: str, help_text: str | None = None) -> tuple[pd.DataFrame, str, str] | None:
    """Hiển thị bộ chọn dữ liệu chuẩn. Trả về (df chuẩn hoá 2 cột, tên thời
    gian, tên giá trị) hoặc None nếu người dùng chưa chọn xong."""
    st.subheader("📂 Chọn dữ liệu")
    if help_text:
        st.caption(help_text)

    source = st.radio(
        "Nguồn dữ liệu", options=["Dùng SKU mẫu của khóa học", "Tải lên CSV riêng"],
        horizontal=True, key=f"{key_prefix}_source",
    )

    if source == "Dùng SKU mẫu của khóa học":
        try:
            raw_df = load_sample_dataset()
        except ValueError as error:
            st.error(str(error))
            return None

        sku_options = list(SAMPLE_SKU_INFO.keys())
        sku_labels = [f"{SAMPLE_SKU_INFO[k]['label']} ({SAMPLE_SKU_INFO[k]['shape']})" for k in sku_options]
        chosen_idx = st.selectbox(
            "Chọn SKU", options=range(len(sku_options)), format_func=lambda i: sku_labels[i],
            key=f"{key_prefix}_sku",
        )
        chosen_sku = sku_options[chosen_idx]
        st.info(f"💡 {SAMPLE_SKU_INFO[chosen_sku]['note']}")

        prepared = prepare_time_series(raw_df, "Thang", chosen_sku)
        return prepared, "Thang", chosen_sku

    uploaded_file = st.file_uploader("Tải file CSV (cần cột thời gian và cột giá trị số)", type=["csv"],
                                      key=f"{key_prefix}_upload")
    if uploaded_file is None:
        st.caption("Chưa có file — hãy tải lên một CSV để tiếp tục.")
        return None

    try:
        raw_df = load_csv_data(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        return None

    columns = raw_df.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        time_column = st.selectbox("Cột thời gian", columns, key=f"{key_prefix}_time_col")
    with col2:
        value_options = [c for c in columns if c != time_column]
        value_column = st.selectbox("Cột giá trị", value_options, key=f"{key_prefix}_value_col")

    try:
        prepared = prepare_time_series(raw_df, time_column, value_column)
    except ValueError as error:
        st.error(str(error))
        return None

    return prepared, time_column, value_column


def show_series_preview(df: pd.DataFrame, n_rows: int = 5) -> None:
    with st.expander("Xem trước dữ liệu", expanded=False):
        st.dataframe(df.head(n_rows), width="stretch")
        st.caption(f"Tổng {len(df)} dòng.")
