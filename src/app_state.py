"""Quản lý session_state của Streamlit — giữ kết quả model ổn định giữa các
lần rerun (ví dụ khi người dùng đổi widget khác không liên quan)."""
from __future__ import annotations

import streamlit as st

DATASET_KEY = "app_dataset"
DATASET_NAME_KEY = "app_dataset_name"
COLUMN_SELECTION_KEY = "app_column_selection"

MODEL_RESULTS_PREFIX = "model_results_"


def save_dataset(df, name: str) -> None:
    st.session_state[DATASET_KEY] = df
    st.session_state[DATASET_NAME_KEY] = name


def get_dataset():
    return st.session_state.get(DATASET_KEY), st.session_state.get(DATASET_NAME_KEY)


def save_model_result(page_key: str, **results) -> None:
    st.session_state[f"{MODEL_RESULTS_PREFIX}{page_key}"] = results


def get_model_result(page_key: str) -> dict | None:
    return st.session_state.get(f"{MODEL_RESULTS_PREFIX}{page_key}")


def clear_model_result(page_key: str) -> None:
    st.session_state.pop(f"{MODEL_RESULTS_PREFIX}{page_key}", None)


def register_method_mape(method_name: str, mape: float, sku_name: str) -> None:
    """Ghi lại MAPE của một phương pháp vào bảng tổng hợp dùng ở trang So Sánh."""
    key = "method_comparison_registry"
    registry = st.session_state.get(key, {})
    registry.setdefault(sku_name, {})[method_name] = mape
    st.session_state[key] = registry


def get_method_registry(sku_name: str) -> dict[str, float]:
    registry = st.session_state.get("method_comparison_registry", {})
    return registry.get(sku_name, {})
