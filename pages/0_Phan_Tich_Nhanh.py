"""Phân tích nhanh — tải lên MỘT FILE BẤT KỲ (không cần theo khuôn ERP_4SKU)
và app tự động: dò cột ngày/giá trị, phân rã, so sánh 4 mô hình Holt-Winters,
tìm nhanh SARIMA, backtest và đưa ra khuyến nghị cuối cùng, kèm xuất Excel.

Trang này KHÔNG thay thế các trang mô hình hóa chi tiết (Chương 3-6) — nó là
một "trợ lý tự động" tổng hợp nhiều bước lại một chỗ cho dữ liệu ngoài
chương trình học, dùng lại đúng các công thức trong `src/` để kết quả luôn
nhất quán với phần còn lại của app.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from src.quick_data_utils import (
    PERIOD_LABELS, load_tabular_bytes, list_excel_sheets, pick_best_sheet,
    auto_detect_columns, build_series, infer_period_and_freq, make_future_index,
    seasonal_pivot, run_decomposition,
)
from src.quick_sample_registry import QUICK_SAMPLE_INFO, QUICK_SAMPLE_GROUPS, load_sample_bytes
from src.naive_smoothing import fit_holt_winters
from src.arima_modeling import determine_seasonal_diff_orders, sarima_quick_search, fit_sarima
from src.metrics import calculate_metrics
from src.validation import chronological_split
from src.plotting import (
    line_chart, forecast_chart_with_ci, decomposition_chart, seasonal_by_year_chart,
    metric_comparison_bar, backtest_chart,
)
from src.app_state import register_method_mape

st.title("⚡ Phân tích nhanh — dữ liệu tự do")
st.caption(
    "Tải lên MỘT FILE Excel/CSV bất kỳ (không cần đúng khuôn ERP_4SKU của khóa học) — app tự dò cột "
    "ngày/giá trị, tự chọn sheet phù hợp nếu file có nhiều sheet, rồi chạy toàn bộ quy trình: phân rã → "
    "Holt-Winters (4 biến thể) → SARIMA → backtest → khuyến nghị. Dùng đúng công thức trong `src/` như "
    "các trang khác của app."
)

HW_COMBOS: list[tuple[str, str]] = [("add", "add"), ("add", "mul"), ("mul", "add"), ("mul", "mul")]
_TREND_LABEL = {"add": "cộng", "mul": "nhân"}


def _fmt_date(ts: pd.Timestamp, period: int) -> str:
    return ts.strftime("%b %Y") if period in (12, 4, 1) else ts.strftime("%d-%m-%Y")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CHỌN DỮ LIỆU
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("1. Chọn dữ liệu")

source = st.radio(
    "Nguồn dữ liệu", ["📦 Dữ liệu mẫu nhanh (7 bộ)", "📤 Tải lên Excel/CSV của bạn"],
    horizontal=True, key="qa_source",
)

file_bytes: bytes | None = None
file_name: str | None = None
source_label = ""

if source == "📦 Dữ liệu mẫu nhanh (7 bộ)":
    for group_title, keys in QUICK_SAMPLE_GROUPS:
        st.markdown(f"**{group_title}**")
        cols = st.columns(len(keys))
        for col, key in zip(cols, keys):
            info = QUICK_SAMPLE_INFO[key]
            with col:
                st.caption(f"{info['label']} · _{info['tag']}_")
    sample_keys = list(QUICK_SAMPLE_INFO.keys())
    chosen_key = st.selectbox(
        "Chọn bộ dữ liệu mẫu", sample_keys,
        format_func=lambda k: f"{QUICK_SAMPLE_INFO[k]['label']} ({QUICK_SAMPLE_INFO[k]['tag']})",
        key="qa_sample_key",
    )
    st.info(f"💡 {QUICK_SAMPLE_INFO[chosen_key]['note']}")
    try:
        file_bytes = load_sample_bytes(chosen_key)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    file_name = QUICK_SAMPLE_INFO[chosen_key]["filename"]
    source_label = f"mẫu:{chosen_key}"

else:
    uploaded = st.file_uploader("Tải file Excel (.xlsx/.xls) hoặc CSV", type=["xlsx", "xls", "csv"], key="qa_upload")
    if uploaded is None:
        st.caption("Chưa có file — hãy tải lên để tiếp tục.")
        st.stop()
    file_bytes = uploaded.read()
    file_name = uploaded.name
    source_label = f"upload:{uploaded.name}"

# ---- Chọn sheet (nếu là Excel nhiều sheet) ----
sheet_name = None
if file_name.lower().endswith((".xlsx", ".xls")):
    try:
        sheets = list_excel_sheets(file_bytes)
    except Exception as error:
        st.error(f"Không đọc được file Excel: {error}")
        st.stop()
    if len(sheets) == 1:
        sheet_name = sheets[0]
    else:
        with st.spinner("Đang chấm điểm các sheet để đề xuất sheet phù hợp nhất..."):
            best_sheet, scores = pick_best_sheet(file_bytes, file_name)
        best_idx = sheets.index(best_sheet)
        skipped = [s for s in sheets if scores.get(s, 0) == 0]
        sheet_name = st.selectbox(
            "📑 Sheet", sheets, index=best_idx, key="qa_sheet",
            help=f"Tự động đề xuất sheet '{best_sheet}' — điểm phù hợp cao nhất (có cột ngày + cột số).",
        )
        if skipped:
            st.caption(f"⚡ Đã tự động bỏ qua sheet không có dữ liệu chuỗi thời gian: {', '.join(skipped)}.")

try:
    raw_df = load_tabular_bytes(file_bytes, file_name, sheet_name)
except Exception as error:
    st.error(f"Không đọc được dữ liệu: {error}")
    st.stop()

if raw_df.empty:
    st.error("File không có dữ liệu.")
    st.stop()

with st.expander("Xem trước dữ liệu thô", expanded=False):
    st.dataframe(raw_df.head(15), width="stretch")
    st.caption(f"Tổng {len(raw_df)} dòng, {len(raw_df.columns)} cột.")

auto_date_col, auto_numeric_cols = auto_detect_columns(raw_df)
all_cols = raw_df.columns.tolist()
if auto_date_col is None or not auto_numeric_cols:
    st.error("Không tìm được cột thời gian hoặc cột số hợp lệ trong dữ liệu này — thử chọn sheet/file khác.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    date_col = st.selectbox("Cột thời gian", all_cols,
                             index=all_cols.index(auto_date_col) if auto_date_col in all_cols else 0,
                             key="qa_date_col")
with col2:
    numeric_candidates = [c for c in all_cols if c != date_col]
    default_value = auto_numeric_cols[0] if auto_numeric_cols[0] != date_col else (
        auto_numeric_cols[1] if len(auto_numeric_cols) > 1 else numeric_candidates[0]
    )
    value_col = st.selectbox("Cột cần dự báo (Y)", numeric_candidates,
                              index=numeric_candidates.index(default_value) if default_value in numeric_candidates else 0,
                              key="qa_value_col")

other_numeric = [c for c in numeric_candidates if c != value_col
                 and pd.to_numeric(raw_df[c], errors="coerce").notna().sum() / max(len(raw_df), 1) > 0.7]
if other_numeric:
    st.caption(
        f"ℹ️ File này còn có cột số khác ({', '.join(f'`{c}`' for c in other_numeric)}) — nếu muốn dùng chúng "
        "làm biến giải thích cho hồi quy đa biến, ghé trang **Regression (Trend · Mùa vụ · Promo)**."
    )

try:
    series_raw = build_series(raw_df, date_col, value_col)
except Exception as error:
    st.error(f"Không dựng được chuỗi thời gian: {error}")
    st.stop()

auto_period, auto_freq = infer_period_and_freq(series_raw.index) if len(series_raw) >= 3 else (12, "MS")

if len(series_raw) < max(2 * auto_period, 8):
    st.error(
        f"Chuỗi chỉ có {len(series_raw)} quan sát — cần tối thiểu {max(2 * auto_period, 8)} "
        "(≈2 chu kỳ mùa vụ) để phân tích đáng tin cậy. Hãy chọn cột/sheet khác hoặc dữ liệu dài hơn."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# 2. THIẾT LẬP
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("2. Thiết lập dự báo")

freq_options = ["MS", "M", "QS", "Q", "W", "D", "AS"]
col1, col2, col3, col4 = st.columns(4)
with col1:
    period = st.number_input("Chu kỳ mùa vụ (s)", min_value=1, max_value=365, value=int(auto_period),
                              key="qa_period",
                              help="Số kỳ trong một chu kỳ mùa vụ — 12 cho dữ liệu tháng, 4 cho dữ liệu quý, "
                                   "7 cho dữ liệu theo ngày trong tuần.")
with col2:
    horizon = st.number_input("Số kỳ dự báo", min_value=1, max_value=90, value=int(period), key="qa_horizon")
with col3:
    test_size = st.number_input("Số kỳ Backtest (Test set)", min_value=2, max_value=90, value=int(period),
                                 key="qa_test_size")
with col4:
    freq = st.selectbox("Tần suất (freq)", freq_options,
                         index=freq_options.index(auto_freq) if auto_freq in freq_options else 0, key="qa_freq")

PLABEL = PERIOD_LABELS.get(period, "Kỳ")

series = series_raw.copy()
try:
    series = series.asfreq(freq)
    if series.isna().any():
        series = series.interpolate()
except Exception:
    series = series_raw

y = series.values
n = len(y)

# reset cac ket qua nang (SARIMA/backtest) khi chuoi du lieu thay doi
series_sig = f"{source_label}|{value_col}|{n}|{series.index[0]}|{series.index[-1]}|{period}|{freq}"
if st.session_state.get("qa_series_sig") != series_sig:
    st.session_state["qa_series_sig"] = series_sig
    for k in ("qa_sarima_results", "qa_sarima_best", "qa_sarima_diag", "qa_backtest"):
        st.session_state.pop(k, None)

# ═══════════════════════════════════════════════════════════════════════════
# 3. TỔNG QUAN
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("3. Tổng quan dữ liệu")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Số quan sát", n)
c2.metric("Từ", series.index.min().strftime("%b %Y"))
c3.metric("Đến", series.index.max().strftime("%b %Y"))
c4.metric("Chu kỳ mùa vụ", f"{period} ({PLABEL})")

fig = line_chart(series.index, {value_col: y}, title=f"Time plot — {value_col}",
                  xaxis_title="Thời gian", yaxis_title=value_col)
st.plotly_chart(fig, width="stretch")

first_q = float(np.mean(y[:max(period, 1)]))
last_q = float(np.mean(y[-max(period, 1):]))
trend_pct = (last_q - first_q) / abs(first_q) * 100 if first_q != 0 else np.nan
trend_word = "tăng" if trend_pct > 1 else ("giảm" if trend_pct < -1 else "ổn định")
st.info(
    f"📊 So sánh trung bình {period} kỳ đầu và {period} kỳ cuối, chuỗi **{trend_word}** khoảng "
    f"**{trend_pct:+.1f}%**. Trung bình toàn chuỗi ≈ **{np.mean(y):,.2f}**, độ lệch chuẩn ≈ **{np.std(y):,.2f}**."
)

# ═══════════════════════════════════════════════════════════════════════════
# 4. PHÂN RÃ
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("4. Phân rã chuỗi thời gian (Decomposition)")

decomp_add = run_decomposition(series, period, "additive")
decomp_mul = run_decomposition(series, period, "multiplicative")
valid_decomps = [d for d in (decomp_add, decomp_mul) if d.status == "ok"]

if not valid_decomps:
    st.warning("Không thể phân rã chuỗi này (quá ngắn hoặc lỗi dữ liệu).")
else:
    cols = st.columns(len(valid_decomps))
    for col, d in zip(cols, valid_decomps):
        with col:
            mode_label = "Cộng (Additive)" if d.mode == "additive" else "Nhân (Multiplicative)"
            fig = decomposition_chart(d.trend.index, d.trend.values, d.seasonal.values, d.resid.values,
                                       title=mode_label)
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Phần dư ≈ **{d.resid_pct:.2f}%** so với mức trung bình chuỗi (đã chuẩn hoá).")
    if decomp_mul.status != "ok":
        st.info(f"ℹ️ Mô hình nhân không khả dụng: {decomp_mul.reason}")

    if len(valid_decomps) == 2:
        recommended = "multiplicative" if decomp_mul.resid_pct < decomp_add.resid_pct else "additive"
        rec_vn = "nhân (multiplicative)" if recommended == "multiplicative" else "cộng (additive)"
        st.markdown(
            f"🧩 **Phân tích:** phần dư mô hình cộng ≈ **{decomp_add.resid_pct:.2f}%**, mô hình nhân ≈ "
            f"**{decomp_mul.resid_pct:.2f}%** (đã chuẩn hoá về % mức trung bình chuỗi để so sánh công bằng — "
            "phần dư cộng là giá trị tuyệt đối quanh 0, phần dư nhân là tỷ số quanh 1 nên KHÔNG so sánh "
            f"trực tiếp được nếu để thô). Mô hình **{rec_vn}** có phần dư nhỏ hơn → phù hợp hơn với "
            f"`{value_col}`. Nếu biên độ mùa vụ TĂNG THEO mức xu hướng, mô hình nhân thường hợp lý hơn; "
            "nếu biên độ ổn định, mô hình cộng phù hợp hơn."
        )

# ═══════════════════════════════════════════════════════════════════════════
# 5. MÙA VỤ THEO NĂM
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("5. Biểu đồ mùa vụ theo năm")

sp = seasonal_pivot(series, period)
fig = seasonal_by_year_chart(sp.pivot, sp.xlabel, title=f"Seasonality of {value_col}")
st.plotly_chart(fig, width="stretch")

period_means = sp.pivot.mean(axis=1)
peak_p, trough_p = int(period_means.idxmax()), int(period_means.idxmin())
st.info(
    f"📅 Trung bình theo {sp.xlabel.lower()}, giá trị **cao nhất** thường rơi vào {sp.xlabel.lower()} "
    f"**{peak_p}** (≈{period_means.max():,.2f}), **thấp nhất** vào {sp.xlabel.lower()} **{trough_p}** "
    f"(≈{period_means.min():,.2f})."
)

# ═══════════════════════════════════════════════════════════════════════════
# 6. HOLT-WINTERS — 4 MÔ HÌNH
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("6. San bằng số mũ — Holt-Winters (4 biến thể)")

hw_rows = []
for trend, seasonal in HW_COMBOS:
    label = f"Xu hướng {_TREND_LABEL[trend]} / Mùa vụ {_TREND_LABEL[seasonal]}"
    try:
        r = fit_holt_winters(y, season_length=period, trend=trend, seasonal=seasonal, future_steps=int(horizon))
        valid = ~np.isnan(r.fitted)
        m = calculate_metrics(y[valid], r.fitted[valid])
        hw_rows.append({"trend": trend, "seasonal": seasonal, "label": label, "status": "ok",
                         "mse": m.mse, "mape": m.mape, "params": r.params, "forecast": r.forecast})
    except (ValueError, Exception) as error:  # statsmodels co the nem loi khac ValueError
        hw_rows.append({"trend": trend, "seasonal": seasonal, "label": label, "status": "error",
                         "reason": str(error)})

hw_table = pd.DataFrame([{
    "Loại": r["label"],
    "α (level)": round(r["params"]["alpha"], 4) if r["status"] == "ok" else "—",
    "β (trend)": round(r["params"]["beta"], 4) if r["status"] == "ok" else "—",
    "γ (seasonal)": round(r["params"]["gamma"], 4) if r["status"] == "ok" else "—",
    "MSE": round(r["mse"], 4) if r["status"] == "ok" else f"⚠ {r.get('reason', 'lỗi')[:60]}",
} for r in hw_rows])

hw_ok = [r for r in hw_rows if r["status"] == "ok"]
if not hw_ok:
    st.error("Không có biến thể Holt-Winters nào chạy được với chuỗi này.")
    st.stop()

best_hw = min(hw_ok, key=lambda r: r["mse"])


def _highlight_best_hw(row):
    is_best = row["Loại"] == best_hw["label"]
    return ["background-color:#d4f7dc;font-weight:700" if is_best else "" for _ in row]


st.dataframe(hw_table.style.apply(_highlight_best_hw, axis=1), width="stretch", hide_index=True)
st.success(f"🏆 Tốt nhất: **{best_hw['label']}** — MSE = **{best_hw['mse']:,.4f}**.")

hw_future_index = make_future_index(series.index[-1], int(horizon), freq)
fig = forecast_chart_with_ci(
    series.index, y, hw_future_index, best_hw["forecast"],
    title=f"Dự báo Holt-Winters — {best_hw['label']}", xaxis_title="Thời gian", yaxis_title=value_col,
)
st.plotly_chart(fig, width="stretch")

with st.expander(f"📋 Bảng dự báo ({int(horizon)} kỳ tới) — Holt-Winters"):
    st.dataframe(pd.DataFrame({
        "Date": [_fmt_date(d, period) for d in hw_future_index],
        "Forecast": np.round(best_hw["forecast"], 2),
    }), width="stretch", hide_index=True)

register_method_mape(f"Holt-Winters ({best_hw['label']})", best_hw["mape"], value_col)

# ═══════════════════════════════════════════════════════════════════════════
# 7. SARIMA — TÌM NHANH
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("7. ARIMA / SARIMA — tìm kiếm nhanh")
st.caption(
    "Quy trình rút gọn: (1) kiểm định ADF chọn bậc sai phân d, D tự động, (2) quét NHANH một danh sách nhỏ "
    "(p,q) — dùng cùng p,q cho phần mùa vụ P,Q để có ngay vài ứng viên hợp lý. Muốn kiểm soát P,Q độc lập "
    "và xem bản đồ nhiệt AIC đầy đủ, dùng trang **ARIMA / SARIMA Lab**."
)

if st.button("🔍 Chạy tìm kiếm SARIMA nhanh", type="primary", key="qa_run_sarima"):
    with st.spinner("Đang kiểm định ADF và quét các ứng viên SARIMA..."):
        d, D, adf_info = determine_seasonal_diff_orders(y, period)
        results = sarima_quick_search(y, period, d, D)
    st.session_state["qa_sarima_diag"] = (d, D, adf_info)
    st.session_state["qa_sarima_results"] = results

if "qa_sarima_diag" in st.session_state:
    d, D, adf_info = st.session_state["qa_sarima_diag"]
    adf_rows = [{"Chuỗi": "Dữ liệu gốc", "ADF p-value": round(adf_info["original"], 4),
                 "Dừng?": "✅ Có" if adf_info["original"] <= 0.05 else "❌ Không"}]
    if "seasonal_diff" in adf_info:
        adf_rows.append({"Chuỗi": f"Sai phân mùa vụ (lag {period})", "ADF p-value": round(adf_info["seasonal_diff"], 4),
                          "Dừng?": "✅ Có" if adf_info["seasonal_diff"] <= 0.05 else "❌ Không"})
    if "seasonal_first_diff" in adf_info:
        adf_rows.append({"Chuỗi": "Sai phân mùa vụ + bậc 1", "ADF p-value": round(adf_info["seasonal_first_diff"], 4),
                          "Dừng?": "✅ Có" if adf_info["seasonal_first_diff"] <= 0.05 else "❌ Không"})
    st.dataframe(pd.DataFrame(adf_rows), width="stretch", hide_index=True)
    st.caption(f"⇒ Bậc sai phân được chọn: **d = {d}**, **D = {D}** (chu kỳ mùa vụ s = {period}).")

    results = st.session_state["qa_sarima_results"]
    sarima_ok = [r for r in results if r["status"] == "ok" and not np.isnan(r["aic"])]

    if not sarima_ok:
        st.error("Không có tổ hợp SARIMA nào hội tụ với chuỗi này.")
    else:
        sarima_table = pd.DataFrame([{
            "order (p,d,q)": str(r["order"]), "seasonal_order (P,D,Q,s)": str(r["seasonal_order"]),
            "AIC": round(r["aic"], 2), "MSE (khớp mẫu)": round(r["mse"], 4) if not np.isnan(r["mse"]) else None,
        } for r in results if r["status"] == "ok"])

        best_sarima = min(sarima_ok, key=lambda r: r["aic"])

        def _highlight_best_sarima(row):
            is_best = (row["order (p,d,q)"] == str(best_sarima["order"])
                       and row["seasonal_order (P,D,Q,s)"] == str(best_sarima["seasonal_order"]))
            return ["background-color:#d4f7dc;font-weight:700" if is_best else "" for _ in row]

        st.dataframe(sarima_table.style.apply(_highlight_best_sarima, axis=1), width="stretch", hide_index=True)
        st.success(
            f"🏆 Tốt nhất: SARIMA{best_sarima['order']}{best_sarima['seasonal_order']} — "
            f"AIC = **{best_sarima['aic']:,.2f}**."
        )
        st.session_state["qa_sarima_best"] = best_sarima

        with st.spinner("Đang huấn luyện lại mô hình tốt nhất và dự báo..."):
            try:
                sarima_fit = fit_sarima(y, order=best_sarima["order"], seasonal_order=best_sarima["seasonal_order"],
                                         future_steps=int(horizon), ci_alpha=0.05)
                st.session_state["qa_sarima_fit"] = sarima_fit
            except ValueError as error:
                st.error(str(error))
                st.session_state.pop("qa_sarima_fit", None)

        if "qa_sarima_fit" in st.session_state:
            sarima_fit = st.session_state["qa_sarima_fit"]
            sarima_future_index = make_future_index(series.index[-1], int(horizon), freq)
            fig = forecast_chart_with_ci(
                series.index, y, sarima_future_index, sarima_fit.forecast_mean,
                sarima_fit.forecast_ci_lower, sarima_fit.forecast_ci_upper,
                title=f"Dự báo SARIMA{best_sarima['order']}{best_sarima['seasonal_order']}",
                xaxis_title="Thời gian", yaxis_title=value_col,
            )
            st.plotly_chart(fig, width="stretch")

            with st.expander(f"📋 Bảng dự báo ({int(horizon)} kỳ tới) — SARIMA"):
                st.dataframe(pd.DataFrame({
                    "Date": [_fmt_date(d_, period) for d_ in sarima_future_index],
                    "Forecast": np.round(sarima_fit.forecast_mean, 2),
                    "Lower CI": np.round(sarima_fit.forecast_ci_lower, 2),
                    "Upper CI": np.round(sarima_fit.forecast_ci_upper, 2),
                }), width="stretch", hide_index=True)

            valid_resid = ~np.isnan(sarima_fit.fitted)
            m_sarima = calculate_metrics(y[valid_resid], sarima_fit.fitted[valid_resid])
            register_method_mape(f"SARIMA{best_sarima['order']}{best_sarima['seasonal_order']}",
                                  m_sarima.mape, value_col)
            st.caption(
                "💬 Dải tô màu là khoảng tin cậy 95% — thường MỞ RỘNG DẦN theo thời gian vì độ bất định "
                "tăng lên khi dự báo xa hơn vào tương lai."
            )
else:
    st.caption("Chưa chạy tìm kiếm SARIMA — bấm nút phía trên để bắt đầu (kết quả sẽ được giữ lại khi bạn "
               "tương tác với các phần khác của trang).")

# ═══════════════════════════════════════════════════════════════════════════
# 8. BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("8. Backtest: Holt-Winters vs SARIMA")

has_sarima = "qa_sarima_best" in st.session_state and "qa_sarima_fit" in st.session_state

if n <= test_size + period:
    st.warning("Chuỗi quá ngắn so với kích thước Test set đã chọn — bỏ qua backtest.")
elif not has_sarima:
    st.info("Hãy chạy **tìm kiếm SARIMA** ở mục 7 trước để có đủ 2 phương pháp cho backtest.")
else:
    if st.button("▶ Chạy Backtest", type="primary", key="qa_run_backtest"):
        split = chronological_split(y, int(test_size))
        train_x = series.index[: split.n_train]
        test_x = series.index[split.n_train:]
        best_sarima = st.session_state["qa_sarima_best"]

        backtest_result = {"train_x": train_x, "test_x": test_x, "test_y": split.test, "forecasts": {}}

        try:
            hw_train = fit_holt_winters(split.train, season_length=period, trend=best_hw["trend"],
                                         seasonal=best_hw["seasonal"], future_steps=int(test_size))
            hw_test_mse = float(calculate_metrics(split.test, hw_train.forecast).mse)
            backtest_result["forecasts"][f"Holt-Winters ({best_hw['label']})"] = (hw_train.forecast, hw_test_mse)
        except Exception as error:
            backtest_result["hw_error"] = str(error)

        try:
            sarima_train = fit_sarima(split.train, order=best_sarima["order"],
                                       seasonal_order=best_sarima["seasonal_order"], future_steps=int(test_size))
            sarima_test_mse = float(calculate_metrics(split.test, sarima_train.forecast_mean).mse)
            label = f"SARIMA{best_sarima['order']}{best_sarima['seasonal_order']}"
            backtest_result["forecasts"][label] = (sarima_train.forecast_mean, sarima_test_mse)
        except Exception as error:
            backtest_result["sarima_error"] = str(error)

        st.session_state["qa_backtest"] = backtest_result

    if "qa_backtest" in st.session_state:
        bt = st.session_state["qa_backtest"]
        if bt.get("hw_error"):
            st.warning(f"Holt-Winters lỗi trên tập train backtest: {bt['hw_error']}")
        if bt.get("sarima_error"):
            st.warning(f"SARIMA lỗi trên tập train backtest: {bt['sarima_error']}")

        if bt["forecasts"]:
            fig = backtest_chart(
                bt["train_x"], y[: len(bt["train_x"])], bt["test_x"], bt["test_y"],
                {name: fc for name, (fc, _) in bt["forecasts"].items()},
                title=f"Backtest {int(test_size)} kỳ cuối",
            )
            st.plotly_chart(fig, width="stretch")

            names = list(bt["forecasts"].keys())
            mses = [mse for _, mse in bt["forecasts"].values()]
            bt_table = pd.DataFrame({"Phương pháp": names, "MSE trên Test set": [round(v, 4) for v in mses]})
            st.dataframe(bt_table, width="stretch", hide_index=True)

            fig2 = metric_comparison_bar(names, mses, title="So sánh MSE trên Test set (thấp hơn = tốt hơn)")
            st.plotly_chart(fig2, width="stretch")

            if len(names) == 2:
                winner_idx = int(np.argmin(mses))
                winner_name = names[winner_idx]
                loser_idx = 1 - winner_idx
                st.success(
                    f"🥇 **Kết luận backtest:** trên {int(test_size)} kỳ gần nhất, **{winner_name}** cho MSE "
                    f"thấp hơn (**{mses[winner_idx]:,.4f}** so với **{mses[loser_idx]:,.4f}** của "
                    f"{names[loser_idx]}). SARIMA thường vượt trội cho dự báo ngắn hạn vì tập trung khai "
                    "thác diễn biến gần nhất, trong khi Holt-Winters thích hợp hơn khi xu hướng/mùa vụ dài "
                    "hạn ổn định, ít bị nhiễu bởi biến động gần đây."
                )
                st.session_state["qa_winner"] = winner_name
                st.session_state["qa_winner_mse"] = mses[winner_idx]

# ═══════════════════════════════════════════════════════════════════════════
# 9. KHUYẾN NGHỊ CUỐI CÙNG & XUẤT EXCEL
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("9. 🎯 Khuyến nghị cuối cùng & xuất Excel")

winner_name = st.session_state.get("qa_winner")
if winner_name is None:
    st.info(
        "Chạy xong mục **8. Backtest** ở trên để app đưa ra khuyến nghị dựa trên sai số kiểm định khách "
        "quan. Trong lúc chờ, bạn vẫn có thể dùng kết quả Holt-Winters hoặc SARIMA ở các mục 6–7 và xuất "
        "Excel bên dưới."
    )
else:
    st.success(f"🏆 **Khuyến nghị: dùng {winner_name} để dự báo `{value_col}` cho {int(horizon)} "
               f"{PLABEL.lower()} tới** — MSE backtest = **{st.session_state['qa_winner_mse']:,.4f}** "
               "(thấp nhất trong các phương pháp đã thử).")

# ---- Xuat Excel ----
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    fc_df = pd.DataFrame(index=hw_future_index)
    fc_df.index.name = "Date"
    fc_df["Holt-Winters"] = best_hw["forecast"]
    if "qa_sarima_fit" in st.session_state:
        sarima_fit = st.session_state["qa_sarima_fit"]
        sarima_future_index = make_future_index(series.index[-1], int(horizon), freq)
        fc_df = fc_df.reindex(sarima_future_index.union(hw_future_index))
        fc_df["SARIMA"] = pd.Series(sarima_fit.forecast_mean, index=sarima_future_index)
    fc_df.to_excel(writer, sheet_name="Forecast")

    hw_table.to_excel(writer, sheet_name="HW model details", index=False)

    if "qa_sarima_results" in st.session_state:
        sarima_export_rows = [{
            "order (p,d,q)": str(r["order"]), "seasonal_order (P,D,Q,s)": str(r["seasonal_order"]),
            "AIC": round(r["aic"], 2) if not np.isnan(r["aic"]) else None,
            "MSE": round(r["mse"], 4) if not np.isnan(r["mse"]) else None, "Status": r["status"],
        } for r in st.session_state["qa_sarima_results"]]
        pd.DataFrame(sarima_export_rows).to_excel(writer, sheet_name="SARIMA model details", index=False)

    if "qa_backtest" in st.session_state and st.session_state["qa_backtest"]["forecasts"]:
        bt = st.session_state["qa_backtest"]
        bt_export = pd.DataFrame({"Phương pháp": list(bt["forecasts"].keys()),
                                    "MSE trên Test set": [round(v, 4) for _, v in bt["forecasts"].values()]})
        bt_export.to_excel(writer, sheet_name="Backtest comparison", index=False)

    series.to_frame(name=value_col).to_excel(writer, sheet_name="Data")

st.download_button(
    "⬇️ Tải xuống Excel (kết quả đầy đủ)", data=excel_buf.getvalue(),
    file_name=f"PhanTichNhanh_{str(value_col)[:20].strip().replace(' ', '_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="qa_download",
)
st.caption(
    "File Excel gồm: dự báo Holt-Winters/SARIMA, chi tiết 4 mô hình Holt-Winters, chi tiết các ứng viên "
    "SARIMA đã thử (nếu đã chạy mục 7), so sánh MSE backtest (nếu đã chạy mục 8), và dữ liệu gốc."
)
