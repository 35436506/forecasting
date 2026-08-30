"""Nạp dữ liệu TỰ DO (bất kỳ file Excel/CSV nào, không theo khuôn ERP_4SKU).

Trang *Phân tích nhanh* dùng module này để hỗ trợ người dùng tải lên một file
Excel/CSV BẤT KỲ — kể cả khi file có nhiều sheet (một số là "sheet hướng dẫn"
cần tự động bỏ qua), header không nằm ở dòng đầu tiên, hoặc cột thời gian ở
định dạng lạ như "2014 JAN". Đây là phần bổ sung cho `src/data_utils.py`
(vốn chỉ phục vụ đúng khuôn bộ mẫu ERP_4SKU hoặc CSV 2 cột đơn giản).

Toàn bộ hàm ở đây THUẦN pandas/numpy — không phụ thuộc Streamlit — theo đúng
quy ước tách UI khỏi logic của các module `src/*` khác trong app.
"""
from __future__ import annotations

import calendar
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PERIOD_LABELS = {12: "Tháng", 4: "Quý", 52: "Tuần", 7: "Ngày trong tuần", 1: "Kỳ"}

_MONTH_MAP: dict[str, int] = {m.upper(): i for i, m in enumerate(calendar.month_abbr) if m}
_MONTH_MAP.update({m.upper(): i for i, m in enumerate(calendar.month_name) if m})


# ──────────────────────────────────────────────────────────────────────────
# Đọc file — tự động dò dòng header và (khi là Excel nhiều sheet) chấm điểm
# từng sheet để đề xuất sheet phù hợp nhất.
# ──────────────────────────────────────────────────────────────────────────

def detect_header_row(raw_df: pd.DataFrame, max_scan: int = 10) -> int:
    """Dò dòng header thực sự: dòng có tỷ lệ text cao, NGAY SAU đó là dòng số."""
    n_cols = raw_df.shape[1]
    if n_cols == 0:
        return 0
    scan = min(max_scan, max(raw_df.shape[0] - 1, 0))
    for i in range(scan):
        row = raw_df.iloc[i]
        text_count = sum(
            1 for v in row
            if v is not None and not (isinstance(v, float) and np.isnan(v))
            and not isinstance(v, (int, float, np.integer, np.floating))
        )
        text_ratio = text_count / n_cols
        if i + 1 < raw_df.shape[0]:
            next_row = raw_df.iloc[i + 1]
            num_count = sum(
                1 for v in next_row
                if isinstance(v, (int, float, np.integer, np.floating))
                and not (isinstance(v, float) and np.isnan(v))
            )
            num_ratio = num_count / n_cols
        else:
            num_ratio = 0
        if text_ratio >= 0.5 and num_ratio >= 0.4:
            return i
    return 0


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() / max(len(df), 1) > 0.7:
                df[col] = converted
    return df


def load_tabular_bytes(file_bytes: bytes, file_name: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Đọc CSV hoặc Excel (một sheet cụ thể) từ bytes, tự dò dòng header."""
    name = file_name.lower()
    if name.endswith(".csv"):
        buf = io.BytesIO(file_bytes)
        try:
            raw = pd.read_csv(buf, header=None, nrows=15)
        except Exception:
            buf = io.BytesIO(file_bytes)
            raw = pd.read_csv(buf, header=None, nrows=15, encoding="latin1")
        header_row = detect_header_row(raw)
        buf = io.BytesIO(file_bytes)
        try:
            df = pd.read_csv(buf, header=header_row)
        except Exception:
            buf = io.BytesIO(file_bytes)
            df = pd.read_csv(buf, header=header_row, encoding="latin1")
    elif name.endswith((".xlsx", ".xls")):
        buf = io.BytesIO(file_bytes)
        xf = pd.ExcelFile(buf)
        sheet = sheet_name or xf.sheet_names[0]
        buf = io.BytesIO(file_bytes)
        raw = pd.read_excel(buf, sheet_name=sheet, header=None, nrows=15)
        header_row = detect_header_row(raw)
        buf = io.BytesIO(file_bytes)
        df = pd.read_excel(buf, sheet_name=sheet, header=header_row)
    else:
        raise ValueError(f"Không hỗ trợ định dạng file '{file_name}' — chỉ nhận .csv, .xlsx, .xls.")

    return _coerce_numeric_columns(df)


def list_excel_sheets(file_bytes: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def try_parse_date_column(col: pd.Series) -> pd.Series:
    """Ép một cột về datetime, chấp nhận cả các định dạng lạ như '2014 JAN'."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return pd.to_datetime(col)
    try:
        return pd.to_datetime(col, errors="raise", format="mixed")
    except Exception:
        pass

    def parse_one(v):
        s = str(v).strip().upper().replace("-", " ").replace("/", " ").replace(".", " ")
        parts = [p for p in s.split() if p]
        if len(parts) == 2:
            a, b = parts
            if a.isdigit() and b in _MONTH_MAP:
                return pd.Timestamp(year=int(a), month=_MONTH_MAP[b], day=1)
            if b.isdigit() and a in _MONTH_MAP:
                return pd.Timestamp(year=int(b), month=_MONTH_MAP[a], day=1)
            if a.isdigit() and b.isdigit() and len(a) == 4:
                try:
                    return pd.Timestamp(year=int(a), month=int(b), day=1)
                except Exception:
                    return pd.NaT
        return pd.NaT

    out = col.apply(parse_one)
    if out.notna().sum() / max(len(out), 1) > 0.8:
        return out
    try:
        return pd.to_datetime(col, errors="coerce", format="mixed")
    except Exception:
        return pd.to_datetime(col, errors="coerce")


def score_sheet(file_bytes: bytes, file_name: str, sheet: str) -> float:
    """Chấm điểm 0–100 cho một sheet: có cột ngày hợp lệ + cột số + đủ dòng
    thì điểm cao; sheet kiểu 'hướng dẫn/tổng quan' hầu như toàn chữ sẽ về 0.
    """
    try:
        df = load_tabular_bytes(file_bytes, file_name, sheet)
        if df is None or df.empty or len(df) < 4:
            return 0.0
        fill_ratio = df.notna().sum().sum() / max(df.size, 1)
        if fill_ratio < 0.3:
            return 0.0
        date_score = 0.0
        for col in df.columns:
            try:
                parsed = try_parse_date_column(df[col])
                ratio = parsed.notna().sum() / max(len(df), 1)
                if ratio > 0.7:
                    date_score = ratio
                    break
            except Exception:
                continue
        if date_score == 0:
            return 0.0
        num_cols = sum(
            1 for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() / max(len(df), 1) > 0.5
        )
        return float(date_score * 40 + min(num_cols, 5) * 10 + min(len(df) / 10, 2) * 5)
    except Exception:
        return 0.0


def pick_best_sheet(file_bytes: bytes, file_name: str) -> tuple[str, dict[str, float]]:
    sheets = list_excel_sheets(file_bytes)
    scores = {s: score_sheet(file_bytes, file_name, s) for s in sheets}
    best = max(scores, key=lambda s: scores[s])
    return best, scores


# ──────────────────────────────────────────────────────────────────────────
# Suy luận cột & chu kỳ mùa vụ, dựng chuỗi thời gian
# ──────────────────────────────────────────────────────────────────────────

def auto_detect_columns(df: pd.DataFrame) -> tuple[str | None, list[str]]:
    """Trả về (cột thời gian khả năng cao nhất, danh sách cột số còn lại)."""
    best_date_col, best_score = None, -1.0
    for col in df.columns:
        parsed = try_parse_date_column(df[col])
        score = parsed.notna().sum() / max(len(df), 1)
        if score > best_score:
            best_score, best_date_col = score, col

    numeric_cols = []
    for col in df.columns:
        if col == best_date_col:
            continue
        conv = pd.to_numeric(df[col], errors="coerce")
        if conv.notna().sum() / max(len(df), 1) > 0.7:
            numeric_cols.append(col)
    return best_date_col, numeric_cols


def infer_period_and_freq(idx: pd.DatetimeIndex) -> tuple[int, str]:
    """Suy luận chu kỳ mùa vụ (12=tháng, 4=quý...) và tần suất pandas."""
    if len(idx) < 3:
        return 12, "MS"
    diffs = (idx[1:] - idx[:-1]).days
    median_days = float(np.median(diffs))
    if 27 <= median_days <= 31:
        return 12, "MS"
    if 85 <= median_days <= 95:
        return 4, "QS"
    if 360 <= median_days <= 370:
        return 1, "AS"
    if 6 <= median_days <= 8:
        return 52, "W"
    if median_days == 1:
        return 7, "D"
    return 12, "MS"


def build_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    """Dựng một pd.Series đã sắp xếp tăng dần theo thời gian, loại trùng lặp/NaN."""
    dates = try_parse_date_column(df[date_col])
    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").values, index=dates, name=value_col)
    s = s[~s.index.isna()].dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def make_future_index(last_date: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    try:
        return pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
    except Exception:
        return pd.date_range(start=last_date, periods=periods + 1, freq="MS")[1:]


@dataclass
class SeasonalPivot:
    pivot: pd.DataFrame
    xlabel: str


def seasonal_pivot(series: pd.Series, period: int) -> SeasonalPivot:
    """Bảng chéo (period-in-year × năm) để vẽ biểu đồ mùa vụ chồng các năm."""
    tmp = series.reset_index()
    tmp.columns = ["date", "value"]
    tmp["year"] = tmp["date"].dt.year
    if period == 12:
        tmp["p"], xlabel = tmp["date"].dt.month, "Tháng"
    elif period == 4:
        tmp["p"], xlabel = tmp["date"].dt.quarter, "Quý"
    else:
        tmp["p"], xlabel = (np.arange(len(tmp)) % period) + 1, "Kỳ"
    pivot = tmp.pivot_table(index="p", columns="year", values="value", aggfunc="mean")
    return SeasonalPivot(pivot=pivot, xlabel=xlabel)


# ──────────────────────────────────────────────────────────────────────────
# Phân rã chuỗi thời gian — so sánh Cộng (Additive) vs Nhân (Multiplicative)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class DecompositionOutcome:
    mode: str
    status: str  # 'ok' | 'skip' | 'error'
    trend: pd.Series | None = None
    seasonal: pd.Series | None = None
    resid: pd.Series | None = None
    resid_pct: float | None = None  # do lech chuan phan du, chuan hoa ve % muc trung binh chuoi
    reason: str = ""


def run_decomposition(series: pd.Series, period: int, mode: str) -> DecompositionOutcome:
    """Phân rã bằng statsmodels.seasonal_decompose. `resid_pct` được CHUẨN HOÁ
    về cùng đơn vị % giữa hai mode (mode cộng: resid_std / mean; mode nhân:
    resid_std đã là tỷ lệ quanh 1, nhân 100) để có thể so sánh công bằng —
    Residual MSE thô của hai mode nằm trên hai thang đo khác nhau nên KHÔNG
    thể so sánh trực tiếp.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    if mode == "multiplicative" and (series <= 0).any():
        return DecompositionOutcome(mode=mode, status="skip",
                                     reason="Chuỗi có giá trị ≤ 0, không áp dụng được mô hình nhân.")
    try:
        res = seasonal_decompose(series, model=mode, period=period, extrapolate_trend="freq")
        resid = res.resid.dropna()
        resid_std = float(resid.std())
        series_mean = float(series.mean())
        resid_pct = (resid_std / series_mean * 100) if mode == "additive" else (resid_std * 100)
        return DecompositionOutcome(mode=mode, status="ok", trend=res.trend, seasonal=res.seasonal,
                                     resid=res.resid, resid_pct=resid_pct)
    except Exception as exc:
        return DecompositionOutcome(mode=mode, status="error", reason=str(exc))
