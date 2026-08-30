"""7 bộ dữ liệu mẫu 'chạy nhanh' cho trang Phân tích nhanh — độc lập với bộ
mẫu ERP_4SKU của khóa học (dùng ở `src/data_utils.py`). Các file thật nằm
trong `data/quick_samples/` (đã giải mã sẵn từ bản demo gốc), nhóm theo mục
đích minh họa để người dùng chọn nhanh.
"""
from __future__ import annotations

from pathlib import Path

QUICK_SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "quick_samples"

QUICK_SAMPLE_INFO: dict[str, dict[str, str]] = {
    "additive": {
        "label": "🏠 Doanh số đồ gia dụng",
        "tag": "Additive",
        "filename": "Doanh_so_do_gia_dung_Additive.xlsx",
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "note": "Mùa vụ dao động ỔN ĐỊNH quanh xu hướng — ví dụ kinh điển cho phân rã CỘNG.",
    },
    "multiplicative": {
        "label": "⚡ Sản lượng điện",
        "tag": "Multiplicative",
        "filename": "San_luong_dien_Multiplicative.xlsx",
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "note": "Biên độ mùa vụ TĂNG THEO mức xu hướng — ví dụ kinh điển cho phân rã NHÂN.",
    },
    "regression": {
        "label": "🏢 Giá thuê văn phòng",
        "tag": "Regression",
        "filename": "Gia_thue_van_phong_Regression.xlsx",
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "note": "Có kèm biến giải thích khác — phù hợp để thử phần hồi quy đa biến.",
    },
    "covid": {
        "label": "✈️ Khách du lịch (giai đoạn COVID)",
        "tag": "Dữ liệu xáo trộn",
        "filename": "Khach_du_lich_Messy_COVID.xlsx",
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "note": "Có cú sốc bất thường (COVID) — thử xem các mô hình ngoại suy phản ứng thế nào.",
    },
    "airpassengers": {
        "label": "🛫 Air Passengers",
        "tag": "Box-Jenkins kinh điển",
        "filename": "AirPassengers.csv",
        "mime": "text/csv",
        "note": "Bộ dữ liệu Box-Jenkins nổi tiếng nhất trong giáo trình chuỗi thời gian.",
    },
    "dailytemp": {
        "label": "🌡️ Daily Min Temperatures (Melbourne)",
        "tag": "Dữ liệu ngày",
        "filename": "daily-min-temperatures.csv",
        "mime": "text/csv",
        "note": "10 năm dữ liệu THEO NGÀY — thử nghiệm chu kỳ mùa vụ khác 12/4.",
    },
    "multireg": {
        "label": "📈 Time Series Regression Dataset",
        "tag": "Nhiều biến X",
        "filename": "Time_Series_Regression_Dataset.xlsx",
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "note": "File Excel có SHEET HƯỚNG DẪN đứng trước — minh họa tính năng tự dò sheet phù hợp.",
    },
}

QUICK_SAMPLE_GROUPS: list[tuple[str, list[str]]] = [
    ("📦 Theo phương pháp phân rã", ["additive", "multiplicative", "regression", "covid"]),
    ("🌍 Dữ liệu quốc tế (CSV)", ["airpassengers", "dailytemp"]),
    ("📊 Hồi quy đa biến", ["multireg"]),
]


def sample_path(key: str) -> Path:
    info = QUICK_SAMPLE_INFO[key]
    return QUICK_SAMPLES_DIR / info["filename"]


def load_sample_bytes(key: str) -> bytes:
    path = sample_path(key)
    if not path.exists():
        raise ValueError(f"Không tìm thấy file dữ liệu mẫu '{path.name}' trong data/quick_samples/.")
    return path.read_bytes()
