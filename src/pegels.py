"""Phân loại hình dạng nhu cầu: CV, %kỳ=0, SLOPE và Ma trận Pegels.

Khớp Lecture Notes Chương 1 — ba chỉ số định lượng quyết định một SKU thuộc
nhóm Ổn định / Mùa vụ-Xu hướng / Rời rạc / Vòng đời ngắn, và gợi ý ô Pegels
tương ứng để chọn đúng biến thể Holt-Winters ở bước sau.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DemandShapeResult:
    cv: float
    pct_zero: float
    slope: float
    shape_label: str
    pegels_hint: str
    explanation: str


def _linear_slope(y: np.ndarray) -> float:
    """SLOPE hồi quy tuyến tính đơn giản Y theo t (t=1..n)."""
    n = len(y)
    t = np.arange(1, n + 1)
    t_mean, y_mean = t.mean(), y.mean()
    numerator = np.sum((t - t_mean) * (y - y_mean))
    denominator = np.sum((t - t_mean) ** 2)
    return float(numerator / denominator) if denominator != 0 else 0.0


def classify_demand_shape(series: pd.Series | np.ndarray) -> DemandShapeResult:
    """Tính CV, %kỳ=0, SLOPE và suy ra nhãn hình dạng nhu cầu + gợi ý ô Pegels."""
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 4:
        raise ValueError("Cần tối thiểu 4 quan sát để phân loại hình dạng nhu cầu.")

    n = len(y)
    mean_y = y.mean()
    std_y = y.std(ddof=0)
    cv = float(std_y / mean_y) if mean_y != 0 else float("inf")
    pct_zero = float(np.mean(y == 0) * 100)
    slope = _linear_slope(y)
    # Dung TAC DONG LUY KE cua trend qua toan chuoi (slope*n) so voi mean,
    # thay vi chi so slope TUNG KY - mot SKU co slope/ky nho nhung KEO DAI
    # qua nhieu ky van tao ra xu huong LON ve tong the (vi du Noi com dien:
    # slope~12.7/thang nhung qua 36 thang la +458, ~61% so voi mean).
    slope_cumulative_pct = abs(slope) * n / mean_y * 100 if mean_y != 0 else 0.0

    # Phat hien "vong doi ngan": zero DON THUAN o DAU chuoi (truoc ra mat),
    # PHAN CON LAI gan nhu khong con zero -- khac voi "roi rac" la zero RAI
    # RAC xuyen suot chuoi (vi du linh kien thay the).
    first_nonzero_idx = int(np.argmax(y > 0)) if np.any(y > 0) else n
    tail_after_launch = y[first_nonzero_idx:]
    pct_zero_after_launch = (
        float(np.mean(tail_after_launch == 0) * 100) if len(tail_after_launch) > 0 else 100.0
    )
    is_short_lifecycle_pattern = (
        first_nonzero_idx >= n * 0.25 and pct_zero_after_launch <= 10 and len(tail_after_launch) < 24
    )

    # Nguyen tac phan loai (khop Lecture Notes Chuong 1), theo THU TU uu tien:
    # 1) Mau "vong doi ngan" (zero dau chuoi, sau do lien tuc co doanh so,
    #    con lai chua du 24 ky) -> Vong doi ngan, kiem tra TRUOC "roi rac"
    # 2) %ky=0 cao (>=30%) VA khong phai mau vong doi ngan -> Roi rac
    # 3) SLOPE luy ke lon (>=20% tong the so voi mean) -> Mua vu - Xu huong
    # 4) Con lai -> On dinh
    if is_short_lifecycle_pattern:
        shape_label = "Vòng đời ngắn (Short Life-cycle)"
        pegels_hint = "Ngoài ma trận Pegels — chưa đủ dữ liệu ước lượng mùa vụ ổn định"
        explanation = (
            f"{pct_zero:.1f}% số kỳ bằng 0, nhưng TẬP TRUNG ở {first_nonzero_idx} kỳ ĐẦU (trước ra mắt) — "
            f"sau đó chỉ còn {len(tail_after_launch)} kỳ có doanh số, hầu như không còn khoảng trống. "
            "Đây là mẫu hình sản phẩm MỚI ra mắt, không phải nhu cầu rời rạc thật. "
            "Tạm dùng Holt hoặc Quadratic Trend (Chương 5) và tái đánh giá khi đủ 24 kỳ có doanh số."
        )
    elif pct_zero >= 30:
        shape_label = "Rời rạc (Intermittent)"
        pegels_hint = "Ngoài ma trận Pegels — dùng họ Croston/SBA/TSB (Chương 6)"
        explanation = (
            f"{pct_zero:.1f}% số kỳ có giá trị 0, RẢI RÁC xuyên suốt chuỗi (không tập trung ở đầu/cuối) — "
            "vượt ngưỡng 30%, nhu cầu phát sinh không đều đặn theo thời gian. "
            "Các phương pháp Exponential Smoothing/ARIMA thông thường sẽ thất bại; "
            "nên chuyển sang Croston/SBA/TSB."
        )
    elif slope_cumulative_pct >= 20:
        shape_label = "Mùa vụ – Xu hướng (Trend-Seasonal)"
        pegels_hint = "Ô (A,A) hoặc (A,M) / (M,A) hoặc (M,M) — dùng Holt-Winters (Chương 3)"
        explanation = (
            f"Xu hướng lũy kế qua {n} kỳ ước tính khoảng {slope_cumulative_pct:.0f}% so với mức trung bình — "
            "xu hướng rõ rệt. Cần xác định thêm mùa vụ là Cộng hay Nhân (xem biên độ dao động qua các năm) "
            "trước khi chọn đúng biến thể Holt-Winters."
        )
    else:
        shape_label = "Ổn định (Stable)"
        pegels_hint = "Ô (N,N) — dùng SES (Chương 3)"
        explanation = (
            f"CV = {cv:.3f} và xu hướng lũy kế chỉ khoảng {slope_cumulative_pct:.0f}% so với mức trung bình — "
            "dao động quanh một mức trung bình không đổi, không có xu hướng hay mùa vụ hệ thống rõ rệt."
        )

    return DemandShapeResult(
        cv=cv, pct_zero=pct_zero, slope=slope,
        shape_label=shape_label, pegels_hint=pegels_hint, explanation=explanation,
    )
