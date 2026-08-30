"""Croston, SBA, TSB — dự báo cho nhu cầu rời rạc (nhiều kỳ = 0).

Khớp Lecture Notes Chương 6, mục "Nhu cầu rời rạc: họ phương pháp Croston".
Cả ba biến thể đều là ứng dụng sáng tạo của công thức SES đã học Chương 3,
áp dụng lên hai chuỗi phụ (quy mô, khoảng cách phát sinh) thay vì chuỗi gốc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CrostonResult:
    variant: str
    forecast: np.ndarray
    alpha: float
    beta: float | None = None


def croston_family(
    series: np.ndarray,
    alpha: float = 0.2,
    beta: float = 0.1,
    variant: str = "croston",
) -> CrostonResult:
    """variant: 'croston' | 'sba' | 'tsb'.

    Croston/SBA tách chuỗi thành quy mô (z) và khoảng cách phát sinh (p),
    mỗi chuỗi chỉ cập nhật khi có nhu cầu phát sinh (y_t > 0).
    TSB thay khoảng cách bằng xác suất phát sinh, cập nhật ở MỌI kỳ — nhờ đó
    "hạ nhiệt" dần khi chuỗi 0 kéo dài, khác với Croston/SBA giữ nguyên mức.
    """
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n < 4:
        raise ValueError("Cần tối thiểu 4 quan sát để chạy Croston/SBA/TSB.")
    if variant not in ("croston", "sba", "tsb"):
        raise ValueError("variant phải là 'croston', 'sba' hoặc 'tsb'.")

    forecasts = np.full(n, np.nan)

    if variant in ("croston", "sba"):
        nonzero_idx = np.where(y > 0)[0]
        if len(nonzero_idx) == 0:
            raise ValueError("Chuỗi không có kỳ nào phát sinh nhu cầu (toàn bộ = 0).")

        first_idx = int(nonzero_idx[0])
        z = y[first_idx]
        p = first_idx + 1
        q = 1

        for i in range(first_idx + 1, n):
            if y[i] > 0:
                z = alpha * y[i] + (1 - alpha) * z
                p = alpha * q + (1 - alpha) * p
                q = 1
            else:
                q += 1
            bias_correction = (1 - alpha / 2) if variant == "sba" else 1.0
            forecasts[i] = bias_correction * z / p if p != 0 else np.nan

        return CrostonResult(variant=variant, forecast=forecasts, alpha=alpha)

    # TSB: cap nhat MOI ky, dung xac suat phat sinh thay vi khoang cach p
    z = y[0] if y[0] > 0 else np.mean(y[y > 0]) if np.any(y > 0) else 0.0
    prob = float(np.mean(y[:max(1, n // 4)] > 0)) or 0.1

    for i in range(1, n):
        indicator = 1.0 if y[i] > 0 else 0.0
        prob = beta * indicator + (1 - beta) * prob
        if y[i] > 0:
            z = alpha * y[i] + (1 - alpha) * z
        forecasts[i] = prob * z

    return CrostonResult(variant=variant, forecast=forecasts, alpha=alpha, beta=beta)


def suggest_variant(series: np.ndarray, recent_window: int = 6) -> dict[str, float | str]:
    """Gợi ý SBA hay TSB dựa trên quy tắc kinh nghiệm ADI/CV² (Syntetos-Boylan)
    và xu hướng tần suất phát sinh gần đây — khớp Lecture Notes Chương 6.
    """
    y = np.asarray(series, dtype=float)
    nonzero_idx = np.where(y > 0)[0]
    if len(nonzero_idx) < 2:
        return {"suggestion": "Không đủ dữ liệu để gợi ý — cần ít nhất 2 kỳ phát sinh nhu cầu."}

    intervals = np.diff(nonzero_idx)
    adi = float(np.mean(intervals)) if len(intervals) > 0 else float(len(y))
    demand_sizes = y[nonzero_idx]
    cv2 = float((np.std(demand_sizes) / np.mean(demand_sizes)) ** 2) if np.mean(demand_sizes) != 0 else 0.0

    recent = y[-recent_window:]
    recent_freq = float(np.mean(recent > 0))
    earlier = y[-2 * recent_window:-recent_window] if len(y) >= 2 * recent_window else recent
    earlier_freq = float(np.mean(earlier > 0)) if len(earlier) > 0 else recent_freq
    declining = recent_freq < earlier_freq - 0.1

    if declining:
        suggestion = "TSB"
        reason = (
            f"Tần suất phát sinh {recent_window} kỳ gần nhất ({recent_freq:.0%}) thấp hơn rõ rệt so với "
            f"{recent_window} kỳ trước đó ({earlier_freq:.0%}) — có dấu hiệu nhu cầu đang suy giảm, nên "
            "dùng TSB để phản ánh đúng rủi ro."
        )
    elif adi >= 1.32 and cv2 < 0.49:
        suggestion = "SBA"
        reason = f"ADI={adi:.2f} (≥1.32) và CV²={cv2:.2f} (<0.49) — nhu cầu rời rạc đều, ưu tiên SBA."
    else:
        suggestion = "SBA (mặc định)"
        reason = f"ADI={adi:.2f}, CV²={cv2:.2f} — chưa rơi đúng vùng quy tắc, dùng SBA làm lựa chọn an toàn."

    return {"adi": adi, "cv2": cv2, "recent_freq": recent_freq, "earlier_freq": earlier_freq,
            "suggestion": suggestion, "reason": reason}
