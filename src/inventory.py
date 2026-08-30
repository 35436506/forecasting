"""Safety Stock, Reorder Point (ROP), Tracking Signal.

Khớp Lecture Notes Chương 6 — biến độ chính xác dự báo (MSE) thành quyết
định tồn kho cụ thể.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z_TABLE = {80: 1.282, 85: 1.440, 90: 1.645, 95: 1.960, 97.5: 2.240, 99: 2.576}


@dataclass
class SafetyStockResult:
    z: float
    service_level: float
    mse: float
    lead_time: float
    safety_stock: float
    rop: float | None = None
    avg_demand_per_period: float | None = None


def safety_stock(mse: float, lead_time: float = 1.0, service_level: float = 95.0) -> SafetyStockResult:
    """SS = z * sqrt(MSE * L). service_level tính bằng %, ví dụ 95 -> z=1.96."""
    if mse < 0:
        raise ValueError("MSE phải không âm.")
    if lead_time <= 0:
        raise ValueError("Lead time phải dương.")

    z = Z_TABLE.get(service_level)
    if z is None:
        from scipy.stats import norm
        z = float(norm.ppf(service_level / 100))

    ss = z * np.sqrt(mse * lead_time)
    return SafetyStockResult(z=z, service_level=service_level, mse=mse, lead_time=lead_time, safety_stock=float(ss))


def reorder_point(
    avg_demand_per_period: float, lead_time: float, mse: float, service_level: float = 95.0,
) -> SafetyStockResult:
    """ROP = d̄ × L + Safety Stock."""
    if avg_demand_per_period < 0:
        raise ValueError("Nhu cầu trung bình mỗi kỳ phải không âm.")

    ss_result = safety_stock(mse=mse, lead_time=lead_time, service_level=service_level)
    rop = avg_demand_per_period * lead_time + ss_result.safety_stock
    ss_result.rop = float(rop)
    ss_result.avg_demand_per_period = avg_demand_per_period
    return ss_result


def holding_cost_per_period(safety_stock_units: float, unit_cost: float, holding_rate_pct: float) -> float:
    """Chi phí vốn tồn kho an toàn mỗi kỳ = SS × giá vốn/đơn vị × tỷ lệ giữ hàng."""
    if safety_stock_units < 0 or unit_cost < 0 or holding_rate_pct < 0:
        raise ValueError("Các tham số đầu vào phải không âm.")
    return safety_stock_units * unit_cost * (holding_rate_pct / 100)
