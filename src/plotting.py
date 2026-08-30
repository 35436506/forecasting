"""Các hàm vẽ biểu đồ Plotly dùng chung cho toàn bộ app — giữ đồng nhất
màu sắc, layout giữa các trang."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

C_DARK = "#1C2833"
C_GREEN = "#00AE41"
C_GOLD = "#F4B400"
C_RED = "#C0392B"
C_BLUE = "#2E86C1"
C_GRAY = "#95A5A6"
C_PURPLE = "#8E44AD"


def line_chart(
    x_values, series: dict[str, np.ndarray], title: str, xaxis_title: str = "", yaxis_title: str = "",
    colors: dict[str, str] | None = None, dash: dict[str, bool] | None = None,
) -> go.Figure:
    """Vẽ nhiều đường trên cùng một biểu đồ. series: {tên: mảng giá trị}."""
    fig = go.Figure()
    default_colors = [C_DARK, C_GREEN, C_GOLD, C_RED, C_BLUE, "#8E44AD", "#16A085"]
    for i, (name, y_values) in enumerate(series.items()):
        color = (colors or {}).get(name, default_colors[i % len(default_colors)])
        is_dashed = (dash or {}).get(name, False)
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode="lines+markers", name=name,
            line=dict(color=color, width=2, dash="dash" if is_dashed else "solid"),
            marker=dict(size=4),
        ))
    fig.update_layout(
        title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        margin=dict(l=20, r=20, t=50, b=20), height=420, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def forecast_chart_with_ci(
    x_history, y_history, x_future, y_forecast,
    y_ci_lower: np.ndarray | None = None, y_ci_upper: np.ndarray | None = None,
    title: str = "Dự báo", xaxis_title: str = "", yaxis_title: str = "",
) -> go.Figure:
    """Chuỗi lịch sử + dự báo tương lai kèm khoảng tin cậy (nếu có)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_history, y=y_history, mode="lines+markers", name="Lịch sử",
                              line=dict(color=C_DARK, width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x_future, y=y_forecast, mode="lines+markers", name="Dự báo",
                              line=dict(color=C_GREEN, width=2.5), marker=dict(size=6, symbol="square")))
    if y_ci_lower is not None and y_ci_upper is not None:
        fig.add_trace(go.Scatter(
            x=list(x_future) + list(x_future)[::-1],
            y=list(y_ci_upper) + list(y_ci_lower)[::-1],
            fill="toself", fillcolor="rgba(0,174,65,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="Khoảng tin cậy", hoverinfo="skip",
        ))
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                       margin=dict(l=20, r=20, t=50, b=20), height=440, hovermode="x unified")
    return fig


def bar_chart(x_values, y_values, title: str, xaxis_title: str = "", yaxis_title: str = "",
              color: str = C_GREEN, highlight_indices: list[int] | None = None) -> go.Figure:
    colors = [color] * len(x_values)
    if highlight_indices:
        for idx in highlight_indices:
            colors[idx] = C_RED
    fig = go.Figure(data=[go.Bar(x=x_values, y=y_values, marker_color=colors)])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                       margin=dict(l=20, r=20, t=50, b=20), height=360)
    return fig


def acf_pacf_charts(lags, acf_values, pacf_values, conf_bound: float) -> tuple[go.Figure, go.Figure]:
    def _make(values, title):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=lags, y=values, marker_color=C_GREEN, width=0.3))
        fig.add_hline(y=conf_bound, line_dash="dash", line_color=C_GRAY)
        fig.add_hline(y=-conf_bound, line_dash="dash", line_color=C_GRAY)
        fig.add_hline(y=0, line_color=C_DARK, line_width=1)
        fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="Hệ số",
                           margin=dict(l=20, r=20, t=50, b=20), height=320)
        return fig

    return _make(acf_values, "ACF"), _make(pacf_values, "PACF")


def heatmap_chart(matrix: np.ndarray, best_idx: tuple[int, int], title: str,
                   xaxis_title: str = "q", yaxis_title: str = "p") -> go.Figure:
    """Bản đồ nhiệt AIC — càng đậm càng thấp (tốt)."""
    text = [[f"{v:.1f}" if not np.isnan(v) else "–" for v in row] for row in matrix]
    fig = go.Figure(data=go.Heatmap(
        z=matrix, text=text, texttemplate="%{text}", colorscale="Greens_r",
        colorbar=dict(title="AIC"),
    ))
    best_p, best_q = best_idx
    fig.add_shape(type="rect", x0=best_q - 0.5, x1=best_q + 0.5, y0=best_p - 0.5, y1=best_p + 0.5,
                  line=dict(color=C_RED, width=3))
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                       margin=dict(l=20, r=20, t=50, b=20), height=420)
    return fig


def tracking_signal_chart(x_values, ts_values: np.ndarray, threshold: float = 4.0) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=ts_values, mode="lines+markers", name="Tracking Signal",
                              line=dict(color=C_DARK, width=2)))
    fig.add_hline(y=threshold, line_dash="dash", line_color=C_RED, annotation_text=f"+{threshold}")
    fig.add_hline(y=-threshold, line_dash="dash", line_color=C_RED, annotation_text=f"-{threshold}")
    fig.add_hline(y=0, line_color=C_GRAY, line_width=1)
    fig.update_layout(title="Tracking Signal theo thời gian", xaxis_title="Kỳ", yaxis_title="TS",
                       margin=dict(l=20, r=20, t=50, b=20), height=380)
    return fig


def method_comparison_bar(method_names: list[str], mape_values: list[float],
                           title: str = "So sánh MAPE giữa các phương pháp") -> go.Figure:
    order = np.argsort(mape_values)
    sorted_names = [method_names[i] for i in order]
    sorted_values = [mape_values[i] for i in order]
    colors = [C_GREEN if v == min(sorted_values) else C_GRAY for v in sorted_values]
    fig = go.Figure(data=[go.Bar(
        x=sorted_names, y=sorted_values, marker_color=colors,
        text=[f"{v:.1f}%" for v in sorted_values], textposition="outside",
    )])
    fig.update_layout(title=title, yaxis_title="MAPE (%)", margin=dict(l=20, r=20, t=50, b=20), height=420)
    return fig


def metric_comparison_bar(names: list[str], values: list[float], title: str,
                           yaxis_title: str = "MSE", value_suffix: str = "") -> go.Figure:
    """Bản tổng quát của `method_comparison_bar` — nhận bất kỳ chỉ số nào
    (MSE, AIC...) thay vì cố định MAPE, dùng cho trang Phân tích nhanh nơi
    cần so sánh nhiều loại chỉ số khác nhau giữa các phương pháp."""
    order = np.argsort(values)
    sorted_names = [names[i] for i in order]
    sorted_values = [values[i] for i in order]
    colors = [C_GREEN if v == min(sorted_values) else C_GRAY for v in sorted_values]
    fig = go.Figure(data=[go.Bar(
        x=sorted_names, y=sorted_values, marker_color=colors,
        text=[f"{v:,.2f}{value_suffix}" for v in sorted_values], textposition="outside",
    )])
    fig.update_layout(title=title, yaxis_title=yaxis_title, margin=dict(l=20, r=20, t=50, b=20), height=420)
    return fig


def decomposition_chart(dates, trend: np.ndarray, seasonal: np.ndarray, resid: np.ndarray,
                         title: str = "Phân rã chuỗi thời gian") -> go.Figure:
    """3 subplot xếp dọc: Trend / Seasonal / Residual — dùng chung trục thời gian."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=("Xu hướng (Trend)", "Mùa vụ (Seasonal)", "Phần dư (Residual)"))
    fig.add_trace(go.Scatter(x=dates, y=trend, mode="lines", line=dict(color=C_BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=seasonal, mode="lines", line=dict(color=C_GREEN, width=1.6)), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=resid, mode="lines", line=dict(color=C_RED, width=1)), row=3, col=1)
    fig.update_layout(title=title, showlegend=False, margin=dict(l=20, r=20, t=60, b=20), height=560)
    return fig


def seasonal_by_year_chart(pivot: pd.DataFrame, xlabel: str, title: str = "Mùa vụ theo năm") -> go.Figure:
    """Một đường cho mỗi năm, trục x là kỳ trong năm (tháng/quý/...) — giúp
    nhận biết trực quan hình dạng mùa vụ có LẶP LẠI ổn định qua các năm hay không."""
    fig = go.Figure()
    palette = [C_DARK, C_GREEN, C_GOLD, C_RED, C_BLUE, C_PURPLE, "#16A085", C_GRAY]
    for i, year in enumerate(pivot.columns):
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[year].values, mode="lines+markers", name=str(year),
            line=dict(color=palette[i % len(palette)], width=1.8), marker=dict(size=5),
        ))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Giá trị",
                       margin=dict(l=20, r=20, t=50, b=20), height=420, hovermode="x unified")
    return fig


def backtest_chart(train_x, train_y, test_x, test_y, forecasts: dict[str, np.ndarray],
                    title: str = "Backtest — so sánh trên tập kiểm định") -> go.Figure:
    """Vẽ Train (xám) + Test thực tế (đen) + N đường dự báo backtest chồng lên nhau."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_x, y=train_y, mode="lines", name="Train",
                              line=dict(color=C_GRAY, width=1.5)))
    fig.add_trace(go.Scatter(x=test_x, y=test_y, mode="lines+markers", name="Test (thực tế)",
                              line=dict(color=C_DARK, width=2.5), marker=dict(size=5)))
    palette = [C_GREEN, C_RED, C_GOLD, C_PURPLE]
    for i, (name, values) in enumerate(forecasts.items()):
        fig.add_trace(go.Scatter(x=test_x, y=values, mode="lines+markers", name=name,
                                  line=dict(color=palette[i % len(palette)], width=2, dash="dash"),
                                  marker=dict(size=5, symbol="square")))
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=50, b=20), height=440, hovermode="x unified")
    return fig
