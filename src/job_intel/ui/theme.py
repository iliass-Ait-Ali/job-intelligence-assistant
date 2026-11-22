"""Design tokens and the Plotly template of the web app.

Colors follow a validated data-viz palette (categorical order is fixed, never cycled; one hue for
magnitude; status colors reserved for states and always paired with a text label). The app uses a
single, light theme (configured in ``.streamlit/config.toml``) so charts are validated against one
surface.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#fcfcfb"
PAGE = "#f6f7f9"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BRAND = "#4338ca"  # UI accent (buttons, links); not a data color

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]  # fmt: skip
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_TEXT = {"good": "#006300", "warning": "#8a5a00", "serious": "#a3461f", "critical": "#b42323"}

FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"


def _template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=13, color=INK_2),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
        margin=dict(l=8, r=16, t=40, b=8),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=INK_2)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_2)),
        bargap=0.25,
    )  # fmt: skip
    return t


pio.templates["jobintel"] = _template()
pio.templates.default = "jobintel"


def hbar(labels, values, *, title: str = "", value_format: str = ".0%", hover: str | None = None, height: int | None = None,
         color: str = SERIES[0]) -> go.Figure:  # fmt: skip
    """Horizontal bar chart, largest on top, one series (no legend: the title names it)."""
    labels, values = list(labels)[::-1], list(values)[::-1]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=color, line=dict(color=SURFACE, width=2),
                                                        cornerradius=4),
        texttemplate=f"%{{x:{value_format}}}", textposition="outside", textfont=dict(color=INK_2, size=11),
        hovertemplate=hover or f"%{{y}}: %{{x:{value_format}}}<extra></extra>", cliponaxis=False,
    ))  # fmt: skip
    fig.update_layout(title=title, height=height or max(260, 24 * len(labels) + 70), showlegend=False,
                      xaxis=dict(showgrid=True, tickformat=value_format, rangemode="tozero"),
                      yaxis=dict(showgrid=False, automargin=True))  # fmt: skip
    return fig
