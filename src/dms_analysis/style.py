import plotly.graph_objects as go
import plotly.io as pio

TEMPLATE = "dms"
WIDTH = 1100
FONT = "DejaVu Sans, Arial, sans-serif"

TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#ffffff"

WEIGHT = "#3d3c39"
SIGNAL_BAND = "rgba(61, 60, 57, 0.10)"
WINDOW_BAND = "rgba(42, 120, 214, 0.10)"
EXCLUDED = "#a8a69f"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#c98500"
VIOLET = "#6250d6"
RED = "#e34948"

CATEGORICAL = (BLUE, ORANGE, AQUA)
DIVERGING = ((0.0, "#2a78d6"), (0.5, "#f2f1ed"), (1.0, "#e34948"))
SEQUENTIAL = ((0.0, "#f2f1ed"), (1.0, "#2a78d6"))


def register() -> None:
    axis = {
        "gridcolor": GRID,
        "linecolor": GRID,
        "zerolinecolor": GRID,
        "tickfont": {"color": TEXT_SECONDARY},
        "title": {"font": {"color": TEXT_SECONDARY}},
    }
    pio.templates[TEMPLATE] = go.layout.Template(
        layout={
            "font": {"family": FONT, "size": 13, "color": TEXT},
            "title": {"font": {"size": 16}, "x": 0, "xanchor": "left"},
            "paper_bgcolor": SURFACE,
            "plot_bgcolor": SURFACE,
            "colorway": list(CATEGORICAL),
            "xaxis": axis,
            "yaxis": axis,
            "legend": {"orientation": "v", "x": 1.01, "xanchor": "left", "y": 1, "yanchor": "top"},
            "margin": {"l": 70, "r": 20, "t": 80, "b": 60},
            "hoverlabel": {"font": {"family": FONT}},
            "width": WIDTH,
        }
    )
    pio.templates.default = f"plotly_white+{TEMPLATE}"


def _mix(low: str, high: str, share: float) -> str:
    start = [int(low[index : index + 2], 16) for index in (1, 3, 5)]
    end = [int(high[index : index + 2], 16) for index in (1, 3, 5)]
    share = min(max(share, 0.0), 1.0)
    red, green, blue = (round(a + (b - a) * share) for a, b in zip(start, end, strict=True))
    return f"#{red:02x}{green:02x}{blue:02x}"


def sequential_css(value: float, low: float, high: float, color: str = BLUE) -> str:
    share = 0.0 if high <= low else (value - low) / (high - low)
    text = "#ffffff" if share > 0.6 else TEXT
    return f"background-color: {_mix('#f7f6f2', color, share * 0.85)}; color: {text}"


def diverging_css(value: float, limit: float) -> str:
    color = RED if value > 0 else BLUE
    return sequential_css(abs(value), 0.0, limit, color)
