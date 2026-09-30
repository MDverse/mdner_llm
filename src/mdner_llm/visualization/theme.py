"""Module for ."""

import plotly.graph_objects as go
from plotly.colors import sample_colorscale

COLOR_PALETTE = [
    "#ffffd9",
    "#edf8b1",
    "#c7e9b4",
    "#7fcdbb",
    "#41b6c4",
    "#1d91c0",
    "#225ea8",
    "#0c2c84",
]


def generate_palette(color_count: int) -> list[str]:
    """Sample a color scale for the requested number of colors.

    Returns
    -------
    list[str]
        A list of hex color codes sampled from the blue-to-purple color scale.
    """
    sample_points = [index / max(color_count - 1, 1) for index in range(color_count)]
    return sample_colorscale(COLOR_PALETTE, sample_points)


def apply_journal_theme(figure: go.Figure, y_axis_title: str) -> None:
    """Apply a publication-ready layout to a Plotly figure."""
    figure.update_layout(
        template="simple_white",
        font={"family": "Arial, Helvetica, sans-serif", "size": 16, "color": "black"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 70, "r": 30, "t": 70, "b": 90},
    )
    figure.update_yaxes(
        title=y_axis_title.capitalize().replace("_", " "),
        range=[0, 1],
        tickfont={"size": 14},
        title_font={"size": 17},
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        ticks="outside",
    )
    figure.update_xaxes(
        tickfont={"size": 16},
        showgrid=False,
        tickangle=0,
        ticks="outside",
    )
