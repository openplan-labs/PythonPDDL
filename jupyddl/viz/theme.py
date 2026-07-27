"""The jupyddl chart theme: one palette, two modes, applied everywhere.

The categorical slots are assigned in a fixed order and never cycled, so a
planner keeps its colour whichever chart it appears in and whichever other
planners share the figure. Both modes were validated for colour-vision
deficiency separation against their own surface; on the light surface three
slots sit below 3:1 contrast, which is why every chart in :mod:`jupyddl.viz`
ships a legend *and* direct labels rather than relying on hue alone.
"""

from __future__ import annotations

# Categorical slots, in assignment order. Never cycle past slot 8: fold the
# rest into "other" or use small multiples.
LIGHT = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": [
        "#2a78d6",  # blue
        "#eb6834",  # orange
        "#1baf7a",  # aqua
        "#eda100",  # yellow
        "#e87ba4",  # magenta
        "#008300",  # green
        "#4a3aa7",  # violet
        "#e34948",  # red
    ],
    "sequential": [
        "#cde2fb",
        "#b7d3f6",
        "#9ec5f4",
        "#86b6ef",
        "#6da7ec",
        "#5598e7",
        "#3987e5",
        "#2a78d6",
        "#256abf",
        "#1c5cab",
        "#184f95",
        "#104281",
        "#0d366b",
    ],
    "good": "#0ca30c",
    "critical": "#d03b3b",
}

DARK = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "text": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": [
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ],
    "sequential": [
        "#0d366b",
        "#104281",
        "#184f95",
        "#1c5cab",
        "#256abf",
        "#2a78d6",
        "#3987e5",
        "#5598e7",
        "#6da7ec",
        "#86b6ef",
        "#9ec5f4",
        "#b7d3f6",
        "#cde2fb",
    ],
    "good": "#0ca30c",
    "critical": "#d03b3b",
}

FONT_STACK = ["DejaVu Sans", "Segoe UI", "Helvetica", "sans-serif"]


def palette(dark: bool = False) -> dict:
    """Return the colour roles for the requested mode."""
    return DARK if dark else LIGHT


def series_color(index: int, dark: bool = False) -> str:
    """Colour for categorical slot ``index`` (stable, never cycled past 8)."""
    slots = palette(dark)["series"]
    return slots[index % len(slots)]


def sequential(fraction: float, dark: bool = False) -> str:
    """Sample the single-hue sequential ramp at ``fraction`` in ``[0, 1]``."""
    ramp = palette(dark)["sequential"]
    if fraction != fraction:  # NaN
        fraction = 0.0
    fraction = min(1.0, max(0.0, float(fraction)))
    return ramp[int(round(fraction * (len(ramp) - 1)))]


def sequential_cmap(dark: bool = False):
    """The sequential ramp as a matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("jupyddl_seq", palette(dark)["sequential"])


def rc_params(dark: bool = False) -> dict:
    """matplotlib rcParams implementing the theme (pass to ``plt.rc_context``)."""
    pal = palette(dark)
    return {
        "figure.facecolor": pal["page"],
        "figure.edgecolor": pal["page"],
        "savefig.facecolor": pal["page"],
        "savefig.edgecolor": pal["page"],
        "axes.facecolor": pal["surface"],
        "axes.edgecolor": pal["axis"],
        "axes.labelcolor": pal["text_secondary"],
        "axes.titlecolor": pal["text"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": pal["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "xtick.color": pal["muted"],
        "ytick.color": pal["muted"],
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.labelcolor": pal["text_secondary"],
        "ytick.labelcolor": pal["text_secondary"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "text.color": pal["text"],
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 9,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": pal["text_secondary"],
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "figure.autolayout": False,
    }
