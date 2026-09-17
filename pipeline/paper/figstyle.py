"""
pipeline/paper/figstyle.py - Shared figure styling.

One place for every colour and size decision, so the figure set reads as one
system and a journal's sizing request is a one-line change.

PALETTE
-------
A colour-temperature axis: a deep teal-blue at the cool end and a burnt amber
at the warm end. The two species take the two poles; ordered scales take one
hue and vary lightness.

Both were checked with the data-viz validator against a white surface:

    species pair (#0E6E96, #C8763C)
        chroma floor pass, CVD separation dE 17.5 (protan) / 26.9 (tritan),
        normal-vision dE 26.6, contrast >= 3:1

    ordinal ramp (#7FB8CE, #2E7E9E, #0A4257)
        lightness monotone, adjacent dL >= 0.06, light end 2.18:1 vs surface,
        hue spread 9 degrees

These are print figures: one light surface, no hover layer, and identity
carried by direct labels and legends rather than by colour alone.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# INK AND SURFACES
# ============================================================================

SURFACE = "#ffffff"
INK_PRIMARY = "#1a1a1a"
INK_SECONDARY = "#4d4d4d"
INK_MUTED = "#8a8a8a"
AXIS = "#bdbdbd"
NEUTRAL_MID = "#f2f1ef"

# ============================================================================
# THE TEMPERATURE POLES
# ============================================================================

COOL = "#0E6E96"
WARM = "#C8763C"

SERIES = {"human": COOL, "vlm": WARM}
SPECIES_LABEL = {"human": "Humans", "vlm": "GPT-5.2"}
SPECIES_MARKER = {"human": "o", "vlm": "s"}

# ============================================================================
# RAMPS
# ============================================================================

# Cool arm, light -> dark.
COOL_RAMP = [
    "#d6e9f1", "#bcdae8", "#a1cbdf", "#7FB8CE", "#5CA4BE", "#3D90AE",
    "#2E7E9E", "#1D6B8B", "#125A77", "#0A4257", "#07303F",
]
# Warm arm, light -> dark.
WARM_RAMP = [
    "#f8e5d5", "#f2d2b4", "#eabf93", "#e0a970", "#d69353", "#C8763C",
    "#b46630", "#9d5626", "#83451D", "#6a3616", "#512810",
]

# Diverging across the temperature axis, for a signed quantity with a
# meaningful zero. Dark at both poles, neutral at the midpoint.
TEMPERATURE = LinearSegmentedColormap.from_list(
    "cool_warm", list(reversed(COOL_RAMP)) + [NEUTRAL_MID] + WARM_RAMP
)

# One hue, light -> dark, for unsigned magnitude.
SEQUENTIAL = LinearSegmentedColormap.from_list("cool_seq", COOL_RAMP)

# ============================================================================
# RCPARAMS
# ============================================================================


def apply_style(base_font: float = 7.0) -> None:
    """
    Install the paper's matplotlib defaults.

    Args:
        base_font: Body font size in points. Journal figures are reproduced
                   small, so this is the one knob worth changing.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 400,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": base_font,
            "axes.titlesize": base_font + 0.5,
            "axes.labelsize": base_font,
            "xtick.labelsize": base_font - 0.5,
            "ytick.labelsize": base_font - 0.5,
            "legend.fontsize": base_font - 0.5,
            "axes.titleweight": "regular",
            "axes.labelcolor": INK_PRIMARY,
            "axes.edgecolor": INK_SECONDARY,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            # No gridlines anywhere in the paper figures. Deliberate: the
            # panels carry few points and ruled lines crossing them read as
            # data. Left off here rather than per axes so it cannot come back
            # one figure at a time.
            "axes.grid": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.4,
            "text.color": INK_PRIMARY,
            "errorbar.capsize": 0,
        }
    )


# ============================================================================
# HELPERS
# ============================================================================


def temperature_color(value: float, max_abs: float) -> tuple:
    """Colour for a signed value on the cool-to-warm axis, zero at neutral."""
    if max_abs <= 0:
        return TEMPERATURE(0.5)
    return TEMPERATURE(0.5 + 0.5 * float(np.clip(value / max_abs, -1.0, 1.0)))


def panel_label(ax, letter: str, dx: float = -0.02, dy: float = 1.04) -> None:
    """Bold lower-case panel letter outside the axes' top-left corner."""
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=mpl.rcParams["font.size"] + 1.5,
        fontweight="bold",
        va="bottom",
        ha="right",
        color=INK_PRIMARY,
    )


def hide_spines(ax, keep: tuple = ("left", "bottom")) -> None:
    """Leave only the named spines visible."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def centred_panel_grid(
    fig,
    n_panels: int,
    n_cols: int,
    **gridspec_kwargs,
) -> list:
    """
    Axes for `n_panels` in rows of `n_cols`, with a short last row centred.

    A five-panel figure in a three-column grid otherwise leaves a hole in one
    corner, which reads as a missing result rather than as a layout. Each panel
    spans two columns of a 2*n_cols grid, so a row holding one fewer panel can
    be offset by a single column and sit centred under the row above.

    Returns the axes in reading order.
    """
    n_rows = -(-n_panels // n_cols)
    gs = fig.add_gridspec(n_rows, 2 * n_cols, **gridspec_kwargs)

    axes = []
    for index in range(n_panels):
        row, col = divmod(index, n_cols)
        in_row = min(n_cols, n_panels - row * n_cols)
        # Centre a short row: half a panel's width per missing panel.
        offset = (n_cols - in_row)
        start = 2 * col + offset
        axes.append(fig.add_subplot(gs[row, start : start + 2]))
    return axes
