"""
figures/figstyle.py - Shared figure styling.

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


def _reference_style(series: str) -> dict:
    """Line style for a reference level: the series colour, thin and dashed."""
    return {"color": SERIES[series], "linewidth": 0.8, "dashes": (4.0, 2.5)}


def reference_line(ax, y: float, series: str = "human") -> None:
    """
    A horizontal line marking a series' level, drawn behind the bars.

    The one exception to the no-lines rule: used where every value is
    expressed relative to that series, so the series itself has become a
    constant and a bar for it would only repeat the axis.
    """
    ax.axhline(y, zorder=1.5, **_reference_style(series))


def species_legend(
    ax,
    order: tuple[str, ...] = ("human", "vlm"),
    reference: str | None = None,
    **placement,
):
    """
    The species key: a solid swatch per series, entries stacked vertically.

    Every figure uses this one key, whether its panels hold curves or bars,
    so the species read the same way throughout. The swatch is the full
    series colour rather than a drawing of the mark, and kept thin so it reads
    as a colour key rather than as a bar. `reference` names a series drawn as
    a reference_line rather than as bars; it heads the key with the line's own
    style. `placement` takes the legend's `loc` and `bbox_to_anchor`.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=SERIES[s], linewidth=0) for s in order]
    labels = [SPECIES_LABEL[s] for s in order]
    if reference is not None:
        handles.insert(0, Line2D([], [], **_reference_style(reference)))
        labels.insert(0, SPECIES_LABEL[reference])

    return ax.legend(
        handles=handles,
        labels=labels,
        ncol=1,
        fontsize=7.4,
        handlelength=2.4,
        handleheight=0.55,
        handletextpad=0.7,
        labelspacing=0.9,
        borderaxespad=0,
        **placement,
    )


def legend_panel(ax, order: tuple[str, ...] = ("human", "vlm")) -> None:
    """
    Use a spare grid slot for the species key.

    A grid of five result panels leaves a sixth slot; the key goes there rather
    than in a strip below the figure, where it would widen the figure's
    footprint for two entries. It sits against the slot's left edge, where a
    panel's y axis would be, so it lines up with the column above it.
    """
    ax.axis("off")
    species_legend(ax, order, loc="center left", bbox_to_anchor=(0.0, 0.5))


def grouped_bars(
    ax,
    categories: list[str],
    values: dict[str, list[float]],
    intervals: dict[str, list[tuple[float, float]]] | None = None,
    order: tuple[str, ...] = ("human", "vlm"),
    bar_width: float = 0.2,
) -> None:
    """
    One group of side-by-side bars per category, one bar per series.

    Bars take the series colours used everywhere else, with no outline; a thin
    white edge separates the bars in a group. `intervals` gives each bar's
    (low, high) and is drawn as a thin whisker in secondary ink, so the same
    95% intervals the curve panels carry as bands are not dropped here.

    The width is per bar, not per group: adding a series widens the group
    rather than thinning every bar. At 0.2, four series still fit within a
    category with room between groups.
    """
    n = len(order)
    width = bar_width
    x = np.arange(len(categories))
    for j, series in enumerate(order):
        offset = (j - (n - 1) / 2) * width
        heights = np.asarray(values[series], dtype=float)
        ax.bar(
            x + offset,
            heights,
            width=width,
            color=SERIES[series],
            edgecolor=SURFACE,
            linewidth=0.4,
            zorder=2,
        )
        if intervals is not None:
            lo = np.array([a for a, _ in intervals[series]], dtype=float)
            hi = np.array([b for _, b in intervals[series]], dtype=float)
            ax.errorbar(
                x + offset,
                heights,
                yerr=[np.clip(heights - lo, 0, None), np.clip(hi - heights, 0, None)],
                fmt="none",
                ecolor=INK_SECONDARY,
                elinewidth=0.5,
                capsize=1.2,
                capthick=0.5,
                zorder=3,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(categories) - 0.5)
