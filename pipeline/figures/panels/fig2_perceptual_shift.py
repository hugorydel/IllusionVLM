"""
Figure 2 - Perceptual shift against signed illusion strength.

Each panel shows, for one illusion, how far the illusion displaces the point of
subjective equality as it strengthens in either of its two spatial directions.

WHERE THE ESTIMATES COME FROM
    by_magnitude.csv, the joint two-direction fit: at each |strength| the 16
    response cells (8 physical differences x 2 illusion directions) are fitted
    as one model in which the two directions share a slope and a lapse pair but
    each takes its own point of subjective equality,

        PSE(d) = side_bias - d * magnitude

    Only `magnitude` is drawn. `side_bias` is a standing preference for one
    response option, present whichever way the surround pushes, and carrying it
    into the figure would shift an observer's whole curve by a constant - on
    Muller-Lyer by +0.151 for the model against -0.017 for humans, enough to
    make comparable illusions look three times apart.

    The seven levels per curve are then smoothed with the same natural cubic
    spline as Figure 3's error rates, at a smaller basis size - fitted across
    the signed axis, weighted by the inverse square of each level's standard
    error. See pipeline/figures/smoothing.py; the weighting is what keeps an
    unidentified level from steering the curve, and the natural spline's linear
    ends are what keep it from overshooting past the last tested level.

X AXIS
    Signed illusion strength normalised to each illusion's own tested maximum,
    spanning -1 to +1 with 0 meaning no illusion. The sign is the illusion's
    SPATIAL DIRECTION - which way the surround displaces the percept - not
    congruency. A direction slice mixes congruent and incongruent trials by
    construction, which is what makes a PSE definable for it, so congruency
    cannot label this axis. Congruency is the axis of Figure 3.

Y AXIS
    The smoothed effect divided by ONE number per illusion - the peak of that
    illusion's smoothed HUMAN curve - with both species divided by the same
    one. The human curve therefore reaches exactly -1 and +1, and the model is
    read directly against it: 1.4 is about half again the human effect, 0.2 is
    a fifth of it. Taking the denominator from the smooth rather than from the
    raw levels is what keeps the human reference at exactly one.

The grid slot after the last illusion holds the legend. Figure 4 summarises
each curve by its average over the tested strengths, from the same fits.

Rod-Frame and Delboeuf are held out; see selection.FIGURES_EXCLUDED.

All explanatory text belongs in the caption, not in the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.figures.selection import DISPLAY_NAMES, FIGURE_ILLUSIONS, species_present
from pipeline.figures.figstyle import (
    INK_PRIMARY,
    SERIES,
    apply_style,
    centred_panel_grid,
    hide_spines,
    legend_panel,
    line_style,
)
from pipeline.figures.smoothing import smooth_magnitude

YLIM = (-1.3, 1.3)  # data and bands reach +/-1.18 with the current illusion set
N_COLS = 3

_Z95 = 1.959964


def load(paper_dir: Path) -> pd.DataFrame:
    """The reportable magnitude levels for the illusions in the figures."""
    magnitudes = pd.read_csv(paper_dir / "by_magnitude.csv")
    return magnitudes[
        magnitudes["illusion"].isin(FIGURE_ILLUSIONS) & magnitudes["reportable"]
    ]


def levels_for(magnitudes: pd.DataFrame, illusion: str, species: str) -> pd.DataFrame:
    """
    One row per |strength| for a single curve, ready to smooth.

    `se` comes from the profile interval on magnitude, which is what lets the
    smooth discount a level the data did not pin down.
    """
    g = magnitudes[
        (magnitudes["illusion"] == illusion) & (magnitudes["species"] == species)
    ]
    if g.empty:
        return g
    k_max = float(g["k_abs"].max())
    return pd.DataFrame(
        {
            "x": g["k_abs"] / k_max,
            "y": g["illusion_magnitude"],
            "se": (g["magnitude_ci_high"] - g["magnitude_ci_low"]) / (2 * _Z95),
        }
    )


def human_reference(magnitudes: pd.DataFrame, illusion: str) -> float:
    """
    The peak of this illusion's smoothed human curve.

    One denominator for both species in the panel, taken from the same smooth
    that is drawn, so the human curve reaches exactly one rather than somewhere
    near it.
    """
    curve = smooth_magnitude(levels_for(magnitudes, illusion, "human"))
    if curve is None:
        return float("nan")
    denom = float(np.abs(curve["fit"]).max())
    return denom if np.isfinite(denom) and denom > 0 else float("nan")


def curves(magnitudes: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Every curve the figure draws, already divided by its illusion's human peak.

    Keyed by illusion, then species. An illusion whose human curve cannot be
    smoothed maps to an empty dict, since nothing in its panel can be scaled.
    """
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for name in FIGURE_ILLUSIONS:
        out[name] = {}
        denom = human_reference(magnitudes, name)
        if not np.isfinite(denom):
            continue
        for species in species_present(magnitudes):
            curve = smooth_magnitude(levels_for(magnitudes, name, species))
            if curve is not None:
                out[name][species] = curve.assign(
                    fit=curve["fit"] / denom,
                    lo=curve["lo"] / denom,
                    hi=curve["hi"] / denom,
                )
    return out


def _draw_curve(ax, curve: pd.DataFrame, species: str) -> None:
    """One species' curve and band."""
    # Band first and unoutlined, so the fitted line stays the strongest mark and
    # the uncertainty reads as texture behind it.
    ax.fill_between(
        curve["x"],
        curve["lo"].clip(*YLIM),
        curve["hi"].clip(*YLIM),
        color=SERIES[species],
        alpha=0.16,
        lw=0,
        zorder=2,
    )
    ax.plot(
        curve["x"],
        curve["fit"].clip(*YLIM),
        lw=1.2,
        zorder=3,
        **line_style(species),
    )


def build(paper_dir: Path, out_path: Path) -> None:
    """Render Figure 2 to `out_path`."""
    apply_style(base_font=7.5)

    magnitudes = load(paper_dir)
    by_illusion = curves(magnitudes)

    # One slot per illusion plus one for the legend.
    fig = plt.figure(figsize=(6.0, 3.5))
    axes = centred_panel_grid(
        fig,
        len(FIGURE_ILLUSIONS) + 1,
        N_COLS,
        left=0.112,
        right=0.985,
        top=0.869,
        bottom=0.11,
        hspace=0.62,
        wspace=0.42,
    )

    for i, name in enumerate(FIGURE_ILLUSIONS):
        ax = axes[i]
        hide_spines(ax)

        for species, curve in by_illusion[name].items():
            _draw_curve(ax, curve, species)
        ax.set_title(DISPLAY_NAMES[name], fontsize=7.4, color=INK_PRIMARY, pad=4)
        ax.set_ylim(*YLIM)
        ax.set_yticks([-1, 0, 1])
        ax.set_xlim(-1.0, 1.0)  # curves start at the y axis, not short of it
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["-1", "0", "1"])
        # Every curve panel carries its own x label, so the layout stays
        # right whether or not the last row is short and centred.
        ax.set_xlabel("Illusion Strength", fontsize=7.2)

        if i % N_COLS == 0:
            ax.set_ylabel("Perceptual Shift", fontsize=7.2)
        else:
            ax.set_yticklabels([])

        ax.text(
            -0.11 if i % N_COLS else -0.34,
            1.18,
            "ABCDEFGH"[i],
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            color=INK_PRIMARY,
            va="top",
            ha="left",
        )

    legend_panel(axes[-1], species_present(magnitudes))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out_path} and {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    build(Path("results/_paper"), Path("results/_paper/figures/fig2_perceptual_shift.png"))
