"""
Figure 3 - Error rate against signed illusion strength.

One panel per illusion, laid out as Figure 2. The x axis is signed illusion
strength normalised to each illusion's own tested maximum, so it runs -1 to +1
in every panel: congruent strengths left of zero, incongruent right. The raw
values are in different stimulus parameters per illusion and are not comparable
across panels; the fraction of the tested range is.

Each curve is a natural cubic spline in signed strength, fitted by binomial GLM
to that species' aggregated response counts, with a quasi-binomial 95% band.
See pipeline/figures/smoothing.py for the basis size and the dispersion
correction, both of which matter to how wide the band is.

Nothing about the psychometric fits enters here, so this is the figure that says
what the perceptual biases in Figure 2 cost in accuracy.

The grid slot after the last illusion holds the legend. Figure 4 summarises
each curve by its incongruent minus its congruent average, from the same fits.

Two illusions are in the dataset but not in this figure; see
selection.FIGURES_EXCLUDED for why.

Absolute error levels depend on task conditions: the human data were collected
under a speeded task and the model was untimed. The change along an illusion's
own axis is the comparable quantity, not the level.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pipeline.figures.selection import DISPLAY_NAMES, FIGURE_ILLUSIONS
from pipeline.figures.figstyle import (
    INK_PRIMARY,
    SERIES,
    apply_style,
    centred_panel_grid,
    hide_spines,
    legend_panel,
)
from pipeline.figures.smoothing import smooth_error_rate

YLIM = (-4.0, 100.0)
N_COLS = 3


def load(paper_dir: Path) -> pd.DataFrame:
    """Aggregated response counts per (species, illusion, strength)."""
    return pd.read_csv(paper_dir / "by_strength.csv")


def cells_for(by_strength: pd.DataFrame, illusion: str, species: str) -> pd.DataFrame:
    """The per-strength counts behind one curve."""
    return by_strength[
        (by_strength["illusion"] == illusion) & (by_strength["species"] == species)
    ]


def curves(by_strength: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Every curve the figure draws, keyed by illusion, then species.

    A species with too few levels to smooth is left out of its illusion's
    dict.
    """
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for name in FIGURE_ILLUSIONS:
        out[name] = {}
        for species in ("human", "vlm"):
            sub = cells_for(by_strength, name, species)
            curve = smooth_error_rate(sub) if not sub.empty else None
            if curve is not None:
                out[name][species] = curve
    return out


def _draw_curve(ax, curve: pd.DataFrame, species: str) -> None:
    """One species' curve and band."""
    # Band first and unoutlined, so the fitted line stays the figure's strongest
    # mark and the uncertainty reads as texture behind it.
    ax.fill_between(
        curve["x"],
        curve["lo"].clip(*YLIM),
        curve["hi"].clip(*YLIM),
        color=SERIES[species],
        alpha=0.16,
        lw=0,
        zorder=2,
    )
    ax.plot(curve["x"], curve["fit"], color=SERIES[species], lw=1.2, zorder=3)


def build(paper_dir: Path, out_path: Path) -> None:
    """Render Figure 3 to `out_path`."""
    apply_style(base_font=7.5)

    by_illusion = curves(load(paper_dir))

    # One slot per illusion plus one for the legend.
    fig = plt.figure(figsize=(6.0, 3.3))
    axes = centred_panel_grid(
        fig,
        len(FIGURE_ILLUSIONS) + 1,
        N_COLS,
        left=0.108,
        right=0.985,
        top=0.861,
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
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(-1.0, 1.0)  # curves start at the y axis, not short of it
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["-1", "0", "1"])
        # Every curve panel carries its own x label, so the layout stays
        # right whether or not the last row is short and centred.
        ax.set_xlabel("Illusion Strength", fontsize=7.2)

        if i % N_COLS == 0:
            ax.set_ylabel("Error Rate (%)", fontsize=7.2)
        else:
            ax.set_yticklabels([])

        ax.text(
            -0.11 if i % N_COLS else -0.31,
            1.18,
            "ABCDEFGH"[i],
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            color=INK_PRIMARY,
            va="top",
            ha="left",
        )

    legend_panel(axes[-1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out_path} and {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    build(Path("results/_paper"), Path("results/_paper/figures/fig3_congruency.png"))
