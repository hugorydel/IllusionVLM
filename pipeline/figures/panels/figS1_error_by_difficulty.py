"""
Figure S1 - Error rate against illusion strength, split by task difficulty.

One row per species, one column per illusion. Each line is one difficulty band:
the eight |difference| levels pooled in adjacent pairs by rank, hardest (the
two smallest differences) to easiest (the two largest). Pooling in pairs is
what lets every band have a value at every strength in both species - the
human design shows only every other difference at each strength, and each pair
contains one level from each alternation.

The data are the raw error rates on the shared grid, not smoothed: this is the
figure that shows what Figure 3's curves summarise. Error rate pools both
signs of the difference at each strength.

Strength is normalised to each illusion's tested maximum, as in Figures 2 and
3: negative congruent, positive incongruent.

All explanatory text belongs in the caption, not in the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pipeline.figures.figstyle import (
    COOL_RAMP,
    INK_PRIMARY,
    SPECIES_LABEL,
    WARM_RAMP,
    apply_style,
    hide_spines,
)
from pipeline.figures.selection import DISPLAY_NAMES, FIGURE_ILLUSIONS
from pipeline.human_comparison.metrics import add_difficulty_bins

N_BANDS = 4
BAND_LABELS = ("Easiest", "Easy", "Hard", "Hardest")  # by difficulty_bin
YLIM = (-4.0, 100.0)

# One shade per difficulty_bin, easiest (lightest) to hardest (darkest), within
# each species' own hue: the hardest band carries the illusion effect, so it
# gets the strongest ink.
RAMPS = {
    "human": [COOL_RAMP[i] for i in (3, 5, 7, 9)],
    "vlm": [WARM_RAMP[i] for i in (3, 5, 7, 9)],
}


def load(paper_dir: Path) -> pd.DataFrame:
    """Shared-grid cells for the illusions in the figures, banded by difficulty."""
    cells = pd.read_csv(paper_dir / "cells.csv")
    cells = cells[cells["shared"] & cells["illusion"].isin(FIGURE_ILLUSIONS)]
    return add_difficulty_bins(cells, n_bins=N_BANDS)


def error_by_band(cells: pd.DataFrame) -> pd.DataFrame:
    """
    Error rate per (species, illusion, strength, band), in percent.

    Errors are positive-option responses where the difference is negative and
    negative-option responses where it is positive.
    """
    cells = cells.assign(
        n_errors=np.where(
            cells["true_diff"] > 0,
            cells["n_trials"] - cells["n_positive"],
            cells["n_positive"],
        )
    )
    out = (
        cells.groupby(["species", "illusion", "illusion_strength", "difficulty_bin"])
        .agg(n_errors=("n_errors", "sum"), n_trials=("n_trials", "sum"))
        .reset_index()
    )
    out["error_pct"] = 100.0 * out["n_errors"] / out["n_trials"]
    out["x"] = out["illusion_strength"] / out.groupby(["species", "illusion"])[
        "illusion_strength"
    ].transform(lambda s: s.abs().max())
    return out


def build(paper_dir: Path, out_path: Path) -> None:
    """Render Figure S1 to `out_path`."""
    apply_style(base_font=7.5)

    errors = error_by_band(load(paper_dir))
    species_rows = ("human", "vlm")
    n_cols = len(FIGURE_ILLUSIONS)

    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(
        len(species_rows),
        n_cols,
        left=0.075,
        right=0.86,
        top=0.9,
        bottom=0.14,
        hspace=0.32,
        wspace=0.14,
    )

    for r, species in enumerate(species_rows):
        for c, illusion in enumerate(FIGURE_ILLUSIONS):
            ax = fig.add_subplot(gs[r, c])
            hide_spines(ax)
            sub = errors[(errors["species"] == species) & (errors["illusion"] == illusion)]
            # Easiest first, so the hardest (darkest) band is drawn on top.
            for band in range(N_BANDS):
                line = sub[sub["difficulty_bin"] == band].sort_values("x")
                ax.plot(
                    line["x"],
                    line["error_pct"],
                    color=RAMPS[species][band],
                    lw=0.9,
                    marker="o",
                    markersize=1.8,
                    markeredgewidth=0,
                    zorder=2 + band,
                )
            ax.set_ylim(*YLIM)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xlim(-1.05, 1.05)
            ax.set_xticks([-1, 0, 1])
            ax.set_xticklabels(["-1", "0", "1"])
            if r == 0:
                ax.set_title(DISPLAY_NAMES[illusion], fontsize=7.4, color=INK_PRIMARY, pad=4)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Illusion Strength", fontsize=7.2)
            if c == 0:
                ax.set_ylabel("Error Rate (%)", fontsize=7.2)
                ax.text(
                    -0.5,
                    1.14,
                    "AB"[r],
                    transform=ax.transAxes,
                    fontsize=8.0,
                    fontweight="bold",
                    color=INK_PRIMARY,
                    va="top",
                    ha="left",
                )
            else:
                ax.set_yticklabels([])

        # One key per row, titled with the species: the row's own ramp, hardest
        # at the top as the lines stack.
        ax.legend(
            handles=[
                Line2D([], [], color=RAMPS[species][band], lw=1.4)
                for band in reversed(range(N_BANDS))
            ],
            labels=[BAND_LABELS[band] for band in reversed(range(N_BANDS))],
            title=SPECIES_LABEL[species],
            title_fontsize=7.4,
            fontsize=7.0,
            loc="center left",
            bbox_to_anchor=(1.08, 0.5),
            alignment="left",
            handlelength=1.8,
            labelspacing=0.5,
            borderaxespad=0,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out_path} and {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    build(
        Path("results/_paper"),
        Path("results/_paper/figures/figS1_error_by_difficulty.png"),
    )
