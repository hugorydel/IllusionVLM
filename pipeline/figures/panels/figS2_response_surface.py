"""
Figure S2 - Response surfaces: the raw choices behind Figures 2 and 3.

One row per species, one column per illusion. Each cell is the proportion of
trials on which the first-named option (Top or Left) was chosen, at one
illusion strength (x) and one signed difference band (y).

The difference axis uses the same four difficulty bands as Figure S1, signed:
the bottom rows are differences favouring the second option, easiest at the
bottom, and the top rows the first option, easiest at the top. Banding fills
every cell in both species; on the raw 16-level axis half of every human
column would be empty, because the human design shows only every other
difference at each strength.

Without an illusion the boundary between the two answers sits at the middle
of the axis. An illusion that shifts perception tilts it: the congruent side
(left) keeps the boundary central and the incongruent side (right) pushes
responses across it.

The colour scale is greyscale on purpose. Blue and orange mean humans and
GPT-5.2 in every other figure, so a hue here would read as a species.

All explanatory text belongs in the caption, not in the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.figures.figstyle import (
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES,
    SPECIES_LABEL,
    apply_style,
)
from pipeline.figures.panels.figS1_error_by_difficulty import N_BANDS
from pipeline.figures.panels.figS1_error_by_difficulty import load as load_banded
from pipeline.figures.selection import DISPLAY_NAMES, FIGURE_ILLUSIONS

CMAP = "Greys"


def surface(cells: pd.DataFrame, species: str, illusion: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Proportion choosing the first option, as a (signed band x strength) grid.

    Returns the grid, rows ordered from the most negative band to the most
    positive, and the sorted strengths normalised to their maximum.
    """
    sub = cells[(cells["species"] == species) & (cells["illusion"] == illusion)]
    agg = (
        sub.groupby(["signed_bin", "illusion_strength"])
        .agg(n_positive=("n_positive", "sum"), n_trials=("n_trials", "sum"))
        .reset_index()
    )
    agg["p"] = agg["n_positive"] / agg["n_trials"]
    grid = agg.pivot(index="signed_bin", columns="illusion_strength", values="p")
    grid = grid.sort_index().sort_index(axis=1)
    strengths = grid.columns.to_numpy(float)
    return grid.to_numpy(float), strengths / np.abs(strengths).max()


def build(paper_dir: Path, out_path: Path) -> None:
    """Render Figure S2 to `out_path`."""
    apply_style(base_font=7.5)

    cells = load_banded(paper_dir)
    species_rows = ("human", "vlm")
    n_cols = len(FIGURE_ILLUSIONS)

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = fig.add_gridspec(
        len(species_rows),
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.06],
        left=0.1,
        right=0.93,
        top=0.9,
        bottom=0.14,
        hspace=0.18,
        wspace=0.1,
    )

    image = None
    for r, species in enumerate(species_rows):
        for c, illusion in enumerate(FIGURE_ILLUSIONS):
            ax = fig.add_subplot(gs[r, c])
            grid, x = surface(cells, species, illusion)
            n_rows = grid.shape[0]
            image = ax.imshow(
                grid,
                origin="lower",
                cmap=CMAP,
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
                interpolation="nearest",
                extent=(-0.5, len(x) - 0.5, -0.5, n_rows - 0.5),
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_edgecolor(INK_SECONDARY)

            # Strength ticks at -1, 0 and 1 of the normalised axis.
            ax.set_xticks([0, (len(x) - 1) / 2, len(x) - 1])
            ax.set_xticklabels(["-1", "0", "1"])
            # Difference ticks at the two ends and the sign change between.
            ax.set_yticks([0, (n_rows - 1) / 2, n_rows - 1])
            ax.set_yticklabels(["-1", "0", "1"])

            if r == 0:
                ax.set_title(DISPLAY_NAMES[illusion], fontsize=7.4, color=INK_PRIMARY, pad=4)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Illusion Strength", fontsize=7.2)
            if c == 0:
                ax.set_ylabel("Physical Difference", fontsize=7.2)
                # The species, in its own colour, as the row's label.
                ax.text(
                    -0.62,
                    0.5,
                    SPECIES_LABEL[species],
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=7.6,
                    color=SERIES[species],
                    va="center",
                    ha="center",
                )
                ax.text(
                    -0.72,
                    1.16,
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

    cax = fig.add_subplot(gs[:, n_cols])
    bar = fig.colorbar(image, cax=cax)
    bar.set_ticks([0, 0.5, 1])
    bar.set_ticklabels(["0", "0.5", "1"])
    bar.outline.set_linewidth(0.5)
    bar.set_label("P(First Option)", fontsize=7.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out_path} and {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    build(
        Path("results/_paper"),
        Path("results/_paper/figures/figS2_response_surface.png"),
    )
