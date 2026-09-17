"""
Figure 3 - Error rate against signed illusion strength.

One panel per illusion, laid out as Figure 2. The x axis is signed illusion
strength normalised to each illusion's own tested maximum, so it runs -1 to +1
in every panel: congruent strengths left of zero, incongruent right. The raw
values are in different stimulus parameters per illusion and are not comparable
across panels; the fraction of the tested range is.

Each curve is a natural cubic spline in signed strength, fitted by binomial GLM
to that species' aggregated response counts, with a quasi-binomial 95% band.
See pipeline/paper/smoothing.py for the basis size and the dispersion
correction, both of which matter to how wide the band is.

Nothing about the psychometric fits enters here, so this is the figure that says
what the perceptual biases in Figure 2 cost in accuracy.

Two illusions are in the dataset but not in this figure; see
conventions.FIGURES_EXCLUDED for why.

Absolute error levels depend on task conditions: the human data were collected
under a speeded task and the model was untimed. The change along an illusion's
own axis is the comparable quantity, not the level.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from pipeline.paper.conventions import DISPLAY_NAMES, FIGURE_ILLUSIONS
from pipeline.paper.figstyle import (
    INK_PRIMARY,
    SERIES,
    SPECIES_LABEL,
    apply_style,
    centred_panel_grid,
    hide_spines,
)
from pipeline.paper.smoothing import smooth_error_rate

YLIM = (-4.0, 100.0)
N_COLS = 3


def _panel_letter(i: int) -> str:
    return "ABCDEF"[i]


def build(paper_dir: Path, out_path: Path) -> None:
    """Render Figure 3 to `out_path`."""
    apply_style(base_font=7.5)

    by_strength = pd.read_csv(paper_dir / "by_strength.csv")

    fig = plt.figure(figsize=(6.0, 3.7))
    axes = centred_panel_grid(
        fig,
        len(FIGURE_ILLUSIONS),
        N_COLS,
        left=0.108,
        right=0.985,
        top=0.876,
        bottom=0.198,
        hspace=0.62,
        wspace=0.42,
    )

    for i, illusion in enumerate(FIGURE_ILLUSIONS):
        ax = axes[i]
        hide_spines(ax)

        for species in ("human", "vlm"):
            sub = by_strength[
                (by_strength["illusion"] == illusion)
                & (by_strength["species"] == species)
            ]
            if sub.empty:
                continue

            curve = smooth_error_rate(sub)
            if curve is None:
                continue

            # Band first and unoutlined, so the fitted line stays the figure's
            # strongest mark and the uncertainty reads as texture behind it.
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
                curve["fit"],
                color=SERIES[species],
                lw=1.2,
                zorder=3,
            )

        ax.set_title(DISPLAY_NAMES[illusion], fontsize=7.4, color=INK_PRIMARY, pad=4)
        ax.set_ylim(*YLIM)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(-1.02, 1.02)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["-1", "0", "1"])
        if i % N_COLS == 0:
            ax.set_ylabel("Error Rate (%)", fontsize=7.2)
        else:
            ax.set_yticklabels([])
        # Every panel carries its own x label: the centred short row means the
        # panels do not line up in columns, so nothing can be inherited.
        ax.set_xlabel("Illusion Strength", fontsize=7.2)

        ax.text(
            -0.11 if i % N_COLS else -0.31,
            1.18,
            _panel_letter(i),
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            color=INK_PRIMARY,
            va="top",
            ha="left",
        )

    fig.legend(
        handles=[
            Line2D([], [], color=SERIES[sp], lw=1.2, label=SPECIES_LABEL[sp])
            for sp in ("human", "vlm")
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        fontsize=7.4,
        handlelength=1.8,
        columnspacing=2.4,
        handletextpad=0.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out_path} and {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    build(Path("results/_paper"), Path("results/_paper/figures/fig3_congruency.png"))
