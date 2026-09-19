"""
Figure S3 - Exact response probabilities against the sampled responses.

For the open-weight models only: pipeline/module_2/local_vlm.py reads each
model's probability of each option directly, once per stimulus, alongside the
100 sampled answers. One row per model, one column per illusion.

    Line  the error rate those probabilities predict at the sampling
          temperature: per stimulus, the probability of the wrong option,
          averaged over the differences tested at each strength.
    Dots  the error rate of the sampled answers at the same strengths, as in
          Figure 3 before smoothing.

Where the dots sit on the line the sampling behaved as it should, and the line
is the same curve without sampling noise. GPT-5.2 has no row: its responses
were collected without stored probabilities.

Strength is normalised to each illusion's tested maximum, as in Figure 3:
negative congruent, positive incongruent. Shared-grid cells only.

All explanatory text belongs in the caption, not in the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pipeline.figures.figstyle import (
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES,
    SPECIES_LABEL,
    apply_style,
    hide_spines,
    line_style,
)
from pipeline.figures.selection import DISPLAY_NAMES, FIGURE_ILLUSIONS, models_present

YLIM = (-4.0, 100.0)

# Figure geometry per model row, in inches, matching Figure S1's.
ROW_HEIGHT = 1.254
TOP_MARGIN = 0.33
BOTTOM_MARGIN = 0.462


def expected_error(probs: pd.DataFrame) -> pd.DataFrame:
    """
    Predicted error rate per (model, illusion, strength), in percent.

    Per stimulus the error probability is P(second option) where the first is
    correct and P(first option) where the second is. Stimuli missing either
    option's probability are left out of their strength's mean.
    """
    probs = probs.dropna(subset=["p_first_sampled"])
    p_error = np.where(
        probs["true_diff"] > 0, 1.0 - probs["p_first_sampled"], probs["p_first_sampled"]
    )
    out = (
        probs.assign(p_error=p_error)
        .groupby(["species", "illusion", "illusion_strength"])["p_error"]
        .mean()
        .mul(100.0)
        .rename("error_pct")
        .reset_index()
    )
    return _normalise(out)


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `x`, the strength over its illusion's tested maximum."""
    return frame.assign(
        x=frame["illusion_strength"]
        / frame.groupby(["species", "illusion"])["illusion_strength"].transform(
            lambda s: s.abs().max()
        )
    )


def build(paper_dir: Path, out_path: Path) -> None:
    """Render Figure S3 to `out_path`, if any model has exact probabilities."""
    path = paper_dir / "probabilities.csv"
    if not path.exists():
        print("  skipped: no model has exact probabilities yet")
        return

    apply_style(base_font=7.5)

    probs = pd.read_csv(path)
    probs = probs[probs["illusion"].isin(FIGURE_ILLUSIONS)]
    expected = expected_error(probs)
    sampled = pd.read_csv(paper_dir / "by_strength.csv")
    sampled = _normalise(sampled.assign(error_pct=100.0 * sampled["err_overall"]))

    models = models_present(probs)
    n_cols = len(FIGURE_ILLUSIONS)
    height = TOP_MARGIN + BOTTOM_MARGIN + ROW_HEIGHT * len(models)
    fig = plt.figure(figsize=(7.2, height))
    gs = fig.add_gridspec(
        len(models),
        n_cols,
        left=0.1,
        right=0.86,
        top=1 - TOP_MARGIN / height,
        bottom=BOTTOM_MARGIN / height,
        hspace=0.32,
        wspace=0.14,
    )

    for r, model in enumerate(models):
        for c, illusion in enumerate(FIGURE_ILLUSIONS):
            ax = fig.add_subplot(gs[r, c])
            hide_spines(ax)

            line = expected[
                (expected["species"] == model) & (expected["illusion"] == illusion)
            ].sort_values("x")
            dots = sampled[
                (sampled["species"] == model) & (sampled["illusion"] == illusion)
            ].sort_values("x")
            ax.plot(line["x"], line["error_pct"], lw=1.2, zorder=3, **line_style(model))
            ax.scatter(
                dots["x"],
                dots["error_pct"],
                s=5,
                color=SERIES[model],
                edgecolors="none",
                zorder=4,
            )

            ax.set_ylim(*YLIM)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xlim(-1.05, 1.05)
            ax.set_xticks([-1, 0, 1])
            ax.set_xticklabels(["-1", "0", "1"])
            if r == 0:
                ax.set_title(DISPLAY_NAMES[illusion], fontsize=7.4, color=INK_PRIMARY, pad=4)
            if r < len(models) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Illusion Strength", fontsize=7.2)
            if c == 0:
                ax.set_ylabel("Error Rate (%)", fontsize=7.2)
                # The model, in its own colour, as the row's label.
                ax.text(
                    -0.62,
                    0.5,
                    SPECIES_LABEL[model],
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=7.6,
                    color=SERIES[model],
                    va="center",
                    ha="center",
                )
                ax.text(
                    -0.72,
                    1.14,
                    "ABCDEFGH"[r],
                    transform=ax.transAxes,
                    fontsize=8.0,
                    fontweight="bold",
                    color=INK_PRIMARY,
                    va="top",
                    ha="left",
                )
            else:
                ax.set_yticklabels([])

            # The mark key, once, beside the first row. Drawn in ink: the
            # colours already name the models.
            if r == 0 and c == n_cols - 1:
                ax.legend(
                    handles=[
                        Line2D([], [], color=INK_SECONDARY, lw=1.2),
                        Line2D([], [], color=INK_SECONDARY, lw=0, marker="o", markersize=2.6),
                    ],
                    labels=["Exact Probability", "Sampled Responses"],
                    fontsize=7.0,
                    loc="center left",
                    bbox_to_anchor=(1.08, 0.5),
                    handlelength=1.8,
                    labelspacing=0.7,
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
        Path("results/_paper/figures/figS3_exact_probabilities.png"),
    )
