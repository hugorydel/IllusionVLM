"""
Figure 4 - Each illusion's effect on GPT-5.2, relative to its effect on humans.

One bar per illusion in each panel: GPT-5.2's summary divided by the human
one, so 1 means the illusion moves the model exactly as much as it moves
people:

    A  perceptual shift: the Figure 2 curve averaged over strengths (0, 1].
       Only the positive branch, because the curve is odd by construction and
       averages to zero over the full axis.
    B  error rate: the Figure 3 curve averaged over incongruent strengths minus
       its average over congruent strengths. The difference is the error the
       illusion adds; a plain average would mostly measure baseline error,
       which depends on humans being speeded and the model untimed.

Both summaries use the whole tested range rather than the value at full
strength, where a smooth is least well pinned. They are computed from the same
spline fits as the curves in Figures 2 and 3, not re-fitted.

UNCERTAINTY
    Each summary's variance comes from its fit's coefficient covariance: exact
    for the shift, which is linear in the coefficients, and by the delta method
    for the error effect. The whisker is then Fieller's 95% interval for the
    ratio, which carries the uncertainty of BOTH the model and the human
    summary. The drawn bands are not used for this, since pointwise bands say
    nothing about how the points along a curve covary.

Humans are 1 by construction in both panels, so they are drawn as a dashed
reference line in the human colour rather than as bars of constant height.
Nothing is lost by it: the human uncertainty is already inside each model's
Fieller interval. Further models slot in as further bars per illusion.

All explanatory text belongs in the caption, not in the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline.figures.selection import DISPLAY_NAMES, FIGURE_ILLUSIONS
from pipeline.figures.figstyle import (
    INK_PRIMARY,
    apply_style,
    grouped_bars,
    hide_spines,
    reference_line,
    species_legend,
)
from pipeline.figures.panels import fig2_perceptual_shift, fig3_congruency
from pipeline.figures.smoothing import error_effect, mean_magnitude

MODELS = ("vlm",)

_Z95 = 1.959964


def fieller_interval(
    a: float, var_a: float, b: float, var_b: float, z: float = _Z95
) -> tuple[float, float]:
    """
    Fieller's confidence interval for a / b, with a and b independent.

    The two summaries come from separate fits to separate data, so they are
    independent. The interval is bounded only when b is clearly away from
    zero, which the human effects are by a wide margin; otherwise it is
    returned as NaN rather than as a misleading finite range.
    """
    denom = b * b - z * z * var_b
    if denom <= 0:
        return float("nan"), float("nan")
    disc = var_a * b * b + var_b * a * a - z * z * var_a * var_b
    half = z * np.sqrt(max(disc, 0.0))
    return (a * b - half) / denom, (a * b + half) / denom


def relative_to_humans(summaries: dict[str, dict[str, tuple[float, float] | None]]):
    """
    Each model's summary as a fraction of the human one, with its interval.

    `summaries` maps illusion -> species -> (estimate, variance). Each
    interval is Fieller's, carrying both the model's and the human
    uncertainty. Returns the display names, then values and intervals keyed
    by model, in the layout grouped_bars takes.
    """
    names: list[str] = []
    values: dict[str, list[float]] = {m: [] for m in MODELS}
    intervals: dict[str, list[tuple[float, float]]] = {m: [] for m in MODELS}
    for illusion, by_species in summaries.items():
        names.append(DISPLAY_NAMES[illusion])
        human = by_species.get("human")
        for model in MODELS:
            est = by_species.get(model)
            if human is None or est is None:
                values[model].append(np.nan)
                intervals[model].append((np.nan, np.nan))
                continue
            values[model].append(est[0] / human[0])
            intervals[model].append(fieller_interval(*est, *human))
    return names, values, intervals


def largest_first(names, values, intervals, by: str = MODELS[0]):
    """
    Reorder a panel's illusions by one series' value, largest first.

    Each panel is sorted on its own values, so the ranking in each panel is
    read left to right without cross-referencing. An illusion the series
    lacks goes last.
    """
    key = np.nan_to_num(np.asarray(values[by], dtype=float), nan=-np.inf)
    order = np.argsort(-key, kind="stable")
    return (
        [names[i] for i in order],
        {s: [v[i] for i in order] for s, v in values.items()},
        {s: [v[i] for i in order] for s, v in intervals.items()},
    )


def build(paper_dir: Path, out_path: Path) -> None:
    """Render Figure 4 to `out_path`."""
    apply_style(base_font=7.5)

    magnitudes = fig2_perceptual_shift.load(paper_dir)
    by_strength = fig3_congruency.load(paper_dir)

    shift = largest_first(*relative_to_humans(
        {
            name: {
                sp: mean_magnitude(
                    fig2_perceptual_shift.levels_for(magnitudes, name, sp)
                )
                for sp in ("human",) + MODELS
            }
            for name in FIGURE_ILLUSIONS
        }
    ))
    error = largest_first(*relative_to_humans(
        {
            name: {
                sp: error_effect(fig3_congruency.cells_for(by_strength, name, sp))
                for sp in ("human",) + MODELS
            }
            for name in FIGURE_ILLUSIONS
        }
    ))

    # One scale for both panels, so B shares A's tick labels. The floor sits
    # just below zero, as in Figure 3, so a whisker ending at zero keeps its cap.
    pairs = [
        pair
        for _, _, intervals in (shift, error)
        for per_model in intervals.values()
        for pair in per_model
    ]
    top = max(1.25, float(np.nanmax([hi for _, hi in pairs])) * 1.04)
    bottom = min(-0.04, float(np.nanmin([lo for lo, _ in pairs])) - 0.02)

    fig = plt.figure(figsize=(6.0, 2.5))
    # Two bar panels; the right margin holds the key.
    gs = fig.add_gridspec(
        1,
        2,
        left=0.085,
        right=0.82,
        top=0.88,
        bottom=0.3,
        wspace=0.12,
    )

    panels = ((shift, "Perceptual Shift"), (error, "Error Rate"))
    for i, ((names, values, intervals), title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        hide_spines(ax)
        reference_line(ax, 1.0)
        grouped_bars(ax, names, values, intervals, order=MODELS)
        ax.set_ylim(bottom, top)
        ax.set_yticks(np.arange(0, top, 0.5))
        ax.set_title(title, fontsize=7.4, color=INK_PRIMARY, pad=4)
        if i == 0:
            ax.set_yticklabels([f"{t:g}" for t in ax.get_yticks()])
            ax.set_ylabel("Relative Illusion Effect", fontsize=7.2)
        else:
            ax.set_yticklabels([])
        ax.text(
            -0.2 if i == 0 else -0.08,
            1.16,
            "AB"[i],
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            color=INK_PRIMARY,
            va="top",
            ha="left",
        )

    # Stacked as in Figures 2 and 3, left-aligned against the last panel.
    species_legend(
        ax, order=MODELS, reference="human", loc="center left", bbox_to_anchor=(1.06, 0.5)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out_path} and {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    build(Path("results/_paper"), Path("results/_paper/figures/fig4_summary.png"))
