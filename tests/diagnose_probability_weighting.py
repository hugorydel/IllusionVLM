"""
tests/diagnose_probability_weighting.py - Average the probabilities or count the answers?

GPT-5.2 returns, for every call, the probability it placed on each response
option. The paper currently ignores those and counts answers: a cell's value is
the share of its 100 calls that chose the positive option. The alternative is to
average the 100 per-call probabilities instead.

WHY THE AVERAGE SHOULD BE BETTER
    Each call samples a reasoning trace, then an answer from the distribution
    that trace implies. Writing p_i for the probability of the positive option
    on call i and A_i for the answer actually returned:

        theta   = E[p_i]          the quantity both estimators target
        theta_1 = mean(A_i)       what the paper uses now
        theta_2 = mean(p_i)       what this script tests

    Both are unbiased for theta. Their variances are theta(1-theta)/n and
    Var(p)/n, and Var(p) = theta(1-theta) - E[p(1-p)], so theta_2 is never
    noisier and is strictly better whenever any call was uncertain. This is
    Rao-Blackwell: theta_1 throws away the answer distribution and keeps only a
    draw from it. The gain per cell is theta(1-theta) / Var(p), reported here as
    the number of calls theta_1 would need to match 100 of theta_2's.

WHY IT MIGHT NOT BE
    The recorded probabilities are coarse. In the central range they land on a
    0.25 log-odds grid, and a call whose alternative fell outside the returned
    top-k is stored as exactly 1 or 0, which overstates its certainty. Both
    distortions push theta_2 towards the extremes, so it can be biased where
    theta_1 is not. Panel A is the check: a systematic gap between the two
    estimators would show as points off the diagonal.

WHAT THIS SCRIPT DOES
    Aggregates the same GPT-5.2 calls both ways, puts each aggregation through
    the unmodified pipeline (pipeline/human_comparison/build_dataset.py), and
    renders Figures 2-4 from each. The human tables are identical in both, so
    every difference between the two figure sets comes from the model's cells.

    Outputs, all under tests/output/probability_weighting/:
        diagnostic.png            the four panels described below
        figures/responses/        Figures 2-4 as the paper draws them now
        figures/probabilities/    the same three from averaged probabilities
        _paper_responses/         the tables behind each figure set
        _paper_probabilities/

    Panels: A the two estimators against each other, per cell; B the effective
    call count, per illusion; C and D the width of Figure 4's Fieller interval
    under each aggregation, for the perceptual shift and the error effect.

ONE APPROXIMATION
    metrics.error_by_strength stores its error count as an integer, so the
    probability-weighted counts are truncated there - at most one error in the
    1,600 trials behind each strength, or 0.03 percentage points. Every other
    table carries the fractional counts through.

Usage:
    python -m tests.diagnose_probability_weighting
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ILLUSIONS
from pipeline.figures.figstyle import (
    AXIS,
    INK_MUTED,
    INK_PRIMARY,
    SERIES,
    SURFACE,
    apply_style,
    hide_spines,
)
from pipeline.figures.panels import (
    fig2_perceptual_shift,
    fig3_congruency,
    fig4_summary,
)
from pipeline.figures.selection import DISPLAY_NAMES, FIGURE_ILLUSIONS
from pipeline.figures.smoothing import error_effect, mean_magnitude
from pipeline.human_comparison import build_dataset

MODEL_KEY = "gpt-5.2"
RESULTS_ROOT = Path("results")
HUMAN_DIR = Path("data/human")
OUT_ROOT = Path("tests/output/probability_weighting")

# The two aggregations, named by the column each writes into n_positive.
AGGREGATIONS = ("responses", "probabilities")


# ============================================================================
# THE CALLS
# ============================================================================


def participants_dir(illusion: str) -> Path:
    """
    The model's per-call records for one illusion.

    Module 2's output moved under results/<model>/ when the pipeline grew a
    model registry; runs from before that sit directly under results/.
    """
    for candidate in (
        RESULTS_ROOT / MODEL_KEY / illusion / "participants",
        RESULTS_ROOT / illusion / "participants",
    ):
        if candidate.is_dir() and any(candidate.glob("participant_*.jsonl")):
            return candidate
    raise FileNotFoundError(f"No participant files for {illusion} under {RESULTS_ROOT}/")


def load_calls(illusion: dict) -> pd.DataFrame:
    """
    Every call for one illusion: the answer given and the probability behind it.

    The stored field is named logprob_<option> but holds a probability, not its
    logarithm.
    """
    positive = illusion["response_options"][0]
    field = f"logprob_{positive}"

    records = []
    for path in sorted(participants_dir(illusion["name"]).glob("participant_*.jsonl")):
        pid = int(path.stem.split("_")[-1])
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            records.append(
                {
                    "participant_id": rec.get("participant_id", pid),
                    "image_id": rec["image_id"],
                    "illusion_strength": rec["illusion_strength"],
                    "true_diff": rec["true_diff"],
                    "answered_positive": float(rec["response"] == positive),
                    # Runs from before the probabilities were recorded have
                    # none; those illusions fall back to counting answers.
                    "p_positive": float(rec[field]) if field in rec else np.nan,
                }
            )

    calls = pd.DataFrame(records)
    return calls.drop_duplicates(subset=["participant_id", "image_id"])


def aggregate(calls: pd.DataFrame) -> pd.DataFrame:
    """One row per (strength, difference) cell, aggregated both ways."""
    out = (
        calls.groupby(["illusion_strength", "true_diff"])
        .agg(
            n_trials=("answered_positive", "count"),
            responses=("answered_positive", "sum"),
            n_with_p=("p_positive", "count"),
            p_mean=("p_positive", "mean"),
            p_var=("p_positive", "var"),
        )
        .reset_index()
    )
    # An illusion whose calls do not all carry a probability cannot be
    # aggregated that way at all, so it keeps its answer counts in both trees
    # and is left out of the diagnostic panels.
    out["probabilities"] = (
        out["n_trials"] * out["p_mean"] if complete_probabilities(out) else out["responses"]
    )
    return out.sort_values(["illusion_strength", "true_diff"]).reset_index(drop=True)


def complete_probabilities(agg: pd.DataFrame) -> bool:
    """Whether every call behind every cell carries a probability."""
    return bool((agg["n_with_p"] == agg["n_trials"]).all())


def write_cells(agg: pd.DataFrame, root: Path, illusion: str, column: str) -> None:
    """Write one aggregation as the psychometric_data.csv Module 3 would have."""
    out_dir = root / MODEL_KEY / illusion
    out_dir.mkdir(parents=True, exist_ok=True)
    table = agg[["illusion_strength", "true_diff", "n_trials"]].copy()
    table["n_positive"] = agg[column]
    table["prop_positive"] = table["n_positive"] / table["n_trials"]
    table.to_csv(out_dir / "psychometric_data.csv", index=False)


# ============================================================================
# THE EFFECT ON FIGURE 4
# ============================================================================


def _ratio_with_interval(pair: dict) -> tuple[float, float]:
    """The model-to-human ratio and its Fieller interval width."""
    model, human = pair[MODEL_KEY], pair["human"]
    if model is None or human is None:
        return float("nan"), float("nan")
    low, high = fig4_summary.fieller_interval(*model, *human)
    return model[0] / human[0], high - low


def summaries(paper_dir: Path) -> pd.DataFrame:
    """Figure 4's two ratios and their interval widths, per illusion."""
    magnitudes = fig2_perceptual_shift.load(paper_dir)
    by_strength = fig3_congruency.load(paper_dir)

    rows = []
    for name in FIGURE_ILLUSIONS:
        shift = {
            sp: mean_magnitude(fig2_perceptual_shift.levels_for(magnitudes, name, sp))
            for sp in ("human", MODEL_KEY)
        }
        error = {
            sp: error_effect(fig3_congruency.cells_for(by_strength, name, sp))
            for sp in ("human", MODEL_KEY)
        }
        shift_ratio, shift_width = _ratio_with_interval(shift)
        error_ratio, error_width = _ratio_with_interval(error)
        rows.append(
            {
                "illusion": name,
                "shift_ratio": shift_ratio,
                "shift_width": shift_width,
                "error_ratio": error_ratio,
                "error_width": error_width,
            }
        )
    return pd.DataFrame(rows).set_index("illusion")


# ============================================================================
# THE DIAGNOSTIC FIGURE
# ============================================================================


def _width_panel(
    ax, widths: dict[str, pd.DataFrame], column: str, title: str, illusions: list[str]
) -> None:
    """One panel of interval widths: two bars per illusion."""
    names = [DISPLAY_NAMES[i] for i in illusions]
    x = np.arange(len(names))
    colours = {"responses": INK_MUTED, "probabilities": SERIES[MODEL_KEY]}
    for j, how in enumerate(AGGREGATIONS):
        heights = [widths[how].loc[i, column] for i in illusions]
        ax.bar(
            x + (j - 0.5) * 0.34,
            heights,
            width=0.34,
            color=colours[how],
            edgecolor=SURFACE,
            linewidth=0.4,
            label=how.capitalize(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylabel("Interval Width")
    ax.set_title(title, fontsize=7.4, color=INK_PRIMARY, pad=4)


def diagnostic_figure(
    cells: pd.DataFrame,
    widths: dict[str, pd.DataFrame],
    illusions: list[str],
    out_path: Path,
) -> None:
    """Render the four diagnostic panels, over the illusions that have probabilities."""
    cells = cells[cells["illusion"].isin(illusions)]
    apply_style(base_font=7.5)
    fig = plt.figure(figsize=(6.6, 5.2))
    gs = fig.add_gridspec(
        2, 2, hspace=0.6, wspace=0.36, left=0.1, right=0.97, top=0.92, bottom=0.12
    )

    # A - do the two estimators agree?
    ax_a = fig.add_subplot(gs[0, 0])
    hide_spines(ax_a)
    ax_a.plot([0, 1], [0, 1], color=AXIS, linewidth=0.6, zorder=1)
    ax_a.scatter(
        cells["responses"] / cells["n_trials"],
        cells["p_mean"],
        s=2.0,
        color=SERIES[MODEL_KEY],
        alpha=0.35,
        linewidths=0,
        zorder=2,
    )
    ax_a.set_xlabel("Share of Answers")
    ax_a.set_ylabel("Mean Call Probability")
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.set_title("Agreement Per Cell", fontsize=7.4, color=INK_PRIMARY, pad=4)

    # B - how much noise does the average remove?
    ax_b = fig.add_subplot(gs[0, 1])
    hide_spines(ax_b)
    medians = [
        cells.loc[cells["illusion"] == name, "equivalent_calls"].median()
        for name in illusions
    ]
    y = np.arange(len(illusions))
    ax_b.barh(
        y, medians, height=0.55, color=SERIES[MODEL_KEY], edgecolor=SURFACE, linewidth=0.4
    )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([DISPLAY_NAMES[i] for i in illusions])
    ax_b.invert_yaxis()
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_xlabel("Equivalent Calls Per 100")
    ax_b.set_title("Median Noise Reduction", fontsize=7.4, color=INK_PRIMARY, pad=4)

    # C, D - what it does to Figure 4's whiskers.
    ax_c = fig.add_subplot(gs[1, 0])
    hide_spines(ax_c)
    _width_panel(ax_c, widths, "shift_width", "Figure 4A: Perceptual Shift", illusions)
    ax_d = fig.add_subplot(gs[1, 1])
    hide_spines(ax_d)
    _width_panel(ax_d, widths, "error_width", "Figure 4B: Error Rate", illusions)
    ax_d.legend(loc="upper right", fontsize=7.0, handlelength=1.2, handleheight=0.6)

    for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D")):
        ax.text(
            -0.2,
            1.14,
            letter,
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color=INK_PRIMARY,
            va="top",
            ha="left",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-5.2's calls...")
    per_illusion = {}
    covered = []
    for illusion in ILLUSIONS:
        name = illusion["name"]
        calls = load_calls(illusion)
        agg = aggregate(calls)
        per_illusion[name] = agg
        has_p = complete_probabilities(agg)
        covered.append(name) if has_p else None
        note = "" if has_p else "  (no probabilities: counts answers in both trees)"
        print(f"  {name}: {len(calls)} calls over {len(agg)} cells{note}")
        for how in AGGREGATIONS:
            write_cells(agg, OUT_ROOT / f"results_{how}", name, how)

    panel_illusions = [i for i in FIGURE_ILLUSIONS if i in covered]
    if not panel_illusions:
        raise SystemExit("No illusion has a probability for every call.")

    # Per-cell comparison. The noise ratio is only defined where the answers
    # varied at all: a cell answered the same way 100 times, from probabilities
    # that never moved, has no noise left to remove.
    cells = pd.concat(
        [agg.assign(illusion=name) for name, agg in per_illusion.items()],
        ignore_index=True,
    )
    share = cells["responses"] / cells["n_trials"]
    binary_var = share * (1.0 - share)
    cells["equivalent_calls"] = np.where(
        (binary_var > 0) & (cells["p_var"] > 0),
        100.0 * binary_var / cells["p_var"],
        np.nan,
    )

    gap = cells["p_mean"] - share
    print(
        f"\n  mean(probability) - share(answers): mean {gap.mean():+.4f}, "
        f"mean |gap| {gap.abs().mean():.4f}, max |gap| {gap.abs().max():.4f}"
    )
    usable = int(cells["equivalent_calls"].notna().sum())
    print(
        f"  noise reduction measurable in {usable}/{len(cells)} cells; "
        f"median equivalent calls per 100: {cells['equivalent_calls'].median():.0f}"
    )

    # Both aggregations through the unmodified pipeline.
    widths = {}
    for how in AGGREGATIONS:
        print(f"\nBuilding the comparison tables from the {how}...")
        paper_dir = OUT_ROOT / f"_paper_{how}"
        build_dataset.build(OUT_ROOT / f"results_{how}", HUMAN_DIR, paper_dir)
        widths[how] = summaries(paper_dir)

        fig_dir = OUT_ROOT / "figures" / how
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nRendering Figures 2-4 from the {how}...")
        fig2_perceptual_shift.build(paper_dir, fig_dir / "fig2_perceptual_shift.png")
        fig3_congruency.build(paper_dir, fig_dir / "fig3_congruency.png")
        fig4_summary.build(paper_dir, fig_dir / "fig4_summary.png")

    print("\nFigure 4 both ways, as ratio [interval width]:")
    print(f"  {'':18s}{'perceptual shift':>36s}{'error rate':>36s}")
    print(
        f"  {'':18s}"
        + "".join(f"{how:>18s}" for _ in range(2) for how in AGGREGATIONS)
    )
    for name in panel_illusions:
        row = {how: widths[how].loc[name] for how in AGGREGATIONS}
        cellsd = "".join(
            f"{row[how][kind + '_ratio']:.2f} [{row[how][kind + '_width']:.2f}]".rjust(18)
            for kind in ("shift", "error")
            for how in AGGREGATIONS
        )
        print(f"  {DISPLAY_NAMES[name]:18s}{cellsd}")

    diagnostic_figure(cells, widths, panel_illusions, OUT_ROOT / "diagnostic.png")
    print(f"\nDone - everything under {OUT_ROOT}/")


if __name__ == "__main__":
    main()
