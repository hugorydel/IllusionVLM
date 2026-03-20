"""
pipeline/module_3/plot_results.py - Psychometric curve and PSE drift plots.

Produces two figures per illusion:

  fig1_psychometric_curves.png
      Overlaid sigmoid curves (one per strength level), observed proportions,
      and PSE markers. X-axis = true physical difference; Y-axis = P(respond
      positive). The positive option label (e.g. "Top") comes from the illusion
      config so the axis is correctly labelled for every illusion.

  fig2_pse_vs_strength.png
      PSE (± SE) as a function of illusion strength, with a linear trend line.
      A non-zero slope is the core result.

Saved to results/<illusion_name>/figures/.
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf

# ============================================================================
# HELPERS
# ============================================================================


def _cumgauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def _strength_colors(strengths: list) -> dict:
    cmap = cm.get_cmap("plasma", len(strengths))
    return {s: cmap(i) for i, s in enumerate(strengths)}


# ============================================================================
# FIGURE 1: PSYCHOMETRIC CURVES
# ============================================================================


def plot_psychometric_curves(
    illusion: dict,
    psych_data: pd.DataFrame,
    pse_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    positive_option = illusion["response_options"][0]
    name = illusion["name"]

    strengths = sorted(psych_data["illusion_strength"].unique())
    colors = _strength_colors(strengths)
    x_smooth = np.linspace(
        psych_data["true_diff"].min(), psych_data["true_diff"].max(), 400
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for strength in strengths:
        color = colors[strength]
        subset = psych_data[psych_data["illusion_strength"] == strength]
        row = pse_summary[pse_summary["illusion_strength"] == strength]

        ax.scatter(
            subset["true_diff"],
            subset["prop_positive"],
            color=color,
            s=35,
            zorder=3,
            alpha=0.75,
        )

        if not row.empty and bool(row.iloc[0]["fit_success"]):
            pse = float(row.iloc[0]["pse"])
            sigma = float(row.iloc[0]["sigma"])
            ax.plot(
                x_smooth,
                _cumgauss(x_smooth, pse, sigma),
                color=color,
                linewidth=1.8,
                label=f"str={strength:+.0f}  PSE={pse:+.3f}",
            )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="Physical equality",
    )
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("True physical difference (positive − negative)", fontsize=12)
    ax.set_ylabel(f"P(respond '{positive_option}')", fontsize=12)
    ax.set_title(f"Psychometric Functions — {name}", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = figures_dir / "fig1_psychometric_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure 1 → {out_path}")
    plt.close(fig)


# ============================================================================
# FIGURE 2: PSE vs. ILLUSION STRENGTH
# ============================================================================


def plot_pse_vs_strength(
    illusion: dict,
    pse_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    name = illusion["name"]
    valid = pse_summary[pse_summary["fit_success"]].copy()

    if valid.empty:
        print(f"  ⚠ No successful fits for {name} — skipping PSE plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(
        valid["illusion_strength"],
        valid["pse"],
        yerr=valid["pse_se"],
        fmt="o-",
        color="#5c4dc9",
        markersize=7,
        linewidth=2,
        capsize=4,
        label="PSE ± SE",
    )

    if len(valid) >= 2:
        z = np.polyfit(valid["illusion_strength"], valid["pse"], 1)
        x_line = np.linspace(
            valid["illusion_strength"].min(), valid["illusion_strength"].max(), 200
        )
        ax.plot(
            x_line,
            np.polyval(z, x_line),
            "--",
            color="#a0a0a0",
            linewidth=1.2,
            label=f"Linear trend (slope={z[0]:+.4f})",
        )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="No bias (PSE = 0)",
    )

    ax.set_xlabel("Illusion strength", fontsize=12)
    ax.set_ylabel("PSE", fontsize=12)
    ax.set_title(f"PSE vs. Illusion Strength — {name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = figures_dir / "fig2_pse_vs_strength.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure 2 → {out_path}")
    plt.close(fig)


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def run_plotting(
    illusion: dict,
    psych_data: pd.DataFrame,
    pse_summary: pd.DataFrame,
    results_root: Path,
) -> None:
    """
    Generate and save all figures for one illusion.

    Args:
        illusion:     Illusion config dict.
        psych_data:   DataFrame from fit_psychometrics.aggregate_psychometric_data.
        pse_summary:  DataFrame from fit_psychometrics.run_fitting.
        results_root: Top-level results directory.
    """
    figures_dir = results_root / illusion["name"] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pse_summary = pse_summary.copy()
    pse_summary["fit_success"] = pse_summary["fit_success"].astype(bool)

    plot_psychometric_curves(illusion, psych_data, pse_summary, figures_dir)
    plot_pse_vs_strength(illusion, pse_summary, figures_dir)
