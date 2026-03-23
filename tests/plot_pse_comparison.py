"""
tests/plot_pse_comparison.py - PSE comparison figure across illusions.

Generates a publication-ready 1×3 panel showing LOESS-smoothed PSE curves
with SE ribbons for MullerLyer, Ponzo, and VerticalHorizontal.

Usage:
    python tests/plot_pse_comparison.py
    python tests/plot_pse_comparison.py --results-root path/to/results
    python tests/plot_pse_comparison.py --out figures/pse_comparison.png

Each subplot shows:
    • Raw PSE points (reliable fits only, failed/outlier fits excluded)
    • ±1 SE ribbon around raw points
    • LOESS smoothed curve through reliable PSE values

Fit exclusion criterion matches pipeline/module_3/plot_results.py:
    A fit is excluded if pse_se > (diff_max − diff_min) for that illusion.

Output: PNG at 300 dpi, greyscale, publication-ready.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

# ============================================================================
# CONFIGURATION
# ============================================================================

# Illusions to plot, in left-to-right order.
# Each entry: (display_name, results_folder_name)
ILLUSIONS = [
    ("Müller-Lyer", "MullerLyer"),
    ("Ponzo", "Ponzo"),
    ("Vertical-Horizontal", "VerticalHorizontal"),
]

# LOESS bandwidth (frac): fraction of data used for each local fit.
# Smaller = more wiggly, larger = smoother. 0.4 is a reasonable default
# for 15 strength levels; increase toward 0.6 if the curve looks too rough.
LOESS_FRAC = 0.4

# SE ribbon clipping: caps ribbon to ±RIBBON_CLIP around each PSE point.
# Prevents a single high-SE outlier from dominating the y-axis scale.
# Set to None to disable clipping.
RIBBON_CLIP = 0.20

# Per-illusion y-axis limits [ymin, ymax].
# Set to None for fully automatic limits.
# Either value in the tuple can be None to fix only one bound
# (e.g. (-0.15, None) pins the bottom while leaving the top auto).
YLIM: dict[str, tuple[float | None, float | None] | None] = {
    "MullerLyer": (-0.07, 0.60),
    "Ponzo": (-0.07, 0.60),
    "VerticalHorizontal": (-0.07, 0.60),
}

# Greyscale palette
COLOR_LINE = "#000000"  # smoothed curve
COLOR_POINTS = "#444444"  # raw PSE dots
COLOR_RIBBON = "#cccccc"  # ±SE shading
COLOR_ZERO = "#888888"  # PSE = 0 reference line

# Publication figure dimensions (inches)
FIG_WIDTH = 10.0
FIG_HEIGHT = 3.4

# ============================================================================
# DATA LOADING & FILTERING
# ============================================================================


def load_pse(illusion_folder: str, results_root: Path) -> pd.DataFrame:
    """
    Load and filter pse_summary.csv for one illusion.

    Excludes fits where pse_se > (diff_max − diff_min), matching the
    classification logic in pipeline/module_3/plot_results.py.

    Returns a DataFrame with columns:
        illusion_strength, pse, pse_se  (reliable fits only)
    """
    pse_path = results_root / illusion_folder / "pse_summary.csv"
    psych_path = results_root / illusion_folder / "psychometric_data.csv"

    if not pse_path.exists():
        raise FileNotFoundError(f"PSE summary not found: {pse_path}")
    if not psych_path.exists():
        raise FileNotFoundError(f"Psychometric data not found: {psych_path}")

    pse = pd.read_csv(pse_path)
    psych = pd.read_csv(psych_path)

    diff_min = float(psych["true_diff"].min())
    diff_max = float(psych["true_diff"].max())
    delta_range = diff_max - diff_min

    # Keep only reliable fits
    reliable = (
        pse["fit_success"].astype(bool)
        & pse["pse"].notna()
        & pse["pse_se"].notna()
        & (pse["pse_se"] <= delta_range)
    )
    df = pse[reliable][["illusion_strength", "pse", "pse_se"]].copy()
    df = df.sort_values("illusion_strength").reset_index(drop=True)
    return df


# ============================================================================
# PLOTTING
# ============================================================================


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    show_ylabel: bool,
    folder: str = "",
) -> None:
    """
    Draw one illusion panel: SE ribbon + raw dots + LOESS curve.

    Args:
        ax:           Matplotlib axes to draw on.
        df:           DataFrame with illusion_strength, pse, pse_se columns.
        title:        Subplot title (illusion name).
        show_ylabel:  Whether to label the y-axis (left panel only).
        folder:       Results folder name, used to look up per-illusion YLIM.
    """
    x = df["illusion_strength"].values
    y = df["pse"].values
    se = df["pse_se"].values

    # ── SE ribbon (clipped to RIBBON_CLIP to prevent outlier spikes) ────────
    if RIBBON_CLIP is not None:
        ribbon_lo = np.clip(y - se, y - RIBBON_CLIP, y + RIBBON_CLIP)
        ribbon_hi = np.clip(y + se, y - RIBBON_CLIP, y + RIBBON_CLIP)
    else:
        ribbon_lo = y - se
        ribbon_hi = y + se
    ax.fill_between(
        x,
        ribbon_lo,
        ribbon_hi,
        color=COLOR_RIBBON,
        linewidth=0,
        zorder=1,
        label="±1 SE",
    )

    # ── Raw PSE points ────────────────────────────────────────────────────────
    ax.scatter(
        x,
        y,
        color=COLOR_POINTS,
        s=18,
        zorder=3,
        linewidths=0,
    )

    # ── LOESS smooth ─────────────────────────────────────────────────────────
    if len(x) >= 4:
        smoothed = lowess(y, x, frac=LOESS_FRAC, it=0, return_sorted=True)
        ax.plot(
            smoothed[:, 0],
            smoothed[:, 1],
            color=COLOR_LINE,
            linewidth=1.8,
            zorder=4,
        )

    # ── PSE = 0 reference line ────────────────────────────────────────────────
    ax.axhline(0, color=COLOR_ZERO, linewidth=0.8, linestyle="--", zorder=2)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Illusion strength", fontsize=9)

    if show_ylabel:
        ax.set_ylabel("PSE (physical difference\nat 50% threshold)", fontsize=9)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.tick_params(axis="both", labelsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Minimal spine style
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

    ax.grid(axis="y", linewidth=0.4, color="#e0e0e0", zorder=0)

    # ── Per-illusion y-axis limits ────────────────────────────────────────────
    ylim = YLIM.get(folder)
    if ylim is not None:
        ymin, ymax = ylim
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)


# ============================================================================
# MAIN
# ============================================================================


def main(results_root: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharey=False,
    )

    for i, (ax, (display_name, folder)) in enumerate(zip(axes, ILLUSIONS)):
        try:
            df = load_pse(folder, results_root)
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping {display_name}: {e}")
            ax.set_visible(False)
            continue

        plot_panel(
            ax=ax,
            df=df,
            title=display_name,
            show_ylabel=(i == 0),
            folder=folder,
        )
        print(f"  ✓ {display_name}: {len(df)} reliable PSE points")

    fig.tight_layout(w_pad=2.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot smoothed PSE comparison across illusions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Path to the results/ directory (default: ./results)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/pse_comparison.png"),
        help="Output PNG path (default: tests/pse_comparison.png)",
    )
    args = parser.parse_args()
    main(results_root=args.results_root, out_path=args.out)
