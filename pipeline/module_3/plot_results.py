"""
pipeline/module_3/plot_results.py - Psychometric curve and PSE drift plots.

Produces three figures per illusion:

  fig1_error_by_difficulty.png
      Error rate vs. illusion strength, one curve per |true_diff| difficulty
      bin. Red = hard (small |Δ|) → green = easy (large |Δ|). Directly
      comparable to Makowski et al. (2023) Figure 3.

  fig2_pse_vs_strength.png
      PSE (± SE) vs. illusion strength with fit-quality classification:
        • Solid marker   — reliable fit, PSE well within tested Δ range
        • Hollow marker  — PSE near the boundary (interpret with caution)
        • Dotted line    — fit failed / PSE outside tested range
      Linear trend fitted through reliable points only.

  fig3_psychometric_curves.png
      Overlaid cumulative Gaussian fits per strength level (diagnostic).
      Colours follow the same Red→Green convention as Figure 1.

Colour convention (consistent across all figures):
  Red    = hard / incongruent  (high positive illusion strength / small |Δ|)
  Green  = easy / congruent    (high negative illusion strength / large |Δ|)

Saved to results/<illusion_name>/figures/.
"""

import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf

# ============================================================================
# CONSTANTS
# ============================================================================

N_DIFFICULTY_BINS = 6

DIFFICULTY_LABELS = [
    "Very Hard",
    "Hard",
    "Medium-Hard",
    "Medium-Easy",
    "Easy",
    "Very Easy",
]


# ============================================================================
# HELPERS
# ============================================================================


def _cumgauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def _difficulty_colormap(n: int) -> list:
    """
    Return n colours from Red→Orange→Teal→Green (hard→easy).

    Skips the pale yellow centre of RdYlGn that is invisible on white.
    Index 0 = red (hardest), index n-1 = green (easiest).
    """
    cmap = cm.get_cmap("RdYlGn")
    half = n // 2
    remainder = n - half
    positions = np.concatenate(
        [
            np.linspace(0.02, 0.42, half),  # red → orange
            np.linspace(0.58, 0.98, remainder),  # teal → green
        ]
    )
    return [cmap(p) for p in positions]


# ============================================================================
# DATA LOADING
# ============================================================================


def _load_participants(participants_dir: Path) -> pd.DataFrame:
    """Load all participant_*.jsonl files into a single DataFrame."""
    jsonl_files = sorted(participants_dir.glob("participant_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No participant_*.jsonl files found in {participants_dir}"
        )
    records = []
    for path in jsonl_files:
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return pd.DataFrame(records)


# ============================================================================
# FIGURE 1: ERROR RATE BY DIFFICULTY
# ============================================================================


def _build_difficulty_bins(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """
    Assign each trial to one of n_bins difficulty bins based on |true_diff|.

    Bins are defined by quantiles of the unique |true_diff| values so that
    each bin spans a roughly equal range of difficulty levels.

    Adds columns: abs_diff, diff_bin (verbal label), diff_bin_mid, diff_bin_idx.
    """
    df = df[df["correct"].notna()].copy()
    df["correct"] = df["correct"].astype(int)
    df["abs_diff"] = df["true_diff"].abs()

    unique_abs = np.sort(df["abs_diff"].unique())
    bin_edges = np.quantile(unique_abs, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1e-9

    bin_labels, bin_mids = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        label = DIFFICULTY_LABELS[i] if i < len(DIFFICULTY_LABELS) else f"Bin {i + 1}"
        bin_labels.append(label)
        bin_mids.append((lo + hi) / 2)

    label_map, mid_map, idx_map = {}, {}, {}
    for val in unique_abs:
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if lo < val <= hi:
                label_map[val] = bin_labels[i]
                mid_map[val] = bin_mids[i]
                idx_map[val] = i
                break

    df["diff_bin"] = df["abs_diff"].map(label_map)
    df["diff_bin_mid"] = df["abs_diff"].map(mid_map)
    df["diff_bin_idx"] = df["abs_diff"].map(idx_map)
    return df


def plot_error_by_difficulty(
    illusion: dict,
    participants_dir: Path,
    figures_dir: Path,
) -> None:
    """
    Figure 1: error rate vs. illusion strength, one curve per difficulty bin.

    Loads raw participant JSONL files so that per-trial correctness is available.
    """
    name = illusion["name"]

    raw_df = _load_participants(participants_dir)
    df = _build_difficulty_bins(raw_df, n_bins=N_DIFFICULTY_BINS)
    df["error"] = 1 - df["correct"]

    agg = (
        df.groupby(["illusion_strength", "diff_bin", "diff_bin_mid", "diff_bin_idx"])
        .agg(prop_error=("error", "mean"), n=("error", "count"))
        .reset_index()
        .sort_values(["diff_bin_mid", "illusion_strength"])
    )

    bins_ordered = (
        agg.drop_duplicates("diff_bin").sort_values("diff_bin_idx")["diff_bin"].tolist()
    )
    colors = _difficulty_colormap(len(bins_ordered))
    bin_colors = {label: colors[i] for i, label in enumerate(bins_ordered)}

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for bin_label in bins_ordered:
        subset = agg[agg["diff_bin"] == bin_label].sort_values("illusion_strength")
        ax.plot(
            subset["illusion_strength"],
            subset["prop_error"],
            color=bin_colors[bin_label],
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            alpha=0.9,
            label=bin_label,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
        label="No illusion (strength = 0)",
    )
    ax.axhline(
        0.5,
        color="grey",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
        label="Chance (50% error)",
    )

    ax.set_xlabel("Illusion strength", fontsize=12)
    ax.set_ylabel("Probability of error", fontsize=12)
    ax.set_title(f"Error Rate vs. Illusion Strength — {name}", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="upper left", title="Task difficulty", title_fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = figures_dir / "fig1_error_by_difficulty.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure 1 → {out_path}")
    plt.close(fig)


# ============================================================================
# FIGURE 2: PSE vs. ILLUSION STRENGTH
# ============================================================================


def plot_pse_vs_strength(
    illusion: dict,
    pse_summary: pd.DataFrame,
    psych_data: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Figure 2: PSE (±SE) vs. illusion strength with fit-quality classification.

      Good fit  — PSE well within the tested Δ range and SE ≤ Δ range width
                  → solid marker
      Edge fit  — PSE within 10% of boundary                → hollow marker
      Failed    — fit failed, PSE outside range, or SE > Δ range width
                  → dotted vline (excluded from trend line)

    Y-axis is clipped to the tested Δ boundary so that a single outlier
    with a runaway SE cannot compress the rest of the plot.
    """
    name = illusion["name"]
    df = pse_summary.copy()
    df["fit_success"] = df["fit_success"].astype(bool)

    diff_min = float(psych_data["true_diff"].min())
    diff_max = float(psych_data["true_diff"].max())
    delta_range = diff_max - diff_min
    margin = 0.10 * delta_range

    def _classify(row):
        if not row["fit_success"] or np.isnan(row["pse"]):
            return "failed"
        # SE larger than the full tested Δ range indicates an unconstrained fit
        if not np.isnan(row["pse_se"]) and row["pse_se"] > delta_range:
            return "failed"
        if row["pse"] <= diff_min + margin or row["pse"] >= diff_max - margin:
            return "edge"
        return "good"

    df["fit_class"] = df.apply(_classify, axis=1)
    good = df[df["fit_class"] == "good"]
    edge = df[df["fit_class"] == "edge"]
    failed = df[df["fit_class"] == "failed"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if not good.empty:
        ax.errorbar(
            good["illusion_strength"],
            good["pse"],
            yerr=good["pse_se"],
            fmt="o-",
            color="#5c4dc9",
            markersize=7,
            linewidth=2,
            capsize=4,
            label="PSE ± SE (reliable fit)",
        )

    if not edge.empty:
        ax.errorbar(
            edge["illusion_strength"],
            edge["pse"],
            yerr=edge["pse_se"],
            fmt="o",
            color="#e08c00",
            markersize=8,
            linewidth=0,
            capsize=4,
            markerfacecolor="none",
            markeredgewidth=2,
            label="PSE near range boundary (interpret with caution)",
        )

    if not failed.empty:
        for _, row in failed.iterrows():
            ax.axvline(
                row["illusion_strength"],
                color="#cc3333",
                linewidth=1.2,
                linestyle=":",
                alpha=0.6,
            )
        ax.plot(
            [],
            [],
            linestyle=":",
            color="#cc3333",
            linewidth=1.5,
            label="Fit failed / SE too large — excluded from trend",
        )

    if len(good) >= 3:
        z = np.polyfit(good["illusion_strength"], good["pse"], 1)
        x_line = np.linspace(
            df["illusion_strength"].min(), df["illusion_strength"].max(), 200
        )
        ax.plot(
            x_line,
            np.polyval(z, x_line),
            "--",
            color="#a0a0a0",
            linewidth=1.2,
            label=f"Linear trend (slope={z[0]:+.4f})",
        )

    ax.axhspan(diff_min, diff_min + margin, alpha=0.07, color="orange")
    ax.axhspan(diff_max - margin, diff_max, alpha=0.07, color="orange")
    ax.axhline(
        diff_min,
        color="orange",
        linewidth=0.7,
        linestyle="--",
        alpha=0.5,
        label=f"Tested Δ boundary (±{diff_max:.2f})",
    )
    ax.axhline(
        diff_max,
        color="orange",
        linewidth=0.7,
        linestyle="--",
        alpha=0.5,
        label="_nolegend_",
    )
    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="No bias (PSE = 0)",
    )

    # Clip y-axis to tested Δ boundary — prevents runaway SE bars from
    # compressing all other points into a flat line.
    y_pad = margin * 2
    ax.set_ylim(diff_min - y_pad, diff_max + y_pad)

    ax.set_xlabel("Illusion strength", fontsize=12)
    ax.set_ylabel("PSE  (physical difference at 50% threshold)", fontsize=12)
    ax.set_title(f"PSE Shift vs. Illusion Strength — {name}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = figures_dir / "fig2_pse_vs_strength.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure 2 → {out_path}")
    plt.close(fig)


# ============================================================================
# FIGURE 3: RAW PSYCHOMETRIC CURVES
# ============================================================================


def plot_psychometric_curves(
    illusion: dict,
    psych_data: pd.DataFrame,
    pse_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Figure 3: small-multiple grid of psychometric curves, one per strength level.

    Each panel shows the scatter of (true_diff, prop_positive) and the
    fitted cumulative Gaussian. A vertical dashed line marks the PSE.
    Colours follow the same Red→Green convention as Figure 1.
    """
    positive_option = illusion["response_options"][0]
    name = illusion["name"]

    strengths = sorted(psych_data["illusion_strength"].unique())
    n = len(strengths)
    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))

    colors_list = list(reversed(_difficulty_colormap(n)))
    colors = {s: colors_list[i] for i, s in enumerate(strengths)}

    x_smooth = np.linspace(
        psych_data["true_diff"].min(), psych_data["true_diff"].max(), 300
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.2, n_rows * 2.8),
        sharex=True,
        sharey=True,
    )
    axes_flat = np.array(axes).flatten()

    for i, strength in enumerate(strengths):
        ax = axes_flat[i]
        color = colors[strength]
        subset = psych_data[psych_data["illusion_strength"] == strength]
        pse_row = pse_summary[pse_summary["illusion_strength"] == strength]

        ax.scatter(
            subset["true_diff"],
            subset["prop_positive"],
            color=color,
            s=28,
            zorder=3,
            alpha=0.8,
        )

        if not pse_row.empty and bool(pse_row.iloc[0]["fit_success"]):
            pse = float(pse_row.iloc[0]["pse"])
            sigma = float(pse_row.iloc[0]["sigma"])
            ax.plot(
                x_smooth, _cumgauss(x_smooth, pse, sigma), color=color, linewidth=1.6
            )
            ax.axvline(pse, color=color, linewidth=0.9, linestyle="--", alpha=0.7)
            ax.set_title(f"str={strength:+.1f}\nPSE={pse:+.3f}", fontsize=7.5)
        else:
            ax.set_title(f"str={strength:+.1f}\n(no fit)", fontsize=7.5)

        ax.axvline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.4)
        ax.axhline(0.5, color="grey", linewidth=0.6, linestyle=":", alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

        # Axis labels only on the border panels
        if i % n_cols == 0:
            ax.set_ylabel(f"P('{positive_option}')", fontsize=7)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("True Δ", fontsize=7)

    # Hide any unused panels
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Psychometric Functions — {name}", fontsize=12, y=1.01)
    fig.tight_layout()

    out_path = figures_dir / "fig3_psychometric_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure 3 → {out_path}")
    plt.close(fig)


# ============================================================================
# FIGURE 4: SLOPE vs. ILLUSION STRENGTH
# ============================================================================


def plot_slope_vs_strength(
    illusion: dict,
    pse_summary: pd.DataFrame,
    psych_data: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Figure 4: slope (1 / (σ√2π)) ± SE vs. illusion strength.

    Uses the same fit-quality classification as Figure 2:
      good   — converged, PSE well within tested Δ range
      edge   — PSE near boundary
      failed — fit failed or SE too large

    A steeper slope (larger value) means the model is more sensitive to
    physical differences at that illusion strength.
    """
    name = illusion["name"]
    diff_vals = psych_data["true_diff"].unique()
    diff_min = float(diff_vals.min())
    diff_max = float(diff_vals.max())
    delta_range = diff_max - diff_min
    margin = 0.10 * delta_range

    df = pse_summary.copy()
    df["fit_success"] = df["fit_success"].astype(bool)

    # Derive slope and slope_SE from sigma
    df["slope"] = np.where(
        df["fit_success"] & df["sigma"].notna() & (df["sigma"] > 0),
        1.0 / (df["sigma"] * np.sqrt(2 * np.pi)),
        np.nan,
    )
    df["slope_se"] = np.where(
        df["fit_success"]
        & df["sigma"].notna()
        & df["sigma_se"].notna()
        & (df["sigma"] > 0),
        df["sigma_se"] / (df["sigma"] ** 2 * np.sqrt(2 * np.pi)),
        np.nan,
    )

    def _classify(row):
        if not row["fit_success"] or np.isnan(row["pse"]):
            return "failed"
        if not np.isnan(row["pse_se"]) and row["pse_se"] > delta_range:
            return "failed"
        if row["pse"] <= diff_min + margin or row["pse"] >= diff_max - margin:
            return "edge"
        return "good"

    df["fit_class"] = df.apply(_classify, axis=1)
    good = df[df["fit_class"] == "good"]
    edge = df[df["fit_class"] == "edge"]
    failed = df[df["fit_class"] == "failed"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if not good.empty:
        ax.errorbar(
            good["illusion_strength"],
            good["slope"],
            yerr=good["slope_se"],
            fmt="o-",
            color="#1a7f5c",
            markersize=7,
            linewidth=2,
            capsize=4,
            label="Slope ± SE (reliable fit)",
        )

    if not edge.empty:
        ax.errorbar(
            edge["illusion_strength"],
            edge["slope"],
            yerr=edge["slope_se"],
            fmt="o",
            color="#e08c00",
            markersize=8,
            linewidth=0,
            capsize=4,
            markerfacecolor="none",
            markeredgewidth=2,
            label="Slope near boundary (interpret with caution)",
        )

    if not failed.empty:
        for _, row in failed.iterrows():
            ax.axvline(
                row["illusion_strength"],
                color="#cc3333",
                linewidth=1.2,
                linestyle=":",
                alpha=0.6,
            )
        ax.plot(
            [],
            [],
            linestyle=":",
            color="#cc3333",
            linewidth=1.5,
            label="Fit failed — excluded from trend",
        )

    if len(good) >= 3:
        z = np.polyfit(good["illusion_strength"], good["slope"], 1)
        x_line = np.linspace(
            df["illusion_strength"].min(), df["illusion_strength"].max(), 200
        )
        ax.plot(
            x_line,
            np.polyval(z, x_line),
            "--",
            color="#a0a0a0",
            linewidth=1.2,
            label=f"Linear trend (slope={z[0]:+.5f})",
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Illusion strength", fontsize=12)
    ax.set_ylabel("Slope  (1 / σ√2π)  at PSE", fontsize=12)
    ax.set_title(f"Psychometric Slope vs. Illusion Strength — {name}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = figures_dir / "fig4_slope_vs_strength.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure 4 → {out_path}")
    plt.close(fig)


# ============================================================================
# FIGURE 5: RESPONSE SURFACE HEATMAP
# ============================================================================


def plot_response_surface(
    illusion: dict,
    psych_data: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Figure 5: heatmap of p(positive response) across the full
    (illusion_strength × true_diff) grid.

    Colour scale is diverging around 0.5 (white = no bias).
    Columns = illusion strength levels, rows = true_diff levels.
    A structured illusion will show a visible left-right shift in each column
    as illusion_strength changes; Contrast/White will appear uniformly coloured.
    """
    name = illusion["name"]
    positive_option = illusion["response_options"][0]

    pivot = psych_data.pivot_table(
        index="true_diff",
        columns="illusion_strength",
        values="prop_positive",
    )
    # Rows: large diff at top (positive = model should choose positive option)
    pivot = pivot.sort_index(ascending=False)

    strengths = pivot.columns.tolist()
    diffs = pivot.index.tolist()

    fig, ax = plt.subplots(
        figsize=(max(8, len(strengths) * 0.6), max(5, len(diffs) * 0.35))
    )

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"P(respond '{positive_option}')", fontsize=10)
    cbar.ax.axhline(0.5, color="black", linewidth=1.0, linestyle="--")

    ax.set_xticks(range(len(strengths)))
    ax.set_xticklabels(
        [f"{s:+.1f}" for s in strengths], fontsize=7, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(diffs)))
    ax.set_yticklabels([f"{d:+.4f}" for d in diffs], fontsize=7)

    ax.set_xlabel("Illusion strength", fontsize=12)
    ax.set_ylabel("True physical difference (Δ)", fontsize=12)
    ax.set_title(f"Response Surface — {name}", fontsize=13)

    # Horizontal line at diff ≈ 0 (nearest value)
    nearest_zero_idx = int(np.argmin(np.abs(np.array(diffs))))
    ax.axhline(
        nearest_zero_idx - 0.5, color="white", linewidth=1.2, linestyle="--", alpha=0.6
    )

    fig.tight_layout()
    out_path = figures_dir / "fig5_response_surface.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure 5 → {out_path}")
    plt.close(fig)


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def run_plotting(
    illusion: dict,
    psych_data: pd.DataFrame,
    pse_summary: pd.DataFrame,
    results_root: Path,
    participants_dir: Path,
) -> None:
    """
    Generate and save all three figures for one illusion.

    Args:
        illusion:         Illusion config dict.
        psych_data:       DataFrame from fit_psychometrics.aggregate_psychometric_data.
        pse_summary:      DataFrame from fit_psychometrics.run_fitting.
        results_root:     Top-level results directory.
        participants_dir: Directory of participant_XX.jsonl (needed for Fig 1).
    """
    figures_dir = results_root / illusion["name"] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pse_summary = pse_summary.copy()
    pse_summary["fit_success"] = pse_summary["fit_success"].astype(bool)

    plot_error_by_difficulty(illusion, participants_dir, figures_dir)
    plot_pse_vs_strength(illusion, pse_summary, psych_data, figures_dir)
    plot_psychometric_curves(illusion, psych_data, pse_summary, figures_dir)
    plot_slope_vs_strength(illusion, pse_summary, psych_data, figures_dir)
    plot_response_surface(illusion, psych_data, figures_dir)
