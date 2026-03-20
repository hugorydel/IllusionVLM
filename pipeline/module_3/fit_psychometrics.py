"""
pipeline/module_3/fit_psychometrics.py - Psychometric function fitting.

Loads all participant JSONL files for one illusion, aggregates response
proportions per (strength, difference) cell, fits a cumulative Gaussian
to each strength level, and extracts the PSE.

The positive response option (response_options[0]) is treated as the
"responded_positive" axis — e.g. "Top", "Left", "Vertical" — mirroring
the ground-truth convention used in compute_correct().

Outputs (written to results/<illusion_name>/):
    psychometric_data.csv   — proportion-positive per (strength, diff) cell
    pse_summary.csv         — fitted PSE ± SE and sigma per strength level
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf

# ============================================================================
# PSYCHOMETRIC FUNCTION
# ============================================================================


def cumulative_gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Cumulative Gaussian (Φ): P(respond positive | x) = Φ((x − μ) / σ)

    Args:
        x:     True physical difference (positive option − negative option).
        mu:    PSE — where P = 0.5.
        sigma: Slope parameter; smaller = steeper.
    """
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


# ============================================================================
# DATA LOADING
# ============================================================================


def load_participants(participants_dir: Path, positive_option: str) -> pd.DataFrame:
    """
    Load all participant JSONL files from participants_dir.

    Args:
        participants_dir: Directory containing participant_XX.jsonl files.
        positive_option:  response_options[0] for this illusion (e.g. "Top").

    Returns:
        DataFrame with columns including 'responded_positive'.

    Raises:
        FileNotFoundError: if directory doesn't exist or no files found.
        ValueError:        if all files are empty.
    """
    if not participants_dir.exists():
        raise FileNotFoundError(
            f"Participants directory not found: {participants_dir}\n"
            "Run Module 2 first."
        )

    jsonl_files = sorted(participants_dir.glob("participant_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No participant_*.jsonl files found in {participants_dir}\n"
            "Run Module 2 first."
        )

    print(f"  Found {len(jsonl_files)} participant file(s)")

    all_records: list[dict] = []
    for path in jsonl_files:
        try:
            pid = int(path.stem.split("_")[-1])
        except ValueError:
            pid = 0

        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        rec.setdefault("participant_id", pid)
                        all_records.append(rec)
                    except json.JSONDecodeError:
                        continue

        print(f"    ✓ {path.name}")

    if not all_records:
        raise ValueError(f"All participant files in {participants_dir} are empty.")

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["participant_id", "image_id"])
    df["responded_positive"] = (df["response"] == positive_option).astype(int)

    n_participants = df["participant_id"].nunique()
    n_cells = df.groupby(["illusion_strength", "true_diff"]).ngroups
    print(
        f"\n  Total responses    : {len(df)}"
        f"\n  Participants       : {n_participants}"
        f"\n  Unique cells       : {n_cells}"
        f"\n  Avg per cell       : {len(df) / n_cells:.1f}\n"
    )
    return df


# ============================================================================
# AGGREGATION
# ============================================================================


def aggregate_psychometric_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute proportion of positive responses per (strength, diff) cell.

    Returns DataFrame with columns:
        illusion_strength, true_diff, n_trials, n_positive, prop_positive
    """
    grouped = (
        df.groupby(["illusion_strength", "true_diff"])
        .agg(
            n_trials=("responded_positive", "count"),
            n_positive=("responded_positive", "sum"),
        )
        .reset_index()
    )
    grouped["prop_positive"] = grouped["n_positive"] / grouped["n_trials"]
    return grouped.sort_values(["illusion_strength", "true_diff"]).reset_index(
        drop=True
    )


# ============================================================================
# PSE FITTING
# ============================================================================


def fit_pse(diff_values: np.ndarray, prop_positive: np.ndarray) -> dict:
    """
    Fit a cumulative Gaussian and return PSE, sigma, and standard errors.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                cumulative_gaussian,
                diff_values,
                prop_positive,
                p0=[0.0, 0.1],
                bounds=([-2.0, 0.001], [2.0, 5.0]),
                maxfev=10_000,
            )
        pse, sigma = popt
        perr = np.sqrt(np.diag(pcov))
        return {
            "pse": round(float(pse), 4),
            "sigma": round(float(sigma), 4),
            "pse_se": round(float(perr[0]), 4),
            "sigma_se": round(float(perr[1]), 4),
            "fit_success": True,
            "note": "",
        }
    except Exception as e:
        return {
            "pse": np.nan,
            "sigma": np.nan,
            "pse_se": np.nan,
            "sigma_se": np.nan,
            "fit_success": False,
            "note": str(e),
        }


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def run_fitting(
    illusion: dict,
    results_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load participants, aggregate, fit PSEs, and save CSVs.

    Args:
        illusion:     Illusion config dict.
        results_root: Top-level results directory (e.g. Path("results")).

    Returns:
        (psychometric_data, pse_summary) DataFrames.
    """
    name = illusion["name"]
    positive_option = illusion["response_options"][0]
    illusion_results_dir = results_root / name
    participants_dir = illusion_results_dir / "participants"

    illusion_results_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_participants(participants_dir, positive_option)

    # Aggregate
    psych_data = aggregate_psychometric_data(df)
    psych_path = illusion_results_dir / "psychometric_data.csv"
    psych_data.to_csv(psych_path, index=False)
    print(f"  ✓ Psychometric data → {psych_path}")

    # Fit
    print(f"\n  {'Strength':>8}  {'PSE':>8}  {'±SE':>7}  {'Sigma':>7}  Status")
    print(f"  {'─'*46}")

    pse_rows = []
    for strength in sorted(df["illusion_strength"].unique()):
        subset = psych_data[psych_data["illusion_strength"] == strength]
        result = fit_pse(subset["true_diff"].values, subset["prop_positive"].values)
        pse_rows.append({"illusion_strength": strength, **result})

        ok = "✓" if result["fit_success"] else "✗"
        pse_str = f"{result['pse']:+.4f}" if not np.isnan(result["pse"]) else "     NaN"
        se_str = (
            f"{result['pse_se']:.4f}" if not np.isnan(result["pse_se"]) else "     NaN"
        )
        sig_str = (
            f"{result['sigma']:.4f}" if not np.isnan(result["sigma"]) else "     NaN"
        )
        note = f"  [{result['note']}]" if result["note"] else ""
        print(
            f"  {strength:>8.1f}  {pse_str:>8}  {se_str:>7}  {sig_str:>7}  {ok}{note}"
        )

    pse_df = pd.DataFrame(pse_rows)
    pse_path = illusion_results_dir / "pse_summary.csv"
    pse_df.to_csv(pse_path, index=False)
    print(f"\n  ✓ PSE summary → {pse_path}")

    return psych_data, pse_df
