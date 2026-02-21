#!/usr/bin/env python3
"""
fit_psychometrics.py - Fit sigmoid psychometric functions and extract PSE (Phase 3)

Reads all participant_XX.jsonl files from results/synthetic_participants/, pools
them into a single DataFrame, and for each illusion strength level fits a
cumulative Gaussian (sigmoid) to the proportion of "Top" responses as a function
of true physical difference.

With 20 synthetic participants × 16 difference levels × 15 strength levels the
aggregated proportions are based on 20 samples per cell, giving a smooth,
human-like psychometric curve suitable for reliable PSE estimation.

The Point of Subjective Equality (PSE) is the x-value where the fitted curve
crosses 50% — i.e., the physical difference needed for the model to be indifferent.
  PSE = 0    → no bias
  PSE > 0    → model perceives top as shorter (needs extra length to seem equal)
  PSE < 0    → model perceives top as longer

Outputs:
    results/pse_summary.csv       — one row per illusion strength level
    results/psychometric_data.csv — proportion-top per (strength, diff) cell

Usage:
    python fit_psychometrics.py [--participants-dir PATH] [--output-dir PATH]

Requirements:
    pip install numpy scipy pandas
"""

import argparse
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
    Cumulative Gaussian (Φ) — the standard psychometric function.

    P("Top" | x) = Φ((x − μ) / σ)

    Args:
        x:     True physical difference (top − bottom line length).
        mu:    PSE — the x-value at which P = 0.5.
        sigma: Slope parameter (JND / √2); smaller = steeper curve.

    Returns:
        Probability of responding "Top".
    """
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


# ============================================================================
# DATA LOADING
# ============================================================================


def load_participant_file(jsonl_path: Path, participant_id: int) -> list[dict]:
    """
    Load a single participant JSONL file, injecting participant_id if absent.

    Args:
        jsonl_path:     Path to the .jsonl file.
        participant_id: Fallback integer ID (used if the record lacks the field).

    Returns:
        List of record dicts.
    """
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                record.setdefault("participant_id", participant_id)
                records.append(record)
    return records


def load_all_participants(participants_dir: Path) -> pd.DataFrame:
    """
    Discover and load all participant_XX.jsonl files from participants_dir.

    Files are matched by the glob pattern ``participant_*.jsonl`` so any
    zero-padded or non-padded naming is accepted.

    Args:
        participants_dir: Directory produced by query_vlm.py.

    Returns:
        Consolidated DataFrame with columns:
            participant_id, image_id, illusion_strength, true_diff,
            response, correct, responded_top

    Raises:
        FileNotFoundError: If the directory doesn't exist or contains no files.
        ValueError:        If all found files are empty.
    """
    if not participants_dir.exists():
        raise FileNotFoundError(
            f"Participants directory not found: {participants_dir}\n"
            "Run query_vlm.py first."
        )

    jsonl_files = sorted(participants_dir.glob("participant_*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            f"No participant_*.jsonl files found in {participants_dir}\n"
            "Run query_vlm.py first."
        )

    print(f"Found {len(jsonl_files)} participant file(s) in {participants_dir}/")

    all_records: list[dict] = []
    for jsonl_path in jsonl_files:
        # Extract a numeric ID from the filename for use as fallback
        stem = jsonl_path.stem  # e.g. 'participant_03'
        try:
            file_id = int(stem.split("_")[-1])
        except ValueError:
            file_id = 0

        records = load_participant_file(jsonl_path, participant_id=file_id)
        all_records.extend(records)
        print(f"  ✓ {jsonl_path.name}  ({len(records)} responses)")

    if not all_records:
        raise ValueError(f"All participant files in {participants_dir} were empty.")

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["participant_id", "image_id"])
    df["responded_top"] = (df["response"] == "Top").astype(int)

    n_participants = df["participant_id"].nunique()
    n_cells = df.groupby(["illusion_strength", "true_diff"]).ngroups
    print(
        f"\n  Total responses   : {len(df)}"
        f"\n  Unique participants: {n_participants}"
        f"\n  Unique cells       : {n_cells}"
        f"\n  Avg per cell       : {len(df) / n_cells:.1f}\n"
    )
    return df


# ============================================================================
# PSYCHOMETRIC DATA AGGREGATION
# ============================================================================


def aggregate_psychometric_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute proportion of "Top" responses per (illusion_strength, true_diff) cell.

    Pooling across all synthetic participants is handled automatically by
    groupby — n_trials will equal N_PARTICIPANTS × (repetitions per cell).

    Returns a DataFrame with columns:
        illusion_strength, true_diff, n_trials, n_top, prop_top
    """
    grouped = (
        df.groupby(["illusion_strength", "true_diff"])
        .agg(
            n_trials=("responded_top", "count"),
            n_top=("responded_top", "sum"),
        )
        .reset_index()
    )
    grouped["prop_top"] = grouped["n_top"] / grouped["n_trials"]
    return grouped.sort_values(["illusion_strength", "true_diff"]).reset_index(
        drop=True
    )


# ============================================================================
# PSE FITTING
# ============================================================================


def fit_pse(
    diff_values: np.ndarray,
    prop_top: np.ndarray,
) -> dict:
    """
    Fit a cumulative Gaussian to the psychometric data and return PSE + slope.

    Args:
        diff_values: Array of true physical differences (x-axis).
        prop_top:    Corresponding proportions of "Top" responses (y-axis).

    Returns:
        Dict with keys: pse, sigma, pse_se, sigma_se, fit_success, note.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                cumulative_gaussian,
                diff_values,
                prop_top,
                p0=[0.0, 0.1],  # initial guess: unbiased, moderate slope
                bounds=([-1.0, 0.001], [1.0, 2.0]),
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
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fit psychometric functions and extract PSE per illusion strength",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--participants-dir",
        type=str,
        default="./results/synthetic_participants",
        help=(
            "Directory containing participant_XX.jsonl files "
            "(default: ./results/synthetic_participants)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save output CSVs (default: ./results)",
    )
    args = parser.parse_args()

    participants_dir = Path(args.participants_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("LOADING SYNTHETIC PARTICIPANT DATA")
    print("=" * 60)

    try:
        df = load_all_participants(participants_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        raise SystemExit(1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    psych_data = aggregate_psychometric_data(df)
    psych_path = output_dir / "psychometric_data.csv"
    psych_data.to_csv(psych_path, index=False)
    print(f"✓ Psychometric data saved to {psych_path}\n")

    # ── Fit PSE per strength level ────────────────────────────────────────────
    print("=" * 60)
    print("PSE FITTING RESULTS")
    print("=" * 60)
    print(f"  {'Strength':>8}  {'PSE':>8}  {'±SE':>8}  {'Sigma':>8}  {'OK':>4}")
    print("  " + "-" * 46)

    pse_rows = []
    for strength in sorted(df["illusion_strength"].unique()):
        subset = psych_data[psych_data["illusion_strength"] == strength]
        result = fit_pse(
            subset["true_diff"].values,
            subset["prop_top"].values,
        )
        pse_rows.append({"illusion_strength": strength, **result})

        ok_str = "✓" if result["fit_success"] else "✗"
        pse_str = f"{result['pse']:+.4f}" if not np.isnan(result["pse"]) else "   NaN"
        se_str = (
            f"{result['pse_se']:.4f}" if not np.isnan(result["pse_se"]) else "   NaN"
        )
        sig_str = (
            f"{result['sigma']:.4f}" if not np.isnan(result["sigma"]) else "   NaN"
        )
        note = f"  [{result['note']}]" if result["note"] else ""
        print(
            f"  {strength:>8.1f}  {pse_str:>8}  {se_str:>8}  {sig_str:>8}  {ok_str:>4}{note}"
        )

    print("=" * 60)

    pse_df = pd.DataFrame(pse_rows)
    pse_path = output_dir / "pse_summary.csv"
    pse_df.to_csv(pse_path, index=False)
    print(f"\n✓ PSE summary saved to {pse_path}")
    print("\nInterpretation:")
    print("  PSE = 0   → no systematic bias at this illusion strength")
    print(
        "  PSE > 0   → model perceives top line as shorter (illusion biases toward bottom)"
    )
    print(
        "  PSE < 0   → model perceives top line as longer  (illusion biases toward top)"
    )
    print("\nCore finding: does PSE shift systematically with illusion strength?")
    print(
        "\nNext step: run plot_results.py to visualise psychometric curves and PSE drift."
    )


if __name__ == "__main__":
    main()
