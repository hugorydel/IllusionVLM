"""
pipeline/module_3/export_diagnostics.py - Diagnostic CSV export for Module 3.

Produces four analysis files per illusion alongside the existing outputs:

  aggregated_responses.csv   — full psychometric surface, one row per
                               (illusion_strength × true_diff) cell
  fit_diagnostics.csv        — per-curve fit quality metrics and warnings
  baseline_summary.csv       — no-illusion (strength=0) sanity checks
  illusion_summary.csv       — one-row-per-illusion usability judgments

All functions take DataFrames already in memory from run_fitting() and
write to results/<illusion_name>/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.special import erf

# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _cumgauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Cumulative Gaussian — local copy to avoid a circular import."""
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def _safe_round(x, decimals: int):
    """Round x unless it is NaN."""
    return round(float(x), decimals) if not np.isnan(x) else np.nan


# ============================================================================
# FILE 1 — AGGREGATED RESPONSES
# ============================================================================


def export_aggregated_responses(
    df: pd.DataFrame,
    psych_data: pd.DataFrame,
    illusion: dict,
    results_dir: Path,
    model: str,
) -> pd.DataFrame:
    """
    One row per (illusion_strength × true_diff) cell.

    Columns:
        model, illusion, illusion_strength, true_diff,
        n_trials, n_positive_response, p_positive_response,
        accuracy, positive_option_label

    accuracy is NaN at true_diff=0 (no ground truth for equal stimuli).
    """
    acc = (
        df[df["true_diff"] != 0]
        .groupby(["illusion_strength", "true_diff"])
        .agg(accuracy=("correct", "mean"))
        .reset_index()
    )

    out = psych_data.rename(
        columns={
            "n_positive": "n_positive_response",
            "prop_positive": "p_positive_response",
        }
    ).merge(acc, on=["illusion_strength", "true_diff"], how="left")

    out.insert(0, "model", model)
    out.insert(1, "illusion", illusion["name"])
    out["positive_option_label"] = illusion["response_options"][0]

    out = out[
        [
            "model",
            "illusion",
            "illusion_strength",
            "true_diff",
            "n_trials",
            "n_positive_response",
            "p_positive_response",
            "accuracy",
            "positive_option_label",
        ]
    ]

    path = results_dir / "aggregated_responses.csv"
    out.to_csv(path, index=False)
    print(f"  ✓ Aggregated responses → {path}")
    return out


# ============================================================================
# FILE 2 — FIT DIAGNOSTICS
# ============================================================================


def export_fit_diagnostics(
    psych_data: pd.DataFrame,
    pse_summary: pd.DataFrame,
    illusion: dict,
    results_dir: Path,
    model: str,
) -> pd.DataFrame:
    """
    One row per fitted psychometric curve (per illusion_strength level).

    Fit quality: R², AIC, BIC computed via binomial log-likelihood
    (the correct noise model for proportion data).

    Warning flags:
        boundary_hit          — PSE within 5 % of the tested Δ range boundary
        monotonicity_warning  — Spearman ρ(prop_positive, true_diff) < 0  AND  p < 0.1
        flat_curve_warning    — max − min prop_positive < 0.1 across the diff range
    """
    diff_vals = psych_data["true_diff"].unique()
    diff_min = float(diff_vals.min())
    diff_max = float(diff_vals.max())
    bdy_margin = 0.05 * (diff_max - diff_min)
    k_params = 2  # PSE (mu) and sigma
    eps = 1e-10

    rows = []
    for strength in sorted(pse_summary["illusion_strength"].unique()):
        fit_row = pse_summary[pse_summary["illusion_strength"] == strength].iloc[0]
        subset = psych_data[psych_data["illusion_strength"] == strength].copy()

        pse = float(fit_row["pse"])
        sigma = float(fit_row["sigma"])
        pse_se = float(fit_row["pse_se"])
        sigma_se = float(fit_row["sigma_se"])
        converged = bool(fit_row["fit_success"])

        # Slope at PSE = derivative of Φ at the inflection point
        if converged and not np.isnan(sigma) and sigma > 0:
            slope = 1.0 / (sigma * np.sqrt(2 * np.pi))
            slope_se = (
                sigma_se / (sigma**2 * np.sqrt(2 * np.pi))
                if not np.isnan(sigma_se)
                else np.nan
            )
        else:
            slope = slope_se = np.nan

        pse_ci_low = pse - 1.96 * pse_se if not np.isnan(pse_se) else np.nan
        pse_ci_high = pse + 1.96 * pse_se if not np.isnan(pse_se) else np.nan
        n_points = len(subset)

        # ── Goodness-of-fit ──────────────────────────────────────────────────
        fit_r2 = aic = bic = np.nan
        if converged and not np.isnan(pse) and not np.isnan(sigma):
            p_hat = _cumgauss(subset["true_diff"].values, pse, sigma)
            p_obs = subset["prop_positive"].values
            ss_res = np.sum((p_obs - p_hat) ** 2)
            ss_tot = np.sum((p_obs - p_obs.mean()) ** 2)
            fit_r2 = _safe_round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else np.nan

            # Binomial log-likelihood: Σ [ k·ln(p̂) + (n−k)·ln(1−p̂) ]
            k = subset["n_positive"].values.astype(float)
            n = subset["n_trials"].values.astype(float)
            p_clipped = np.clip(p_hat, eps, 1 - eps)
            ll = float(np.sum(k * np.log(p_clipped) + (n - k) * np.log(1 - p_clipped)))
            aic = _safe_round(-2 * ll + 2 * k_params, 3)
            bic = _safe_round(-2 * ll + np.log(n_points) * k_params, 3)

        # ── Warning flags ────────────────────────────────────────────────────
        boundary_hit = bool(
            converged
            and not np.isnan(pse)
            and (pse <= diff_min + bdy_margin or pse >= diff_max - bdy_margin)
        )

        mono_warning = False
        if len(subset) >= 4:
            rho, p_val = scipy_stats.spearmanr(
                subset["true_diff"], subset["prop_positive"]
            )
            mono_warning = bool(rho < 0 and p_val < 0.1)

        flat_warning = bool(
            subset["prop_positive"].max() - subset["prop_positive"].min() < 0.1
        )

        rows.append(
            {
                "model": model,
                "illusion": illusion["name"],
                "illusion_strength": strength,
                "n_points_used": n_points,
                "PSE": _safe_round(pse, 4),
                "PSE_SE": _safe_round(pse_se, 4),
                "PSE_CI_low": _safe_round(pse_ci_low, 4),
                "PSE_CI_high": _safe_round(pse_ci_high, 4),
                "slope": _safe_round(slope, 6),
                "slope_SE": _safe_round(slope_se, 6),
                "sigma": _safe_round(sigma, 4),
                "sigma_SE": _safe_round(sigma_se, 4),
                "converged": converged,
                "fit_status": str(fit_row["note"]),
                "fit_R2": fit_r2,
                "AIC": aic,
                "BIC": bic,
                "boundary_hit": boundary_hit,
                "monotonicity_warning": mono_warning,
                "flat_curve_warning": flat_warning,
            }
        )

    diag_df = pd.DataFrame(rows)
    path = results_dir / "fit_diagnostics.csv"
    diag_df.to_csv(path, index=False)
    print(f"  ✓ Fit diagnostics     → {path}")
    return diag_df


# ============================================================================
# FILE 3 — BASELINE SUMMARY  (strength = 0 only)
# ============================================================================


def export_baseline_summary(
    df: pd.DataFrame,
    pse_summary: pd.DataFrame,
    illusion: dict,
    results_dir: Path,
    model: str,
) -> pd.DataFrame:
    """
    Sanity-check summary for the no-illusion condition (illusion_strength = 0).

    baseline_side_bias_index: overall proportion of positive responses at
    strength=0 minus 0.5.  Range −0.5 to +0.5; 0 = no side bias.
    """
    base = df[df["illusion_strength"] == 0].copy()
    pse_row = pse_summary[pse_summary["illusion_strength"] == 0]

    baseline_pse = float(pse_row["pse"].iloc[0]) if not pse_row.empty else np.nan
    baseline_sigma = float(pse_row["sigma"].iloc[0]) if not pse_row.empty else np.nan
    baseline_slope = (
        1.0 / (baseline_sigma * np.sqrt(2 * np.pi))
        if not np.isnan(baseline_sigma) and baseline_sigma > 0
        else np.nan
    )

    nonzero = base[base["true_diff"] != 0].copy()
    baseline_accuracy_overall = (
        float(nonzero["correct"].mean()) if len(nonzero) > 0 else np.nan
    )

    baseline_accuracy_small_diffs = np.nan
    if len(nonzero) > 0:
        nonzero["abs_diff"] = nonzero["true_diff"].abs()
        tercile_cut = nonzero["abs_diff"].quantile(1 / 3)
        small = nonzero[nonzero["abs_diff"] <= tercile_cut]
        baseline_accuracy_small_diffs = (
            float(small["correct"].mean()) if len(small) > 0 else np.nan
        )

    # Overall positive-response rate at strength=0 minus 0.5
    baseline_side_bias_index = _safe_round(
        float(base["responded_positive"].mean()) - 0.5, 4
    )

    row = {
        "model": model,
        "illusion": illusion["name"],
        "baseline_PSE": _safe_round(baseline_pse, 4),
        "baseline_slope": _safe_round(baseline_slope, 6),
        "baseline_accuracy_overall": _safe_round(baseline_accuracy_overall, 4),
        "baseline_accuracy_small_diffs": _safe_round(baseline_accuracy_small_diffs, 4),
        "baseline_side_bias_index": baseline_side_bias_index,
    }

    out = pd.DataFrame([row])
    path = results_dir / "baseline_summary.csv"
    out.to_csv(path, index=False)
    print(f"  ✓ Baseline summary    → {path}")
    return out


# ============================================================================
# FILE 4 — ILLUSION SUMMARY
# ============================================================================


def export_illusion_summary(
    fit_diag_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    psych_data: pd.DataFrame,
    illusion: dict,
    results_dir: Path,
    model: str,
) -> pd.DataFrame:
    """
    One-row synthesis of whether this illusion is psychophysically usable.

    usable_for_PSE_shift:    fit_success_rate ≥ 0.6  AND  PSE moves
                             monotonically (Spearman ρ ≥ 0.3)
    usable_for_shape_effect: fit_success_rate ≥ 0.6  AND  slope varies
                             across strength levels (slope_range > 0)

    monotonic_trend_error: Spearman ρ between illusion_strength and mean
    error rate per strength, derived from psych_data (no raw df needed).
    """
    name = illusion["name"]
    reliable = fit_diag_df[fit_diag_df["converged"] & fit_diag_df["PSE"].notna()].copy()

    fit_success_rate = _safe_round(float(fit_diag_df["converged"].mean()), 4)

    baseline_PSE_abs = np.nan
    if not baseline_df.empty:
        bpse = baseline_df["baseline_PSE"].iloc[0]
        if not np.isnan(bpse):
            baseline_PSE_abs = _safe_round(abs(float(bpse)), 4)

    pse_range = (
        _safe_round(float(reliable["PSE"].max() - reliable["PSE"].min()), 4)
        if len(reliable) >= 2
        else np.nan
    )

    valid_slopes = reliable["slope"].dropna()
    slope_range = (
        _safe_round(float(valid_slopes.max() - valid_slopes.min()), 6)
        if len(valid_slopes) >= 2
        else np.nan
    )

    pos_pse = reliable[reliable["illusion_strength"] > 0]["PSE"]
    neg_pse = reliable[reliable["illusion_strength"] < 0]["PSE"]
    asymmetry_index = (
        _safe_round(float(pos_pse.mean() - neg_pse.mean()), 4)
        if len(pos_pse) > 0 and len(neg_pse) > 0
        else np.nan
    )

    monotonic_trend_PSE = np.nan
    if len(reliable) >= 4:
        rho, _ = scipy_stats.spearmanr(reliable["illusion_strength"], reliable["PSE"])
        monotonic_trend_PSE = _safe_round(float(rho), 4)

    # Mean error rate per strength derived from psych_data
    # accuracy: if true_diff > 0, correct = responded positive (prop_positive);
    #           if true_diff < 0, correct = 1 − prop_positive
    err_rows = []
    for strength, grp in psych_data.groupby("illusion_strength"):
        nonzero = grp[grp["true_diff"] != 0]
        if len(nonzero) == 0:
            continue
        acc = np.where(
            nonzero["true_diff"].values > 0,
            nonzero["prop_positive"].values,
            1 - nonzero["prop_positive"].values,
        )
        err_rows.append(
            {"illusion_strength": strength, "mean_error": 1 - float(acc.mean())}
        )
    err_df = pd.DataFrame(err_rows)

    monotonic_trend_error = np.nan
    if len(err_df) >= 4:
        rho_e, _ = scipy_stats.spearmanr(
            err_df["illusion_strength"], err_df["mean_error"]
        )
        monotonic_trend_error = _safe_round(float(rho_e), 4)

    usable_for_PSE_shift = bool(
        fit_success_rate >= 0.6
        and not np.isnan(pse_range)
        and pse_range > 0
        and not np.isnan(monotonic_trend_PSE)
        and monotonic_trend_PSE >= 0.3
    )
    usable_for_shape_effect = bool(
        fit_success_rate >= 0.6 and not np.isnan(slope_range) and slope_range > 0
    )

    row = {
        "model": model,
        "illusion": name,
        "baseline_PSE_abs": baseline_PSE_abs,
        "fit_success_rate": fit_success_rate,
        "PSE_range_across_strength": pse_range,
        "slope_range_across_strength": slope_range,
        "asymmetry_index": asymmetry_index,
        "monotonic_trend_PSE": monotonic_trend_PSE,
        "monotonic_trend_error": monotonic_trend_error,
        "usable_for_PSE_shift": usable_for_PSE_shift,
        "usable_for_shape_effect": usable_for_shape_effect,
    }

    out = pd.DataFrame([row])
    path = results_dir / "illusion_summary.csv"
    out.to_csv(path, index=False)
    print(f"  ✓ Illusion summary    → {path}")
    return out
