"""
pipeline/paper/metrics.py - Measures derived from the response cells.

THE SLICE
---------
In this stimulus set the SIGN of illusion_strength encodes congruency relative
to the true difference, not a fixed spatial direction (see conventions.py).
Grouping trials by signed strength therefore mixes trials in which the illusion
points in opposite spatial directions, and the pooled response function is not
monotonic in delta by construction.

Slicing on the illusion's spatial direction recovers a monotonic arrangement:

    direction = -sign(illusion_strength) * sign(true_diff)
    magnitude = |illusion_strength|

Within one (magnitude, direction) group the illusion points the same way in
every cell while delta still spans both signs, because the delta < 0 cells come
from the incongruent strength and the delta > 0 cells from the congruent one of
the same magnitude.

THE MEASURE
-----------
Both directions at a level are fitted jointly, sharing a slope and lapses and
differing only by a shift (psychometrics.fit_illusion_magnitude):

    illusion_magnitude   half the separation between the two directions' PSEs
    side_bias            their midpoint

Fitting jointly rather than differencing two independent PSEs uses every cell
at the level to determine the slope, so an estimate and a profile interval
exist at every level, including those where one direction alone is too noisy
to locate its own PSE.

Delta denotes a different physical quantity per illusion (a length ratio, a
size transform, a brightness percentage), so `magnitude_pct` rescales the
magnitude to a percentage of that illusion's largest tested |delta|. Values at
or above 100% mean the separation exceeds the tested range and is bounded from
below rather than measured.

Error-rate summaries further down are descriptive only; they do not separate
the illusion effect from a response bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.paper.psychometrics import (
    assess_sliced_fit,
    fit_illusion_magnitude,
    fit_psychometric,
)

# Minimum number of populated delta levels needed to fit one sliced cell.
MIN_LEVELS_PER_SLICE = 5

# Width of the 95% interval on magnitude_pct, in percentage points, above
# which a level is treated as unconstrained rather than estimated. Across this
# dataset the median width is 5 and the 90th percentile 13, with a separate
# group at 34 and above; nothing falls between.
MAX_CI_WIDTH_PCT = 25.0

# Peak accuracy, at the easiest difference tested with the illusion switched
# off, below which an illusion is reported as not performed for that observer.
# An illusion effect is not defined when the observer cannot do the underlying
# discrimination at any tested difference.
#
# This replaces a gate on pooled baseline d'. Pooled d' averages accuracy over
# the whole difference grid, so it reports where that grid sits relative to the
# observer's threshold as much as whether the observer can do the task: an
# observer with a coarse but perfectly ordered psychometric function scores low
# simply because most of the levels fall below its threshold. Rod-Frame is the
# case in point - on the shared grid GPT-5.2 pools to d' = 0.51, yet it is at
# 0.90 on the largest shared tilt and at ceiling beyond it, with accuracy rising
# monotonically throughout. Peak accuracy asks what the gate is meant to ask:
# was this task performed at any tested difference? Acuity is then reported
# separately, and on its own scale, by threshold_75.
MIN_PEAK_ACCURACY = 0.75

# Worse-of ordering when combining the two illusion directions of one level.
_STATUS_RANK = {"measured": 0, "bounded": 1, "unreliable": 2}


def _worse_status(a: str, b: str) -> str:
    """The less trustworthy of two slice statuses."""
    return a if _STATUS_RANK.get(a, 9) >= _STATUS_RANK.get(b, 9) else b


# ============================================================================
# SLICING
# ============================================================================


def add_illusion_direction(cells: pd.DataFrame) -> pd.DataFrame:
    """
    Add the illusion's spatial direction and normalised magnitude.

    Zero-strength rows carry no illusion direction and are dropped; the
    baseline is characterised separately by `baseline_summary`.

    Adds:
        k_abs      normalised magnitude 1..7 (0 excluded)
        direction  +1 or -1, the side the illusion pushes the percept toward
    """
    out = cells[cells["illusion_strength"] != 0].copy()
    out["k_abs"] = out.groupby(["species", "illusion"])[
        "illusion_strength"
    ].transform(lambda s: (s.abs() / s.abs().max() * 7).round())
    out["direction"] = (
        -np.sign(out["illusion_strength"]) * np.sign(out["true_diff"])
    ).astype(int)
    return out


# ============================================================================
# PER-DIRECTION FITS
# ============================================================================


def pse_by_direction(sliced: pd.DataFrame) -> pd.DataFrame:
    """
    Fit one psychometric function per (species, illusion, k_abs, direction).

    Args:
        sliced: Output of `add_illusion_direction`.

    Returns:
        One row per slice with the fit, its profile interval and its
        diagnostics from `assess_sliced_fit`.
    """
    rows = []
    keys = ["species", "illusion", "k_abs", "direction"]
    for (species, illusion, k_abs, direction), grp in sliced.groupby(keys):
        grp = grp.sort_values("true_diff")
        if len(grp) < MIN_LEVELS_PER_SLICE:
            continue
        fit = fit_psychometric(
            grp["true_diff"].values,
            grp["n_positive"].values,
            grp["n_trials"].values,
        )
        diag = assess_sliced_fit(
            grp["true_diff"].values,
            grp["n_positive"].values,
            grp["n_trials"].values,
            fit,
        )
        # Where the response saturates the fitted PSE is an extrapolation,
        # so the grid edge is substituted and the derived magnitude becomes a
        # lower bound.
        pse = (
            diag["bound_pse"]
            if diag["status"] == "bounded" and np.isfinite(diag["bound_pse"])
            else fit["pse"]
        )

        rows.append(
            {
                "species": species,
                "illusion": illusion,
                "k_abs": float(k_abs),
                "direction": int(direction),
                "n_levels": len(grp),
                "n_trials": int(grp["n_trials"].sum()),
                "pse": pse,
                "pse_fitted": fit["pse"],
                "crosses_half": diag["crosses_half"],
                "pse_ci_low": fit["pse_ci_low"],
                "pse_ci_high": fit["pse_ci_high"],
                "sigma": fit["sigma"],
                # Kept so figures can redraw the fitted curve without refitting.
                "lapse_lo": fit["lapse_lo"],
                "lapse_hi": fit["lapse_hi"],
                "fit_r2": fit["fit_r2"],
                "converged": fit["converged"],
                "fit_status": diag["status"],
                "reportable": diag["reportable"],
                "max_drawdown": diag["max_drawdown"],
                "pse_abs_frac": diag["pse_abs_frac"],
            }
        )
    return pd.DataFrame(rows)


def illusion_magnitude(
    sliced: pd.DataFrame, jnd: dict | None = None
) -> pd.DataFrame:
    """
    Fit magnitude and side bias jointly at every (species, illusion, k_abs).

    Both illusion directions are fitted together with a shared slope and
    shared lapses, so every level yields an estimate and a profile interval.
    A level with no illusion returns a magnitude near zero whose interval
    spans zero.

    `magnitude_pct` expresses the magnitude as a percentage of that illusion's
    largest tested |delta|, which removes the per-illusion delta units and
    makes levels comparable across illusions.

    `saturated` marks a level at which neither direction's response crosses
    0.5 inside the tested range: the separation is then bounded from below
    rather than measured.
    """
    rows = []
    keys = ["species", "illusion", "k_abs"]
    for (species, illusion, k_abs), grp in sliced.groupby(keys):
        if grp["direction"].nunique() < 2:
            continue
        grid_max = float(grp["true_diff"].abs().max())
        strength_abs = float(grp["illusion_strength"].abs().mean())
        fit = fit_illusion_magnitude(
            grp["true_diff"].values,
            grp["direction"].values,
            grp["n_positive"].values,
            grp["n_trials"].values,
        )
        scale = 100.0 / grid_max if grid_max > 0 else np.nan
        rows.append(
            {
                "species": species,
                "illusion": illusion,
                "k_abs": float(k_abs),
                "n_cells": len(grp),
                "n_trials": int(grp["n_trials"].sum()),
                "grid_max": round(grid_max, 5),
                "illusion_strength_abs": round(strength_abs, 5),
                "illusion_magnitude": fit["magnitude"],
                "magnitude_ci_low": fit["magnitude_ci_low"],
                "magnitude_ci_high": fit["magnitude_ci_high"],
                "magnitude_pct": round(fit["magnitude"] * scale, 3)
                if np.isfinite(fit["magnitude"])
                else np.nan,
                "magnitude_pct_ci_low": round(fit["magnitude_ci_low"] * scale, 3)
                if np.isfinite(fit["magnitude_ci_low"])
                else np.nan,
                "magnitude_pct_ci_high": round(fit["magnitude_ci_high"] * scale, 3)
                if np.isfinite(fit["magnitude_ci_high"])
                else np.nan,
                "side_bias": fit["side_bias"],
                "sigma": fit["sigma"],
                "fit_r2": fit["fit_r2"],
                "converged": fit["converged"],
                "saturated": fit["saturated"],
            }
        )
    out = pd.DataFrame(rows)

    # Magnitude on the shared yardstick. The denominator depends only on the
    # illusion, never on the species or the strength level, so neighbouring
    # levels are not rescaled relative to one another.
    if jnd:
        denom = out["illusion"].map(jnd)
        for src, dst in (
            ("illusion_magnitude", "magnitude_jnd"),
            ("magnitude_ci_low", "magnitude_jnd_ci_low"),
            ("magnitude_ci_high", "magnitude_jnd_ci_high"),
        ):
            out[dst] = (out[src] / denom).round(4)
        out["baseline_jnd"] = denom.round(5)

    out["reportable"] = out["converged"] & out["illusion_magnitude"].notna()
    # A separation wider than the tested range cannot be located within it,
    # and the optimiser's own bound sits at 150%.
    out["exceeds_grid"] = out["magnitude_pct"].abs() >= 100.0
    out["ci_width_pct"] = (
        out["magnitude_pct_ci_high"] - out["magnitude_pct_ci_low"]
    ).round(3)
    out["unconstrained"] = out["ci_width_pct"] > MAX_CI_WIDTH_PCT
    out["fit_status"] = np.where(
        ~out["reportable"],
        "unreliable",
        np.where(
            out["unconstrained"],
            "unconstrained",
            np.where(out["exceeds_grid"] | out["saturated"], "bounded", "measured"),
        ),
    )
    return out.sort_values(keys).reset_index(drop=True)


def illusion_magnitude_from_pairs(by_direction: pd.DataFrame) -> pd.DataFrame:
    """
    Differencing alternative to `illusion_magnitude`, kept for diagnostics.

    The level inherits the worse of its two directions:

        measured    both directions interpolated
        bounded     at least one direction saturated, so the magnitude is a
                    lower bound: the true PSE lies beyond the grid edge
                    substituted for it
        unreliable  either direction failed assess_sliced_fit

    Returns one row per (species, illusion, k_abs).
    """
    rows = []
    for (species, illusion, k_abs), grp in by_direction.groupby(
        ["species", "illusion", "k_abs"]
    ):
        pos = grp[grp["direction"] == 1]
        neg = grp[grp["direction"] == -1]
        if pos.empty or neg.empty:
            continue
        pos, neg = pos.iloc[0], neg.iloc[0]
        if not np.isfinite(pos["pse"]) or not np.isfinite(neg["pse"]):
            continue

        # Half-width of the interval on each PSE, propagated to the magnitude
        # and the bias, both of which are averages of the two.
        def halfwidth(row) -> float:
            lo, hi = row["pse_ci_low"], row["pse_ci_high"]
            return (hi - lo) / 2.0 if np.isfinite(lo) and np.isfinite(hi) else np.nan

        hw = 0.5 * np.sqrt(halfwidth(pos) ** 2 + halfwidth(neg) ** 2)

        rows.append(
            {
                "species": species,
                "illusion": illusion,
                "k_abs": float(k_abs),
                "illusion_magnitude": round(
                    (float(neg["pse"]) - float(pos["pse"])) / 2.0, 5
                ),
                "side_bias": round((float(neg["pse"]) + float(pos["pse"])) / 2.0, 5),
                "magnitude_ci_halfwidth": round(hw, 5) if np.isfinite(hw) else np.nan,
                "pse_pos": pos["pse"],
                "pse_neg": neg["pse"],
                "fit_r2_min": min(pos["fit_r2"], neg["fit_r2"]),
                "fit_status": _worse_status(pos["fit_status"], neg["fit_status"]),
                "is_lower_bound": bool(
                    _worse_status(pos["fit_status"], neg["fit_status"]) == "bounded"
                ),
                "reportable": bool(
                    _worse_status(pos["fit_status"], neg["fit_status"])
                    in ("measured", "bounded")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["species", "illusion", "k_abs"]
    ).reset_index(drop=True)


# ============================================================================
# BASELINE COMPETENCE
# ============================================================================


def baseline_jnd(cells: pd.DataFrame, species: str = "human") -> dict:
    """
    Each illusion's just-noticeable difference for one species.

    Fitted at zero illusion strength, so it is the dispersion of the
    psychometric function with no illusion present: the smallest difference
    that observer resolves reliably, in that illusion's delta units.

    Used as the yardstick for `magnitude_jnd`. A single species supplies the
    denominator for both, so the scale is identical on either side of a
    comparison; dividing each observer by its own JND would instead rescale
    the two sides differently and confound illusion size with precision.
    """
    base = cells[(cells["illusion_strength"] == 0) & (cells["species"] == species)]
    out = {}
    for illusion, grp in base.groupby("illusion"):
        fit = fit_psychometric(
            grp["true_diff"].values, grp["n_positive"].values, grp["n_trials"].values
        )
        out[illusion] = fit["sigma"]
    return out


def _threshold_75(abs_diff: np.ndarray, accuracy: np.ndarray) -> float:
    """
    Smallest difference at which accuracy first reaches 0.75, interpolated.

    Model-free, so it does not depend on a psychometric fit converging. Returns
    NaN when the observer never reaches 0.75 at any tested difference.
    """
    order = np.argsort(abs_diff)
    x, y = abs_diff[order], accuracy[order]
    if y.size and y[0] >= 0.75:
        return float(x[0])
    for i in range(1, len(x)):
        if y[i] >= 0.75 > y[i - 1]:
            t = (0.75 - y[i - 1]) / (y[i] - y[i - 1])
            return float(x[i - 1] + t * (x[i] - x[i - 1]))
    return float("nan")


def baseline_competence(cells: pd.DataFrame) -> pd.DataFrame:
    """
    Whether the underlying discrimination is performed with the illusion off.

    Everything here is measured at zero illusion strength, so it describes the
    task itself rather than any illusion.

        peak_accuracy   best accuracy at any tested difference. Gates the
                        per-illusion scalars: below MIN_PEAK_ACCURACY the
                        illusion is marked as not performed for that observer.
        threshold_75    smallest difference resolved at 75% correct, in that
                        illusion's delta units. The acuity measure: two
                        observers can both reach ceiling and still differ
                        several-fold here.
        baseline_d_prime
                        retained as a descriptive column only. It pools over
                        every difference level the species was shown, and the
                        two species were not always shown the same levels, so
                        it is not comparable across species or illusions and
                        must not be used as a gate.

    Returns one row per (species, illusion).
    """
    from scipy.stats import norm

    rows = []
    base = cells[cells["illusion_strength"] == 0]
    for (species, illusion), grp in base.groupby(["species", "illusion"]):
        pos = grp[grp["true_diff"] > 0]
        neg = grp[grp["true_diff"] < 0]
        if pos.empty or neg.empty:
            continue

        # Log-linear correction keeps z finite at rates of exactly 0 or 1.
        h = (pos["n_positive"].sum() + 0.5) / (pos["n_trials"].sum() + 1.0)
        f = (neg["n_positive"].sum() + 0.5) / (neg["n_trials"].sum() + 1.0)
        d_prime = float(norm.ppf(h) - norm.ppf(f))

        by_level = grp.assign(
            abs_diff=grp["true_diff"].abs().round(5),
            n_correct=np.where(
                grp["true_diff"] > 0,
                grp["n_positive"],
                grp["n_trials"] - grp["n_positive"],
            ),
        ).groupby("abs_diff")[["n_correct", "n_trials"]].sum()
        accuracy = (by_level["n_correct"] / by_level["n_trials"]).to_numpy()
        peak = float(accuracy.max())

        rows.append(
            {
                "species": species,
                "illusion": illusion,
                "peak_accuracy": round(peak, 4),
                "threshold_75": round(
                    _threshold_75(by_level.index.to_numpy(float), accuracy), 5
                ),
                "baseline_d_prime": round(d_prime, 4),
                "baseline_criterion": round(-0.5 * float(norm.ppf(h) + norm.ppf(f)), 4),
                "can_do_task": bool(peak >= MIN_PEAK_ACCURACY),
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# PER-ILLUSION SCALARS
# ============================================================================


def illusion_scalars(
    magnitudes: pd.DataFrame, cells: pd.DataFrame, competence: pd.DataFrame
) -> pd.DataFrame:
    """
    Reduce each (species, illusion) to the scalars the figures need.

    Returns:
        magnitude_at_max      illusion magnitude at the largest reportable
                              |strength|, in delta units
        k_at_max              which |strength| that was
        magnitude_trend_rho   Spearman rho of magnitude against |strength|,
                              over reportable levels only. This is the
                              cross-illusion currency: a graded illusion has
                              rho near +1 whatever the delta units.
        n_reportable          how many of the 7 magnitudes survived gating
        side_bias_mean        mean side bias over reportable levels
        baseline_error        error rate at zero strength
        baseline_side_bias    proportion of positive responses at zero
                              strength, minus 0.5
    """
    comp = competence.set_index(["species", "illusion"])

    rows = []
    for (species, illusion), grp in magnitudes.groupby(["species", "illusion"]):
        can_do = bool(comp.loc[(species, illusion), "can_do_task"]) if (
            species,
            illusion,
        ) in comp.index else True

        ok = grp[grp["reportable"] & ~grp["unconstrained"]].sort_values("k_abs")

        if len(ok) >= 3:
            rho = float(spearmanr(ok["k_abs"], ok["illusion_magnitude"]).statistic)
        else:
            rho = np.nan

        at_max = ok.iloc[-1] if not ok.empty else None

        base = cells[
            (cells["species"] == species)
            & (cells["illusion"] == illusion)
            & (cells["illusion_strength"] == 0)
        ]
        if base.empty:
            baseline_error = baseline_side_bias = np.nan
        else:
            err = np.where(
                base["true_diff"] > 0,
                1.0 - base["n_positive"] / base["n_trials"],
                base["n_positive"] / base["n_trials"],
            )
            baseline_error = float(np.average(err, weights=base["n_trials"]))
            baseline_side_bias = float(
                base["n_positive"].sum() / base["n_trials"].sum() - 0.5
            )

        rows.append(
            {
                "species": species,
                "illusion": illusion,
                "magnitude_at_max": (
                    float(at_max["illusion_magnitude"]) if at_max is not None else np.nan
                ),
                "k_at_max": float(at_max["k_abs"]) if at_max is not None else np.nan,
                "magnitude_trend_rho": round(rho, 4) if np.isfinite(rho) else np.nan,
                "n_reportable": int(len(ok)),
                "n_measured": int((ok["fit_status"] == "measured").sum())
                if not ok.empty
                else 0,
                "n_bounded": int((ok["fit_status"] == "bounded").sum())
                if not ok.empty
                else 0,
                "n_unconstrained": int(grp["unconstrained"].sum()),
                "n_levels": int(len(grp)),
                "can_do_task": can_do,
                "side_bias_mean": (
                    round(float(ok["side_bias"].mean()), 5) if not ok.empty else np.nan
                ),
                "baseline_error": round(baseline_error, 5)
                if np.isfinite(baseline_error)
                else np.nan,
                "baseline_side_bias": round(baseline_side_bias, 5)
                if np.isfinite(baseline_side_bias)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# DESCRIPTIVE ERROR RATES (secondary - never the headline)
# ============================================================================


def wilson_interval(n_success: float, n_total: float, z: float = 1.96) -> tuple:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because error rates here sit near
    0 on congruent trials and near 1 on strong incongruent ones, where the
    normal interval runs outside [0, 1].
    """
    if n_total <= 0:
        return (np.nan, np.nan)
    p = n_success / n_total
    denom = 1.0 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half = (
        z / denom * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def error_by_strength(cells: pd.DataFrame) -> pd.DataFrame:
    """
    Error rate per signed strength, with its split by the sign of delta.

    `err_overall` is the plain proportion of errors over every trial at that
    strength, with a Wilson 95% interval. `asymmetry` splits the two sides of
    zero.

    Error rate confounds the illusion effect with any response bias, and its
    absolute level depends on task conditions, so it does not substitute for
    the magnitude in illusion_magnitude.
    """
    rows = []
    for (species, illusion, strength), grp in cells.groupby(
        ["species", "illusion", "illusion_strength"]
    ):
        pos = grp[grp["true_diff"] > 0]
        neg = grp[grp["true_diff"] < 0]
        if pos.empty or neg.empty:
            continue
        err_pos = 1.0 - pos["n_positive"].sum() / pos["n_trials"].sum()
        err_neg = neg["n_positive"].sum() / neg["n_trials"].sum()

        # Errors are positive-option responses below zero and negative-option
        # responses above it.
        n_err = float(
            (pos["n_trials"].sum() - pos["n_positive"].sum()) + neg["n_positive"].sum()
        )
        n_tot = float(grp["n_trials"].sum())
        ci_lo, ci_hi = wilson_interval(n_err, n_tot)

        rows.append(
            {
                "species": species,
                "illusion": illusion,
                "illusion_strength": float(strength),
                "err_pos_diff": round(float(err_pos), 5),
                "err_neg_diff": round(float(err_neg), 5),
                "err_overall": round(n_err / n_tot, 5),
                "err_ci_low": round(ci_lo, 5),
                "err_ci_high": round(ci_hi, 5),
                "asymmetry": round(float(err_pos - err_neg), 5),
                "n_errors": int(n_err),
                "n_trials": int(n_tot),
            }
        )
    return pd.DataFrame(rows)


def add_difficulty_bins(cells: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """
    Bin |true_diff| into equal-count difficulty bands by rank, per illusion.

    The human design is interleaved: each non-zero strength samples every
    other difference level, so the raw grid leaves half of every human row
    empty. Pooling adjacent |delta| ranks in pairs populates every cell.

    Adds `abs_rank` (1 = largest |delta|), `difficulty_bin` (0 = easiest) and
    `signed_bin`, which sorts as signed delta does.
    """
    out = []
    for _, grp in cells.groupby("illusion"):
        grp = grp.copy()
        levels = np.sort(grp["true_diff"].abs().round(5).unique())[::-1]
        rank = {lvl: i + 1 for i, lvl in enumerate(levels)}
        grp["abs_rank"] = grp["true_diff"].abs().round(5).map(rank)
        per_bin = int(np.ceil(len(levels) / n_bins))
        grp["difficulty_bin"] = ((grp["abs_rank"] - 1) // per_bin).astype(int)
        grp["signed_bin"] = np.sign(grp["true_diff"]).astype(int) * (
            n_bins - grp["difficulty_bin"]
        )
        out.append(grp)
    return pd.concat(out, ignore_index=True)
