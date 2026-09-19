"""
pipeline/human_comparison/psychometrics.py - Corrected psychometric fitting core.

Shared by the VLM pipeline (module_3) and the human re-analysis so that both
species are measured with identical machinery.

Differences from the original module_3.fit_psychometrics.fit_pse:

  1. Bounds and initial values are derived from each illusion's own delta
     grid. The original hardcoded mu in [-2, 2] in absolute delta units,
     which is narrower than the tested range for RodFrame (+/-13.7) and
     Contrast (+/-17.5) - those PSEs were pinned at the bound.

  2. Two lapse parameters let the asymptotes fall short of 0/1. Without them
     a strength level where the observer has lost the signal entirely is
     fitted as a shallow sigmoid with an arbitrary mu, rather than being
     identified as uninformative.

  3. Binomial maximum likelihood rather than least squares on proportions,
     so cells are weighted by trial count and boundary proportions (0.0 /
     1.0) are handled correctly.

  4. Uncertainty comes from a profile-likelihood interval rather than a
     Hessian SE. When the response is a clean step the likelihood has a ridge
     as sigma approaches zero, leaving the Hessian singular; the profile
     interval instead returns the bracket the delta grid actually supports.

  5. "converged" means the optimiser succeeded AND the PSE is off its bounds
     AND the curve is neither flat nor at chance. The original returned
     success whenever curve_fit did not raise, which could report a fit that
     never left p0.

A PSE requires a response function that is monotonic in delta. Callers should
slice the data with metrics.add_illusion_direction and gate each fit on
assess_sliced_fit.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import erf
from scipy.stats import spearmanr

# ============================================================================
# REGIME THRESHOLDS
# ============================================================================
# Accuracy on the easiest half of the delta grid below which the observer is
# treated as having no usable signal (chance = 0.5).
BREAKDOWN_ACC_EASY = 0.65
# Largest permitted drop in P(positive) as delta increases. A monotonic
# observer has 0; a full reversal through the middle of the range approaches 1.
REVERSAL_MAX_DRAWDOWN = 0.50
# A drawdown must also exceed this many standard errors of the two cells that
# produce it, so sparse cells cannot manufacture a reversal.
REVERSAL_MIN_SE = 3.0
# Accuracy on the hardest half of the delta grid below which errors are
# treated as concentrated at small differences (an illusion-driven bias).
BIASED_ACC_HARD = 0.60

# Ordered by how much they compromise the PSE.
REGIMES = ("monotonic", "bias-dominated", "reversal", "breakdown")
PSE_ESTIMABLE_REGIMES = ("monotonic", "bias-dominated")

# ---------------------------------------------------------------------------
# Thresholds for DIRECTION-SLICED cells (see assess_sliced_fit).
# ---------------------------------------------------------------------------
# Minimum R2 on the proportion scale for a sliced fit to be reportable.
SLICED_R2_FLOOR = 0.80
# Maximum permitted fall in P(positive) as delta increases. Sliced cells hold
# the illusion's spatial direction constant, so the response function should be
# monotonic; this tolerates sampling noise, not a genuine reversal.
SLICED_MAX_DRAWDOWN = 0.35

# A sliced cell ends up in exactly one of these.
#
#   measured    the response crosses 0.5 inside the tested delta range, so the
#               PSE is interpolated
#   bounded     the response does not cross 0.5 in range, so the PSE lies
#               outside the grid and the derived magnitude is a lower bound
#   unreliable  the response crosses 0.5 but is non-monotonic beyond
#               SLICED_MAX_DRAWDOWN, or the sigmoid fit is below
#               SLICED_R2_FLOOR
FIT_STATUSES = ("measured", "bounded", "unreliable")

EPS = 1e-10


# ============================================================================
# PSYCHOMETRIC FUNCTION
# ============================================================================


def psychometric(
    x: np.ndarray,
    mu: float,
    sigma: float,
    lapse_lo: float = 0.0,
    lapse_hi: float = 0.0,
) -> np.ndarray:
    """
    Cumulative Gaussian with independent lower and upper lapse rates.

        P(positive | x) = l_lo + (1 - l_lo - l_hi) * Phi((x - mu) / sigma)

    With both lapses at 0 this reduces to the plain cumulative Gaussian, so
    fits remain comparable to the original 2-parameter model.

    Args:
        x:        Signed physical difference (positive option - negative option).
        mu:       PSE - the delta at which the underlying Phi reaches 0.5.
        sigma:    Dispersion; smaller = steeper.
        lapse_lo: Lower asymptote.
        lapse_hi: Distance of the upper asymptote below 1.
    """
    phi = 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))
    return lapse_lo + (1.0 - lapse_lo - lapse_hi) * phi


def _neg_log_lik(
    params: np.ndarray, x: np.ndarray, k: np.ndarray, n: np.ndarray
) -> float:
    """Binomial negative log-likelihood for params = (mu, sigma, l_lo, l_hi)."""
    p = psychometric(x, *params)
    p = np.clip(p, EPS, 1.0 - EPS)
    return -float(np.sum(k * np.log(p) + (n - k) * np.log(1.0 - p)))


# ============================================================================
# INITIALISATION AND BOUNDS
# ============================================================================


def _grid_geometry(x: np.ndarray) -> tuple[float, float, float]:
    """Return (min, max, range) of the tested delta grid."""
    lo, hi = float(np.min(x)), float(np.max(x))
    rng = hi - lo
    if rng <= 0:
        rng = max(abs(hi), 1.0)
    return lo, hi, rng


def _bounds_for_grid(x: np.ndarray) -> list[tuple[float, float]]:
    """
    Parameter bounds scaled to the illusion's own delta grid.

    mu is allowed to sit somewhat outside the tested range - a PSE beyond the
    grid is a meaningful (if poorly constrained) result and should be flagged
    by params_on_bound, not silently clipped to an arbitrary constant.
    """
    lo, hi, rng = _grid_geometry(x)
    return [
        (lo - 0.25 * rng, hi + 0.25 * rng),  # mu
        (rng / 500.0, rng * 3.0),  # sigma
        (0.0, 0.45),  # lapse_lo
        (0.0, 0.45),  # lapse_hi
    ]


def _initial_guess(x: np.ndarray, p: np.ndarray) -> list[float]:
    """Data-driven start: mu at the observed 0.5 crossing, sigma from the grid."""
    _, _, rng = _grid_geometry(x)
    order = np.argsort(x)
    xs, ps = x[order], p[order]

    mu0 = float(np.median(xs))
    crossings = np.where(np.diff(np.sign(ps - 0.5)) != 0)[0]
    if len(crossings) > 0:
        i = crossings[0]
        denom = ps[i + 1] - ps[i]
        frac = (0.5 - ps[i]) / denom if abs(denom) > EPS else 0.5
        mu0 = float(xs[i] + frac * (xs[i + 1] - xs[i]))

    return [mu0, rng / 10.0, 0.02, 0.02]


# ============================================================================
# FITTING
# ============================================================================


def fit_psychometric(
    diff_values: np.ndarray,
    n_positive: np.ndarray,
    n_trials: np.ndarray,
) -> dict:
    """
    Fit the 4-parameter psychometric function by binomial maximum likelihood.

    Args:
        diff_values: Signed delta per cell.
        n_positive:  Count of positive-option responses per cell.
        n_trials:    Total responses per cell.

    Returns:
        dict with pse, sigma, lapse_lo, lapse_hi, their SEs, fit_r2,
        converged, params_on_bound and note.
    """
    x = np.asarray(diff_values, dtype=float)
    k = np.asarray(n_positive, dtype=float)
    n = np.asarray(n_trials, dtype=float)
    p_obs = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)

    failed = {
        "pse": np.nan,
        "sigma": np.nan,
        "lapse_lo": np.nan,
        "lapse_hi": np.nan,
        "pse_se": np.nan,
        "sigma_se": np.nan,
        "pse_ci_low": np.nan,
        "pse_ci_high": np.nan,
        "fit_r2": np.nan,
        "converged": False,
        "pse_identified": False,
        "sigma_identified": False,
        "params_on_bound": False,
    }

    if len(x) < 4 or np.all(n <= 0):
        return {**failed, "note": "too few usable cells"}

    bounds = _bounds_for_grid(x)
    p0 = _initial_guess(x, p_obs)
    p0 = [min(max(v, b[0]), b[1]) for v, b in zip(p0, bounds)]

    try:
        res = minimize(
            _neg_log_lik,
            p0,
            args=(x, k, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 20_000, "ftol": 1e-12},
        )
    except Exception as exc:  # pragma: no cover - optimiser safety net
        return {**failed, "note": f"optimiser error: {exc}"}

    mu, sigma, lo, hi = (float(v) for v in res.x)
    rng = _grid_geometry(x)[2]

    # Standard errors from the numerical Hessian of the negative log-likelihood.
    # These are undefined when the response is a clean step, because the
    # likelihood then has a ridge as sigma approaches zero; the profile
    # interval below is the reliable interval and is what figures should use.
    pse_se = sigma_se = np.nan
    try:
        hess = _numerical_hessian(_neg_log_lik, res.x, (x, k, n))
        cov = np.linalg.inv(hess)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        pse_se, sigma_se = float(se[0]), float(se[1])
    except (np.linalg.LinAlgError, ValueError):
        pass

    ci_low, ci_high = _profile_ci_mu(x, k, n, bounds, res.x, float(res.fun))

    # Goodness of fit on the proportion scale, for comparability with the
    # original diagnostics.
    p_hat = psychometric(x, mu, sigma, lo, hi)
    ss_res = float(np.sum((p_obs - p_hat) ** 2))
    ss_tot = float(np.sum((p_obs - np.nanmean(p_obs)) ** 2))
    fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Each bound means something different, so they are flagged separately
    # rather than collapsed into one "on bound" warning.
    #
    #   mu on bound        - PSE fell outside the searchable range; unusable.
    #   sigma at floor     - the JND is finer than the Δ grid can resolve. The
    #                        slope is unreportable but the PSE is still
    #                        bracketed by adjacent Δ levels, so it is kept.
    #   sigma at ceiling   - the curve is essentially flat; PSE meaningless.
    #   lapse at 0.45      - responding is near chance; PSE meaningless.
    #   lapse at 0.0       - an ordinary optimum (no lapses needed), not a
    #                        pathology, so it is not flagged at all.
    pse_on_bound = abs(mu - bounds[0][0]) < 1e-6 or abs(mu - bounds[0][1]) < 1e-6
    sigma_at_floor = abs(sigma - bounds[1][0]) < 1e-6
    sigma_at_ceiling = abs(sigma - bounds[1][1]) < 1e-6
    lapse_saturated = abs(lo - 0.45) < 1e-6 or abs(hi - 0.45) < 1e-6

    # The PSE is identified when the profile interval is finite and narrower
    # than half the tested grid.
    ci_width = (
        ci_high - ci_low if np.isfinite(ci_low) and np.isfinite(ci_high) else np.inf
    )
    identified = bool(np.isfinite(ci_width) and ci_width < 0.5 * rng)

    converged = bool(
        res.success
        and not pse_on_bound
        and not sigma_at_ceiling
        and not lapse_saturated
    )

    notes = []
    if not res.success:
        notes.append("optimiser did not converge")
    if pse_on_bound:
        notes.append("PSE on bound")
    if sigma_at_floor:
        notes.append("JND finer than grid resolution")
    if sigma_at_ceiling:
        notes.append("curve flat")
    if lapse_saturated:
        notes.append("responding near chance")
    if not identified:
        notes.append("PSE interval wider than half the grid")

    return {
        "pse": round(mu, 5),
        "sigma": round(sigma, 5),
        "lapse_lo": round(lo, 5),
        "lapse_hi": round(hi, 5),
        "pse_se": round(pse_se, 5) if np.isfinite(pse_se) else np.nan,
        "sigma_se": round(sigma_se, 5) if np.isfinite(sigma_se) else np.nan,
        "pse_ci_low": round(ci_low, 5) if np.isfinite(ci_low) else np.nan,
        "pse_ci_high": round(ci_high, 5) if np.isfinite(ci_high) else np.nan,
        "fit_r2": round(fit_r2, 5) if np.isfinite(fit_r2) else np.nan,
        "converged": converged,
        "pse_identified": identified,
        "sigma_identified": bool(not sigma_at_floor and not sigma_at_ceiling),
        "params_on_bound": bool(pse_on_bound or sigma_at_ceiling or lapse_saturated),
        "note": "; ".join(notes),
    }


# 95% interval from the likelihood-ratio criterion for a single parameter.
_CHI2_95_1DF = 3.841459


def _profile_ci_mu(
    x: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    bounds: list[tuple[float, float]],
    best: np.ndarray,
    nll_min: float,
) -> tuple[float, float]:
    """
    Profile-likelihood 95% interval for mu.

    The remaining parameters are re-optimised at each candidate mu, and the
    endpoints are found by root-finding on the deviance rather than on a fixed
    grid — a sharply determined PSE has an interval far narrower than any
    practical grid step, which would otherwise collapse to a single node.

    Unlike a Hessian SE this stays well defined when the response is a clean
    step, where it returns the bracket the data actually support.
    """
    mu_lo, mu_hi = bounds[0]
    mu_hat = float(best[0])

    def excess_deviance(mu_try: float) -> float:
        """Deviance above the LR threshold; negative inside the interval."""
        try:
            r = minimize(
                lambda q: _neg_log_lik(np.array([mu_try, *q]), x, k, n),
                best[1:],
                method="L-BFGS-B",
                bounds=bounds[1:],
                options={"maxiter": 2_000, "ftol": 1e-10},
            )
            return 2.0 * (float(r.fun) - nll_min) - _CHI2_95_1DF
        except Exception:  # pragma: no cover - optimiser safety net
            return np.nan

    def find_edge(limit: float) -> float:
        """Locate the deviance crossing between mu_hat and `limit`."""
        f_limit = excess_deviance(limit)
        if not np.isfinite(f_limit) or f_limit <= 0:
            # Never crosses within the searchable range.
            return limit
        try:
            return float(brentq(excess_deviance, mu_hat, limit, xtol=1e-9, rtol=1e-9))
        except (ValueError, RuntimeError):
            return limit

    return find_edge(mu_lo), find_edge(mu_hi)


def _numerical_hessian(fn, params: np.ndarray, args: tuple, eps: float = 1e-5):
    """Central-difference Hessian of fn at params."""
    p = np.asarray(params, dtype=float)
    k = len(p)
    h = np.maximum(np.abs(p) * eps, eps)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            pp, pm, mp, mm = p.copy(), p.copy(), p.copy(), p.copy()
            pp[i] += h[i]
            pp[j] += h[j]
            pm[i] += h[i]
            pm[j] -= h[j]
            mp[i] -= h[i]
            mp[j] += h[j]
            mm[i] -= h[i]
            mm[j] -= h[j]
            H[i, j] = H[j, i] = (
                fn(pp, *args) - fn(pm, *args) - fn(mp, *args) + fn(mm, *args)
            ) / (4.0 * h[i] * h[j])
    return H


# ============================================================================
# MODEL-FREE REGIME CLASSIFICATION
# ============================================================================


def classify_regime(
    diff_values: np.ndarray,
    n_positive: np.ndarray,
    n_trials: np.ndarray,
) -> dict:
    """
    Classify one CONGRUENCY-POOLED cell's response function.

    SCOPE. This function describes cells grouped by SIGNED illusion strength.
    In this stimulus set the sign of the strength encodes congruency relative
    to the true difference, so such a group mixes trials in which the illusion
    points in opposite spatial directions and the pooled response function is
    not monotonic in delta by construction.

    It is retained only for describing that pooled view, and must not be used
    to decide whether a PSE is estimable: slice with
    metrics.add_illusion_direction and use assess_sliced_fit instead. Its
    accuracy-based "breakdown" test in particular does not apply to sliced
    cells.

    These labels describe the shape of the pooled response function, not the
    magnitude of any bias.

        monotonic       - rises with delta and is accurate at large |delta|;
                          PSE estimable
        bias-dominated  - accurate at large |delta| but errors concentrated at
                          small |delta|; PSE estimable
        reversal        - P(positive) falls substantially as delta increases,
                          so no monotonic function describes the data
        breakdown       - inaccurate even at the largest |delta|; responses
                          carry little information about delta

    Returns the classification plus the statistics behind it.
    """
    x = np.asarray(diff_values, dtype=float)
    k = np.asarray(n_positive, dtype=float)
    n = np.asarray(n_trials, dtype=float)

    order = np.argsort(x)
    x, k, n = x[order], k[order], n[order]
    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)

    valid = np.isfinite(p) & (x != 0)
    if valid.sum() < 4:
        return {
            "regime": "breakdown",
            "acc_easy": np.nan,
            "acc_hard": np.nan,
            "max_drawdown": np.nan,
            "drawdown_se": np.nan,
            "rho_diff_response": np.nan,
        }

    xv, pv, nv = x[valid], p[valid], n[valid]

    # Accuracy: the positive option is correct when delta > 0.
    acc = np.where(xv > 0, pv, 1.0 - pv)
    easy = np.abs(xv) >= np.median(np.abs(xv))
    acc_easy = float(np.mean(acc[easy])) if easy.any() else np.nan
    acc_hard = float(np.mean(acc[~easy])) if (~easy).any() else np.nan

    # Largest fall in P(positive) below its running maximum as delta
    # increases, with the binomial SE of the two cells that produce it.
    # Requiring the fall to clear a multiple of that SE keeps a single sparse
    # cell from registering as a reversal.
    running_max = np.maximum.accumulate(pv)
    drops = running_max - pv
    j = int(np.argmax(drops))
    max_drawdown = float(drops[j])
    i = int(np.argmax(running_max[: j + 1] == running_max[j]))
    drawdown_se = float(
        np.sqrt(
            pv[i] * (1 - pv[i]) / max(nv[i], 1) + pv[j] * (1 - pv[j]) / max(nv[j], 1)
        )
    )

    rho = spearmanr(xv, pv).statistic if len(xv) >= 4 else np.nan

    reversal = max_drawdown > REVERSAL_MAX_DRAWDOWN and (
        drawdown_se <= 0 or max_drawdown > REVERSAL_MIN_SE * drawdown_se
    )

    if np.isfinite(acc_easy) and acc_easy < BREAKDOWN_ACC_EASY:
        regime = "breakdown"
    elif reversal:
        regime = "reversal"
    elif np.isfinite(acc_hard) and acc_hard < BIASED_ACC_HARD:
        regime = "bias-dominated"
    else:
        regime = "monotonic"

    return {
        "regime": regime,
        "acc_easy": round(acc_easy, 4) if np.isfinite(acc_easy) else np.nan,
        "acc_hard": round(acc_hard, 4) if np.isfinite(acc_hard) else np.nan,
        "max_drawdown": round(max_drawdown, 4),
        "drawdown_se": round(drawdown_se, 4),
        "rho_diff_response": round(float(rho), 4) if np.isfinite(rho) else np.nan,
    }


def joint_psychometric(
    x: np.ndarray,
    direction: np.ndarray,
    magnitude: float,
    side_bias: float,
    sigma: float,
    lapse_lo: float = 0.0,
    lapse_hi: float = 0.0,
) -> np.ndarray:
    """
    One psychometric function per illusion direction, sharing every parameter
    except a single shift.

        PSE(d) = side_bias - d * magnitude
        P(positive | x, d) = Phi((x - PSE(d)) / sigma), with lapses

    `magnitude` is therefore half the separation between the two directions'
    PSEs, and `side_bias` their midpoint - the same definitions used when the
    two are fitted separately.

    Args:
        x:         Signed delta per cell.
        direction: +1 or -1 per cell.
        magnitude: Half the PSE separation between directions.
        side_bias: Midpoint of the two PSEs.
        sigma:     Shared dispersion.
        lapse_lo:  Shared lower asymptote.
        lapse_hi:  Shared distance of the upper asymptote below 1.
    """
    mu = side_bias - direction * magnitude
    phi = 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))
    return lapse_lo + (1.0 - lapse_lo - lapse_hi) * phi


def _joint_neg_log_lik(
    params: np.ndarray,
    x: np.ndarray,
    d: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
) -> float:
    """Binomial negative log-likelihood for the joint two-direction model."""
    p = joint_psychometric(x, d, *params)
    p = np.clip(p, EPS, 1.0 - EPS)
    return -float(np.sum(k * np.log(p) + (n - k) * np.log(1.0 - p)))


def fit_illusion_magnitude(
    diff_values: np.ndarray,
    direction: np.ndarray,
    n_positive: np.ndarray,
    n_trials: np.ndarray,
) -> dict:
    """
    Estimate illusion magnitude and side bias from both directions at once.

    Fitting the two directions jointly with a shared slope and shared lapses
    constrains the estimate far more than differencing two independent PSEs:
    it uses every cell at the level to determine the slope, so a magnitude and
    an interval are returned even where one direction alone is too noisy to
    locate its own PSE. An absent illusion returns a magnitude near zero with
    an interval spanning zero, rather than no estimate.

    Args:
        diff_values: Signed delta per cell, both directions pooled.
        direction:   +1 or -1 per cell.
        n_positive:  Positive-option response counts.
        n_trials:    Trial counts.

    Returns:
        dict with magnitude, side_bias, sigma, the lapses, the profile
        interval on magnitude, fit_r2, converged and saturated.
    """
    x = np.asarray(diff_values, dtype=float)
    d = np.asarray(direction, dtype=float)
    k = np.asarray(n_positive, dtype=float)
    n = np.asarray(n_trials, dtype=float)

    failed = {
        "magnitude": np.nan,
        "side_bias": np.nan,
        "sigma": np.nan,
        "lapse_lo": np.nan,
        "lapse_hi": np.nan,
        "magnitude_ci_low": np.nan,
        "magnitude_ci_high": np.nan,
        "fit_r2": np.nan,
        "converged": False,
        "saturated": False,
    }
    if len(x) < 6 or np.all(n <= 0) or len(np.unique(d)) < 2:
        return {**failed, "note": "too few usable cells"}

    grid_max = float(np.max(np.abs(x)))
    _, _, rng = _grid_geometry(x)
    bounds = [
        (-1.5 * grid_max, 1.5 * grid_max),  # magnitude
        (-1.5 * grid_max, 1.5 * grid_max),  # side bias
        (rng / 500.0, rng * 3.0),  # sigma
        (0.0, 0.45),  # lapse_lo
        (0.0, 0.45),  # lapse_hi
    ]
    p0 = [0.1 * grid_max, 0.0, rng / 10.0, 0.02, 0.02]

    try:
        res = minimize(
            _joint_neg_log_lik,
            p0,
            args=(x, d, k, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 20_000, "ftol": 1e-12},
        )
    except Exception as exc:  # pragma: no cover - optimiser safety net
        return {**failed, "note": f"optimiser error: {exc}"}

    m, c, sigma, lo, hi = (float(v) for v in res.x)

    ci_low, ci_high = _profile_ci_joint(x, d, k, n, bounds, res.x, float(res.fun))

    p_obs = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    p_hat = joint_psychometric(x, d, m, c, sigma, lo, hi)
    ss_res = float(np.sum((p_obs - p_hat) ** 2))
    ss_tot = float(np.sum((p_obs - np.nanmean(p_obs)) ** 2))
    fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Saturation: neither direction's observed response crosses 0.5 inside the
    # tested range, so the separation is only bounded from below.
    saturated = True
    for side in (1.0, -1.0):
        sel = d == side
        if sel.sum() >= 2:
            pv = p_obs[sel]
            pv = pv[np.isfinite(pv)]
            if len(pv) >= 2 and pv.min() < 0.5 < pv.max():
                saturated = False
                break

    return {
        "magnitude": round(m, 5),
        "side_bias": round(c, 5),
        "sigma": round(sigma, 5),
        "lapse_lo": round(lo, 5),
        "lapse_hi": round(hi, 5),
        "magnitude_ci_low": round(ci_low, 5) if np.isfinite(ci_low) else np.nan,
        "magnitude_ci_high": round(ci_high, 5) if np.isfinite(ci_high) else np.nan,
        "fit_r2": round(fit_r2, 5) if np.isfinite(fit_r2) else np.nan,
        "converged": bool(res.success),
        "saturated": bool(saturated),
        "note": "" if res.success else "optimiser did not converge",
    }


def _profile_ci_joint(
    x: np.ndarray,
    d: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    bounds: list[tuple[float, float]],
    best: np.ndarray,
    nll_min: float,
) -> tuple[float, float]:
    """Profile-likelihood 95% interval for the magnitude parameter."""
    m_lo, m_hi = bounds[0]
    m_hat = float(best[0])

    def excess_deviance(m_try: float) -> float:
        try:
            r = minimize(
                lambda q: _joint_neg_log_lik(np.array([m_try, *q]), x, d, k, n),
                best[1:],
                method="L-BFGS-B",
                bounds=bounds[1:],
                options={"maxiter": 2_000, "ftol": 1e-10},
            )
            return 2.0 * (float(r.fun) - nll_min) - _CHI2_95_1DF
        except Exception:  # pragma: no cover - optimiser safety net
            return np.nan

    def find_edge(limit: float) -> float:
        f_limit = excess_deviance(limit)
        if not np.isfinite(f_limit) or f_limit <= 0:
            return limit
        try:
            return float(brentq(excess_deviance, m_hat, limit, xtol=1e-9, rtol=1e-9))
        except (ValueError, RuntimeError):
            return limit

    return find_edge(m_lo), find_edge(m_hi)


def assess_sliced_fit(
    diff_values: np.ndarray,
    n_positive: np.ndarray,
    n_trials: np.ndarray,
    fit: dict,
) -> dict:
    """
    Diagnose a fit to a DIRECTION-SLICED cell.

    The test is whether the observed response crosses 0.5 inside the tested
    delta range. If it does, the PSE is interpolated. If it does not, the PSE
    lies outside the grid and the fitted value is an extrapolation, so the
    grid edge is substituted and the derived magnitude becomes a lower bound.

    Returns one of FIT_STATUSES: "measured", "bounded" or "unreliable".

    For a bounded cell the direction of saturation gives the sign of the
    bound: a response that stays above 0.5 throughout means the positive
    option was always chosen, so the PSE sits below the smallest tested delta.

    Accuracy is not tested, only the shape of the response. A monotonic
    response function with a large PSE shift produces low accuracy on one side
    of zero while remaining well formed, and an accuracy test would reject it.

    Args:
        diff_values: Signed delta per cell.
        n_positive:  Positive-option response counts.
        n_trials:    Trial counts.
        fit:         The corresponding fit_psychometric result.

    Returns:
        dict with status, crosses_half, bound_pse (the grid-edge value to use
        when the cell is bounded), max_drawdown, fit_r2 and pse_abs_frac.
    """
    x = np.asarray(diff_values, dtype=float)
    k = np.asarray(n_positive, dtype=float)
    n = np.asarray(n_trials, dtype=float)

    order = np.argsort(x)
    x, k, n = x[order], k[order], n[order]
    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    valid = np.isfinite(p)

    max_drawdown = (
        float(np.max(np.maximum.accumulate(p[valid]) - p[valid]))
        if valid.sum() >= 2
        else np.nan
    )

    grid_max = float(np.max(np.abs(x))) if len(x) else np.nan
    pse = fit.get("pse", np.nan)
    pse_abs_frac = (
        abs(float(pse)) / grid_max
        if np.isfinite(pse) and np.isfinite(grid_max) and grid_max > 0
        else np.nan
    )
    r2 = fit.get("fit_r2", np.nan)

    pv = p[valid]
    crosses_half = bool(valid.sum() >= 2 and pv.min() < 0.5 < pv.max())

    # Where the response saturates, the PSE sits beyond the grid on the side
    # the observer never chose: always-positive responding puts it below the
    # smallest delta, and vice versa.
    bound_pse = np.nan
    if not crosses_half and valid.sum() >= 2 and np.isfinite(grid_max):
        bound_pse = -grid_max if float(np.median(pv)) > 0.5 else grid_max

    misshapen = (np.isfinite(max_drawdown) and max_drawdown > SLICED_MAX_DRAWDOWN) or (
        not np.isfinite(r2) or r2 < SLICED_R2_FLOOR
    )

    if not crosses_half:
        # A saturated response is monotone by construction; only a genuinely
        # erratic one is downgraded further.
        status = (
            "unreliable"
            if np.isfinite(max_drawdown) and max_drawdown > SLICED_MAX_DRAWDOWN
            else "bounded"
        )
    elif misshapen:
        status = "unreliable"
    else:
        status = "measured"

    return {
        "status": status,
        "crosses_half": crosses_half,
        "bound_pse": round(bound_pse, 5) if np.isfinite(bound_pse) else np.nan,
        "max_drawdown": round(max_drawdown, 4) if np.isfinite(max_drawdown) else np.nan,
        "fit_r2": r2,
        "pse_abs_frac": round(pse_abs_frac, 4) if np.isfinite(pse_abs_frac) else np.nan,
        "reportable": status in ("measured", "bounded"),
    }


def pse_is_interpretable(regime: str, converged: bool, identified: bool) -> bool:
    """
    Whether a fitted PSE should be reported for this cell.

    Requires all three of: a response function a monotonic model can describe,
    an optimiser result off its bounds, and a profile interval narrow enough to
    constrain the estimate.
    """
    return bool(converged and identified and regime in PSE_ESTIMABLE_REGIMES)
