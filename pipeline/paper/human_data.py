"""
pipeline/paper/human_data.py - Human comparison data from the Illusion Game.

Loads the trial-level data of Makowski et al. (2023) Study 2 and reshapes it
into exactly the schema the VLM pipeline produces, so that both species can be
put through the same psychometric fitting.

Source
------
    https://github.com/RealityBending/IllusionGameValidation
    data/study2_part1.csv, data/study2_part2.csv   (256 participants)
    data/study3.csv                                (participant-level scores)

Why the grids line up
---------------------
config.ILLUSIONS was derived from the stimulus manifests of this same study,
so the human and VLM grids share all 15 illusion strengths and the core 8
absolute difference values for every illusion we test. Where our grid was
extended (extra MullerLyer / VerticalHorizontal / RodFrame differences, extra
VerticalHorizontal strengths), those levels simply have no human counterpart
and are dropped by `restrict_to_shared_grid`.

Reconstructing the response
---------------------------
The published data record `Error` rather than the key pressed in a form we can
map directly onto our positive/negative option. They also record
`Illusion_Side`, the sign of the signed difference built into the stimulus, and
`Illusion_Difference`, its magnitude. Together these recover both axes:

    signed_diff        = Illusion_Difference * Illusion_Side
    responded_positive = (Illusion_Side == +1) XOR (Error == 1)

i.e. a correct response means the participant chose the side that was in fact
larger, and an error means they chose the other one. `validate_reconstruction`
checks the result behaves like a psychometric function before it is used.

Sign convention
---------------
Handled entirely by `conventions.apply_canonical_signs`; this module does not
flip anything itself. The published human strengths turn out to be canonical
already for all ten illusions, contrary to the changelog-derived note in
config.py - see conventions.py for the evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Illusion_Type in the human data -> our config.ILLUSIONS name.
HUMAN_TO_OURS = {
    "Contrast": "Contrast",
    "Delboeuf": "Delboeuf",
    "Ebbinghaus": "Ebbinghaus",
    "Müller-Lyer": "MullerLyer",
    "Poggendorff": "Poggendorff",
    "Ponzo": "Ponzo",
    "Rod-Frame": "RodFrame",
    "Vertical-Horizontal": "VerticalHorizontal",
    "White": "White",
    "Zöllner": "Zollner",
}

USED_COLUMNS = [
    "Participant",
    "Illusion_Type",
    "Illusion_Strength",
    "Illusion_Difference",
    "Illusion_Side",
    "Illusion_Effect",
    "Error",
    "RT",
]


# ============================================================================
# LOADING
# ============================================================================


def load_trials(data_dir: Path) -> pd.DataFrame:
    """
    Load and normalise both halves of the human Study 2 trial data.

    Args:
        data_dir: Directory holding study2_part1.csv and study2_part2.csv.

    Returns:
        Trial-level DataFrame with columns illusion, participant_id,
        illusion_strength, true_diff, responded_positive, correct and rt.

    Raises:
        FileNotFoundError: if either half is missing.
    """
    parts = []
    for n in (1, 2):
        path = data_dir / f"study2_part{n}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Human data not found: {path}\n"
                "Download from https://github.com/RealityBending/"
                "IllusionGameValidation/tree/main/data"
            )
        parts.append(pd.read_csv(path, usecols=USED_COLUMNS))

    df = pd.concat(parts, ignore_index=True)

    df["illusion"] = df["Illusion_Type"].map(HUMAN_TO_OURS)
    unmapped = df.loc[df["illusion"].isna(), "Illusion_Type"].unique()
    if len(unmapped):
        raise ValueError(f"Unmapped Illusion_Type values: {list(unmapped)}")

    # Error and Illusion_Side arrive as strings in places; coerce both.
    df["Error"] = pd.to_numeric(df["Error"], errors="coerce")
    df["Illusion_Side"] = pd.to_numeric(df["Illusion_Side"], errors="coerce")
    df["Illusion_Strength"] = pd.to_numeric(df["Illusion_Strength"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Error", "Illusion_Side", "Illusion_Strength"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} trial(s) with missing response or stimulus data")

    # Recover the signed difference and the response (see module docstring).
    df["true_diff"] = df["Illusion_Difference"] * df["Illusion_Side"]
    df["responded_positive"] = (
        (df["Illusion_Side"] > 0) ^ (df["Error"] == 1)
    ).astype(int)
    df["correct"] = (1 - df["Error"]).astype(int)

    out = df.rename(
        columns={
            "Participant": "participant_id",
            "Illusion_Strength": "illusion_strength",
            "Illusion_Effect": "congruency",
            "RT": "rt",
        }
    )
    return out[
        [
            "illusion",
            "participant_id",
            "illusion_strength",
            "true_diff",
            "responded_positive",
            "correct",
            "congruency",
            "rt",
        ]
    ].reset_index(drop=True)


# ============================================================================
# VALIDATION
# ============================================================================


def validate_reconstruction(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Check the reconstructed response axis behaves like a psychometric function.

    At zero illusion strength P(positive) must rise with the signed difference
    and overall accuracy must sit well above chance. If the response were
    reconstructed with the wrong polarity this would come out inverted, so a
    negative correlation here means the derivation is wrong rather than that
    the participants were.

    Returns one row per illusion with the check statistics and a `passed` flag.
    """
    rows = []
    for illusion, grp in trials.groupby("illusion"):
        base = grp[grp["illusion_strength"] == 0]
        cells = (
            base.groupby("true_diff")["responded_positive"].mean().sort_index()
            if len(base)
            else pd.Series(dtype=float)
        )
        rho = (
            float(np.corrcoef(cells.index.values, cells.values)[0, 1])
            if len(cells) >= 4
            else np.nan
        )
        acc = float(grp["correct"].mean())
        rows.append(
            {
                "illusion": illusion,
                "n_trials": len(grp),
                "n_participants": grp["participant_id"].nunique(),
                "baseline_cells": len(cells),
                "baseline_rho_diff_response": round(rho, 4),
                "overall_accuracy": round(acc, 4),
                "passed": bool(np.isfinite(rho) and rho > 0.8 and acc > 0.6),
            }
        )
    return pd.DataFrame(rows).sort_values("illusion").reset_index(drop=True)


# ============================================================================
# AGGREGATION TO THE VLM SCHEMA
# ============================================================================


def aggregate_psychometric_data(trials: pd.DataFrame, illusion: str) -> pd.DataFrame:
    """
    Collapse one illusion's human trials to the psychometric_data.csv schema.

    Returns columns illusion_strength, true_diff, n_trials, n_positive,
    prop_positive - identical to the VLM pipeline's output, so the same fitting
    and plotting code applies unchanged.
    """
    sub = trials[trials["illusion"] == illusion]
    if sub.empty:
        raise ValueError(f"No human trials for illusion {illusion!r}")

    grouped = (
        sub.groupby(["illusion_strength", "true_diff"])
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


def restrict_to_shared_grid(
    vlm_data: pd.DataFrame, human_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Cut both species down to the (strength, difference) cells they share.

    Our grid extends beyond the human one for some illusions. Comparisons must
    be made on the shared cells; the extra VLM levels are still reported, but
    separately and never as part of a human contrast.

    Returns (vlm_shared, human_shared, info) where info records what was
    dropped so captions can state it.
    """

    def key(df):
        return set(zip(df["illusion_strength"].round(5), df["true_diff"].round(5)))

    shared = key(vlm_data) & key(human_data)

    def take(df):
        mask = [
            (s, d) in shared
            for s, d in zip(df["illusion_strength"].round(5), df["true_diff"].round(5))
        ]
        return df[mask].reset_index(drop=True)

    v, h = take(vlm_data), take(human_data)
    info = {
        "n_shared_cells": len(shared),
        "n_vlm_only_cells": len(key(vlm_data) - shared),
        "n_human_only_cells": len(key(human_data) - shared),
        "shared_strengths": sorted({s for s, _ in shared}),
        "shared_diffs": sorted({d for _, d in shared}),
    }
    return v, h, info
