"""
pipeline/human_comparison/build_dataset.py - Assemble the tidy tables the figures read.

Puts both species through identical processing and writes the result to
results/_paper/ so that figure code never touches raw data and every figure is
guaranteed to be drawn from the same numbers.

Outputs
-------
    cells.csv         per (species, illusion, strength, true_diff) response
                      cell, with the illusion direction and |strength| added
    by_direction.csv  one psychometric fit per (species, illusion, |strength|,
                      illusion direction) - the primary analysis
    by_magnitude.csv  per (species, illusion, |strength|): illusion magnitude
                      and side bias, from the two direction fits
    competence.csv    per (species, illusion): peak accuracy and the 75%
                      threshold with the illusion off, plus the can_do_task gate
    slope_lrt.csv     likelihood-ratio test of the shared slope and lapses in
                      the by_magnitude fit, one row per level
    scalars.csv       one row per (species, illusion): magnitude at the
                      strongest reportable level, its trend against strength,
                      and baseline competence
    by_strength.csv   descriptive error rates per SIGNED strength, for the
                      congruency figure only
    shared_grid.csv   what the species share, and what was dropped
    human_scores.csv  per-participant human sensitivity scores (study3)

All comparisons are computed on the shared (strength, difference) grid. Our
grid extends beyond the human one for several illusions; those extra cells are
kept in cells.csv, flagged `shared=False`, and excluded from everything that
contrasts the two species.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.human_comparison.conventions import ILLUSION_ORDER, apply_canonical_signs
from pipeline.human_comparison.human_data import (
    aggregate_psychometric_data,
    load_trials,
    restrict_to_shared_grid,
    validate_reconstruction,
)
from pipeline.human_comparison.metrics import (
    add_illusion_direction,
    baseline_competence,
    baseline_jnd,
    error_by_strength,
    illusion_magnitude,
    illusion_scalars,
    pse_by_direction,
)
from pipeline.human_comparison.robustness import slope_asymmetry

OUT_DIR = Path("results/_paper")


# ============================================================================
# LOADING PER SPECIES
# ============================================================================


def load_vlm_cells(results_root: Path) -> pd.DataFrame:
    """Load every illusion's VLM response cells, canonically signed."""
    frames = []
    for illusion in ILLUSION_ORDER:
        path = results_root / illusion / "psychometric_data.csv"
        if not path.exists():
            print(f"  ! missing {path}, skipping {illusion}")
            continue
        frames.append(pd.read_csv(path).assign(illusion=illusion))
    if not frames:
        raise FileNotFoundError(f"No VLM psychometric_data.csv found under {results_root}")
    return apply_canonical_signs(pd.concat(frames, ignore_index=True), "vlm")


def load_human_cells(human_dir: Path) -> pd.DataFrame:
    """Load every illusion's human response cells, canonically signed."""
    trials = load_trials(human_dir)

    checks = validate_reconstruction(trials)
    failed = checks[~checks["passed"]]
    if not failed.empty:
        raise ValueError(
            "Human response reconstruction failed its sanity check for: "
            f"{list(failed['illusion'])}\n{failed.to_string(index=False)}"
        )

    frames = [
        aggregate_psychometric_data(trials, illusion).assign(illusion=illusion)
        for illusion in ILLUSION_ORDER
    ]
    return apply_canonical_signs(pd.concat(frames, ignore_index=True), "human")


# ============================================================================
# MAIN
# ============================================================================


def build(results_root: Path, human_dir: Path, out_dir: Path = OUT_DIR) -> dict:
    """Build every table and write it to `out_dir`. Returns the frames."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading VLM cells...")
    vlm = load_vlm_cells(results_root)
    print(f"  {len(vlm)} cells across {vlm['illusion'].nunique()} illusions")

    print("Loading human cells...")
    human = load_human_cells(human_dir)
    print(f"  {len(human)} cells across {human['illusion'].nunique()} illusions")

    # Mark the shared grid per illusion.
    shared_rows, v_parts, h_parts = [], [], []
    for illusion in ILLUSION_ORDER:
        v = vlm[vlm["illusion"] == illusion]
        h = human[human["illusion"] == illusion]
        if v.empty or h.empty:
            continue
        vs, hs, info = restrict_to_shared_grid(v, h)
        shared_rows.append({"illusion": illusion, **{
            k: val for k, val in info.items() if not isinstance(val, list)
        }})
        v_parts.append(vs)
        h_parts.append(hs)

    shared = pd.DataFrame(shared_rows)

    def tag_shared(full: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
        keys = set(
            zip(
                kept["illusion"],
                kept["illusion_strength"].round(5),
                kept["true_diff"].round(5),
            )
        )
        full = full.copy()
        full["shared"] = [
            (i, s, d) in keys
            for i, s, d in zip(
                full["illusion"],
                full["illusion_strength"].round(5),
                full["true_diff"].round(5),
            )
        ]
        return full

    vlm = tag_shared(vlm, pd.concat(v_parts, ignore_index=True))
    human = tag_shared(human, pd.concat(h_parts, ignore_index=True))

    cells = pd.concat(
        [vlm.assign(species="vlm"), human.assign(species="human")], ignore_index=True
    )

    # Everything comparative is computed on the shared grid only.
    shared_cells = cells[cells["shared"]].copy()

    print("Slicing by illusion direction...")
    sliced = add_illusion_direction(shared_cells)
    print(
        f"  {len(sliced)} cells in "
        f"{sliced.groupby(['species','illusion','k_abs','direction']).ngroups} slices"
    )

    print("Fitting one psychometric function per slice...")
    by_direction = pse_by_direction(sliced)
    ok = int(by_direction["reportable"].sum())
    print(f"  {len(by_direction)} fits, {ok} reportable")
    for status, n in by_direction["fit_status"].value_counts().items():
        print(f"    {status:18s} {n}")

    print("Fitting magnitude jointly across both directions...")
    jnd = baseline_jnd(shared_cells, species="human")
    magnitudes = illusion_magnitude(sliced, jnd=jnd)
    print(
        f"  {len(magnitudes)} (illusion, strength) levels, "
        f"{int(magnitudes['reportable'].sum())} estimated, "
        f"{int(magnitudes['saturated'].sum())} bounded"
    )

    print("Assessing baseline competence...")
    competence = baseline_competence(shared_cells)
    for _, r in competence[~competence["can_do_task"]].iterrows():
        print(f"    {r['species']}/{r['illusion']}: peak accuracy = "
              f"{r['peak_accuracy']:.2f} - task not performed")

    print("Computing per-illusion scalars...")
    scalars = illusion_scalars(magnitudes, shared_cells, competence)

    print("Computing descriptive error rates...")
    by_strength = error_by_strength(shared_cells)

    print("Testing the shared-slope assumption behind the magnitude fit...")
    slope_lrt = slope_asymmetry(sliced)
    n_reject = int(slope_lrt["rejects_shared_slope"].sum())
    print(f"    {n_reject}/{len(slope_lrt)} levels reject a shared slope "
          "(Holm-corrected)")
    for sp, grp in slope_lrt.groupby("species"):
        print(f"      {sp}: {int(grp['rejects_shared_slope'].sum())}/{len(grp)}")

    # Carry the slice labels into the published cells table.
    cells = cells.merge(
        sliced[["species", "illusion", "illusion_strength", "true_diff", "k_abs", "direction"]],
        on=["species", "illusion", "illusion_strength", "true_diff"],
        how="left",
    )

    human_scores = pd.read_csv(human_dir / "study3.csv")

    frames = {
        "cells": cells,
        "by_direction": by_direction,
        "by_magnitude": magnitudes,
        "scalars": scalars,
        "competence": competence,
        "by_strength": by_strength,
        "slope_lrt": slope_lrt,
        "shared_grid": shared,
        "human_scores": human_scores,
    }
    for name, frame in frames.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"  wrote {path}  ({len(frame)} rows)")

    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--human",
        type=Path,
        required=True,
        help="Directory holding study2_part1.csv, study2_part2.csv, study3.csv",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    build(args.results, args.human, args.out)


if __name__ == "__main__":
    main()
