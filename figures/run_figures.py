"""
figures/run_figures.py - Build the paper's figure set end to end.

Usage:
    python -m figures.run_figures --human <dir-with-study2-and-study3-csvs>

Rebuilds results/_paper/*.csv from the VLM participant data and the human
Illusion Game data, then renders every main figure to
results/_paper/figures/ as both PNG (400 dpi) and PDF (vector, for
submission).

Pass --skip-dataset to re-render figures from the tables already on disk,
which is the fast path while iterating on layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from figures import build_dataset
from figures.panels import (
    fig1_paradigm,
    fig2_perceptual_shift,
    fig3_congruency,
    fig4_summary,
)

PAPER_DIR = Path("results/_paper")
FIG_DIR = PAPER_DIR / "figures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human",
        type=Path,
        help="Directory holding study2_part1.csv, study2_part2.csv, study3.csv "
        "(required unless --skip-dataset)",
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--stimuli", type=Path, default=Path("stimuli"))
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Re-render figures from the existing results/_paper tables",
    )
    args = parser.parse_args()

    if not args.skip_dataset:
        if args.human is None:
            parser.error("--human is required unless --skip-dataset is passed")
        print("=" * 70)
        print("Building dataset")
        print("=" * 70)
        build_dataset.build(args.results, args.human, PAPER_DIR)

    print()
    print("=" * 70)
    print("Rendering figures")
    print("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFigure 1 - paradigm")
    try:
        fig1_paradigm.build(args.stimuli, FIG_DIR / "fig1_paradigm.png")
    except FileNotFoundError as e:
        # stimuli/ is gitignored, so a fresh clone will not have the images.
        print(f"  skipped: {e}")

    print("\nFigure 2 - perceptual shift against strength")
    fig2_perceptual_shift.build(PAPER_DIR, FIG_DIR / "fig2_perceptual_shift.png")

    print("\nFigure 3 - congruency error rates")
    fig3_congruency.build(PAPER_DIR, FIG_DIR / "fig3_congruency.png")

    print("\nFigure 4 - both measures relative to humans")
    fig4_summary.build(PAPER_DIR, FIG_DIR / "fig4_summary.png")

    print(f"\nDone. Figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
