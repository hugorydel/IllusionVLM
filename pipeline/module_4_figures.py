"""
pipeline/module_4_figures.py - Module 4: human comparison and paper figures.

Runs after Module 3, whose results/<model>/<illusion>/psychometric_data.csv
files are the models' input here. Two stages:

  1. pipeline/human_comparison/ puts every model and the human Illusion Game
     data through identical processing and writes the comparison tables to
     results/_paper/*.csv.
  2. pipeline/figures/ reads only those tables and renders every paper figure
     to results/_paper/figures/, as PNG (400 dpi) and PDF (vector, for
     submission):

         fig1_paradigm             the task and stimuli
         fig2_perceptual_shift     perceptual shift against illusion strength
         fig3_congruency           error rate against illusion strength
         fig4_summary              both measures relative to humans
         figS1_error_by_difficulty error rate by difficulty band (supplement)
         figS2_response_surface    raw choices per condition (supplement)
         figS3_exact_probabilities exact option probabilities against the
                                   sampled answers, for the open models
                                   (supplement; only once any have run)

Every figure draws humans and each model in config.MODELS that has results.

Usage:
    python -m pipeline.module_4_figures                 # rebuild tables, render all
    python -m pipeline.module_4_figures --skip-dataset  # render from existing tables

The human data are Makowski et al. (2023) study2_part1.csv, study2_part2.csv
and study3.csv, from RealityBending/IllusionGameValidation, expected in
data/human/ unless --human says otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.figures.panels import (
    fig1_paradigm,
    fig2_perceptual_shift,
    fig3_congruency,
    fig4_summary,
    figS1_error_by_difficulty,
    figS2_response_surface,
    figS3_exact_probabilities,
)
from pipeline.human_comparison import build_dataset

RESULTS_ROOT = Path("results")
STIMULI_ROOT = Path("stimuli")
HUMAN_DIR = Path("data/human")
PAPER_DIR = RESULTS_ROOT / "_paper"
FIG_DIR = PAPER_DIR / "figures"


def render_figures(paper_dir: Path = PAPER_DIR, stimuli_root: Path = STIMULI_ROOT) -> None:
    """Render every paper figure from the tables in `paper_dir`."""
    fig_dir = paper_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Figure 1 - paradigm")
    try:
        fig1_paradigm.build(stimuli_root, fig_dir / "fig1_paradigm.png", paper_dir)
    except FileNotFoundError as e:
        # stimuli/ is gitignored, so a fresh clone will not have the images.
        print(f"  skipped: {e}")

    print("\n  Figure 2 - perceptual shift against strength")
    fig2_perceptual_shift.build(paper_dir, fig_dir / "fig2_perceptual_shift.png")

    print("\n  Figure 3 - error rate against strength")
    fig3_congruency.build(paper_dir, fig_dir / "fig3_congruency.png")

    print("\n  Figure 4 - both measures relative to humans")
    fig4_summary.build(paper_dir, fig_dir / "fig4_summary.png")

    print("\n  Figure S1 - error rate by difficulty")
    figS1_error_by_difficulty.build(paper_dir, fig_dir / "figS1_error_by_difficulty.png")

    print("\n  Figure S2 - response surfaces")
    figS2_response_surface.build(paper_dir, fig_dir / "figS2_response_surface.png")

    print("\n  Figure S3 - exact probabilities")
    figS3_exact_probabilities.build(paper_dir, fig_dir / "figS3_exact_probabilities.png")


def run(
    human_dir: Path = HUMAN_DIR,
    skip_dataset: bool = False,
    results_root: Path = RESULTS_ROOT,
    stimuli_root: Path = STIMULI_ROOT,
) -> None:
    """
    Build the comparison tables, then render every paper figure.

    Args:
        human_dir:    Directory holding the Illusion Game CSVs.
        skip_dataset: Re-render from the tables already in results/_paper.
        results_root: Top-level results directory (Module 3 outputs).
        stimuli_root: Stimulus images, for Figure 1.
    """
    paper_dir = results_root / "_paper"

    if not skip_dataset:
        if not human_dir.exists():
            raise FileNotFoundError(
                f"Human data not found in {human_dir}. Put study2_part1.csv, "
                "study2_part2.csv and study3.csv there, or pass skip_dataset=True "
                "to re-render from the existing tables."
            )
        print("\n  Building the human comparison tables...")
        build_dataset.build(results_root, human_dir, paper_dir)

    render_figures(paper_dir, stimuli_root)
    print(f"\n✓ Module 4 complete — figures saved under {paper_dir / 'figures'}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--human", type=Path, default=HUMAN_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--stimuli", type=Path, default=STIMULI_ROOT)
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Re-render figures from the existing results/_paper tables",
    )
    args = parser.parse_args()
    run(args.human, args.skip_dataset, args.results, args.stimuli)


if __name__ == "__main__":
    main()
