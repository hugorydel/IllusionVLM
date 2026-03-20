"""
pipeline/module_3_analyse.py - Module 3: Psychometric fitting and visualisation.

Iterates the illusion registry. For each illusion:
  1. Loads all participant JSONL files from results/<name>/participants/
  2. Fits cumulative Gaussian psychometric functions per strength level
  3. Extracts PSE values and saves pse_summary.csv + psychometric_data.csv
  4. Generates fig1_psychometric_curves.png and fig2_pse_vs_strength.png

Skip logic:
  If both pse_summary.csv and the two figures already exist for an illusion,
  that illusion is skipped unless force=True is passed.
"""

from pathlib import Path

from pipeline.module_3.fit_psychometrics import run_fitting
from pipeline.module_3.plot_results import run_plotting

RESULTS_ROOT = Path("results")


def _is_complete(illusion_name: str) -> bool:
    """Return True if all expected outputs already exist for this illusion."""
    base = RESULTS_ROOT / illusion_name
    expected = [
        base / "pse_summary.csv",
        base / "psychometric_data.csv",
        base / "figures" / "fig1_psychometric_curves.png",
        base / "figures" / "fig2_pse_vs_strength.png",
    ]
    return all(p.exists() for p in expected)


def run(illusions: list[dict], force: bool = False) -> None:
    """
    Fit and plot results for all illusions in the registry.

    Args:
        illusions: List of illusion config dicts (from config.ILLUSIONS).
        force:     Refit and replot even if outputs already exist.
    """
    print(f"\nAnalysing {len(illusions)} illusion(s)...")

    for illusion in illusions:
        name = illusion["name"]
        print(f"\n  {'━' * 50}")
        print(f"  {name}")
        print(f"  {'━' * 50}")

        if not force and _is_complete(name):
            print(f"  ✓ Already complete — skipping. (Use force=True to rerun.)")
            continue

        try:
            psych_data, pse_summary = run_fitting(illusion, RESULTS_ROOT)
            run_plotting(illusion, psych_data, pse_summary, RESULTS_ROOT)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ✗ Skipping {name}: {e}")
            continue

    print(f"\n✓ Module 3 complete — outputs saved under {RESULTS_ROOT}/")
