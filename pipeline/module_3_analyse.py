"""
pipeline/module_3_analyse.py - Module 3: Psychometric fitting and visualisation.

Iterates the illusion registry. For each illusion:
  1. Loads all participant JSONL files from results/<n>/participants/
  2. Fits cumulative Gaussian psychometric functions per strength level
  3. Extracts PSE values and saves pse_summary.csv + psychometric_data.csv
  4. Generates fig1_error_by_difficulty.png, fig2_pse_vs_strength.png,
     and fig3_psychometric_curves.png

Skip logic:
  If all expected outputs already exist for an illusion, that illusion is
  skipped unless force=True is passed.
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
        base / "figures" / "fig1_error_by_difficulty.png",
        base / "figures" / "fig2_pse_vs_strength.png",
        base / "figures" / "fig3_psychometric_curves.png",
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

        participants_dir = RESULTS_ROOT / name / "participants"

        try:
            psych_data, pse_summary = run_fitting(illusion, RESULTS_ROOT)
            run_plotting(
                illusion=illusion,
                psych_data=psych_data,
                pse_summary=pse_summary,
                results_root=RESULTS_ROOT,
                participants_dir=participants_dir,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"  ✗ Skipping {name}: {e}")
            continue

    print(f"\n✓ Module 3 complete — outputs saved under {RESULTS_ROOT}/")
