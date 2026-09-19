"""
pipeline/module_3_analyse.py - Module 3: Psychometric fitting per illusion.

Iterates the illusion registry. For each illusion:
  1. Loads all participant JSONL files from results/<n>/participants/
  2. Aggregates them into response cells (psychometric_data.csv), which is
     Module 4's input for the model
  3. Fits cumulative Gaussian psychometric functions per strength level and
     saves pse_summary.csv with the diagnostic CSVs

Figures are Module 4's job; this module writes tables only.

Skip logic:
  If all expected outputs already exist for an illusion, that illusion is
  skipped unless force=True is passed.
"""

from pathlib import Path

from pipeline.module_3.fit_psychometrics import run_fitting

RESULTS_ROOT = Path("results")


def _is_complete(illusion_name: str) -> bool:
    """
    Return True if all expected outputs exist AND are newer than all participant files.

    If any participant file is newer than the oldest output, new data has been
    added since the last analysis run and a rerun is needed.
    """
    base = RESULTS_ROOT / illusion_name
    expected = [
        base / "pse_summary.csv",
        base / "psychometric_data.csv",
        base / "aggregated_responses.csv",
        base / "fit_diagnostics.csv",
        base / "baseline_summary.csv",
        base / "illusion_summary.csv",
    ]

    # All outputs must exist
    if not all(p.exists() for p in expected):
        return False

    # Find the oldest output modification time
    oldest_output_mtime = min(p.stat().st_mtime for p in expected)

    # Find all participant files
    participants_dir = base / "participants"
    participant_files = (
        list(participants_dir.glob("participant_*.jsonl"))
        if participants_dir.exists()
        else []
    )

    if not participant_files:
        return True  # No participant files yet — outputs are up to date

    # If any participant file is newer than the oldest output, rerun needed
    newest_participant_mtime = max(p.stat().st_mtime for p in participant_files)
    return newest_participant_mtime <= oldest_output_mtime


def run(illusions: list[dict], force: bool = False) -> None:
    """
    Fit and export results for all illusions in the registry.

    Args:
        illusions: List of illusion config dicts (from config.ILLUSIONS).
        force:     Refit even if outputs already exist.
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
            run_fitting(illusion, RESULTS_ROOT)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ✗ Skipping {name}: {e}")
            continue

    print(f"\n✓ Module 3 complete — outputs saved under {RESULTS_ROOT}/")
