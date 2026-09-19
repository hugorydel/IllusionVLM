"""
pipeline/module_3_analyse.py - Module 3: Psychometric fitting per illusion.

Iterates every model in config.MODELS that has results, and every illusion in
the registry. For each (model, illusion):
  1. Loads all participant JSONL files from results/<model>/<illusion>/participants/
  2. Aggregates them into response cells (psychometric_data.csv), which is
     Module 4's input for the model
  3. Fits cumulative Gaussian psychometric functions per strength level and
     saves pse_summary.csv with the diagnostic CSVs

Figures are Module 4's job; this module writes tables only.

Skip logic:
  If all expected outputs already exist and are newer than the participant
  files, that (model, illusion) is skipped unless force=True is passed.
"""

from pathlib import Path

from config import MODELS
from pipeline.module_3.fit_psychometrics import run_fitting

RESULTS_ROOT = Path("results")


def _is_complete(base: Path) -> bool:
    """
    Return True if all expected outputs exist AND are newer than all participant files.

    If any participant file is newer than the oldest output, new data has been
    added since the last analysis run and a rerun is needed.
    """
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
    Fit and export results for every model with results, for each illusion.

    Args:
        illusions: List of illusion config dicts (from config.ILLUSIONS).
        force:     Refit even if outputs already exist.
    """
    models = [m for m in MODELS if (RESULTS_ROOT / m["key"]).exists()]
    print(
        f"\nAnalysing {len(illusions)} illusion(s) for {len(models)} model(s): "
        f"{', '.join(m['label'] for m in models)}"
    )

    for model in models:
        model_root = RESULTS_ROOT / model["key"]
        for illusion in illusions:
            name = illusion["name"]
            print(f"\n  {'━' * 50}")
            print(f"  {model['label']} — {name}")
            print(f"  {'━' * 50}")

            if not (model_root / name / "participants").exists():
                print("  – No responses yet — skipping.")
                continue

            if not force and _is_complete(model_root / name):
                print(f"  ✓ Already complete — skipping. (Use force=True to rerun.)")
                continue

            try:
                run_fitting(illusion, model_root, model=model["label"])
            except (FileNotFoundError, ValueError) as e:
                print(f"  ✗ Skipping {name}: {e}")
                continue

    print(f"\n✓ Module 3 complete — outputs saved under {RESULTS_ROOT}/<model>/")
