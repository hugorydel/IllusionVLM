"""
pipeline/module_2_query.py - Module 2: VLM querying (real-time).

Iterates the illusion registry. For each illusion:
  1. Discovers all stimuli in stimuli/<name>/
  2. Counts complete participant files already in results/<name>/participants/
  3. Generates only the remaining (N_PARTICIPANTS - already_complete) participants
  4. Each participant's responses are written to their own JSONL file

Skip logic:
  A participant file is considered "complete" if it contains a response for
  every image in the current stimulus grid. Partially-complete files have only
  their missing images queried.

For the cheaper Batch API alternative, use pipeline/module_2/batch_vlm.py.
"""

import asyncio
import getpass
import json
import sys
from pathlib import Path

from config import (
    JPEG_QUALITY,
    MAX_CONCURRENCY,
    MAX_DIMENSIONS,
    MAX_TOKENS,
    MODEL,
    N_PARTICIPANTS,
    TEMPERATURE,
)
from pipeline.module_2.batch_processor import BatchProcessor
from pipeline.module_2.query import VLMQuerier
from pipeline.utils import discover_images

RESULTS_ROOT = Path("results")
STIMULI_ROOT = Path("stimuli")


# ============================================================================
# RESULT MANAGEMENT
# ============================================================================


def _participants_dir(illusion_name: str) -> Path:
    return RESULTS_ROOT / illusion_name / "participants"


def _errors_dir(illusion_name: str) -> Path:
    return RESULTS_ROOT / illusion_name / "errors"


def _load_done_images(jsonl_path: Path) -> set[str]:
    """Return the set of image_ids already recorded in a participant file."""
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["image_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    except Exception as e:
        print(f"  Warning: Could not read {jsonl_path}: {e}")
    return done


# ============================================================================
# PER-PARTICIPANT RUN
# ============================================================================


async def run_participant(
    participant_id: int,
    total_participants: int,
    all_images: list[str],
    image_dir: Path,
    out_dir: Path,
    err_dir: Path,
    querier: VLMQuerier,
    max_concurrency: int,
    max_dimension: int,
    jpeg_quality: int,
) -> tuple[int, int]:
    """
    Run one participant over whatever images are missing from their file.

    Returns:
        (n_processed, n_errors)
    """
    out_path = out_dir / f"participant_{participant_id:02d}.jsonl"
    err_path = err_dir / f"participant_{participant_id:02d}_errors.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        out_path.write_text("")
    if not err_path.exists():
        err_path.write_text("")

    already_done = _load_done_images(out_path)
    to_process = [img for img in all_images if img not in already_done]

    if not to_process:
        print(f"  [P{participant_id:02d}] ✓ Already complete — skipping.")
        return 0, 0

    if already_done:
        print(
            f"  [P{participant_id:02d}] Resuming — "
            f"{len(already_done)} done, {len(to_process)} remaining"
        )

    print(f"\n{'─' * 60}")
    print(
        f"  Participant {participant_id:02d} / {total_participants}  ({len(to_process)} images)"
    )
    print(f"{'─' * 60}")

    processor = BatchProcessor(
        querier=querier,
        image_dir=image_dir,
        output_path=out_path,
        errors_path=err_path,
        participant_id=participant_id,
        max_concurrency=max_concurrency,
        max_dimension=max_dimension,
        jpeg_quality=jpeg_quality,
    )
    await processor.process_batch(to_process)
    return processor.processed, processor.errors


# ============================================================================
# PER-ILLUSION QUERY
# ============================================================================


async def query_illusion(
    illusion: dict,
    querier: VLMQuerier,
    n_participants: int,
    max_concurrency: int,
    max_dimension: int,
    jpeg_quality: int,
) -> None:
    name = illusion["name"]
    image_dir = STIMULI_ROOT / name
    out_dir = _participants_dir(name)
    err_dir = _errors_dir(name)

    try:
        all_images = discover_images(
            image_dir,
            name,
            strengths=illusion.get("strengths"),
            differences=illusion.get("differences"),
        )
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return

    print(
        f"\n  {name} — {len(all_images)} stimuli, target {n_participants} participants"
    )

    total_processed = total_errors = 0

    for pid in range(1, n_participants + 1):
        n_ok, n_err = await run_participant(
            participant_id=pid,
            total_participants=n_participants,
            all_images=all_images,
            image_dir=image_dir,
            out_dir=out_dir,
            err_dir=err_dir,
            querier=querier,
            max_concurrency=max_concurrency,
            max_dimension=max_dimension,
            jpeg_quality=jpeg_quality,
        )
        total_processed += n_ok
        total_errors += n_err

    print(f"\n  ✓ {name} complete: {total_processed} responses, {total_errors} errors")
    if total_errors:
        print(f"    Error logs: {err_dir}/")


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================


def run(illusions: list[dict], dry_run: bool = False) -> None:
    """
    Query the VLM for all illusions in the registry.

    Args:
        illusions: List of illusion config dicts (from config.ILLUSIONS).
        dry_run:   Print the plan without making any API calls.
    """
    print(f"\nQuerying VLM for {len(illusions)} illusion(s)...")
    print(f"  Model            : {MODEL}")
    print(f"  Temperature      : {TEMPERATURE}")
    print(f"  Target N         : {N_PARTICIPANTS} participants")
    print(f"  Max concurrency  : {MAX_CONCURRENCY}")

    if dry_run:
        print("\n[DRY RUN] Would process:")
        for ill in illusions:
            print(f"  {ill['name']}: {N_PARTICIPANTS} participants × stimulus grid")
        return

    api_key = getpass.getpass("\nEnter your OpenAI API key (input hidden): ")
    if not api_key or not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    print("  ✓ API key received")

    async def _run_all():
        for illusion in illusions:
            querier = VLMQuerier(
                api_key=api_key.strip(),
                model=MODEL,
                temperature=TEMPERATURE,
                illusion=illusion,
                max_tokens=MAX_TOKENS,
            )
            await query_illusion(
                illusion=illusion,
                querier=querier,
                n_participants=N_PARTICIPANTS,
                max_concurrency=MAX_CONCURRENCY,
                max_dimension=MAX_DIMENSIONS,
                jpeg_quality=JPEG_QUALITY,
            )

    asyncio.run(_run_all())
    print(f"\n✓ Module 2 complete — results saved under {RESULTS_ROOT}/")
