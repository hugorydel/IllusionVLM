#!/usr/bin/env python3
"""
batch_vlm.py - Submit and retrieve OpenAI Batch API jobs (Phase 2, async variant)

The Batch API is 3.4× cheaper than real-time querying for GPT-5.2, at the cost of a
delayed output.  This module handles three distinct phases:

  submit    Scan the output directory for existing participant files, compute
            which participant IDs are missing up to --n-participants, build a
            batch JSONL request file, upload it, and submit the batch job.
            Saves a state file (batch_state.json) so the other commands know
            which batch to poll.

  status    Check the current status of the submitted batch.

  download  Once the batch completes, download the results and write one
            participant_XX.jsonl per participant into the output directory,
            in the same format produced by query_vlm.py.

Gap-filling behaviour
---------------------
Only participant IDs that are genuinely absent from the output directory are
queued, regardless of the numbering of existing files.  Example:

    Existing files : participant_076.jsonl … participant_100.jsonl  (25 files)
    --n-participants 100
    → Missing IDs  : 1 … 75  (submitted to the batch)
    → NOT          : 101 … 125  (that would exceed the target)

This means you can freely mix real-time (query_vlm.py) and batch runs without
creating duplicate participant files.

Batch request format
--------------------
Each line in the uploaded JSONL is a /v1/chat/completions request.  Note that
the Batch API does not support /v1/responses (the newer endpoint used by
query_vlm.py), so this module uses chat.completions instead — the model output
and structured JSON schema behaviour are identical.

Custom ID encoding
------------------
    custom_id = "p{participant_id:03d}|{image_id}"

The pipe (|) is used as a separator because image IDs already contain
underscores.  The download phase splits on the first | to recover both fields.

State file
----------
    results/batch_state.json   (location overridable with --state-file)

Stores: batch_id, input_file_id, submitted_at, participant_ids,
n_participants_target, output_dir.

Usage:
    python batch_vlm.py submit   --n-participants 100
    python batch_vlm.py status
    python batch_vlm.py download

Options (all subcommands):
    --output-dir PATH     Participant JSONL directory  (default: ./results/synthetic_participants)
    --state-file PATH     Batch state JSON             (default: ./results/batch_state.json)

Options (submit only):
    --image-dir PATH      Stimulus PNG directory       (default: ./stimuli)
    --n-participants N    Target total participants    (required)
    --model NAME          OpenAI model                 (default: from parameters.py)
    --temperature FLOAT   Sampling temperature         (default: 0.5)
    --max-dimension N     Resize images above this     (default: 1024)
    --jpeg-quality N      JPEG quality 1-100           (default: 90)
    --batch-dir PATH      Staging dir for batch JSONL  (default: ./results/batch_staging)
    --dry-run             Show plan without uploading
"""

import argparse
import getpass
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from model_parameters import MAX_TOKENS, OPENAI_MODEL, TEMPERATURE
from openai import OpenAI
from PIL import Image
from query_vlm import compute_correct, discover_images, parse_image_id, preprocess_image
from response_schema import response_schema
from vlm_prompt import vlm_prompt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_MODEL = OPENAI_MODEL
STATE_FILENAME = "batch_state.json"

# Translate response_schema (responses-API format) to chat.completions format
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": response_schema["name"],
        "strict": response_schema["strict"],
        "schema": response_schema["schema"],
    },
}

# ============================================================================
# GAP-FILLING: which participant IDs are missing?
# ============================================================================


def existing_participant_ids(output_dir: Path) -> set[int]:
    """
    Scan output_dir for participant_*.jsonl files and return the set of their
    numeric IDs.  Only files that are non-empty are counted as complete.

    Examples:
        participant_01.jsonl  → 1
        participant_100.jsonl → 100
    """
    ids: set[int] = set()
    if not output_dir.exists():
        return ids
    for path in output_dir.glob("participant_*.jsonl"):
        if path.stat().st_size == 0:
            continue  # empty file = incomplete run, treat as missing
        stem = path.stem  # e.g. 'participant_03'
        try:
            ids.add(int(stem.split("_")[-1]))
        except ValueError:
            continue
    return ids


def missing_participant_ids(output_dir: Path, n_participants: int) -> list[int]:
    """
    Return the sorted list of participant IDs in [1, n_participants] that do
    not yet have a non-empty file in output_dir.
    """
    target = set(range(1, n_participants + 1))
    have = existing_participant_ids(output_dir)
    return sorted(target - have)


# ============================================================================
# BATCH JSONL PREPARATION
# ============================================================================


def make_custom_id(participant_id: int, image_id: str) -> str:
    """Encode participant + image into a single batch custom_id string."""
    return f"p{participant_id:03d}|{image_id}"


def parse_custom_id(custom_id: str) -> tuple[int, str]:
    """Decode a custom_id back into (participant_id, image_id)."""
    pid_str, image_id = custom_id.split("|", 1)
    return int(pid_str[1:]), image_id  # strip leading 'p'


def build_single_request(
    participant_id: int,
    image_id: str,
    image_base64: str,
    model: str,
    temperature: float,
) -> dict:
    """
    Build one /v1/chat/completions batch request dict for a single image.

    The Batch API does NOT support /v1/responses, so we use chat.completions
    format here.  Output is structurally identical to query_vlm.py.
    """
    return {
        "custom_id": make_custom_id(participant_id, image_id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "max_tokens": MAX_TOKENS,
            "response_format": _RESPONSE_FORMAT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vlm_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        },
    }


def prepare_batch_file(
    missing_ids: list[int],
    image_ids: list[str],
    image_dir: Path,
    batch_dir: Path,
    model: str,
    temperature: float,
    max_dimension: int,
    jpeg_quality: int,
) -> Path:
    """
    Build the batch input JSONL and write it to batch_dir.

    Returns the path to the written file.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = batch_dir / f"batch_input_{timestamp}.jsonl"

    total = len(missing_ids) * len(image_ids)
    print(
        f"\nPreparing batch JSONL: {len(missing_ids)} participants × "
        f"{len(image_ids)} images = {total} requests"
    )
    print(f"  Writing to: {out_path}")

    count = 0
    with open(out_path, "w") as f:
        for pid in missing_ids:
            for image_id in image_ids:
                image_path = image_dir / f"{image_id}.png"
                image_base64 = preprocess_image(
                    image_path,
                    max_dimension=max_dimension,
                    jpeg_quality=jpeg_quality,
                )
                request = build_single_request(
                    pid, image_id, image_base64, model, temperature
                )
                f.write(json.dumps(request) + "\n")
                count += 1
                if count % 500 == 0:
                    print(f"  ... {count}/{total} requests written")

    print(f"  ✓ {count} requests written")
    return out_path


# ============================================================================
# STATE FILE
# ============================================================================


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"  ✓ State saved to {state_path}")


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        print(f"Error: State file not found: {state_path}")
        print("Run 'python batch_vlm.py submit' first.")
        sys.exit(1)
    with open(state_path, "r") as f:
        return json.load(f)


# ============================================================================
# SUBCOMMAND: submit
# ============================================================================


def cmd_submit(args) -> None:
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    batch_dir = Path(args.batch_dir)
    state_path = Path(args.state_file)

    # ── Discover stimuli ───────────────────────────────────────────────────
    try:
        all_images = discover_images(image_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # ── Compute missing participant IDs ────────────────────────────────────
    missing = missing_participant_ids(output_dir, args.n_participants)
    have = existing_participant_ids(output_dir)

    print("\n" + "=" * 60)
    print("BATCH SUBMISSION PLAN")
    print("=" * 60)
    print(f"  Target participants : {args.n_participants}")
    print(f"  Already complete    : {len(have)}  {sorted(have) if have else '(none)'}")
    print(f"  To submit           : {len(missing)}  {missing}")
    print(f"  Images per part.    : {len(all_images)}")
    print(f"  Total requests      : {len(missing) * len(all_images)}")
    print(f"  Model               : {args.model}")
    print(f"  Temperature         : {args.temperature}")
    print("=" * 60)

    if not missing:
        print("\n✓ All participants already complete — nothing to submit.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would submit the above batch. Exiting.")
        return

    # ── Build batch JSONL ──────────────────────────────────────────────────
    batch_path = prepare_batch_file(
        missing_ids=missing,
        image_ids=all_images,
        image_dir=image_dir,
        batch_dir=batch_dir,
        model=args.model,
        temperature=args.temperature,
        max_dimension=args.max_dimension,
        jpeg_quality=args.jpeg_quality,
    )

    # ── API key ────────────────────────────────────────────────────────────
    print()
    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
    if not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    client = OpenAI(api_key=api_key.strip())

    # ── Upload file ────────────────────────────────────────────────────────
    print("\nUploading batch input file...")
    with open(batch_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  ✓ File uploaded: {uploaded.id}  ({uploaded.filename})")

    # ── Submit batch ───────────────────────────────────────────────────────
    print("Submitting batch job...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"VLM illusion study — participants {missing[0]}–{missing[-1]}"
        },
    )
    print(f"  ✓ Batch submitted: {batch.id}")
    print(f"  Status            : {batch.status}")
    print(f"  Completion window : ~24h")

    # ── Save state ─────────────────────────────────────────────────────────
    state = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "participant_ids": missing,
        "n_participants_target": args.n_participants,
        "output_dir": str(output_dir),
        "model": args.model,
        "status": batch.status,
    }
    save_state(state_path, state)

    print(f"\nNext steps:")
    print(f"  Check status : python batch_vlm.py status")
    print(f"  Download     : python batch_vlm.py download  (once status = completed)")


# ============================================================================
# SUBCOMMAND: status
# ============================================================================


def cmd_status(args) -> None:
    state_path = Path(args.state_file)
    state = load_state(state_path)

    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
    if not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    client = OpenAI(api_key=api_key.strip())

    batch = client.batches.retrieve(state["batch_id"])

    print("\n" + "=" * 60)
    print("BATCH STATUS")
    print("=" * 60)
    print(f"  Batch ID          : {batch.id}")
    print(f"  Status            : {batch.status}")
    print(f"  Submitted at      : {state['submitted_at']}")
    print(f"  Participants      : {state['participant_ids']}")
    if batch.request_counts:
        rc = batch.request_counts
        print(f"  Requests total    : {rc.total}")
        print(f"  Requests completed: {rc.completed}")
        print(f"  Requests failed   : {rc.failed}")
    if batch.output_file_id:
        print(f"  Output file ID    : {batch.output_file_id}")
    print("=" * 60)

    if batch.status == "completed":
        print(
            "\n✓ Batch complete — run 'python batch_vlm.py download' to retrieve results."
        )
    elif batch.status == "failed":
        print("\n✗ Batch failed. Check the OpenAI dashboard for details.")
    else:
        print(f"\n  Still processing. Check again later.")

    # Update status in state file
    state["status"] = batch.status
    if batch.output_file_id:
        state["output_file_id"] = batch.output_file_id
    save_state(state_path, state)


# ============================================================================
# SUBCOMMAND: download
# ============================================================================


def parse_batch_response(line: str) -> Optional[dict]:
    """
    Parse one line of the batch output JSONL and return a participant record,
    or None if the request failed.

    Each output line has the shape:
        {
            "custom_id": "p001|MullerLyer_str+049_diff+0.46000",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": "{\"image_id\":...,\"response\":...}"}}]
                }
            },
            "error": null
        }
    """
    obj = json.loads(line)

    if obj.get("error") is not None:
        return None

    response_body = obj.get("response", {}).get("body", {})
    if response_body.get("error"):
        return None

    choices = response_body.get("choices", [])
    if not choices:
        return None

    content = choices[0]["message"]["content"]
    parsed = json.loads(content)

    participant_id, image_id = parse_custom_id(obj["custom_id"])
    strength, true_diff = parse_image_id(image_id)
    correct = compute_correct(parsed["response"], true_diff)

    return {
        "participant_id": participant_id,
        "image_id": image_id,
        "illusion_strength": strength,
        "true_diff": true_diff,
        "response": parsed["response"],
        "correct": correct,
    }


def cmd_download(args) -> None:
    state_path = Path(args.state_file)
    state = load_state(state_path)
    output_dir = Path(args.output_dir or state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if state.get("status") != "completed" and "output_file_id" not in state:
        print("Batch does not appear to be completed yet.")
        print("Run 'python batch_vlm.py status' to check.")
        print(
            "(If you believe it is complete, re-run status first to refresh the state file.)"
        )
        sys.exit(1)

    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
    if not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    client = OpenAI(api_key=api_key.strip())

    # Retrieve output file ID (refresh from API in case state is stale)
    batch = client.batches.retrieve(state["batch_id"])
    if batch.status != "completed":
        print(
            f"Batch status is '{batch.status}', not 'completed'. Cannot download yet."
        )
        sys.exit(1)

    output_file_id = batch.output_file_id
    print(f"\nDownloading results (file: {output_file_id})...")
    raw = client.files.content(output_file_id).text
    lines = [l for l in raw.splitlines() if l.strip()]
    print(f"  ✓ {len(lines)} response lines received")

    # ── Parse and bucket by participant ───────────────────────────────────
    records_by_participant: dict[int, list[dict]] = {}
    n_ok = 0
    n_fail = 0

    for line in lines:
        record = parse_batch_response(line)
        if record is None:
            n_fail += 1
            continue
        pid = record["participant_id"]
        records_by_participant.setdefault(pid, []).append(record)
        n_ok += 1

    print(f"  ✓ Parsed: {n_ok} successful, {n_fail} failed")

    # ── Write participant files ────────────────────────────────────────────
    print(f"\nWriting participant files to {output_dir}/")
    for pid in sorted(records_by_participant):
        out_path = output_dir / f"participant_{pid:02d}.jsonl"
        records = sorted(
            records_by_participant[pid],
            key=lambda r: (r["illusion_strength"], r["true_diff"]),
        )
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"  ✓ participant_{pid:02d}.jsonl  ({len(records)} responses)")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Participants written : {len(records_by_participant)}")
    print(f"  Failed requests      : {n_fail}")
    print(f"  Output directory     : {output_dir}/")
    print("=" * 60)
    print("\nNext step: run fit_psychometrics.py to aggregate and fit PSEs.")

    # Mark state as downloaded
    state["status"] = "downloaded"
    save_state(state_path, state)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API wrapper for VLM illusion study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Shared options ─────────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--output-dir",
        type=str,
        default="./results/synthetic_participants",
        help="Participant JSONL directory (default: ./results/synthetic_participants)",
    )
    shared.add_argument(
        "--state-file",
        type=str,
        default="./results/batch_state.json",
        help="Batch state file (default: ./results/batch_state.json)",
    )

    # ── submit ─────────────────────────────────────────────────────────────
    p_submit = subparsers.add_parser(
        "submit",
        parents=[shared],
        help="Prepare and submit a batch job for missing participants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_submit.add_argument(
        "--n-participants",
        type=int,
        required=True,
        help="Target total number of participants (fills gaps up to this number)",
    )
    p_submit.add_argument(
        "--image-dir",
        type=str,
        default="./stimuli",
        help="Directory of stimulus PNGs (default: ./stimuli)",
    )
    p_submit.add_argument(
        "--batch-dir",
        type=str,
        default="./results/batch_staging",
        help="Staging directory for batch JSONL files (default: ./results/batch_staging)",
    )
    p_submit.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL})",
    )
    p_submit.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})",
    )
    p_submit.add_argument(
        "--max-dimension",
        type=int,
        default=1024,
        help="Max image dimension before resizing (default: 1024)",
    )
    p_submit.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for API upload (default: 90)",
    )
    p_submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without uploading or submitting",
    )

    # ── status ─────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "status",
        parents=[shared],
        help="Check the status of the submitted batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── download ───────────────────────────────────────────────────────────
    subparsers.add_parser(
        "download",
        parents=[shared],
        help="Download completed batch results and write participant files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "download":
        cmd_download(args)


if __name__ == "__main__":
    main()
