"""
pipeline/module_2/batch_vlm.py - OpenAI Batch API wrapper for the illusion pipeline.

Handles the three phases of an OpenAI Batch API job:

  submit    Identify missing (participant, image) pairs for a given illusion,
            build batch JSONL request files, upload, and submit. Saves state
            to results/<model>/<illusion>/_batch_tmp/batch_state.json.

  status    Poll the submitted batch(es) and print current progress.

  download  Download results, write participant_XX.jsonl files to
            results/<model>/<illusion>/participants/, and report coverage against the
            target grid. A batch can reach a terminal status with some of its
            requests failed or expired; those appear only in the batch's error
            file, so it is read and tallied here. _batch_tmp/ is deleted only
            when every request is accounted for -- otherwise the state and the
            cached output/error files are kept so the gap can be diagnosed and
            refilled by re-running submit.

Batch API constraint:
  The Batch API targets /v1/chat/completions (not /v1/responses), so logprobs
  are extracted from choices[0].logprobs.content in the download phase.

Cleanup:
  _batch_tmp/ is removed after a fully complete download so results/ stays
  clean, and preserved whenever anything is missing.

Standalone CLI usage (per-illusion):
    python -m pipeline.module_2.batch_vlm --illusion MullerLyer submit --n-participants 100
    python -m pipeline.module_2.batch_vlm --illusion MullerLyer status
    python -m pipeline.module_2.batch_vlm --illusion MullerLyer download
"""

import argparse
import getpass
import json
import math
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import (
    ILLUSIONS,
    JPEG_QUALITY,
    MAX_BATCH_BYTES,
    MAX_BATCH_REQUESTS,
    MAX_DIMENSIONS,
    MAX_TOKENS,
    MODEL,
    TEMPERATURE,
)
from pipeline.module_2.response_schema import make_chat_completions_schema
from pipeline.utils import (
    compute_correct,
    discover_images,
    parse_filename,
    preprocess_image,
)

if __import__("sys").stdout.encoding != "utf-8":
    import sys as _sys

    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(_sys.stderr, "reconfigure"):
        _sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# config.MODEL's results live in their own folder beside the other models';
# see config.MODELS.
RESULTS_ROOT = Path("results") / MODEL


# ============================================================================
# PATHS
# ============================================================================


def participants_dir(illusion_name: str) -> Path:
    return RESULTS_ROOT / illusion_name / "participants"


def batch_tmp_dir(illusion_name: str) -> Path:
    return RESULTS_ROOT / illusion_name / "_batch_tmp"


def state_path(illusion_name: str) -> Path:
    return batch_tmp_dir(illusion_name) / "batch_state.json"


# ============================================================================
# STATE FILE
# ============================================================================


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(path: Path) -> dict:
    if not path.exists():
        print(f"Error: State file not found: {path}")
        print("Run 'submit' first.")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


# ============================================================================
# CLEANUP
# ============================================================================


def cleanup_batch_tmp(illusion_name: str) -> None:
    """
    Remove the _batch_tmp/ directory for this illusion entirely.

    Called after a successful download once all participant files are written.
    """
    tmp = batch_tmp_dir(illusion_name)
    if tmp.exists():
        shutil.rmtree(tmp)
        print(f"  ✓ Cleaned up {tmp}/")


# ============================================================================
# GAP-FILLING
# ============================================================================


def _fmt_ranges(ids: list[int]) -> str:
    """Collapse a sorted id list to compact ranges, e.g. '26-85, 90'."""
    if not ids:
        return ""
    out, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = i
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(out)


def get_missing_requests(
    illusion_name: str,
    n_participants: int,
    all_images: list[str],
) -> dict[int, list[str]]:
    """
    Scan participants/ and return {participant_id: [missing_image_ids]}.

    Only participants with at least one missing image are included.
    """
    out_dir = participants_dir(illusion_name)
    requests: dict[int, list[str]] = {}

    for pid in range(1, n_participants + 1):
        path = out_dir / f"participant_{pid:02d}.jsonl"
        done: set[str] = set()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            done.add(json.loads(line)["image_id"])
                        except Exception:
                            pass
        missing = [img for img in all_images if img not in done]
        if missing:
            requests[pid] = missing

    return requests


# ============================================================================
# BATCH JSONL PREPARATION
# ============================================================================


def make_custom_id(participant_id: int, image_id: str) -> str:
    return f"p{participant_id:03d}|{image_id}"


def parse_custom_id(custom_id: str) -> tuple[int, str]:
    pid_str, image_id = custom_id.split("|", 1)
    return int(pid_str[1:]), image_id


def build_single_request(
    participant_id: int,
    image_id: str,
    image_base64: str,
    prompt: str,
    response_format: dict,
    model: str,
    temperature: float,
) -> dict:
    return {
        "custom_id": make_custom_id(participant_id, image_id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": MAX_TOKENS,  # must cover reasoning tokens; gpt-5.2 reasons on /v1/chat/completions regardless of reasoning_effort
            "response_format": response_format,
            "logprobs": True,
            "top_logprobs": 2,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
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


def prepare_batch_files(
    illusion: dict,
    requests_to_make: dict[int, list[str]],
    image_dir: Path,
    model: str,
    temperature: float,
    max_dimension: int,
    jpeg_quality: int,
) -> list[tuple[Path, list[int]]]:
    """
    Build one or more batch input JSONL files, splitting at MAX_BATCH_BYTES.

    Returns list of (path, participant_ids) tuples.
    """
    tmp = batch_tmp_dir(illusion["name"])
    tmp.mkdir(parents=True, exist_ok=True)

    response_format = make_chat_completions_schema(illusion["response_options"])
    prompt = illusion["prompt"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Pre-encode unique images
    needed: set[str] = set()
    for imgs in requests_to_make.values():
        needed.update(imgs)

    print(f"  Pre-encoding {len(needed)} unique images...")
    encoded: dict[str, str] = {
        img_id: preprocess_image(
            image_dir / f"{img_id}.png",
            max_dimension=max_dimension,
            jpeg_quality=jpeg_quality,
        )
        for img_id in needed
    }

    total = sum(len(v) for v in requests_to_make.values())
    print(
        f"  Building batch JSONL(s): {total} requests across {len(requests_to_make)} participants"
    )

    sub_batches: list[tuple[Path, list[int]]] = []
    file_index = 1
    current_ids: list[int] = []
    current_bytes = 0
    current_count = 0
    current_path: Optional[Path] = None
    current_fh = None

    def _open_new():
        nonlocal file_index, current_path, current_fh
        current_path = tmp / f"batch_input_{timestamp}_{file_index:02d}.jsonl"
        file_index += 1
        current_fh = open(current_path, "w", encoding="utf-8")

    _open_new()

    for pid, img_list in requests_to_make.items():
        pid_lines = [
            json.dumps(
                build_single_request(
                    pid,
                    img_id,
                    encoded[img_id],
                    prompt,
                    response_format,
                    model,
                    temperature,
                )
            )
            + "\n"
            for img_id in img_list
        ]
        pid_bytes = sum(len(ln.encode()) for ln in pid_lines)

        if current_ids and (
            (current_bytes + pid_bytes) > MAX_BATCH_BYTES
            or (current_count + len(pid_lines)) > MAX_BATCH_REQUESTS
        ):
            current_fh.close()
            sub_batches.append((current_path, list(current_ids)))
            current_ids = []
            current_bytes = 0
            current_count = 0
            _open_new()

        for ln in pid_lines:
            current_fh.write(ln)
        current_ids.append(pid)
        current_bytes += pid_bytes
        current_count += len(pid_lines)

    current_fh.close()
    sub_batches.append((current_path, list(current_ids)))

    print(f"  {len(sub_batches)} sub-batch file(s) prepared")
    return sub_batches


# ============================================================================
# LOGPROB EXTRACTION FOR BATCH RESPONSES
# ============================================================================


def _extract_batch_logprobs(
    choices: list[dict],
    response_options: list[str],
) -> dict[str, float]:
    """
    Extract binary logprobs from a chat.completions choices block.

    ``choices[0].logprobs.content`` is a list of token-level dicts.
    We scan for the response option token and renormalise.
    """
    try:
        # `logprobs` is present but null when none were returned, so it cannot
        # be assumed to be a mapping.
        logprobs = choices[0].get("logprobs") or {}
        logprobs_content = logprobs.get("content") or []
        target = set(response_options)

        choice_entry = next(
            entry for entry in logprobs_content if entry.get("token") in target
        )

        raw = {
            lp["token"]: math.exp(lp["logprob"])
            for lp in choice_entry.get("top_logprobs", [])
        }
        probs = {opt: raw.get(opt, 1e-10) for opt in response_options}
        total = sum(probs.values())
        if total <= 0:
            return {opt: 0.5 for opt in response_options}
        return {opt: round(p / total, 6) for opt, p in probs.items()}

    except (AttributeError, StopIteration, KeyError, TypeError, ValueError):
        return {opt: 0.5 for opt in response_options}


# ============================================================================
# RESPONSE PARSING
# ============================================================================


def recover_answer(content: str, response_options: list[str]) -> Optional[str]:
    """
    Recover the chosen option from content that is not valid JSON.

    A response truncated by the token limit looks like '{"response":"Le' - the
    model has already committed to an option but the closing quote and brace
    never arrived. The option is recovered only when exactly one of the
    permitted options matches what is there, so an ambiguous fragment is
    discarded rather than guessed.

    Args:
        content: The raw message content.
        response_options: The permitted answers for this illusion.

    Returns:
        The matching option, or None if zero or several match.
    """
    if not content:
        return None
    # The fragment after the last quote is what the model was part-way through.
    fragment = content.split('"')[-1].strip().lower()
    if not fragment:
        return None
    hits = [opt for opt in response_options if opt.lower().startswith(fragment)]
    if len(hits) == 1:
        return hits[0]
    # Fall back to a whole word appearing anywhere in the content.
    hits = [
        opt
        for opt in response_options
        if re.search(rf"\b{re.escape(opt)}\b", content, re.IGNORECASE)
    ]
    return hits[0] if len(hits) == 1 else None


def parse_batch_response(
    line: str,
    response_options: list[str],
    failures: Optional[Counter] = None,
) -> Optional[dict]:
    """
    Parse one line of batch output JSONL into a participant record.

    Never raises: a single malformed response must not abandon the rest of a
    downloaded batch. Returns None when the record cannot be used, and tallies
    the reason in `failures` when one is supplied.
    """

    def note(reason: str) -> None:
        if failures is not None:
            failures[reason] += 1
        return None

    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return note("output line not readable as JSON")

    if obj.get("error") is not None:
        return note("request returned an error")

    response_body = obj.get("response", {}).get("body", {})
    if response_body.get("error"):
        return note("response body carried an error")

    choices = response_body.get("choices", [])
    if not choices:
        return note("no choices in response")

    finish_reason = choices[0].get("finish_reason", "unknown")
    content = choices[0].get("message", {}).get("content")
    if not content or not content.strip():
        return note(f"empty content (finish_reason={finish_reason})")

    try:
        answer = json.loads(content)["response"]
    except (json.JSONDecodeError, KeyError, TypeError):
        answer = recover_answer(content, response_options)
        if answer is None:
            return note(f"content not parsable (finish_reason={finish_reason})")
        if failures is not None:
            failures[
                f"recovered from unparsable content (finish_reason={finish_reason})"
            ] += 1

    if answer not in response_options:
        return note(f"answer {answer!r} is not a permitted option")

    try:
        participant_id, image_id = parse_custom_id(obj["custom_id"])
        _, strength, true_diff = parse_filename(image_id)
    except (KeyError, ValueError, TypeError) as exc:
        return note(f"could not identify the stimulus ({type(exc).__name__})")

    parsed = {"response": answer}
    correct = compute_correct(parsed["response"], true_diff, response_options)

    probs = _extract_batch_logprobs(choices, response_options)
    opt_a, opt_b = response_options

    return {
        "participant_id": participant_id,
        "image_id": image_id,
        "illusion_strength": strength,
        "true_diff": true_diff,
        "response": parsed["response"],
        "correct": correct,
        f"logprob_{opt_a}": probs[opt_a],
        f"logprob_{opt_b}": probs[opt_b],
    }


# ============================================================================
# SUBCOMMAND FUNCTIONS
# ============================================================================


def cmd_submit(illusion: dict, args) -> None:
    name = illusion["name"]
    image_dir = Path(args.image_dir) / name
    out_dir = participants_dir(name)
    s_path = state_path(name)

    try:
        all_images = discover_images(
            image_dir,
            name,
            strengths=illusion.get("strengths"),
            differences=illusion.get("differences"),
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    requests_to_make = get_missing_requests(name, args.n_participants, all_images)
    total = sum(len(v) for v in requests_to_make.values())

    print("\n" + "=" * 60)
    print(f"BATCH SUBMISSION — {name}")
    print("=" * 60)
    print(f"  Target participants  : {args.n_participants}")
    print(f"  Need updating        : {len(requests_to_make)}")
    print(f"  Missing requests     : {total}")
    print(f"  Model                : {args.model}")
    print(f"  Temperature          : {args.temperature}")
    print("=" * 60)

    if not requests_to_make:
        print("\n[OK] All participants already complete.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would submit the above. Exiting.")
        return

    sub_batch_files = prepare_batch_files(
        illusion=illusion,
        requests_to_make=requests_to_make,
        image_dir=image_dir,
        model=args.model,
        temperature=args.temperature,
        max_dimension=args.max_dimension,
        jpeg_quality=args.jpeg_quality,
    )

    print()
    api_key = getattr(args, "api_key", None) or getpass.getpass(
        "Enter your OpenAI API key (input hidden): "
    )
    if not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    client = OpenAI(api_key=api_key.strip())

    submitted = []
    for i, (batch_path, pids) in enumerate(sub_batch_files, 1):
        print(f"\nSub-batch {i}/{len(sub_batch_files)}: {len(pids)} participants")
        with open(batch_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        print(f"  Uploaded: {uploaded.id}")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": (
                    f"VLM illusion study — {name} — "
                    f"participants {pids[0]}-{pids[-1]} "
                    f"(sub-batch {i}/{len(sub_batch_files)})"
                )
            },
        )
        print(f"  Submitted: {batch.id}  status={batch.status}")
        submitted.append(
            {
                "batch_id": batch.id,
                "input_file_id": uploaded.id,
                "participant_ids": pids,
                "status": batch.status,
                "output_file_id": None,
            }
        )

    state = {
        "illusion_name": name,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "n_participants_target": args.n_participants,
        "model": args.model,
        "batches": submitted,
        "status": "submitted",
    }
    save_state(s_path, state)
    print(f"\n  ✓ State saved to {s_path}")
    print("\nNext steps:")
    print(
        f"  Status   : python -m pipeline.module_2.batch_vlm --illusion {name} status"
    )
    print(
        f"  Download : python -m pipeline.module_2.batch_vlm --illusion {name} download"
    )


def cmd_status(illusion: dict, args) -> None:
    name = illusion["name"]
    s_path = state_path(name)
    state = load_state(s_path)

    api_key = getattr(args, "api_key", None) or getpass.getpass(
        "Enter your OpenAI API key (input hidden): "
    )
    if not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    client = OpenAI(api_key=api_key.strip())

    batches = state.get("batches", [])
    print("\n" + "=" * 60)
    print(f"BATCH STATUS — {name}  ({len(batches)} sub-batch(es))")
    print("=" * 60)
    print(f"  Submitted at : {state['submitted_at']}")
    print(f"  Model        : {state.get('model', 'unknown')}")

    all_complete = True
    for i, b in enumerate(batches, 1):
        batch = client.batches.retrieve(b["batch_id"])
        b["status"] = batch.status
        if batch.output_file_id:
            b["output_file_id"] = batch.output_file_id
        if batch.status != "completed":
            all_complete = False

        print(f"\n  Sub-batch {i}/{len(batches)}:")
        print(f"    ID           : {batch.id}")
        print(f"    Status       : {batch.status}")
        print(f"    Participants : {b['participant_ids']}")
        if batch.request_counts:
            rc = batch.request_counts
            print(
                f"    Requests     : {rc.completed}/{rc.total} done, {rc.failed} failed"
            )
            if rc.failed or (batch.status in ("completed", "expired") and rc.completed < rc.total):
                n_lost = rc.total - rc.completed
                print(
                    f"    [!] PARTIAL   : {n_lost:,} request(s) did not return. "
                    f"Download keeps what arrived; re-run submit for the rest."
                )
        if batch.output_file_id:
            print(f"    Output file  : {batch.output_file_id}")
        if getattr(batch, "error_file_id", None):
            print(f"    Error file   : {batch.error_file_id}")

    print("\n" + "=" * 60)
    if all_complete:
        print(f"\n[OK] All sub-batches complete — run download.")
    else:
        n_done = sum(1 for b in batches if b["status"] == "completed")
        print(f"\n  {n_done}/{len(batches)} complete. Check again later.")

    save_state(s_path, state)


def fetch_error_file(client, batch, cache_dir: Path) -> Counter:
    """
    Download and tally a batch's per-request error file.

    A batch reaches a terminal status even when individual requests inside it
    fail or expire; those requests appear only in `error_file_id` and never in
    the output file. Without reading it, a partial batch is indistinguishable
    from a complete one and the missing trials are silently lost.

    Returns a Counter of error reasons (empty if there is no error file).
    """
    reasons: Counter = Counter()
    error_file_id = getattr(batch, "error_file_id", None)
    if not error_file_id:
        return reasons

    cache = cache_dir / f"errors_{batch.id}.jsonl"
    if cache.exists():
        raw = cache.read_text(encoding="utf-8")
    else:
        raw = client.files.content(error_file_id).text
        cache.write_text(raw, encoding="utf-8")

    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            reasons["unparseable error line"] += 1
            continue
        err = (rec.get("response") or {}).get("body", {}).get("error") or rec.get(
            "error"
        )
        if isinstance(err, dict):
            code = err.get("code") or err.get("type") or "unknown"
            message = (err.get("message") or "")[:120]
            reasons[f"{code}: {message}" if message else str(code)] += 1
        else:
            reasons["unknown error shape"] += 1

    return reasons


def cmd_download(illusion: dict, args) -> None:
    name = illusion["name"]
    s_path = state_path(name)
    state = load_state(s_path)
    out_dir = participants_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    response_options = illusion["response_options"]

    batches = state.get("batches", [])
    if not batches:
        print("Error: No batches in state. Re-run submit.")
        sys.exit(1)

    api_key = getattr(args, "api_key", None) or getpass.getpass(
        "Enter your OpenAI API key (input hidden): "
    )
    if not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    client = OpenAI(api_key=api_key.strip())

    records_by_participant: dict[int, list[dict]] = {}
    total_ok = total_fail = 0
    failures: Counter = Counter()

    skipped_batches = 0
    incomplete_batches = 0
    for i, b in enumerate(batches, 1):
        print(f"\nSub-batch {i}/{len(batches)}: {b['batch_id']}")

        # The batch object is retrieved even when the output is cached, because
        # it carries request_counts and error_file_id — the only way to tell a
        # partial batch from a complete one.
        batch = client.batches.retrieve(b["batch_id"])
        b["status"] = batch.status

        # 'expired' is terminal and still exposes whatever finished inside the
        # 24h window, so its partial output is worth keeping.
        if batch.status not in ("completed", "expired"):
            print(f"  [SKIP] Status is '{batch.status}' — not yet terminal.")
            skipped_batches += 1
            continue

        rc = batch.request_counts
        if rc:
            print(
                f"  Status '{batch.status}': "
                f"{rc.completed}/{rc.total} completed, {rc.failed} failed"
            )
            if rc.failed or rc.completed < rc.total:
                incomplete_batches += 1

        err_reasons = fetch_error_file(client, batch, state_path(name).parent)
        if err_reasons:
            print(f"  Error file: {sum(err_reasons.values()):,} failed request(s)")
            for reason, n in err_reasons.most_common(5):
                print(f"    {n:>7,}  {reason}")

        # A terminal batch's output is cached on disk before anything is
        # parsed, so a malformed response cannot cost a second download of a
        # file this size.
        cache = state_path(name).parent / f"output_{b['batch_id']}.jsonl"
        if cache.exists():
            raw = cache.read_text(encoding="utf-8")
            print(f"  Using cached output ({cache.name})")
        elif batch.output_file_id:
            raw = client.files.content(batch.output_file_id).text
            cache.write_text(raw, encoding="utf-8")
            print(f"  Downloaded and cached to {cache.name}")
        else:
            print("  [WARN] No output file — every request in this sub-batch failed.")
            raw = ""

        lines = [ln for ln in raw.splitlines() if ln.strip()]
        print(f"  {len(lines)} response lines received")

        n_ok = n_fail = 0
        for line in lines:
            try:
                record = parse_batch_response(line, response_options, failures)
            except Exception as exc:  # one bad line must not end the download
                failures[f"unexpected {type(exc).__name__} while parsing"] += 1
                record = None
            if record is None:
                n_fail += 1
                continue
            records_by_participant.setdefault(record["participant_id"], []).append(
                record
            )
            n_ok += 1
        print(f"  Parsed: {n_ok} ok, {n_fail} failed")
        total_ok += n_ok
        total_fail += n_fail

    # Write / merge participant files
    print(f"\nWriting participant files to {out_dir}/")
    for pid in sorted(records_by_participant):
        out_path = out_dir / f"participant_{pid:02d}.jsonl"

        existing: dict[str, dict] = {}
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            existing[rec["image_id"]] = rec
                        except Exception:
                            pass

        for rec in records_by_participant[pid]:
            existing[rec["image_id"]] = rec

        all_recs = sorted(
            existing.values(),
            key=lambda r: (r["illusion_strength"], r["true_diff"]),
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in all_recs:
                f.write(json.dumps(rec) + "\n")

        n_new = len(records_by_participant[pid])
        print(f"  participant_{pid:02d}.jsonl  ({n_new} new → {len(all_recs)} total)")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Participants written : {len(records_by_participant)}")
    print(f"  Total successful     : {total_ok}")
    print(f"  Total failed         : {total_fail}")
    if failures:
        print("\n  Breakdown:")
        for reason, n in failures.most_common():
            print(f"    {n:>7,}  {reason}")

    # Coverage against the target grid, so a shortfall is stated outright rather
    # than left to be discovered later in analysis.
    target = state.get("n_participants_target")
    if target:
        try:
            image_dir = Path(getattr(args, "image_dir", "stimuli")) / name
            expected = len(
                discover_images(
                    image_dir,
                    name,
                    strengths=illusion.get("strengths"),
                    differences=illusion.get("differences"),
                )
            )
        except FileNotFoundError:
            expected = None

        if expected:
            short = []
            for pid in range(1, target + 1):
                p = out_dir / f"participant_{pid:02d}.jsonl"
                n = sum(1 for ln in p.open(encoding="utf-8") if ln.strip()) if p.exists() else 0
                if n < expected:
                    short.append((pid, n))
            print(f"\n  Coverage : {target - len(short)}/{target} participants complete "
                  f"({expected} trials each)")
            if short:
                missing_trials = sum(expected - n for _, n in short)
                absent = [pid for pid, n in short if n == 0]
                print(f"  [!] {len(short)} participant(s) incomplete — "
                      f"{missing_trials:,} trial(s) missing")
                if absent:
                    print(f"      no data at all for participants: {_fmt_ranges(absent)}")
                print("      Re-run 'submit' to queue exactly these, then 'download'.")
    print("=" * 60)

    # Only clean up _batch_tmp/ when every sub-batch reached a terminal status
    # AND every request inside them came back. Deleting the state file and the
    # cached output/error files is what makes a partial batch unrecoverable and
    # undiagnosable after the fact, so it is gated on the request counts rather
    # than on batch status alone.
    if skipped_batches == 0 and incomplete_batches == 0:
        cleanup_batch_tmp(name)
    else:
        if skipped_batches:
            print(
                f"\n  [!] {skipped_batches} sub-batch(es) not yet terminal — state file preserved."
            )
            print("  Run '--batch download' again once all batches have finished.")
        if incomplete_batches:
            print(
                f"\n  [!] {incomplete_batches} sub-batch(es) returned fewer responses than requested."
            )
            print(
                "  The missing trials were NOT retrieved. Re-run 'submit' to queue only"
            )
            print(
                "  what is still missing (it diffs against participants/ before sending),"
            )
            print("  then 'download' again. _batch_tmp/ has been preserved for audit.")


# ============================================================================
# STANDALONE CLI
# ============================================================================


def _get_illusion(name: str) -> dict:
    matches = [ill for ill in ILLUSIONS if ill["name"] == name]
    if not matches:
        names = [ill["name"] for ill in ILLUSIONS]
        print(f"Error: Unknown illusion '{name}'. Available: {', '.join(names)}")
        sys.exit(1)
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API wrapper for VLM illusion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--illusion",
        required=True,
        metavar="NAME",
        help="Illusion name (must match an entry in config.ILLUSIONS)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)

    # submit
    p_sub = subparsers.add_parser("submit", parents=[shared])
    p_sub.add_argument("--n-participants", type=int, required=True)
    p_sub.add_argument("--image-dir", type=str, default="./stimuli")
    p_sub.add_argument("--model", type=str, default=None)
    p_sub.add_argument("--temperature", type=float, default=TEMPERATURE)
    p_sub.add_argument("--max-dimension", type=int, default=MAX_DIMENSIONS)
    p_sub.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    p_sub.add_argument("--dry-run", action="store_true")

    # status / download
    subparsers.add_parser("status", parents=[shared])
    subparsers.add_parser("download", parents=[shared])

    args = parser.parse_args()

    # Resolve model default here so it can reference the illusion
    from config import MODEL

    if hasattr(args, "model") and args.model is None:
        args.model = MODEL

    illusion = _get_illusion(args.illusion)

    if args.command == "submit":
        cmd_submit(illusion, args)
    elif args.command == "status":
        cmd_status(illusion, args)
    elif args.command == "download":
        cmd_download(illusion, args)


if __name__ == "__main__":
    main()
