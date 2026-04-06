#!/usr/bin/env python3
"""
run_pipeline.py - Top-level pipeline runner.

Runs Module 1 → Module 2 → Module 3 across all illusions registered in
config.py, or a filtered subset via --illusion and --modules.

REAL-TIME QUERYING (default):
    python run_pipeline.py                         # full pipeline, all illusions
    python run_pipeline.py --illusion MullerLyer   # single illusion
    python run_pipeline.py --modules 1             # stimulus generation only
    python run_pipeline.py --modules 2 3           # query + analyse, skip generation
    python run_pipeline.py --modules 3 4           # any subset of modules
    python run_pipeline.py --force                 # regenerate existing stimuli (M1)
    python run_pipeline.py --dry-run               # print plan, no API calls (M2)

BATCH API (~50% cheaper, results in up to 24h):
    python run_pipeline.py --batch submit          # M1 + submit jobs to OpenAI
    python run_pipeline.py --batch status          # check job progress
    python run_pipeline.py --batch download        # download results + run M3

    Add --illusion NAME to any batch command to restrict to one illusion.
    Add --modules to batch submit to control whether M1 runs (e.g. --modules 2).
    Add --dry-run to 'batch submit' to preview without API calls.
"""

import argparse
import sys
import types

from config import (
    ILLUSIONS,
    JPEG_QUALITY,
    MAX_DIMENSIONS,
    MODEL,
    N_PARTICIPANTS,
    TEMPERATURE,
)

ALL_MODULES = [1, 2, 3]


def main():
    parser = argparse.ArgumentParser(
        description="VLM Illusion Pipeline — generates stimuli, queries VLM, fits PSEs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--illusion",
        type=str,
        default=None,
        metavar="NAME",
        help="Run pipeline for a single illusion only (e.g. MullerLyer)",
    )
    parser.add_argument(
        "--modules",
        type=int,
        nargs="+",
        default=ALL_MODULES,
        metavar="N",
        help="Which modules to run, e.g. --modules 1 3 (default: all)",
    )
    parser.add_argument(
        "--batch",
        type=str,
        choices=["submit", "status", "download"],
        default=None,
        metavar="PHASE",
        help="Use OpenAI Batch API instead of real-time querying. "
        "PHASE must be one of: submit, status, download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of existing stimuli in Module 1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the querying plan without making any API calls",
    )
    args = parser.parse_args()

    # ── Validate modules ──────────────────────────────────────────────────────
    invalid = [m for m in args.modules if m not in ALL_MODULES]
    if invalid:
        print(f"Error: Unknown module(s): {invalid}. Valid modules: {ALL_MODULES}")
        sys.exit(1)

    modules = set(args.modules)

    # ── Illusion selection ────────────────────────────────────────────────────
    illusions = ILLUSIONS
    if args.illusion:
        illusions = [ill for ill in ILLUSIONS if ill["name"] == args.illusion]
        if not illusions:
            names = [ill["name"] for ill in ILLUSIONS]
            print(f"Error: Unknown illusion '{args.illusion}'.")
            print(f"Available: {', '.join(names)}")
            sys.exit(1)

    # ── BATCH MODE ────────────────────────────────────────────────────────────
    if args.batch is not None:
        _run_batch(illusions, args, modules)
        return

    # ── REAL-TIME MODE ────────────────────────────────────────────────────────
    from pipeline.module_1_generate import run as generate
    from pipeline.module_2_query import run as query
    from pipeline.module_3_analyse import run as analyse

    active = sorted(modules)
    print("=" * 60)
    print("VLM ILLUSION PIPELINE  (real-time)")
    print("=" * 60)
    print(f"  Illusions : {[ill['name'] for ill in illusions]}")
    print(f"  Modules   : {[f'M{m}' for m in active]}")
    print("=" * 60 + "\n")

    if 1 in modules:
        print("\n" + "━" * 60)
        print("MODULE 1 — STIMULUS GENERATION")
        print("━" * 60)
        generate(illusions, force=args.force)

    if 2 in modules:
        print("\n" + "━" * 60)
        print("MODULE 2 — VLM QUERYING (real-time)")
        print("━" * 60)
        query(illusions, dry_run=args.dry_run)

    if 3 in modules:
        print("\n" + "━" * 60)
        print("MODULE 3 — PSYCHOMETRIC ANALYSIS")
        print("━" * 60)
        analyse(illusions)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


# ============================================================================
# BATCH MODE DISPATCHER
# ============================================================================


def _run_batch(illusions: list[dict], args, modules: set[int]) -> None:
    """
    Route --batch submit/status/download to batch_vlm.py functions.

    Builds a synthetic args namespace matching what batch_vlm's cmd_* functions
    expect, so no changes are needed in batch_vlm.py.
    """
    from pipeline.module_1_generate import run as generate
    from pipeline.module_2.batch_vlm import cmd_download, cmd_status, cmd_submit
    from pipeline.module_3_analyse import run as analyse

    phase = args.batch

    print("=" * 60)
    print(f"VLM ILLUSION PIPELINE  (batch / {phase})")
    print("=" * 60)
    print(f"  Illusions : {[ill['name'] for ill in illusions]}")
    print("=" * 60 + "\n")

    if phase == "submit":
        if 1 in modules:
            print("\n" + "━" * 60)
            print("MODULE 1 — STIMULUS GENERATION")
            print("━" * 60)
            generate(illusions, force=args.force)

        batch_args = types.SimpleNamespace(
            n_participants=N_PARTICIPANTS,
            image_dir="./stimuli",
            model=MODEL,
            temperature=TEMPERATURE,
            max_dimension=MAX_DIMENSIONS,
            jpeg_quality=JPEG_QUALITY,
            dry_run=args.dry_run,
        )

        print("\n" + "━" * 60)
        print("MODULE 2 — BATCH SUBMIT")
        print("━" * 60)
        for illusion in illusions:
            print(f"\n  Submitting batch for: {illusion['name']}")
            cmd_submit(illusion, batch_args)

        print("\n" + "=" * 60)
        print("BATCH SUBMIT COMPLETE")
        print("=" * 60)
        ill_flag = f" --illusion {illusions[0]['name']}" if len(illusions) == 1 else ""
        print(f"\nCheck progress  : python run_pipeline.py --batch status{ill_flag}")
        print(f"Download results: python run_pipeline.py --batch download{ill_flag}")

    elif phase == "status":
        batch_args = types.SimpleNamespace()
        for illusion in illusions:
            print(f"\n{'━' * 60}")
            cmd_status(illusion, batch_args)

    elif phase == "download":
        batch_args = types.SimpleNamespace()
        for illusion in illusions:
            print(f"\n{'━' * 60}")
            print(f"  Downloading: {illusion['name']}")
            print(f"{'━' * 60}")
            cmd_download(illusion, batch_args)

        print("\n" + "━" * 60)
        print("MODULE 3 — PSYCHOMETRIC ANALYSIS")
        print("━" * 60)
        analyse(illusions)

        print("\n" + "=" * 60)
        print("BATCH DOWNLOAD + ANALYSIS COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
