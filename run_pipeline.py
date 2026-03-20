#!/usr/bin/env python3
"""
run_pipeline.py - Top-level pipeline runner.

Runs Module 1 → Module 2 → Module 3 across all illusions registered in
config.py, or a filtered subset via --illusion.

Usage:
    python run_pipeline.py                        # full pipeline, all illusions
    python run_pipeline.py --illusion MullerLyer  # single illusion
    python run_pipeline.py --skip-generate        # skip stimulus generation
    python run_pipeline.py --skip-query           # skip VLM querying
    python run_pipeline.py --skip-analyse         # skip fitting + plotting
    python run_pipeline.py --force                # regenerate existing stimuli (M1)
    python run_pipeline.py --dry-run              # print plan, no API calls (M2)
"""

import argparse
import sys

from pipeline.module_1_generate import run as generate
from pipeline.module_2_query import run as query
from pipeline.module_3_analyse import run as analyse

from config import ILLUSIONS


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
        "--skip-generate",
        action="store_true",
        help="Skip Module 1 (stimulus generation)",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Skip Module 2 (VLM querying)",
    )
    parser.add_argument(
        "--skip-analyse",
        action="store_true",
        help="Skip Module 3 (psychometric fitting + plotting)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of existing stimuli in Module 1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the querying plan without making any API calls (Module 2)",
    )
    args = parser.parse_args()

    # ── Illusion selection ────────────────────────────────────────────────────
    illusions = ILLUSIONS
    if args.illusion:
        illusions = [ill for ill in ILLUSIONS if ill["name"] == args.illusion]
        if not illusions:
            names = [ill["name"] for ill in ILLUSIONS]
            print(f"Error: Unknown illusion '{args.illusion}'.")
            print(f"Available: {', '.join(names)}")
            sys.exit(1)

    print("=" * 60)
    print("VLM ILLUSION PIPELINE")
    print("=" * 60)
    print(f"  Illusions : {[ill['name'] for ill in illusions]}")
    print(
        f"  Modules   : "
        f"{'[M1] ' if not args.skip_generate else ''}"
        f"{'[M2] ' if not args.skip_query else ''}"
        f"{'[M3]' if not args.skip_analyse else ''}"
        or "  (none — all skipped)"
    )
    print("=" * 60 + "\n")

    # ── Module 1: Generate stimuli ────────────────────────────────────────────
    if not args.skip_generate:
        print("\n" + "━" * 60)
        print("MODULE 1 — STIMULUS GENERATION")
        print("━" * 60)
        generate(illusions, force=args.force)

    # ── Module 2: Query VLM ───────────────────────────────────────────────────
    if not args.skip_query:
        print("\n" + "━" * 60)
        print("MODULE 2 — VLM QUERYING")
        print("━" * 60)
        query(illusions, dry_run=args.dry_run)

    # ── Module 3: Fit + plot ──────────────────────────────────────────────────
    if not args.skip_analyse:
        print("\n" + "━" * 60)
        print("MODULE 3 — PSYCHOMETRIC ANALYSIS")
        print("━" * 60)
        analyse(illusions)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
