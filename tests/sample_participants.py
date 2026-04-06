"""
tests/sample_participants.py - Sample a small subset of participant data for testing.

For each illusion that has participant files, picks 2 at random and copies
them to tests/sample_data/ with the illusion name prepended to the filename.

Usage:
    python tests/sample_participants.py
    python tests/sample_participants.py --n 3          # pick 3 per illusion
    python tests/sample_participants.py --seed 42      # reproducible draw
    python tests/sample_participants.py --out tests/my_sample  # custom output dir
"""

import argparse
import random
import shutil
from pathlib import Path

RESULTS_ROOT = Path("results")
DEFAULT_OUT = Path("tests/sample_data")
DEFAULT_N = 2


def sample_illusion(
    illusion_name: str,
    n: int,
    out_dir: Path,
    rng: random.Random,
) -> int:
    """
    Copy n randomly chosen participant files for one illusion into out_dir,
    prefixed with the illusion name.

    Returns the number of files actually copied.
    """
    participants_dir = RESULTS_ROOT / illusion_name / "participants"
    if not participants_dir.exists():
        print(f"  [{illusion_name}] No participants dir — skipping.")
        return 0

    available = sorted(participants_dir.glob("participant_*.jsonl"))
    if not available:
        print(f"  [{illusion_name}] No participant files found — skipping.")
        return 0

    if len(available) < n:
        print(
            f"  [{illusion_name}] Only {len(available)} file(s) available "
            f"(requested {n}) — copying all."
        )
        chosen = available
    else:
        chosen = rng.sample(available, n)

    copied = 0
    for src in chosen:
        dest = out_dir / f"{illusion_name.lower()}_{src.name}"
        shutil.copy2(src, dest)
        print(f"  [{illusion_name}] {src.name} → {dest.name}")
        copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a random sample of participant files to tests/sample_data/",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        metavar="N",
        help=f"Participants to sample per illusion (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for reproducible sampling (default: random)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        metavar="DIR",
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # Discover illusions from results/ — any subdirectory with a participants/ folder.
    illusion_dirs = sorted(
        d
        for d in RESULTS_ROOT.iterdir()
        if d.is_dir() and (d / "participants").exists()
    )

    if not illusion_dirs:
        print(f"No illusion results found under {RESULTS_ROOT}/")
        return

    print(f"Sampling {args.n} participant(s) per illusion → {args.out}/\n")

    total = 0
    for illusion_dir in illusion_dirs:
        total += sample_illusion(illusion_dir.name, args.n, args.out, rng)

    print(f"\n✓ {total} file(s) copied to {args.out}/")


if __name__ == "__main__":
    main()
