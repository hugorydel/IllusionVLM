"""
tests/generate_example_stimuli.py - Generate example MullerLyer stimuli for inspection.

Produces three pairs of stimuli (congruent / incongruent fin direction) at:
    - Strong illusion  (illusion_strength = +49)
    - Mild illusion    (illusion_strength = +21)
    - No illusion      (illusion_strength =   0)

All stimuli use difference = 0 (lines are physically equal in length).
The positive/negative strength pair at each level lets you verify that the
fin direction correctly flips the perceptual bias.

Output layout:
    tests/example_stimuli/
        MullerLyer_example_strong_pos.png   (str=+49, diff=0)
        MullerLyer_example_strong_neg.png   (str=-49, diff=0)
        MullerLyer_example_mild_pos.png     (str=+21, diff=0)
        MullerLyer_example_mild_neg.png     (str=-21, diff=0)
        MullerLyer_example_none.png         (str=  0, diff=0)

Usage:
    python tests/generate_example_stimuli.py
    python tests/generate_example_stimuli.py --out tests/example_stimuli
"""

import argparse

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pyllusion

# ============================================================================
# STIMULUS DEFINITIONS
# ============================================================================

STIMULI = [
    # (label,            illusion_strength, difference)
    ("strong_pos", +49, 0),
    ("strong_neg", -49, 0),
    ("mild_pos", +21, 0),
    ("mild_neg", -21, 0),
    ("none", 0, 0),
]


# ============================================================================
# GENERATION
# ============================================================================


def generate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating MullerLyer example stimuli → {out_dir}/\n")
    print(f"  {'Label':<16} {'Strength':>9} {'Diff':>6}  Path")
    print(f"  {'─'*60}")

    for label, strength, diff in STIMULI:
        filename = f"MullerLyer_example_{label}.png"
        out_path = out_dir / filename

        illusion = pyllusion.MullerLyer(illusion_strength=strength, difference=diff)
        illusion.to_image().save(out_path)

        print(f"  {label:<16} {strength:>+9}  {diff:>+5.2f}  {out_path}")

    print(f"\n✓ {len(STIMULI)} stimuli written to {out_dir}/")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MullerLyer example stimuli at strong/mild/no illusion strength.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/example_stimuli"),
        help="Output directory for PNGs (default: tests/example_stimuli)",
    )
    args = parser.parse_args()
    generate(args.out)
