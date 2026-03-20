"""
pipeline/module_1_generate.py - Module 1: Stimulus generation.

Iterates the illusion registry, generates a strength × difference grid of
PNG stimuli for each illusion using pyllusion, and saves them to
stimuli/<illusion_name>/.

Skip logic: any image that already exists is not regenerated unless
force=True is passed.

Parallelism: uses multiprocessing.Pool so all images within an illusion
are generated concurrently across CPU cores.
"""

import matplotlib

matplotlib.use("Agg")  # must precede any other matplotlib/pyllusion import

from multiprocessing import Pool
from pathlib import Path

import pyllusion

from pipeline.utils import make_filename

STIMULI_ROOT = Path("stimuli")


# ============================================================================
# WORKER  (top-level so multiprocessing can pickle it)
# ============================================================================


def _generate_one(args: tuple) -> str:
    """
    Generate and save a single stimulus image.

    Args:
        args: (illusion_name, pyllusion_class, output_dir, strength, diff, force)

    Returns:
        Status string starting with '[done]' or '[skip]'.
    """
    illusion_name, pyllusion_class, output_dir, strength, diff, force = args
    filename = make_filename(illusion_name, strength, diff)
    out_path = Path(output_dir) / filename

    if out_path.exists() and not force:
        return f"[skip]  {filename}"

    cls = getattr(pyllusion, pyllusion_class)
    illusion = cls(illusion_strength=strength, difference=diff)
    illusion.to_image().save(out_path)
    return f"[done]  {filename}"


# ============================================================================
# PER-ILLUSION GENERATION
# ============================================================================


def generate_illusion(illusion: dict, force: bool = False) -> None:
    """
    Generate the full stimulus grid for one illusion.

    Args:
        illusion: Illusion config dict from config.py.
        force:    Regenerate images that already exist.
    """
    name = illusion["name"]
    strengths = illusion["strengths"]
    differences = illusion["differences"]
    total = len(strengths) * len(differences)

    output_dir = STIMULI_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {name}")
    print(f"  {'─' * 40}")
    print(f"  Strengths  : {len(strengths)} levels")
    print(f"  Differences: {len(differences)} levels")
    print(f"  Total      : {total} images")
    print(f"  Output     : {output_dir}/\n")

    task_args = [
        (name, illusion["pyllusion_class"], str(output_dir), strength, diff, force)
        for strength in strengths
        for diff in differences
    ]

    with Pool() as pool:
        results = pool.map(_generate_one, task_args)

    for line in results:
        print(f"    {line}")

    generated = sum(1 for r in results if r.startswith("[done]"))
    skipped = sum(1 for r in results if r.startswith("[skip]"))

    print(f"\n  ✓ {name}: {generated} generated, {skipped} skipped")


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================


def run(illusions: list[dict], force: bool = False) -> None:
    """
    Generate stimuli for all illusions in the registry.

    Args:
        illusions: List of illusion config dicts (from config.ILLUSIONS).
        force:     Regenerate images even if they already exist.
    """
    print(f"\nGenerating stimuli for {len(illusions)} illusion(s)...")

    for illusion in illusions:
        generate_illusion(illusion, force=force)

    print(f"\n✓ Module 1 complete — stimuli saved under {STIMULI_ROOT}/")
