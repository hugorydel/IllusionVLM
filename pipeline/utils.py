"""
pipeline/utils.py - Shared utilities used across all pipeline modules.

Includes:
  - make_filename / parse_filename  : stimulus filename encoding/decoding
  - preprocess_image                : load, resize, base64-encode a stimulus PNG
  - compute_correct                 : ground-truth correctness given response_options
"""

import base64
import io
import math
from pathlib import Path
from typing import Optional

from PIL import Image

# ============================================================================
# FILENAME ENCODING / DECODING
# ============================================================================


def make_filename(illusion_name: str, strength: float, diff: float) -> str:
    """
    Encode illusion parameters into a filename for downstream parsing.

    Sign and magnitude are handled separately so zero-padding is always
    applied to the magnitude only, giving consistent 3-digit fields:
        strength -49 → '-049'   strength +7 → '+007'

    Diff is stored at 5 decimal places to preserve the full precision of
    the IllusionGameValidation parameter set.

    Examples:
        make_filename('MullerLyer',  50,   0.2)    → 'MullerLyer_str+050_diff+0.20000.png'
        make_filename('Ebbinghaus', -49,  -0.3587) → 'Ebbinghaus_str-049_diff-0.35870.png'
    """
    s_sign = "-" if strength < 0 else "+"
    d_sign = "-" if diff < 0 else "+"
    abs_s = abs(strength)
    # Use integer formatting for whole-number strengths (e.g. MullerLyer: 7, 14, 49)
    # and float formatting for fractional strengths (e.g. Ebbinghaus: 0.29, 0.58;
    # VerticalHorizontal: 9.5, 28.5). Preserves existing filenames for integer illusions.
    strength_str = f"{int(abs_s):03d}" if abs_s == int(abs_s) else f"{abs_s:.5f}"
    return (
        f"{illusion_name}_str{s_sign}{strength_str}" f"_diff{d_sign}{abs(diff):.5f}.png"
    )


def parse_filename(stem: str) -> tuple[str, float, float]:
    """
    Reverse of make_filename — extract (illusion_name, strength, diff) from a stem.

    Strips the known 'str' / 'diff' prefixes by position.

    Examples:
        parse_filename('MullerLyer_str+050_diff+0.20000') → ('MullerLyer', 50.0,  0.2)
        parse_filename('Ebbinghaus_str-049_diff-0.35870') → ('Ebbinghaus', -49.0, -0.3587)
    """
    illusion_name, str_part, diff_part = stem.split("_")
    strength = float(str_part[3:])  # strip leading 'str'
    diff = float(diff_part[4:])  # strip leading 'diff'
    return illusion_name, strength, diff


def discover_images(
    image_dir: Path,
    illusion_name: str,
    strengths: list[float] | None = None,
    differences: list[float] | None = None,
) -> list[str]:
    """
    Discover stimulus PNGs for a given illusion, filtered to only the
    (strength, difference) pairs defined in the config.

    Without filters this would return everything on disk, including stimuli
    from prior runs with different grids — causing participants to be queried
    on images outside the current experiment design.

    Args:
        image_dir:    Directory containing stimulus PNGs.
        illusion_name: Illusion name prefix (e.g. 'MullerLyer').
        strengths:    Whitelist of illusion_strength values from config.
                      If None, all strengths on disk are accepted.
        differences:  Whitelist of true_diff values from config.
                      If None, all differences on disk are accepted.

    Returns:
        List of image ID stems matching the config grid, sorted by
        (illusion_strength, true_diff).

    Raises:
        FileNotFoundError: if image_dir doesn't exist or no matching files found.
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    pattern = f"{illusion_name}_str*_diff*.png"
    png_files = list(image_dir.glob(pattern))

    if not png_files:
        raise FileNotFoundError(
            f"No stimuli ({pattern}) found in {image_dir}. "
            "Run Module 1 (generate) first."
        )

    # Build tolerance sets for float comparison
    strength_set = {round(s, 5) for s in strengths} if strengths is not None else None
    diff_set = {round(d, 5) for d in differences} if differences is not None else None

    image_ids = []
    for f in png_files:
        _, s, d = parse_filename(f.stem)
        if strength_set is not None and round(s, 5) not in strength_set:
            continue
        if diff_set is not None and round(d, 5) not in diff_set:
            continue
        image_ids.append(f.stem)

    if not image_ids:
        raise FileNotFoundError(
            f"No stimuli matching the configured grid found in {image_dir}. "
            "Run Module 1 (generate) first."
        )

    image_ids.sort(key=lambda s: parse_filename(s)[1:])  # sort by (strength, diff)
    return image_ids


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================


def preprocess_image(
    image_path: Path,
    max_dimension: int = 512,
    jpeg_quality: int = 90,
) -> str:
    """
    Load, resize if needed, and encode image as base64 JPEG.

    Args:
        image_path:    Path to stimulus PNG.
        max_dimension: Max pixel dimension; image is downscaled if exceeded.
        jpeg_quality:  JPEG quality for encoding (1–100).

    Returns:
        Base64-encoded JPEG string.
    """
    img = Image.open(image_path)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    width, height = img.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        img = img.resize(
            (int(width * scale), int(height * scale)),
            Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ============================================================================
# GROUND-TRUTH CORRECTNESS
# ============================================================================


def compute_correct(
    response: str,
    true_diff: float,
    response_options: list[str],
) -> Optional[int]:
    """
    Determine whether the model's response matches physical ground truth.

    Convention:
        true_diff > 0 → response_options[0] is correct  (e.g. "Top" / "Left")
        true_diff < 0 → response_options[1] is correct  (e.g. "Bottom" / "Right")
        true_diff = 0 → no correct answer                (returns None)

    Args:
        response:         The model's response string.
        true_diff:        Signed physical difference (positive = options[0] is longer).
        response_options: [positive_option, negative_option] from illusion config.

    Returns:
        1 if correct, 0 if incorrect, None if true_diff == 0.
    """
    if true_diff > 0:
        return 1 if response == response_options[0] else 0
    elif true_diff < 0:
        return 1 if response == response_options[1] else 0
    else:
        return None


# ============================================================================
# LOGPROB EXTRACTION
# ============================================================================


def extract_binary_logprobs(
    logprobs_list: list,
    response_options: list[str],
) -> dict[str, float]:
    """
    Find the forced-choice response token in a logprobs list and return
    renormalised probabilities for both options.

    With JSON-schema-constrained output the choice token is embedded inside
    a JSON string (e.g. ``..."response": "Top"...``). We scan the token list
    for the first token that matches either response option and read its
    top_logprobs distribution.

    The raw probabilities are renormalised over just the two options to give
    a clean binary distribution that sums to 1.

    Args:
        logprobs_list:    List of token logprob objects from the Responses API
                          (``response.output[0].content[0].logprobs``).
        response_options: [option_a, option_b] from illusion config.

    Returns:
        {option_a: prob_a, option_b: prob_b}  (values sum to 1.0)
        Falls back to {option_a: 0.5, option_b: 0.5} if extraction fails.
    """
    target = set(response_options)

    try:
        choice_entry = next(entry for entry in logprobs_list if entry.token in target)

        raw = {lp.token: math.exp(lp.logprob) for lp in choice_entry.top_logprobs}

        # Keep only the two target tokens (top_logprobs may include others)
        probs = {opt: raw.get(opt, 1e-10) for opt in response_options}

        total = sum(probs.values())
        return {opt: round(p / total, 6) for opt, p in probs.items()}

    except (StopIteration, AttributeError, ValueError):
        # Graceful fallback — equal probability
        return {opt: 0.5 for opt in response_options}
