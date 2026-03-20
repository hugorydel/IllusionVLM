"""
config.py - Pipeline configuration: model parameters and illusion registry.

Every illusion entry contains everything the pipeline needs to know about it:
    name             : used for folder naming, display, and filename prefixes
    pyllusion_class  : string looked up on the pyllusion module via getattr()
    strengths        : list of illusion strength values (floats, signed)
    differences      : list of physical difference values (floats, signed)
    response_options : [positive_option, negative_option]
                       response_options[0] → correct when true_diff > 0
                       response_options[1] → correct when true_diff < 0
    prompt           : full forced-choice question string for this illusion

To add a new illusion, append a dict to ILLUSIONS following this template.

─────────────────────────────────────────────────────────────────────────────
PARAMETER GRIDS — SOURCE AND RATIONALE
─────────────────────────────────────────────────────────────────────────────
Strength and difference values are derived from the per-illusion grids used
in Makowski et al. (2023) IllusionGameValidation study2, recovered from the
stimulus manifests at:
    https://github.com/RealityBending/IllusionGameValidation

Each illusion has an illusion-specific `strength_unit` and a set of 8
positive difference values. The full grid is:

    strengths   = [unit × k  for k in range(-7, 8)]    → 15 levels
    differences = [-pos_diffs reversed] + [pos_diffs]  → 16 levels
                                                        → 240 stimuli total

The difference values are NOT interchangeable across illusions because
`difference` has a different geometric meaning per illusion in Pyllusion
(e.g. multiplicative line-length ratio for MullerLyer/Ponzo, square-root
size transform for Delboeuf/Ebbinghaus, angular displacement for Zollner,
brightness percentage for Contrast/White, etc.).

─────────────────────────────────────────────────────────────────────────────
SIGN-CONVENTION NOTE (Zollner, Poggendorff, White, Contrast)
─────────────────────────────────────────────────────────────────────────────
Pyllusion's later changelog records that the sign of `illusion_strength`
was reversed for Zollner, Poggendorff, White, and Contrast to fix an
unintended congruent/incongruent labelling error identified during the
IllusionGameValidation work.

For our purposes this is a non-issue for stimulus generation — latest
Pyllusion renders geometrically correct images. The only consequence is
that when *comparing* our VLM PSE curves against Makowski's human data
for those four illusions, the sign of `illusion_strength` on one side of
the comparison must be flipped (x-axis mirror). This is handled at
comparison/plotting time; no changes to stimuli or querying are needed.
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL = "gpt-5.2"
TEMPERATURE = 0.3
REASONING_EFFORT = "none"  # "none"/"low"/"medium"/"high" for reasoning models; None for non-reasoning models (e.g. gpt-4o)
N_PARTICIPANTS = 30
MAX_TOKENS = 200  # Limit is 200 because some image titles are 50-100 tokens long; this limit accomodates them.
MAX_CONCURRENCY = 100
MAX_DIMENSIONS = 512
MAX_BATCH_BYTES: int = 190 * 1024 * 1024  # 190 MB — safely under OpenAI 200 MB limit
JPEG_QUALITY = 90

# ============================================================================
# GRID HELPERS
# ============================================================================


def _strengths(unit: float) -> list[float]:
    """Generate 15 signed strength levels from a per-illusion unit step."""
    return [round(unit * k, 5) for k in range(-7, 8)]


def _differences(positive: list[float]) -> list[float]:
    """Mirror 8 positive difference values to produce a symmetric 16-level grid."""
    return [-round(x, 5) for x in reversed(positive)] + [round(x, 5) for x in positive]


# ============================================================================
# ILLUSION REGISTRY
# ============================================================================

ILLUSIONS = [
    # {
    #     "name": "MullerLyer",
    #     "pyllusion_class": "MullerLyer",
    #     "strengths": _strengths(7.0),
    #     "differences": _differences(
    #         [0.04, 0.06565, 0.10044, 0.14575, 0.20297, 0.27349, 0.3587, 0.46]
    #     )
    #     + [0.57877, 0.71641, 0.87429, 1.05382],
    #     "response_options": ["Top", "Bottom"],
    #     "prompt": (
    #         "Look at the two red horizontal lines in this image.\n\n"
    #         "Which red line looks longer — the TOP one or the BOTTOM one?\n\n"
    #         'Answer with only "Top" or "Bottom".'
    #     ),
    # },
    # {
    #     "name": "Ebbinghaus",
    #     "pyllusion_class": "Ebbinghaus",
    #     "strengths": _strengths(0.29),
    #     "differences": _differences(
    #         [0.07, 0.11066, 0.16462, 0.23378, 0.32001, 0.4252, 0.55124, 0.7]
    #     ),
    #     "response_options": ["Left", "Right"],
    #     "prompt": (
    #         "Look at the two red circles in the centre of each group in this image.\n\n"
    #         "Which central red circle looks bigger — the LEFT one or the RIGHT one?\n\n"
    #         'Answer with only "Left" or "Right".'
    #     ),
    # },
    # {
    #     "name": "Ponzo",
    #     "pyllusion_class": "Ponzo",
    #     "strengths": _strengths(3.6),
    #     "differences": _differences(
    #         [
    #             0.04,
    #             0.052,
    #             0.06565,
    #             0.082,
    #             0.10044,
    #             0.123,
    #             0.14575,
    #             0.173,
    #             0.20297,
    #             0.238,
    #             0.27349,
    #             0.315,
    #             0.3587,
    #             0.41,
    #             0.46,
    #         ]
    #     )
    #     + [0.56, 0.67],
    #     "response_options": ["Top", "Bottom"],
    #     "prompt": (
    #         "Look at the two red horizontal lines in this image.\n\n"
    #         "Which red line looks longer — the TOP one or the BOTTOM one?\n\n"
    #         'Answer with only "Top" or "Bottom".'
    #     ),
    # },
    # {
    #     "name": "VerticalHorizontal",
    #     "pyllusion_class": "VerticalHorizontal",
    #     "strengths": _strengths(9.5),
    #     "differences": _differences(
    #         [0.03, 0.04772, 0.06953, 0.09544, 0.12544, 0.15953, 0.19772, 0.24]
    #     )
    #     + [0.32, 0.42, 0.54, 0.70],
    #     "response_options": ["Left", "Right"],
    #     "prompt": (
    #         "Look at the two red lines in this image.\n\n"
    #         "Which red line looks longer — the LEFT one or the RIGHT one?\n\n"
    #         'Answer with only "Left" or "Right".'
    #     ),
    # },
    # {
    #     "name": "Zollner",
    #     "pyllusion_class": "Zollner",
    #     "strengths": _strengths(11.0),
    #     "differences": _differences(
    #         [0.15, 0.32141, 0.58988, 0.97717, 1.50505, 2.19531, 3.0697, 4.15]
    #     ),
    #     # NOTE: illusion_strength sign convention was reversed in a later
    #     # Pyllusion patch. Flip the x-axis when comparing against Makowski
    #     # human data for this illusion.
    #     "response_options": ["Top", "Bottom"],
    #     "prompt": (
    #         "Look at the two long red diagonal lines in this image.\n\n"
    #         "Which long red line appears to tilt more to the right at the top — "
    #         "the TOP line or the BOTTOM line?\n\n"
    #         'Answer with only "Top" or "Bottom".'
    #     ),
    # },
    # {
    #     "name": "Poggendorff",
    #     "pyllusion_class": "Poggendorff",
    #     "strengths": _strengths(6.4),
    #     "differences": _differences(
    #         [0.02, 0.03538, 0.05713, 0.08636, 0.12415, 0.17162, 0.22987, 0.3]
    #     ),
    #     # NOTE: illusion_strength sign convention was reversed in a later
    #     # Pyllusion patch. Flip the x-axis when comparing against Makowski
    #     # human data for this illusion.
    #     "response_options": ["Left", "Right"],
    #     "prompt": (
    #         "Look at the two red line segments on either side of the vertical bar.\n\n"
    #         "Do the two red segments appear to be part of the same diagonal line, "
    #         "or does one appear higher than the other?\n\n"
    #         "Which segment appears higher — the LEFT one or the RIGHT one?\n\n"
    #         'Answer with only "Left" or "Right".'
    #     ),
    # },
    # {
    #     "name": "RodFrame",
    #     "pyllusion_class": "RodFrame",
    #     "strengths": _strengths(2.0),
    #     "differences": _differences(
    #         [0.06, 0.34882, 0.87661, 1.64336, 2.64907, 3.89375, 5.37739, 7.1]
    #     ),
    #     "response_options": ["Left", "Right"],
    #     "prompt": (
    #         "Look at the red rod (line) inside the tilted frame in this image.\n\n"
    #         "Is the red rod tilted to the LEFT or to the RIGHT of vertical?\n\n"
    #         'Answer with only "Left" or "Right".'
    #     ),
    # },
    # {
    #     "name": "Delboeuf",
    #     "pyllusion_class": "Delboeuf",
    #     "strengths": _strengths(0.31),
    #     "differences": _differences(
    #         [0.07, 0.11066, 0.16462, 0.23378, 0.32001, 0.4252, 0.55124, 0.7]
    #     ),
    #     "response_options": ["Left", "Right"],
    #     "prompt": (
    #         "Look at the two red circles in this image.\n\n"
    #         "Which red circle looks bigger — the LEFT one or the RIGHT one?\n\n"
    #         'Answer with only "Left" or "Right".'
    #     ),
    # },
    # {
    #     "name": "Contrast",
    #     "pyllusion_class": "Contrast",
    #     "strengths": _strengths(4.5),
    #     "differences": _differences(
    #         [3.0, 4.33568, 5.91661, 7.74279, 9.81421, 12.13089, 14.69282, 17.5]
    #     ),
    #     # NOTE: illusion_strength sign convention was reversed in a later
    #     # Pyllusion patch. Flip the x-axis when comparing against Makowski
    #     # human data for this illusion.
    #     "response_options": ["Top", "Bottom"],
    #     "prompt": (
    #         "Look at the two red squares in this image.\n\n"
    #         "Which red square looks lighter (brighter) — the TOP one or the BOTTOM one?\n\n"
    #         'Answer with only "Top" or "Bottom".'
    #     ),
    # },
    # {
    #     "name": "White",
    #     "pyllusion_class": "White",
    #     "strengths": _strengths(2.5),
    #     "differences": _differences(
    #         [3.0, 4.33568, 5.91661, 7.74279, 9.81421, 12.13089, 14.69282, 17.5]
    #     ),
    #     # NOTE: illusion_strength sign convention was reversed in a later
    #     # Pyllusion patch. Flip the x-axis when comparing against Makowski
    #     # human data for this illusion.
    #     "response_options": ["Top", "Bottom"],
    #     "prompt": (
    #         "Look at the two grey rectangles in this image.\n\n"
    #         "Which grey rectangle looks lighter (brighter) — the TOP one or the BOTTOM one?\n\n"
    #         'Answer with only "Top" or "Bottom".'
    #     ),
    # },
]
