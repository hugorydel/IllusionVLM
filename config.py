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
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL = "gpt-5.2"
TEMPERATURE = 0.3
N_PARTICIPANTS = 100
MAX_TOKENS = 50
MAX_CONCURRENCY = 20
MAX_DIMENSIONS = 512
MAX_BATCH_BYTES: int = 190 * 1024 * 1024  # 190 MB — safely under OpenAI 200 MB limit
JPEG_QUALITY = 90

# ============================================================================
# ILLUSION REGISTRY
# ============================================================================

# Strength and difference values from RealityBending/IllusionGameValidation
# study2/stimuli/stimuli_part{1,2}.js — used across all illusions for
# direct comparability with the human reference dataset.

_STRENGTHS = [
    -49.0,
    -42.0,
    -35.0,
    -28.0,
    -21.0,
    -14.0,
    -7.0,
    0.0,
    7.0,
    14.0,
    21.0,
    28.0,
    35.0,
    42.0,
    49.0,
]

_DIFFERENCES = [
    -0.46,
    -0.3587,
    -0.27349,
    -0.20297,
    -0.14575,
    -0.10044,
    -0.06565,
    -0.04,
    0.04,
    0.06565,
    0.10044,
    0.14575,
    0.20297,
    0.27349,
    0.3587,
    0.46,
]

ILLUSIONS = [
    {
        "name": "MullerLyer",
        "pyllusion_class": "MullerLyer",
        "strengths": _STRENGTHS,
        "differences": _DIFFERENCES,
        "response_options": ["Top", "Bottom"],
        "prompt": (
            "Look at the two red horizontal lines in this image.\n\n"
            "Which red line looks longer — the TOP one or the BOTTOM one?\n\n"
            'Answer with only "Top" or "Bottom".'
        ),
    },
    # {
    #     "name": "Ebbinghaus",
    #     "pyllusion_class": "Ebbinghaus",
    #     "strengths": _STRENGTHS,
    #     "differences": _DIFFERENCES,
    #     "response_options": ["Left", "Right"],
    #     "prompt": (
    #         "Look at the two central circles in this image.\n\n"
    #         "Which central circle looks bigger — the LEFT one or the RIGHT one?\n\n"
    #         'Answer with only "Left" or "Right".'
    #     ),
    # },
    # {
    #     "name": "VerticalHorizontal",
    #     "pyllusion_class": "VerticalHorizontal",
    #     "strengths": _STRENGTHS,
    #     "differences": _DIFFERENCES,
    #     "response_options": ["Vertical", "Horizontal"],
    #     "prompt": (
    #         "Look at the two lines in this image.\n\n"
    #         "Which line looks longer — the VERTICAL one or the HORIZONTAL one?\n\n"
    #         'Answer with only "Vertical" or "Horizontal".'
    #     ),
    # },
    # {
    #     "name": "Zollner",
    #     "pyllusion_class": "Zollner",
    #     "strengths": _STRENGTHS,
    #     "differences": _DIFFERENCES,
    #     "response_options": ["Top", "Bottom"],
    #     "prompt": (
    #         "Look at the two long diagonal lines in this image.\n\n"
    #         "Which long line appears to tilt more to the right at the top — "
    #         "the TOP line or the BOTTOM line?\n\n"
    #         'Answer with only "Top" or "Bottom".'
    #     ),
    # },
]
