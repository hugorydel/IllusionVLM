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
N_PARTICIPANTS = 100
MAX_TOKENS = 200  # Limit is 200 because some illusion names are 50-100 tokens long; this limit accommodates them.
MAX_CONCURRENCY = 100
MAX_DIMENSIONS = 512
MAX_BATCH_BYTES: int = 190 * 1024 * 1024  # 190 MB — safely under OpenAI 200 MB limit
# Cap on requests per sub-batch. The byte cap alone allowed ~20k requests in a
# single batch, which is one point of failure against the 24h completion
# window: when it ran out, the unprocessed remainder was lost in one block.
# Smaller independent sub-batches bound that loss and complete more reliably.
MAX_BATCH_REQUESTS: int = 6000
JPEG_QUALITY = 90

# ============================================================================
# MODELS COMPARED
# ============================================================================
# Every model whose responses the analysis reads, in the order the figures
# list them. `key` names the model's folder under results/; `backend` says which
# Module 2 path produces its responses:
#     "openai" - pipeline/module_2_query.py or the batch API (MODEL above)
#     "vllm"   - pipeline/module_2/local_vlm.py, run on a rented GPU
# `n_gpus` is the tensor-parallel size vLLM needs to hold the model in bf16 on
# 80 GB cards. All open models are the Instruct (non-thinking) variants.
MODELS = [
    {"key": "gpt-5.2", "label": "GPT-5.2", "backend": "openai"},
    {
        "key": "qwen3-vl-2b",
        "label": "Qwen3-VL-2B",
        "backend": "vllm",
        "hf_id": "Qwen/Qwen3-VL-2B-Instruct",
        "n_gpus": 1,
    },
    {
        "key": "qwen3-vl-8b",
        "label": "Qwen3-VL-8B",
        "backend": "vllm",
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "n_gpus": 1,
    },
    {
        "key": "qwen3-vl-32b",
        "label": "Qwen3-VL-32B",
        "backend": "vllm",
        "hf_id": "Qwen/Qwen3-VL-32B-Instruct",
        "n_gpus": 1,
    },
    {
        "key": "internvl3.5-2b",
        "label": "InternVL3.5-2B",
        "backend": "vllm",
        "hf_id": "OpenGVLab/InternVL3_5-2B",
        "n_gpus": 1,
    },
    {
        "key": "internvl3.5-8b",
        "label": "InternVL3.5-8B",
        "backend": "vllm",
        "hf_id": "OpenGVLab/InternVL3_5-8B",
        "n_gpus": 1,
    },
    {
        "key": "internvl3.5-38b",
        "label": "InternVL3.5-38B",
        "backend": "vllm",
        "hf_id": "OpenGVLab/InternVL3_5-38B",
        "n_gpus": 2,
    },
]

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
    {
        "name": "MullerLyer",
        "pyllusion_class": "MullerLyer",
        "strengths": _strengths(7.0),
        "differences": _differences(
            [0.04, 0.06565, 0.10044, 0.14575, 0.20297, 0.27349, 0.3587, 0.46]
        ),
        "response_options": ["Top", "Bottom"],
        "prompt": (
            "Look at the two red horizontal lines in this image.\n\n"
            "Which red line looks longer — the TOP one or the BOTTOM one?\n\n"
            'Answer with only "Top" or "Bottom".'
        ),
    },
    {
        "name": "Ebbinghaus",
        "pyllusion_class": "Ebbinghaus",
        "strengths": _strengths(0.29),
        "differences": _differences(
            [0.07, 0.11066, 0.16462, 0.23378, 0.32001, 0.4252, 0.55124, 0.7]
        ),
        "response_options": ["Left", "Right"],
        "prompt": (
            "Look at the two red circles in the centre of each group in this image.\n\n"
            "Which central red circle looks bigger — the LEFT one or the RIGHT one?\n\n"
            'Answer with only "Left" or "Right".'
        ),
    },
    {
        "name": "Ponzo",
        "pyllusion_class": "Ponzo",
        "strengths": _strengths(3.6),
        "differences": _differences(
            [0.04, 0.06565, 0.10044, 0.14575, 0.20297, 0.27349, 0.3587, 0.46]
        ),
        "response_options": ["Top", "Bottom"],
        "prompt": (
            "Look at the two red horizontal lines in this image.\n\n"
            "Which red line looks longer — the TOP one or the BOTTOM one?\n\n"
            'Answer with only "Top" or "Bottom".'
        ),
    },
    {
        "name": "VerticalHorizontal",
        "pyllusion_class": "VerticalHorizontal",
        "strengths": _strengths(9.5),
        "differences": _differences(
            [0.03, 0.04772, 0.06953, 0.09544, 0.12544, 0.15953, 0.19772, 0.24]
        ),
        "response_options": ["Left", "Right"],
        "prompt": (
            "Look at the two red lines in this image.\n\n"
            "Which red line looks longer — the LEFT one or the RIGHT one?\n\n"
            'Answer with only "Left" or "Right".'
        ),
    },
    {
        "name": "Contrast",
        "pyllusion_class": "Contrast",
        "strengths": _strengths(4.5),
        "differences": _differences(
            [3.0, 4.33568, 5.91661, 7.74279, 9.81421, 12.13089, 14.69282, 17.5]
        ),
        # NOTE: Flip the x-axis when comparing against Makowski human data for this illusion.
        "response_options": ["Top", "Bottom"],
        "prompt": (
            "Look at the two small grey rectangles in this image.\n\n"
            "Which grey rectangle looks lighter (brighter) — the TOP one or the BOTTOM one?\n\n"
            'Answer with only "Top" or "Bottom".'
        ),
    },
]
