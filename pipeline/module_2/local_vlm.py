"""
pipeline/module_2/local_vlm.py - Module 2 for open-weight models, run with vLLM.

Queries one model from config.MODELS (backend "vllm") on every stimulus, and
writes its responses in exactly the format the GPT-5.2 pipeline writes, so
Modules 3 and 4 treat it like any other model:

    results/<model>/<illusion>/participants/participant_XX.jsonl
    results/<model>/<illusion>/probabilities.csv
    results/<model>/run_info.json

WHAT EACH MODEL SEES
    The same request GPT-5.2 got: the stimulus preprocessed by
    pipeline.utils.preprocess_image (512 px JPEG), then the illusion's prompt
    text followed by the image in one user message, at config.TEMPERATURE.
    The answer is constrained to exactly the two response options, as GPT-5.2's
    was by its JSON schema. Each model then applies its own recommended image
    preprocessing (Qwen3-VL: 192 visual tokens; InternVL3.5: dynamic 448 px
    tiling, 13 tiles for a 4:3 image). Nothing is resized to suit a model.

ONE ENCODING, N RESPONSES
    Each stimulus is sent once and N answers are sampled from it
    (SamplingParams.n). The answer is a single token, so these are the same in
    distribution as N separate calls, but the image is encoded once rather than
    N times. Answer i to every stimulus is written as participant i.

EXACT PROBABILITIES
    From the same pass, the log-probabilities of the two options' first tokens
    are read before any sampling or constraint is applied, and stored per
    stimulus in probabilities.csv:

        logprob_first, logprob_second   raw log-probabilities (NaN if an option
                                        is outside the top LOGPROBS tokens)
        other_mass                      probability on tokens that start
                                        neither option
        p_first                         P(first option) over the two, at T = 1
        p_first_sampled                 the same at the sampling temperature,
                                        i.e. the expected proportion of the N
                                        sampled answers that choose it

SAMPLING SETTINGS ARE EXPLICIT
    vLLM otherwise applies each checkpoint's own generation_config (Qwen3-VL
    ships top_k 20, top_p 0.8), which would make the models incomparable with
    each other and with GPT-5.2. The engine is created with
    generation_config="vllm" and temperature and top_p are set here.

RUNNING IT (on a rented GPU; there is no GPU on the analysis machine)
    pipeline/module_2/run_open_models.sh wraps this module for a pod session:
    it checks the code is committed, unpacks the stimuli, installs vLLM,
    checks the GPU count, runs each model, records the environment and packs
    the results. On a pod with CUDA and enough GPU memory (one 80 GB card for
    every model except InternVL3.5-38B, which needs two - config n_gpus):

        bash pipeline/module_2/run_open_models.sh --pilot qwen3-vl-2b internvl3.5-2b
        bash pipeline/module_2/run_open_models.sh qwen3-vl-2b qwen3-vl-8b ...

    The pilot runs one illusion with 10 answers per stimulus into
    results/_pilot/, so it never mixes with real data. It prints the
    image-token count, the share of valid answers, and how closely sampled
    proportions follow the exact probabilities. Copy results/<model>/ back
    into this repository and run
        python run_pipeline.py --modules 3 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ILLUSIONS,
    JPEG_QUALITY,
    MAX_DIMENSIONS,
    MODELS,
    N_PARTICIPANTS,
    TEMPERATURE,
)
from pipeline.utils import compute_correct, discover_images, parse_filename, preprocess_image

STIMULI_ROOT = Path("stimuli")
RESULTS_ROOT = Path("results")
PILOT_ROOT = RESULTS_ROOT / "_pilot"

# One word plus an end token; the constraint admits nothing longer.
MAX_ANSWER_TOKENS = 8
# How many of the first token's alternatives to read log-probabilities for.
LOGPROBS = 20
MAX_MODEL_LEN = 8192
SEED = 20260919

PILOT_ILLUSION = "MullerLyer"
PILOT_N = 10


# ============================================================================
# REQUESTS
# ============================================================================


def build_messages(prompt: str, image_base64: str) -> list[dict]:
    """The single user turn GPT-5.2 received: prompt text, then the image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    ]


def _choice_constraint(options: list[str]) -> dict:
    """SamplingParams keyword restricting output to one of `options`."""
    # vLLM renamed guided decoding to structured outputs; accept either.
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(choice=list(options))}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(choice=list(options))}


def sampling_params(options: list[str], n: int, seed: int):
    """Explicit sampling settings for one stimulus; nothing left to defaults."""
    from vllm import SamplingParams

    return SamplingParams(
        n=n,
        temperature=TEMPERATURE,
        top_p=1.0,
        max_tokens=MAX_ANSWER_TOKENS,
        logprobs=LOGPROBS,
        seed=seed,
        **_choice_constraint(options),
    )


def load_engine(model: dict, gpus: int | None = None):
    """
    A vLLM engine for `model`, with the checkpoint's generation defaults off.

    `gpus` overrides the model's n_gpus, for a pod whose single card is large
    enough for a model config splits across two (InternVL3.5-38B on one 96 GB
    or 141 GB GPU, say).
    """
    from vllm import LLM

    return LLM(
        model=model["hf_id"],
        tensor_parallel_size=gpus or model.get("n_gpus", 1),
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        limit_mm_per_prompt={"image": 1},
        generation_config="vllm",
        seed=SEED,
    )


# ============================================================================
# READING THE OUTPUT
# ============================================================================


def option_logprobs(top: dict, options: list[str]) -> tuple[list[float], float]:
    """
    Log-probability of each option's first token, and the mass on neither.

    `top` maps token id -> an object with `.logprob` and `.decoded_token`, as
    vLLM returns for one generated position. A token counts towards an option
    when its text, stripped and lower-cased, is a non-empty prefix of that
    option and of no other ("T" for Top, "Bot" for Bottom). Several such
    tokens are pooled. An option with no token in `top` gets NaN.
    """
    names = [o.lower() for o in options]
    pooled: list[list[float]] = [[] for _ in options]
    for entry in top.values():
        text = (entry.decoded_token or "").strip().lower()
        if not text:
            continue
        hits = [i for i, name in enumerate(names) if name.startswith(text)]
        if len(hits) == 1:
            pooled[hits[0]].append(entry.logprob)

    lps = []
    for vals in pooled:
        if vals:
            m = max(vals)
            lps.append(m + math.log(sum(math.exp(v - m) for v in vals)))
        else:
            lps.append(float("nan"))
    known = [math.exp(v) for v in lps if not math.isnan(v)]
    other = max(0.0, 1.0 - sum(known)) if len(known) == len(options) else float("nan")
    return lps, other


def p_first(lp_first: float, lp_second: float, temperature: float) -> float:
    """P(first option) over the two options at `temperature`."""
    if math.isnan(lp_first) or math.isnan(lp_second):
        return float("nan")
    return 1.0 / (1.0 + math.exp((lp_second - lp_first) / temperature))


# ============================================================================
# ONE ILLUSION
# ============================================================================


def run_illusion(
    llm,
    illusion: dict,
    n: int,
    out_root: Path,
    stimuli_root: Path = STIMULI_ROOT,
    make_params=sampling_params,
) -> dict:
    """
    Query every stimulus of one illusion and write its outputs.

    `llm` is anything with vLLM's `chat(messages, sampling_params, use_tqdm)`
    interface and `make_params` builds its per-stimulus settings, so the
    bookkeeping here can be checked without a GPU.

    Returns a summary for run_info.json and the pilot report.
    """
    name = illusion["name"]
    options = illusion["response_options"]
    image_dir = stimuli_root / name
    image_ids = discover_images(
        image_dir, name, illusion["strengths"], illusion["differences"]
    )

    conversations, params = [], []
    for i, image_id in enumerate(image_ids):
        b64 = preprocess_image(image_dir / f"{image_id}.png", MAX_DIMENSIONS, JPEG_QUALITY)
        conversations.append(build_messages(illusion["prompt"], b64))
        params.append(make_params(options, n, SEED + i))

    outputs = llm.chat(conversations, params, use_tqdm=True)

    out_dir = out_root / name
    part_dir = out_dir / "participants"
    err_dir = out_dir / "errors"
    part_dir.mkdir(parents=True, exist_ok=True)

    records: dict[int, list[dict]] = {pid: [] for pid in range(1, n + 1)}
    errors: dict[int, list[dict]] = {}
    prob_rows, prompt_tokens = [], []

    for image_id, output in zip(image_ids, outputs):
        _, strength, true_diff = parse_filename(image_id)
        prompt_tokens.append(len(output.prompt_token_ids))

        for pid, sample in enumerate(output.outputs, start=1):
            response = sample.text.strip()
            record = {
                "image_id": image_id,
                "illusion_strength": strength,
                "true_diff": true_diff,
                "response": response,
            }
            if response in options:
                record["correct"] = compute_correct(response, true_diff, options)
                records[pid].append(record)
            else:
                errors.setdefault(pid, []).append(record)

        first = output.outputs[0].logprobs
        (lp_a, lp_b), other = option_logprobs(first[0] if first else {}, options)
        prob_rows.append(
            {
                "image_id": image_id,
                "illusion_strength": strength,
                "true_diff": true_diff,
                "logprob_first": lp_a,
                "logprob_second": lp_b,
                "other_mass": other,
                "p_first": p_first(lp_a, lp_b, 1.0),
                "p_first_sampled": p_first(lp_a, lp_b, TEMPERATURE),
                "temperature": TEMPERATURE,
                "observed_first": float(
                    np.mean([s.text.strip() == options[0] for s in output.outputs])
                ),
            }
        )

    for pid, rows in records.items():
        with open(part_dir / f"participant_{pid:02d}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    if errors:
        err_dir.mkdir(parents=True, exist_ok=True)
        for pid, rows in errors.items():
            with open(err_dir / f"participant_{pid:02d}_errors.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    probs = pd.DataFrame(prob_rows)
    probs.to_csv(out_dir / "probabilities.csv", index=False)

    n_total = len(image_ids) * n
    n_invalid = sum(len(v) for v in errors.values())
    return {
        "illusion": name,
        "n_stimuli": len(image_ids),
        "n_samples": n,
        "valid_share": 1.0 - n_invalid / n_total,
        "prompt_tokens": [min(prompt_tokens), max(prompt_tokens)],
        "probabilities_missing": int(probs["p_first"].isna().sum()),
        "mean_abs_sampled_vs_exact": float(
            (probs["observed_first"] - probs["p_first_sampled"]).abs().mean()
        ),
    }


# ============================================================================
# ENTRY POINT
# ============================================================================


def provenance() -> dict:
    """The code and hardware a run used, for run_info.json."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
            return done.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    try:
        import torch

        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        gpus = None
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_modified_files": None if status is None else bool(status),
        "gpus": gpus,
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
    }


def run(
    model_key: str,
    pilot: bool = False,
    illusion: str | None = None,
    n: int | None = None,
    gpus: int | None = None,
) -> None:
    """Run one open model over the illusion registry (or the pilot subset)."""
    matches = [m for m in MODELS if m["key"] == model_key]
    if not matches or matches[0].get("backend") != "vllm":
        local = [m["key"] for m in MODELS if m.get("backend") == "vllm"]
        sys.exit(f"Unknown open model {model_key!r}. Choose one of: {', '.join(local)}")
    model = matches[0]

    if pilot:
        illusions = [i for i in ILLUSIONS if i["name"] == (illusion or PILOT_ILLUSION)]
        n = n or PILOT_N
        out_root = PILOT_ROOT / model["key"]
    else:
        illusions = [i for i in ILLUSIONS if illusion is None or i["name"] == illusion]
        n = n or N_PARTICIPANTS
        out_root = RESULTS_ROOT / model["key"]

    import vllm

    n_gpus = gpus or model.get("n_gpus", 1)
    print(f"Loading {model['hf_id']} on {n_gpus} GPU(s)...")
    llm = load_engine(model, n_gpus)

    summaries = []
    for ill in illusions:
        print(f"\n  {model['label']} — {ill['name']}: {n} answers per stimulus")
        summary = run_illusion(llm, ill, n, out_root)
        summaries.append(summary)
        print(
            f"    valid answers {summary['valid_share']:.1%} | "
            f"prompt tokens {summary['prompt_tokens'][0]}-{summary['prompt_tokens'][1]} | "
            f"sampled vs exact P, mean |diff| {summary['mean_abs_sampled_vs_exact']:.3f} | "
            f"stimuli without both option logprobs {summary['probabilities_missing']}"
        )

    info = {
        "model": model,
        **provenance(),
        "vllm_version": vllm.__version__,
        "temperature": TEMPERATURE,
        "top_p": 1.0,
        "n_samples": n,
        "tensor_parallel_size": n_gpus,
        "seed": SEED,
        "max_answer_tokens": MAX_ANSWER_TOKENS,
        "image_preprocessing": {"max_dimension": MAX_DIMENSIONS, "jpeg_quality": JPEG_QUALITY},
        "pilot": pilot,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "illusions": summaries,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\n✓ {model['label']} complete — outputs under {out_root}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model key from config.MODELS")
    parser.add_argument("--pilot", action="store_true", help="One illusion, 10 answers, into results/_pilot/")
    parser.add_argument("--illusion", default=None, help="Restrict to one illusion")
    parser.add_argument("--n", type=int, default=None, help="Answers per stimulus")
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Override the model's n_gpus, e.g. 1 for a 38B model on one 96 GB card",
    )
    args = parser.parse_args()
    run(args.model, pilot=args.pilot, illusion=args.illusion, n=args.n, gpus=args.gpus)


if __name__ == "__main__":
    main()
