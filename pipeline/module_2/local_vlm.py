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

ONE FORWARD PASS PER STIMULUS
    The answer is a single token: the constraint admits only the two options,
    and their first tokens differ. Asking the model N times would therefore
    repeat one deterministic forward pass N times to re-roll the same die, so
    each stimulus is sent once and its N answers are drawn from the response
    distribution that pass reports. This is the same sampling with the
    redundant recomputation removed; what it saves is the image encoding,
    which for InternVL3.5 is 3,328 of the 3,374 prompt tokens. Answer i to
    every stimulus is written as participant i, as before.

    The draws are reproducible: each stimulus seeds its own generator from
    DRAW_SEED and its image id, so they do not depend on the order stimuli
    are run in or on how many are run at once.

    --sampled instead generates every answer (SamplingParams.n). It pays one
    forward pass per answer - for InternVL3.5, a hundred encodings of the same
    image - and exists so the drawn answers can be checked against generated
    ones whenever that is worth the GPU time.

THE GENERATED ANSWER IS KEPT AS A CHECK
    One answer per stimulus is always generated rather than drawn, and two
    things are verified against it. Per stimulus: that its first token
    identifies the option its full text spells, which is the premise the
    drawing rests on, and a first token matching both options or neither
    fails the run. Per illusion: that the generated answers choose the first
    option about as often as the exact probabilities say they should, which
    is reported with the standard error it should be judged against.

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
                                        i.e. the proportion of answers that
                                        choose it, and what they are drawn from
        observed_first                  the realised proportion among the N
                                        answers written for this stimulus
        probe_first                     1 if the one generated answer chose the
                                        first option, 0 if it chose the second

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

    The pilot runs one illusion into results/_pilot/, so it never mixes with
    real data. It prints the image-token count, the share of valid answers,
    and how far the generated answers fall from the exact probabilities.
    Copy results/<model>/ back
    into this repository and run
        python run_pipeline.py --modules 3 4
"""

from __future__ import annotations

import argparse
import hashlib
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
# Separate from SEED so that regenerating the draws never silently changes
# what the model was asked.
DRAW_SEED = 20260920

PILOT_ILLUSION = "MullerLyer"
# A pilot costs what a real run costs now, so it uses the real answer count.
PILOT_N = N_PARTICIPANTS


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


def first_token_option(top: dict, token_id: int, options: list[str]) -> str | None:
    """
    The option a generated first token identifies, or None if it identifies
    no single one. `top` is that position's log-probability dict, which
    always contains the token actually generated.
    """
    entry = (top or {}).get(token_id)
    text = (entry.decoded_token or "").strip().lower() if entry is not None else ""
    hits = [o for o in options if text and o.lower().startswith(text)]
    return hits[0] if len(hits) == 1 else None


# ============================================================================
# DRAWING THE ANSWERS
# ============================================================================


def draw_rng(image_id: str) -> np.random.Generator:
    """The generator for one stimulus' draws, fixed by DRAW_SEED and its id."""
    digest = hashlib.sha256(f"{DRAW_SEED}:{image_id}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "big"))


def draw_answers(p: float, n: int, options: list[str], rng) -> list[str]:
    """N answers drawn from the model's own distribution over the two options."""
    return [options[0] if u < p else options[1] for u in rng.random(n)]


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
    sampled: bool = False,
) -> dict:
    """
    Query every stimulus of one illusion and write its outputs.

    `llm` is anything with vLLM's `chat(messages, sampling_params, use_tqdm)`
    interface and `make_params` builds its per-stimulus settings, so the
    bookkeeping here can be checked without a GPU.

    With `sampled`, all N answers are generated; otherwise one is generated and
    N are drawn from the distribution it reports.

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
        params.append(make_params(options, n if sampled else 1, SEED + i))

    outputs = llm.chat(conversations, params, use_tqdm=True)

    out_dir = out_root / name
    part_dir = out_dir / "participants"
    err_dir = out_dir / "errors"
    part_dir.mkdir(parents=True, exist_ok=True)

    records: dict[int, list[dict]] = {pid: [] for pid in range(1, n + 1)}
    errors: dict[int, list[dict]] = {}
    prob_rows, prompt_tokens, ambiguous = [], [], []

    for image_id, output in zip(image_ids, outputs):
        _, strength, true_diff = parse_filename(image_id)
        prompt_tokens.append(len(output.prompt_token_ids))

        # The generated answer: the source of the probabilities either way, and
        # the check that its first token settles which option it is.
        probe = output.outputs[0]
        top = (probe.logprobs or [{}])[0]
        (lp_a, lp_b), other = option_logprobs(top, options)
        p_sampled = p_first(lp_a, lp_b, TEMPERATURE)

        probe_text = probe.text.strip()
        by_token = (
            first_token_option(top, probe.token_ids[0], options)
            if probe.token_ids
            else None
        )
        if probe_text not in options or by_token != probe_text:
            ambiguous.append(
                {"image_id": image_id, "answer": probe_text, "first_token_option": by_token}
            )

        if sampled:
            answers = [s.text.strip() for s in output.outputs]
        elif math.isnan(p_sampled):
            # Nothing to draw from; counted in probabilities_missing and left
            # out of every participant's file.
            answers = []
        else:
            answers = draw_answers(p_sampled, n, options, draw_rng(image_id))

        for pid, response in enumerate(answers, start=1):
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

        prob_rows.append(
            {
                "image_id": image_id,
                "illusion_strength": strength,
                "true_diff": true_diff,
                "logprob_first": lp_a,
                "logprob_second": lp_b,
                "other_mass": other,
                "p_first": p_first(lp_a, lp_b, 1.0),
                "p_first_sampled": p_sampled,
                "temperature": TEMPERATURE,
                "observed_first": (
                    float(np.mean([a == options[0] for a in answers]))
                    if answers
                    else float("nan")
                ),
                "probe_first": (
                    float(probe_text == options[0]) if probe_text in options else float("nan")
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
    if ambiguous:
        err_dir.mkdir(parents=True, exist_ok=True)
        with open(err_dir / "first_token_ambiguous.jsonl", "w", encoding="utf-8") as f:
            for row in ambiguous:
                f.write(json.dumps(row) + "\n")

    probs = pd.DataFrame(prob_rows)
    probs.to_csv(out_dir / "probabilities.csv", index=False)

    # How far the generated answers fall from the exact probabilities across
    # the illusion's stimuli. One answer is too few to judge a stimulus by, but
    # their mean is not: under those probabilities it has standard error
    # sqrt(sum p(1 - p)) / K, which is reported beside it.
    usable = probs.dropna(subset=["p_first_sampled", "probe_first"])
    p_exact = usable["p_first_sampled"].to_numpy()
    gap = float(usable["probe_first"].mean() - p_exact.mean()) if len(p_exact) else float("nan")
    gap_se = (
        float(np.sqrt((p_exact * (1.0 - p_exact)).sum()) / len(p_exact))
        if len(p_exact)
        else float("nan")
    )

    summary = {
        "illusion": name,
        "n_stimuli": len(image_ids),
        "n_samples": n,
        "answers": "generated" if sampled else "drawn",
        "valid_share": sum(len(v) for v in records.values()) / (len(image_ids) * n),
        "prompt_tokens": [min(prompt_tokens), max(prompt_tokens)],
        "probabilities_missing": int(probs["p_first_sampled"].isna().sum()),
        "first_token_ambiguous": len(ambiguous),
        "generated_vs_exact": gap,
        "generated_vs_exact_se": gap_se,
        "mean_abs_observed_vs_exact": float(
            (probs["observed_first"] - probs["p_first_sampled"]).abs().mean()
        ),
    }

    if ambiguous and not sampled:
        raise RuntimeError(
            f"{name}: {len(ambiguous)} of {len(image_ids)} answers are not settled by "
            f"their first token, which is what drawing the other answers assumes. "
            f"See {err_dir / 'first_token_ambiguous.jsonl'}, and rerun this model "
            f"with --sampled to generate every answer instead."
        )
    return summary


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
    sampled: bool = False,
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
        how = "generated" if sampled else "drawn"
        print(f"\n  {model['label']} — {ill['name']}: {n} answers per stimulus, {how}")
        summary = run_illusion(llm, ill, n, out_root, sampled=sampled)
        summaries.append(summary)
        print(
            f"    valid answers {summary['valid_share']:.1%} | "
            f"prompt tokens {summary['prompt_tokens'][0]}-{summary['prompt_tokens'][1]} | "
            f"generated vs exact P {summary['generated_vs_exact']:+.3f} "
            f"(SE {summary['generated_vs_exact_se']:.3f}) | "
            f"stimuli without both option logprobs {summary['probabilities_missing']} | "
            f"first token ambiguous {summary['first_token_ambiguous']}"
        )

    info = {
        "model": model,
        **provenance(),
        "vllm_version": vllm.__version__,
        "temperature": TEMPERATURE,
        "top_p": 1.0,
        "n_samples": n,
        "answers": "generated" if sampled else "drawn",
        "draw_seed": None if sampled else DRAW_SEED,
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
    parser.add_argument("--pilot", action="store_true", help="One illusion, into results/_pilot/")
    parser.add_argument("--illusion", default=None, help="Restrict to one illusion")
    parser.add_argument("--n", type=int, default=None, help="Answers per stimulus")
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="Generate every answer instead of drawing them (one forward pass each)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Override the model's n_gpus, e.g. 1 for a 38B model on one 96 GB card",
    )
    args = parser.parse_args()
    run(
        args.model,
        pilot=args.pilot,
        illusion=args.illusion,
        n=args.n,
        gpus=args.gpus,
        sampled=args.sampled,
    )


if __name__ == "__main__":
    main()
