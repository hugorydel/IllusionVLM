"""
tests/diagnose_image_resolution.py - Does shrinking the images cost GPT-5.2 accuracy?

The study sends every stimulus downscaled from 800x600 to 512x384
(config.MAX_DIMENSIONS). At the smallest Ebbinghaus difference that leaves the
two red discs 1.6 px apart in diameter, and GPT-5.2 is at chance (50%) where
humans are at 70%. This diagnostic asks the no-illusion images again at three
resolutions:

    512    the study's setting: rendered at 800x600, downscaled to 512x384
    800    the same rendering at its own size, not downscaled
    1600   re-rendered by Pyllusion at 1600x1200, so the difference is drawn
           with twice the pixels

If accuracy at the smallest differences rises with resolution, the downscaling
is what held the model at chance. If it stays flat, the limit is the model's.

Each request is built by the Batch API's own builder and parsed by its own
parser (pipeline/module_2/batch_vlm.py), so it is the request the study sent,
only sent synchronously rather than as a batch. Stimuli are rendered fresh by
Pyllusion, which reproduces the study's images pixel for pixel.

Usage, from VLM_Analysis/:
    python -m tests.diagnose_image_resolution --dry-run   # plan only, no API calls
    python -m tests.diagnose_image_resolution             # Ebbinghaus, 10 runs per image
    python -m tests.diagnose_image_resolution --illusion MullerLyer --runs 20

The default run is 16 images x 3 resolutions x 10 runs = 480 requests. The API
key is asked for when the run starts and is never written anywhere. Trial
records are saved to tests/output/.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyllusion import

import pandas as pd
import pyllusion
from openai import AsyncOpenAI

from config import ILLUSIONS, JPEG_QUALITY, MAX_DIMENSIONS, MODEL, TEMPERATURE
from pipeline.module_2.batch_vlm import (
    build_single_request,
    make_custom_id,
    parse_batch_response,
)
from pipeline.module_2.response_schema import make_chat_completions_schema
from pipeline.utils import make_filename, preprocess_image

OUT_DIR = Path("tests/output")
PAPER_DIR = Path("results/_paper")

# name -> (render size, max dimension sent). "512" is the study's setting.
CONDITIONS = {
    "512": ((800, 600), MAX_DIMENSIONS),
    "800": ((800, 600), 800),
    "1600": ((1600, 1200), 1600),
}

MAX_RETRIES = 4


def render(illusion: dict, strength: float, stim_dir: Path) -> dict[str, dict[str, str]]:
    """
    Render and encode every difference at one strength, in every condition.

    Returns {condition: {image_id: base64 JPEG}}.
    """
    encoded: dict[str, dict[str, str]] = {c: {} for c in CONDITIONS}
    cls = getattr(pyllusion, illusion["pyllusion_class"])
    for cond, ((width, height), max_dim) in CONDITIONS.items():
        folder = stim_dir / f"{width}x{height}"
        folder.mkdir(parents=True, exist_ok=True)
        for diff in illusion["differences"]:
            name = make_filename(illusion["name"], strength, diff)
            path = folder / name
            if not path.exists():
                cls(illusion_strength=strength, difference=diff).to_image(
                    width=width, height=height
                ).save(path)
            encoded[cond][path.stem] = preprocess_image(path, max_dim, JPEG_QUALITY)
    return encoded


async def ask(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    illusion: dict,
    response_format: dict,
    cond: str,
    run: int,
    image_id: str,
    image_base64: str,
) -> dict | None:
    """One request, built and parsed exactly as the Batch API path does."""
    request = build_single_request(
        run, image_id, image_base64, illusion["prompt"], response_format, MODEL, TEMPERATURE
    )
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                reply = await client.chat.completions.create(**request["body"])
                break
            except Exception as e:  # noqa: BLE001 - any API failure is retried
                if attempt == MAX_RETRIES - 1:
                    print(f"  ✗ {cond} run {run} {image_id}: {str(e)[:100]}")
                    return None
                await asyncio.sleep(2.0 * 2**attempt)

    # Wrapped as a batch output line so the study's own parser reads it.
    line = json.dumps(
        {
            "custom_id": make_custom_id(run, image_id),
            "response": {"body": reply.model_dump()},
        }
    )
    record = parse_batch_response(line, illusion["response_options"])
    if record is None:
        return None
    return {"condition": cond, **record}


async def run_all(
    illusion: dict, encoded: dict[str, dict[str, str]], runs: int, api_key: str, concurrency: int
) -> pd.DataFrame:
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    response_format = make_chat_completions_schema(illusion["response_options"])
    tasks = [
        ask(client, semaphore, illusion, response_format, cond, run, image_id, b64)
        for cond, images in encoded.items()
        for image_id, b64 in images.items()
        for run in range(1, runs + 1)
    ]
    records = [r for r in await asyncio.gather(*tasks) if r is not None]
    print(f"  {len(records)}/{len(tasks)} responses usable")
    return pd.DataFrame(records)


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for k successes in n trials, as percentages."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def study_accuracy(illusion: str, strength: float) -> pd.Series | None:
    """The study's own accuracy per |difference| at this strength, if built."""
    path = PAPER_DIR / "cells.csv"
    if not path.exists():
        return None
    cells = pd.read_csv(path)
    cells = cells[
        (cells["species"] == "vlm")
        & (cells["illusion"] == illusion)
        & (cells["illusion_strength"].round(5) == round(strength, 5))
    ]
    if cells.empty:
        return None
    correct = cells["n_positive"].where(cells["true_diff"] > 0, cells["n_trials"] - cells["n_positive"])
    grouped = cells.assign(correct=correct, abs_diff=cells["true_diff"].abs().round(5))
    agg = grouped.groupby("abs_diff")[["correct", "n_trials"]].sum()
    return 100 * agg["correct"] / agg["n_trials"]


def summarise(df: pd.DataFrame, illusion: dict, strength: float) -> None:
    """Accuracy per |difference| and condition, with the study's for reference."""
    opt_a, opt_b = illusion["response_options"]
    df = df.assign(
        abs_diff=df["true_diff"].abs().round(5),
        p_correct=df[f"logprob_{opt_a}"].where(df["true_diff"] > 0, df[f"logprob_{opt_b}"]),
    )
    table = (100 * df.pivot_table(index="abs_diff", columns="condition", values="correct")).round(0)
    table = table[[c for c in CONDITIONS if c in table.columns]]
    reference = study_accuracy(illusion["name"], strength)
    if reference is not None:
        table.insert(0, "study (512)", reference.round(0))
    print(f"\n  Accuracy (%) by |difference|, {illusion['name']} at strength {strength}:")
    print(textwrap.indent(table.to_string(), "    "))

    print("\n  Overall, per condition:")
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        k, n = sub["correct"].sum(), len(sub)
        lo, hi = wilson(k, n)
        print(
            f"    {cond:>5}: {100 * k / n:5.1f}% correct (95% CI {lo:.0f}-{hi:.0f}%, "
            f"n = {n}), mean P(correct) from logprobs {sub['p_correct'].mean():.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--illusion", default="Ebbinghaus")
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=10, help="runs per image per condition")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="render and plan only")
    args = parser.parse_args()

    illusion = next((d for d in ILLUSIONS if d["name"] == args.illusion), None)
    if illusion is None:
        parser.error(f"unknown illusion {args.illusion!r}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    encoded = render(illusion, args.strength, OUT_DIR / "stimuli" / illusion["name"])
    n_requests = sum(len(images) for images in encoded.values()) * args.runs

    print(f"\n  {illusion['name']}, strength {args.strength}, model {MODEL}, temperature {TEMPERATURE}")
    for cond, ((w, h), max_dim) in CONDITIONS.items():
        kb = sum(len(b) for b in encoded[cond].values()) * 3 / 4 / 1024 / len(encoded[cond])
        sent = f"{min(w, max_dim * w // max(w, h))}x{min(h, max_dim * h // max(w, h))}"
        print(f"    {cond:>5}: rendered {w}x{h}, sent {sent}, ~{kb:.0f} KB per image")
    print(f"  {n_requests} requests ({len(illusion['differences'])} images x {len(CONDITIONS)} conditions x {args.runs} runs)")

    if args.dry_run:
        print("\n  Dry run: no requests sent.")
        return

    api_key = getpass.getpass("\n  OpenAI API key (input hidden): ").strip()
    if not api_key:
        raise SystemExit("No API key provided.")

    df = asyncio.run(run_all(illusion, encoded, args.runs, api_key, args.concurrency))
    if df.empty:
        raise SystemExit("No usable responses.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"resolution_{illusion['name']}_{stamp}.csv"
    df.to_csv(out, index=False)
    summarise(df, illusion, args.strength)
    print(f"\n  Trial records: {out}")


if __name__ == "__main__":
    main()
