"""
pipeline/module_2/batch_processor.py - BatchProcessor: concurrent image querying.

Wraps a VLMQuerier with a semaphore-bounded asyncio task pool, JSONL file
writing with async locks, and per-image progress logging.
"""

import asyncio
import json
from pathlib import Path

import aiofiles

from pipeline.module_2.query import VLMQuerier
from pipeline.utils import preprocess_image


class BatchProcessor:
    """
    Processes a list of stimulus images concurrently for one participant.

    Uses an asyncio.Semaphore to bound the number of in-flight API requests,
    and async file locks to safely write results from multiple coroutines.
    """

    def __init__(
        self,
        querier: VLMQuerier,
        image_dir: Path,
        output_path: Path,
        errors_path: Path,
        participant_id: int,
        max_concurrency: int = 20,
        max_dimension: int = 512,
        jpeg_quality: int = 90,
    ):
        """
        Args:
            querier:          Shared VLMQuerier instance.
            image_dir:        Directory containing stimulus PNGs.
            output_path:      Path to write participant JSONL records.
            errors_path:      Path to write error records.
            participant_id:   Integer participant identifier (for logging).
            max_concurrency:  Max simultaneous API requests.
            max_dimension:    Max pixel dimension before resizing.
            jpeg_quality:     JPEG encoding quality (1–100).
        """
        self.querier = querier
        self.image_dir = image_dir
        self.output_path = output_path
        self.errors_path = errors_path
        self.participant_id = participant_id
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

        self.processed = 0
        self.errors = 0
        self._results_lock = asyncio.Lock()
        self._errors_lock = asyncio.Lock()

    async def process_single_image(self, image_id: str) -> None:
        """Load, encode, query, and log result for one stimulus."""
        async with self.semaphore:
            try:
                image_path = self.image_dir / f"{image_id}.png"
                image_base64 = preprocess_image(
                    image_path,
                    max_dimension=self.max_dimension,
                    jpeg_quality=self.jpeg_quality,
                )

                result = await self.querier.query_image(image_id, image_base64)
                result["participant_id"] = self.participant_id

                async with self._results_lock:
                    async with aiofiles.open(self.output_path, "a") as f:
                        await f.write(json.dumps(result) + "\n")

                self.processed += 1
                correct_str = (
                    "✓"
                    if result["correct"] == 1
                    else ("✗" if result["correct"] == 0 else "–")
                )
                opt_a, opt_b = self.querier._response_options
                lp_a = result.get(f"logprob_{opt_a}", 0.5)
                lp_b = result.get(f"logprob_{opt_b}", 0.5)
                print(
                    f"  [P{self.participant_id:02d} | {self.processed:03d}] "
                    f"str={result['illusion_strength']:+.0f} "
                    f"diff={result['true_diff']:+.2f} "
                    f"→ {result['response']:<10} {correct_str}  "
                    f"({opt_a}:{lp_a:.2f} / {opt_b}:{lp_b:.2f})"
                )

            except Exception as e:
                error_record = {
                    "participant_id": self.participant_id,
                    "image_id": image_id,
                    "error": str(e),
                }

                async with self._errors_lock:
                    async with aiofiles.open(self.errors_path, "a") as f:
                        await f.write(json.dumps(error_record) + "\n")

                self.errors += 1
                safe_err = str(e).encode("ascii", "backslashreplace").decode("ascii")
                print(
                    f"  ✗ ERROR [P{self.participant_id:02d}] {image_id}: {safe_err[:100]}"
                )

    async def process_batch(self, image_ids: list[str]) -> None:
        """Process all images concurrently, bounded by the semaphore."""
        tasks = [self.process_single_image(img_id) for img_id in image_ids]
        await asyncio.gather(*tasks)
