"""
pipeline/module_2/query.py - VLMQuerier: async OpenAI client with retry + logprobs.

Handles a single image query against the Responses API, including:
  - Structured output enforcement via json_schema
  - top_logprobs=2 to capture the binary probability distribution
  - Exponential backoff with rate-limit awareness
  - Logprob extraction and renormalisation over the two response options
"""

import asyncio
import json
from typing import Any

from openai import AsyncOpenAI

from pipeline.module_2.response_schema import make_schema
from pipeline.utils import compute_correct, extract_binary_logprobs, parse_filename


class VLMQuerier:
    """
    Async OpenAI Responses API client with retry logic and logprob extraction.

    One instance is shared across all participants for a given illusion run;
    it is initialised with the illusion-specific prompt and response options.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        illusion: dict,
        max_tokens: int = 50,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
    ):
        """
        Args:
            api_key:      OpenAI API key.
            model:        Model string (e.g. "gpt-4o").
            temperature:  Sampling temperature.
            illusion:     Illusion config dict from config.ILLUSIONS.
            max_tokens:   Max output tokens (small — response is one word).
            max_retries:  Number of retry attempts before raising.
            initial_retry_delay: Base delay (seconds) for exponential backoff.
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.illusion = illusion
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

        self._prompt = illusion["prompt"]
        self._response_options = illusion["response_options"]
        self._schema = make_schema(self._response_options)

    async def query_image(self, image_id: str, image_base64: str) -> dict[str, Any]:
        """
        Query the model on a single stimulus with exponential backoff retry.

        Args:
            image_id:     Stimulus filename stem (e.g. 'MullerLyer_str+049_diff+0.46000').
            image_base64: Base64-encoded JPEG string.

        Returns:
            Record dict with keys:
                image_id, illusion_strength, true_diff, response, correct,
                logprob_<option_a>, logprob_<option_b>

        Raises:
            Exception: If all retries are exhausted.
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.responses.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": self._prompt},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                                },
                            ],
                        }
                    ],
                    text={
                        "format": self._schema,
                        "logprobs": True,
                        "top_logprobs": 2,
                    },
                )

                if not response.output_text:
                    raise ValueError(f"Empty output_text. Full response: {response}")

                parsed = json.loads(response.output_text)

                # Ground-truth fields
                _, strength, true_diff = parse_filename(image_id)
                correct = compute_correct(
                    parsed["response"], true_diff, self._response_options
                )

                # Logprob extraction
                try:
                    logprobs_list = response.output[0].content[0].logprobs
                    probs = extract_binary_logprobs(
                        logprobs_list, self._response_options
                    )
                except (IndexError, AttributeError):
                    probs = {opt: 0.5 for opt in self._response_options}

                opt_a, opt_b = self._response_options
                return {
                    "image_id": image_id,
                    "illusion_strength": strength,
                    "true_diff": true_diff,
                    "response": parsed["response"],
                    "correct": correct,
                    f"logprob_{opt_a}": probs[opt_a],
                    f"logprob_{opt_b}": probs[opt_b],
                }

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = "rate_limit" in error_msg.lower() or "429" in error_msg

                if attempt < self.max_retries - 1:
                    delay = self.initial_retry_delay * (2**attempt)
                    if is_rate_limit:
                        delay *= 2

                    safe_error = error_msg.encode("ascii", "backslashreplace").decode(
                        "ascii"
                    )
                    print(
                        f"  Retry {attempt + 1}/{self.max_retries} for {image_id} "
                        f"(waiting {delay:.1f}s): {safe_error[:100]}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"All retries exhausted: {error_msg}")
