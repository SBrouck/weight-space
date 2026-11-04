"""Generate one thinking-mode answer with MLX and report speed and whether thinking closed.

Uses the paper's sampling (temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5).
Usage: uv run python scripts/speed_test.py models/qwen3.5-0.8B-base [--max-tokens 2048]
"""

from __future__ import annotations

import argparse
import time

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

PROBLEM = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
    "in May. How many clips did Natalia sell altogether in April and May?"
)
SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROBLEM + SUFFIX}],
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )
    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
    processors = make_logits_processors(presence_penalty=1.5)

    text, last = "", None
    for last in stream_generate(
        model, tokenizer, prompt, max_tokens=args.max_tokens,
        sampler=sampler, logits_processors=processors,
    ):
        text += last.text

    closed = "</think>" in text
    print(f"generated {last.generation_tokens} tokens at {last.generation_tps:.1f} tok/s "
          f"(prompt {last.prompt_tps:.1f} tok/s, peak mem {last.peak_memory:.2f} GB)")
    print(f"thinking closed: {closed}   hit max tokens: {last.generation_tokens >= args.max_tokens}")
    print("--- tail of output ---")
    print(text[-600:])


if __name__ == "__main__":
    main()
