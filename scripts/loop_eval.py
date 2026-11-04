"""Measure the thinking-loop rate and accuracy of a model on GSM8K-style problems, batched with MLX.

A sample is "cut" when it reaches --max-tokens without emitting </think> or the end-of-turn token.
Accuracy compares the last \\boxed{} answer with the reference (numeric when both parse).
A sample counts as a loop when its repetition score (see `repetition`) is at least 0.5.
Problems without an "answer" field (open prompts) get no math suffix and no accuracy.
Results are appended per batch, so rerunning the same command resumes after an interruption.

Usage: uv run python scripts/loop_eval.py models/qwen3.5-0.8B-base --n 64 --max-tokens 4096 --tag base
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def last_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...}, handling nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j].strip()
    return None


def same_answer(pred: str | None, ref: str) -> bool:
    """Compare answers numerically when possible, otherwise as normalised strings."""
    if pred is None:
        return False
    norm = lambda s: re.sub(r"[\s$,]|\\!|\\text\{[^}]*\}", "", s)
    p, r = norm(pred), norm(ref)
    try:
        return abs(float(p) - float(r)) < 1e-6
    except ValueError:
        return p == r


def repetition(ids: list[int], window: int = 1024, n: int = 16) -> float:
    """Share of n-grams in the last `window` tokens that already occurred earlier in that window.

    Near 0 for productive text, close to 1 for a verbatim loop. Works on cut samples too,
    so it separates a loop from a long but progressing chain of thought.
    """
    tail = ids[-window:]
    seen, repeats, total = set(), 0, 0
    for i in range(len(tail) - n + 1):
        gram = tuple(tail[i : i + n])
        repeats += gram in seen
        seen.add(gram)
        total += 1
    return repeats / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--data", default="data/gsm8k_train.jsonl")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1, help="samples per problem")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f][args.offset : args.offset + args.n]
    if args.samples > 1:  # one row per (problem, sample), ids suffixed #s
        problems = [{**p, "id": f"{p['id']}#{s}"} for p in problems for s in range(args.samples)]

    # Resume: rows already written for this tag are kept and their problems skipped.
    out_path = f"results/loop_eval_{args.tag}.jsonl"
    done: dict[str, dict] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {r["id"]: r for r in map(json.loads, f)}
    todo = [p for p in problems if p["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to generate")

    if todo:
        model, tokenizer = load(args.model)
        think_end = tokenizer.convert_tokens_to_ids("</think>")
        sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
        processors = make_logits_processors(presence_penalty=1.5)

    # One chunk = one batch, appended to the jsonl as soon as it finishes.
    for start in range(0, len(todo), args.batch):
        chunk = todo[start : start + args.batch]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p["problem"] + (SUFFIX if "answer" in p else "")}],
                add_generation_prompt=True, enable_thinking=True,
            )
            for p in chunk
        ]
        t0 = time.perf_counter()
        resp = batch_generate(
            model, tokenizer, prompts, max_tokens=args.max_tokens, return_token_ids=True,
            sampler=sampler, logits_processors=processors, completion_batch_size=args.batch,
        )
        per_sample = (time.perf_counter() - t0) / len(chunk)

        # Full prompt and generated token ids, for activation capture later (phase 1).
        with open(f"results/texts_{args.tag}.jsonl", "a", encoding="utf-8") as f:
            for p, prompt, ids in zip(chunk, prompts, resp.token_ids):
                f.write(json.dumps({"id": p["id"], "prompt_ids": prompt, "gen_ids": ids}) + "\n")

        with open(out_path, "a", encoding="utf-8") as f:
            for p, text, ids in zip(chunk, resp.texts, resp.token_ids):
                closed = think_end in ids
                pred = last_boxed(text.split("</think>")[-1]) if closed else last_boxed(text)
                row = {
                    "id": p["id"], "tokens": len(ids), "closed_thinking": closed,
                    "cut": len(ids) >= args.max_tokens and not closed,
                    "repetition": round(repetition(ids), 3),
                    "pred": pred, "answer": p.get("answer"),
                    "correct": same_answer(pred, p["answer"]) if "answer" in p else None,
                    "seconds": round(per_sample, 2),
                    "tail": text[-1500:],
                }
                done[p["id"]] = row
                f.write(json.dumps(row) + "\n")
        print(f"chunk {start // args.batch + 1}: {len(chunk)} samples saved")

    rows = [done[p["id"]] for p in problems if p["id"] in done]
    graded = [r for r in rows if r["correct"] is not None]
    n = len(rows)
    seconds = sum(r.get("seconds", 0) for r in rows)
    summary = {
        "tag": args.tag, "model": args.model, "n": n, "max_tokens": args.max_tokens,
        "accuracy": sum(r["correct"] for r in graded) / len(graded) if graded else None,
        "cut_rate": sum(r["cut"] for r in rows) / n,
        "closed_rate": sum(r["closed_thinking"] for r in rows) / n,
        "loop_rate": sum(r["repetition"] >= 0.5 for r in rows) / n,
        "mean_repetition": sum(r["repetition"] for r in rows) / n,
        "mean_tokens": sum(r["tokens"] for r in rows) / n,
        "total_tokens": sum(r["tokens"] for r in rows),
        "seconds": round(seconds, 1),
        "throughput_tok_s": round(sum(r["tokens"] for r in rows) / seconds, 1) if seconds else None,
    }
    with open(f"results/loop_eval_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
