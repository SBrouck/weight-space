"""Estimate a "conclude" direction per layer: what the MLPs write just before the model closes thinking.

Positive set: the 64 tokens before </think> in samples that close their thinking (repetition < 0.5).
Negative set: positions inside the loop of looping samples (as in capture_directions.py).
Per layer: c = unit(mean(conclude) - mean(spiral)) of the MLP output, AUC on held-out samples,
cosine with the loop direction (if --loop-dirs is given), and per-neuron mean activations in both
sets, saved as act_pos_<l> / act_neg_<l> for apply_edit.py --mode amplify.

Usage: uv run python scripts/capture_conclude.py models/qwen3.5-2B-base \
           --tags 2b_base_open_n32_t16k 2b_base_gsm_n32_t16k --loop-dirs results/dirs_2b.npz \
           --out results/conclude_2b.npz
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
from mlx_lm import load

from capture_directions import WINDOW, auc, forward_capture, loop_onset

BEFORE = 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--per-set", type=int, default=48)
    parser.add_argument("--loop-dirs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    samples = []
    for tag in args.tags:
        evals = {r["id"]: r for r in map(json.loads, open(f"results/loop_eval_{tag}.jsonl"))}
        for t in map(json.loads, open(f"results/texts_{tag}.jsonl")):
            samples.append({**t, **evals[t["id"]], "uid": f"{tag}:{t['id']}"})

    model, tokenizer = load(args.model)
    text_model = model.language_model
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    data = {"pos": [], "neg": []}  # (uid, {layer: (acts, outs)})

    for s in samples:
        gen, n_prompt = s["gen_ids"], len(s["prompt_ids"])
        if s["closed_thinking"] and s["repetition"] < 0.5:
            close = gen.index(think_end)
            if close < BEFORE + 16:
                continue
            name, ps = "pos", list(range(close - BEFORE, close))
        elif s["repetition"] >= 0.5:
            t0 = loop_onset(gen)
            if t0 is None:
                continue
            span = list(range(t0 + WINDOW, min(t0 + WINDOW + 1536, len(gen))))
            if len(span) < 64:
                continue
            name, ps = "neg", rng.sample(span, min(args.per_set, len(span)))
        else:
            continue
        ps = sorted(ps)
        tokens = s["prompt_ids"] + gen[: ps[-1] + 1]
        captured = forward_capture(text_model, tokens, {n_prompt + p for p in ps})
        data[name].append((s["uid"], captured))
        print(f"{s['uid']}: {name}={len(ps)}  ({len(tokens)} tokens read)")

    # Held-out split by sample, independently in each set.
    split = {}
    for name in data:
        uids = sorted(u for u, _ in data[name])
        rng.shuffle(uids)
        split[name] = set(uids[: len(uids) // 2])
    print(f"\n{len(data['pos'])} concluding samples, {len(data['neg'])} looping samples")

    def stack(name: str, layer: int, dev: bool | None, which: int = 1) -> np.ndarray:
        return np.concatenate([d[layer][which] for u, d in data[name]
                               if dev is None or (u in split[name]) == dev])

    loop_dirs = np.load(args.loop_dirs) if args.loop_dirs else None
    out = {}
    print("\nlayer  AUC(holdout)  cos(c, -loop_u)")
    for l in range(len(text_model.model.layers)):
        c = stack("pos", l, True).mean(0) - stack("neg", l, True).mean(0)
        c /= np.linalg.norm(c)
        score = auc(stack("pos", l, False) @ c, stack("neg", l, False) @ c)
        cos = float(-c @ loop_dirs[f"u_{l}"]) if loop_dirs is not None else float("nan")
        print(f"{l:5d}  {score:12.3f}  {cos:15.3f}")
        out[f"u_{l}"] = c
        out[f"auc_{l}"] = score
        out[f"act_pos_{l}"] = stack("pos", l, None, which=0).mean(0)
        out[f"act_neg_{l}"] = stack("neg", l, None, which=0).mean(0)
    np.savez(args.out, **out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
