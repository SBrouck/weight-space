"""Estimate a "loop direction" per layer from saved generations, by teacher-forced reading.

For each saved sample (results/texts_<tag>.jsonl, joined with loop_eval_<tag>.jsonl):
- loop samples (repetition >= 0.5): find the loop onset t0, then take positions before the loop
  ("productive") and positions inside it ("loop");
- clean samples (thinking closed, repetition < 0.1): take generated positions as "clean".

The text is re-read in chunks with a KV cache, layer by layer, recording at each MLP:
the neuron activations (input of down_proj) and what the MLP writes to the residual stream.
Per layer: d = mean(loop) - mean(productive) of the MLP output, its separation on held-out samples
(AUC of the projection), its cosine with mean(loop) - mean(clean), and per-neuron statistics
used to pick which down_proj columns to edit.

Usage: uv run python scripts/capture_directions.py models/qwen3.5-2B-base \
           --tags 2b_base_open_n32_t16k 2b_base_gsm_n32_t16k --out results/dirs_2b.npz
"""

from __future__ import annotations

import argparse
import json
import random

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

WINDOW, NGRAM = 512, 16


def loop_onset(ids: list[int]) -> int | None:
    """First index i such that the window ids[i-WINDOW:i] repeats at least half of its 16-grams."""
    for end in range(WINDOW, len(ids) + 1, 64):
        tail = ids[end - WINDOW : end]
        seen, rep = set(), 0
        for k in range(len(tail) - NGRAM + 1):
            g = tuple(tail[k : k + NGRAM])
            rep += g in seen
            seen.add(g)
        if rep / (len(tail) - NGRAM + 1) >= 0.5:
            return end - WINDOW  # the loop is already running at the start of this window
    return None


def forward_capture(text_model, tokens: list[int], keep: set[int], chunk: int = 2048):
    """Teacher-forced pass; return {layer: (acts[k, inter], outs[k, hidden])} for positions in `keep`."""
    inner = text_model.model
    cache = text_model.make_cache()
    acts = {i: [] for i in range(len(inner.layers))}
    outs = {i: [] for i in range(len(inner.layers))}
    for start in range(0, len(tokens), chunk):
        ids = tokens[start : start + chunk]
        sel = [p - start for p in range(start, start + len(ids)) if p in keep]
        h = inner.embed_tokens(mx.array([ids]))
        fa_mask = create_attention_mask(h, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
        for i, (layer, c) in enumerate(zip(inner.layers, cache)):
            x_norm = layer.input_layernorm(h)
            if layer.is_linear:
                r = layer.linear_attn(x_norm, ssm_mask, c)
            else:
                r = layer.self_attn(x_norm, fa_mask, c)
            h = h + r
            mlp = layer.mlp
            z = layer.post_attention_layernorm(h)
            a = nn.silu(mlp.gate_proj(z)) * mlp.up_proj(z)
            o = mlp.down_proj(a)
            h = h + o
            if sel:
                idx = mx.array(sel)
                acts[i].append(a[0, idx].astype(mx.float32))
                outs[i].append(o[0, idx].astype(mx.float32))
        mx.eval(h, *[t[-1] for t in acts.values() if t], *[t[-1] for t in outs.values() if t])
    return {
        i: (np.array(mx.concatenate(acts[i])), np.array(mx.concatenate(outs[i])))
        for i in acts if acts[i]
    }


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (Mann-Whitney)."""
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort() + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--per-set", type=int, default=48, help="positions sampled per set per sample")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    samples = []
    for tag in args.tags:
        evals = {r["id"]: r for r in map(json.loads, open(f"results/loop_eval_{tag}.jsonl"))}
        for t in map(json.loads, open(f"results/texts_{tag}.jsonl")):
            samples.append({**t, **evals[t["id"]], "uid": f"{tag}:{t['id']}"})

    model, _ = load(args.model)
    text_model = model.language_model
    data = {"loop": [], "prod": [], "clean": []}  # per sample: (uid, {layer: (acts, outs)})

    for s in samples:
        gen, n_prompt = s["gen_ids"], len(s["prompt_ids"])
        sets: dict[str, list[int]] = {}
        if s["repetition"] >= 0.5:
            t0 = loop_onset(gen)
            if t0 is None or t0 < 600:
                continue
            prod = range(64, t0 - 256)
            loop = range(t0 + WINDOW, min(t0 + WINDOW + 1536, len(gen)))
            if len(prod) < 64 or len(loop) < 64:
                continue
            sets = {"prod": rng.sample(list(prod), min(args.per_set, len(prod))),
                    "loop": rng.sample(list(loop), min(args.per_set, len(loop)))}
            end = max(loop)
        elif s["closed_thinking"] and s["repetition"] < 0.1:
            span = range(64, len(gen))
            sets = {"clean": rng.sample(list(span), min(args.per_set, len(span)))}
            end = len(gen) - 1
        else:
            continue

        tokens = s["prompt_ids"] + gen[: end + 1]
        keep = {n_prompt + p for ps in sets.values() for p in ps}
        captured = forward_capture(text_model, tokens, keep)
        order = sorted(keep)
        for name, ps in sets.items():
            rows = [order.index(n_prompt + p) for p in ps]
            data[name].append((s["uid"], {i: (a[rows], o[rows]) for i, (a, o) in captured.items()}))
        print(f"{s['uid']}: {', '.join(f'{k}={len(v)}' for k, v in sets.items())}  ({len(tokens)} tokens read)")

    # Dev/holdout split by loop sample (prod and loop of a sample stay together).
    loop_uids = sorted({u for u, _ in data["loop"]})
    rng.shuffle(loop_uids)
    dev = set(loop_uids[: len(loop_uids) // 2])
    print(f"\n{len(loop_uids)} loop samples ({len(dev)} dev), {len(data['clean'])} clean samples")

    n_layers = len(text_model.model.layers)
    stack = lambda name, part, layer, uids=None, which=1: np.concatenate(
        [d[layer][which] for u, d in data[name] if (uids is None or (u in uids) == part)]
    )
    out = {}
    print("\nlayer  |d|/|loop|  AUC(holdout)  cos(d, loop-clean)")
    for l in range(n_layers):
        loop_dev, prod_dev = stack("loop", True, l, dev), stack("prod", True, l, dev)
        loop_ho, prod_ho = stack("loop", False, l, dev), stack("prod", False, l, dev)
        d = loop_dev.mean(0) - prod_dev.mean(0)
        u = d / np.linalg.norm(d)
        score = auc(loop_ho @ u, prod_ho @ u)
        d_clean = stack("loop", True, l).mean(0) - stack("clean", True, l).mean(0)
        cos = float(u @ d_clean / np.linalg.norm(d_clean))
        rel = np.linalg.norm(d) / np.linalg.norm(loop_dev.mean(0))
        print(f"{l:5d}  {rel:9.3f}  {score:12.3f}  {cos:18.3f}")
        # Neuron stats on all loop samples: mean activation in loop vs productive positions.
        a_loop = stack("loop", True, l, which=0).mean(0)
        a_prod = stack("prod", True, l, which=0).mean(0)
        out[f"u_{l}"] = u
        out[f"auc_{l}"] = score
        out[f"cos_clean_{l}"] = cos
        out[f"act_loop_{l}"] = a_loop
        out[f"act_prod_{l}"] = a_prod
    np.savez(args.out, **out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
