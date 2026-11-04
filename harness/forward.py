"""Forward-pass tool for the editor agent. Every call costs one pass from the budget.

  ./fwd generate --model edited|base|both --ids ids.txt [--max-tokens T]
      Answer up to MAX_PROBLEMS dev problems (thinking on, the paper's sampling). "both" runs base
      and edited on the same problems for a paired comparison. Writes passes/fwd_<k>/summary.json,
      rows_<model>.jsonl (per problem: tokens, closed, cut, repetition, correct, tail) and
      texts_<model>.jsonl (token ids, for ./fwd read).
  ./fwd read --texts passes/fwd_<k>/texts_base.jsonl --layers 8-20 [--positions tail|spread] [--per 48]
      Teacher-forced read of up to MAX_TEXTS saved texts with the current edit applied. Records,
      per layer, the MLP neuron activations and MLP outputs at sampled generated positions.
      Writes passes/fwd_<k>/read.npz (keys acts_<layer>, outs_<layer>, sample_index, position).
  ./fwd status
      Budget used and left (free).

Environment: WS_WORKSPACE (agent folder), WS_BASE_MODEL (base model dir), WS_BUDGET, WS_MAX_TOKENS.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from capture_directions import forward_capture  # noqa: E402
from loop_eval import SUFFIX, last_boxed, repetition, same_answer  # noqa: E402

WS = os.environ["WS_WORKSPACE"]
BASE_MODEL = os.environ["WS_BASE_MODEL"]
BUDGET = int(os.environ.get("WS_BUDGET", "12"))
MAX_TOKENS = int(os.environ.get("WS_MAX_TOKENS", "8192"))
MAX_PROBLEMS, MAX_TEXTS = 32, 32
LEDGER = os.path.join(WS, ".budget.json")


def ledger() -> dict:
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return {"budget": BUDGET, "used": 0, "passes": []}


def spend(kind: str, detail: dict) -> int:
    """Reserve one pass; return its index. Refuses when the budget is exhausted."""
    led = ledger()
    if led["used"] >= led["budget"]:
        sys.exit(f"budget exhausted ({led['used']}/{led['budget']} passes used)")
    led["used"] += 1
    k = led["used"]
    led["passes"].append({"k": k, "kind": kind, "detail": detail, "started": time.time()})
    json.dump(led, open(LEDGER, "w"), indent=2)
    return k


def edited_model_dir() -> str:
    """Write base + edit/*.npy to a scratch checkpoint and return its path (base if no edit)."""
    edit_dir = os.path.join(WS, "edit")
    names = sorted(f[:-4] for f in os.listdir(edit_dir) if f.endswith(".npy")) if os.path.isdir(edit_dir) else []
    if not names:
        return BASE_MODEL
    out = os.path.join(WS, ".edited_ckpt")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    shard = next(f for f in os.listdir(BASE_MODEL) if f.endswith(".safetensors"))
    weights = mx.load(os.path.join(BASE_MODEL, shard))
    for n in names:
        weights[n] = mx.array(np.load(os.path.join(edit_dir, f"{n}.npy"))).astype(mx.bfloat16)
    for f in os.listdir(BASE_MODEL):
        if not f.endswith(".safetensors") and os.path.isfile(os.path.join(BASE_MODEL, f)):
            shutil.copy(os.path.join(BASE_MODEL, f), out)
    mx.save_safetensors(os.path.join(out, shard), weights, metadata={"format": "pt"})
    return out


def run_generation(model_dir: str, problems: list[dict], max_tokens: int):
    model, tok = load(model_dir)
    think_end = tok.convert_tokens_to_ids("</think>")
    prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + SUFFIX}],
                                       add_generation_prompt=True, enable_thinking=True)
               for p in problems]
    resp = batch_generate(model, tok, prompts, max_tokens=max_tokens, return_token_ids=True,
                          sampler=make_sampler(temp=0.6, top_p=0.95, top_k=20),
                          logits_processors=make_logits_processors(presence_penalty=1.5),
                          completion_batch_size=len(problems))
    rows, texts = [], []
    for p, prompt, text, ids in zip(problems, prompts, resp.texts, resp.token_ids):
        closed = think_end in ids
        pred = last_boxed(text.split("</think>")[-1]) if closed else last_boxed(text)
        rows.append({"id": p["id"], "tokens": len(ids), "closed": closed,
                     "cut": len(ids) >= max_tokens and not closed,
                     "repetition": round(repetition(ids), 3), "pred": pred,
                     "correct": same_answer(pred, p["answer"]), "tail": text[-800:]})
        texts.append({"id": p["id"], "prompt_ids": prompt, "gen_ids": ids})
    return rows, texts


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, "accuracy": sum(r["correct"] for r in rows) / n,
            "closed": sum(r["closed"] for r in rows) / n, "cut": sum(r["cut"] for r in rows) / n,
            "loop": sum(r["repetition"] >= 0.5 for r in rows) / n,
            "mean_tokens": round(sum(r["tokens"] for r in rows) / n)}


def cmd_generate(args) -> None:
    dev = {p["id"]: p for p in map(json.loads, open(os.path.join(WS, "dev.jsonl")))}
    ids = [line.strip() for line in open(args.ids) if line.strip()]
    unknown = [i for i in ids if i not in dev]
    if unknown:
        sys.exit(f"unknown dev ids: {unknown[:5]}")
    if len(ids) > MAX_PROBLEMS:
        sys.exit(f"at most {MAX_PROBLEMS} problems per pass (got {len(ids)})")
    max_tokens = min(args.max_tokens or MAX_TOKENS, MAX_TOKENS)
    k = spend("generate", {"model": args.model, "n": len(ids), "max_tokens": max_tokens})
    out = os.path.join(WS, "passes", f"fwd_{k}")
    os.makedirs(out, exist_ok=True)
    problems = [dev[i] for i in ids]
    summary = {"pass": k, "max_tokens": max_tokens, "edit": None}
    which = ["base", "edited"] if args.model == "both" else [args.model]
    results = {}
    for m in which:
        model_dir = BASE_MODEL if m == "base" else edited_model_dir()
        if m == "edited":
            summary["edit"] = sorted(os.listdir(os.path.join(WS, "edit"))) if os.path.isdir(os.path.join(WS, "edit")) else []
        t0 = time.perf_counter()
        rows, texts = run_generation(model_dir, problems, max_tokens)
        results[m] = rows
        summary[m] = {**summarise(rows), "seconds": round(time.perf_counter() - t0)}
        with open(os.path.join(out, f"rows_{m}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        with open(os.path.join(out, f"texts_{m}.jsonl"), "w") as f:
            f.writelines(json.dumps(t) + "\n" for t in texts)
    if args.model == "both":
        b, e = results["base"], results["edited"]
        summary["paired"] = {"rescued": sum(not x["correct"] and y["correct"] for x, y in zip(b, e)),
                             "regressed": sum(x["correct"] and not y["correct"] for x, y in zip(b, e))}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


def cmd_read(args) -> None:
    texts = [json.loads(l) for l in open(args.texts)]
    if len(texts) > MAX_TEXTS:
        sys.exit(f"at most {MAX_TEXTS} texts per pass (got {len(texts)})")
    lo, hi = (int(x) for x in args.layers.split("-"))
    k = spend("read", {"texts": args.texts, "n": len(texts), "layers": args.layers})
    out = os.path.join(WS, "passes", f"fwd_{k}")
    os.makedirs(out, exist_ok=True)
    model, _ = load(edited_model_dir())
    rng = random.Random(k)
    arrays: dict[str, list] = {f"{w}_{l}": [] for l in range(lo, hi + 1) for w in ("acts", "outs")}
    index, positions = [], []
    for i, t in enumerate(texts):
        gen = t["gen_ids"]
        span = range(max(0, len(gen) - 1024), len(gen)) if args.positions == "tail" else range(len(gen))
        ps = sorted(rng.sample(list(span), min(args.per, len(span))))
        n_prompt = len(t["prompt_ids"])
        cap = forward_capture(model.language_model, t["prompt_ids"] + gen[: ps[-1] + 1],
                              {n_prompt + p for p in ps})
        for l in range(lo, hi + 1):
            arrays[f"acts_{l}"].append(cap[l][0].astype(np.float16))
            arrays[f"outs_{l}"].append(cap[l][1].astype(np.float16))
        index += [i] * len(ps)
        positions += ps
    np.savez(os.path.join(out, "read.npz"), sample_index=np.array(index), position=np.array(positions),
             **{key: np.concatenate(v) for key, v in arrays.items()})
    print(json.dumps({"pass": k, "texts": len(texts), "positions": len(index),
                      "file": os.path.relpath(os.path.join(out, "read.npz"), WS)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="fwd")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", choices=["base", "edited", "both"], required=True)
    g.add_argument("--ids", required=True)
    g.add_argument("--max-tokens", type=int)
    r = sub.add_parser("read")
    r.add_argument("--texts", required=True)
    r.add_argument("--layers", default="0-23")
    r.add_argument("--positions", choices=["tail", "spread"], default="spread")
    r.add_argument("--per", type=int, default=48)
    sub.add_parser("status")
    args = parser.parse_args()
    os.chdir(WS)
    if args.cmd == "status":
        led = ledger()
        print(json.dumps({"used": led["used"], "budget": led["budget"], "max_tokens": MAX_TOKENS,
                          "max_problems": MAX_PROBLEMS, "max_texts": MAX_TEXTS}, indent=2))
    elif args.cmd == "generate":
        cmd_generate(args)
    else:
        cmd_read(args)


if __name__ == "__main__":
    main()
