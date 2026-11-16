"""Recompute every number quoted in the README from the versioned files in results/ and runs/.

Needs neither MLX nor the model weights. Writes results/summary.json and prints it section by
section. Paired comparisons use a two-sided exact sign test on rescued vs regressed problems.

Usage: uv run python scripts/report.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics

from grading import same_answer

THINK_END = 248069  # id of "</think>" in the Qwen3.5 tokenizer (tokenizer.json, added_tokens)
LOOP = 0.5  # repetition score at or above which a sample counts as a verbatim loop


def rows(tag: str) -> list[dict]:
    return [json.loads(l) for l in open(f"results/loop_eval_{tag}.jsonl", encoding="utf-8")]


def share(xs: list[bool]) -> float:
    return round(sum(map(bool, xs)) / len(xs), 4)


def sign_test(a: int, b: int) -> float:
    """Two-sided exact sign test for a rescued and b regressed problems."""
    n, k = a + b, min(a, b)
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 3) if n else 1.0


def paired(base: list[dict], edited: list[dict], key: str) -> dict:
    b = {r["id"]: bool(r[key]) for r in base}
    e = {r["id"]: bool(r[key]) for r in edited}
    assert b.keys() == e.keys(), "paired runs must cover the same problems"
    rescued = sum(e[i] and not b[i] for i in b)
    regressed = sum(b[i] and not e[i] for i in b)
    return {"rescued": rescued, "regressed": regressed, "sign_test_p": sign_test(rescued, regressed)}


def rates(r: list[dict]) -> dict:
    out = {"n": len(r), "cut": share([x["cut"] for x in r]), "closed": share([x["closed_thinking"] for x in r]),
           "mean_tokens": round(statistics.mean(x["tokens"] for x in r))}
    if r[0].get("answer") is not None:
        out["accuracy"] = share([x["correct"] for x in r])
    if "strict_correct" in r[0]:
        out["strict_accuracy"] = share([x["strict_correct"] for x in r])
    if "repetition" in r[0]:
        out["loop"] = share([x["repetition"] >= LOOP for x in r])
    return out


def span(layers: list[dict], key: str, lo: int, hi: int) -> list[float]:
    vals = [x[key] for x in layers if lo <= x["layer"] <= hi]
    return [round(min(vals), 3), round(max(vals), 3)]


def main() -> None:
    s: dict = {}

    # Section 3.1: anatomy of the published edit (from scripts/diff_weights.py).
    a = json.load(open("results/edit_anatomy.json"))
    t = a["tensors"]
    s["anatomy"] = {
        "changed_tensors": f"{a['n_changed_tensors']} of {a['n_tensors']}",
        "changed_share_pct": round(100 * a["changed_share"], 4),
        "per_tensor": {k.split(".")[3]: {"cols": f"{v['changed_cols']} of {v['shape'][1]}",
                                         "relative_change": round(v["relative_change"], 3),
                                         "rank1_share": round(v["rank1_share"], 4),
                                         "corr": round(v["corr_coeff_projection"], 4),
                                         "k_median": round(v["k_median"], 3)} for k, v in t.items()},
        "abs_cos_between_layers": [round(c["abs_cos"], 3) for c in a["direction_cosines"]],
    }

    # Section 3.2: effect of the token budget, Qwen3.5-0.8B on GSM8K.
    budgets = {"4k": ("base_n64_t4096", "wsp_n64_t4096"), "8k": ("08_base_n32_t8k", "08_wsp_n32_t8k"),
               "16k": ("base_n32_t16k", "wsp_n32_t16k")}
    s["budget"] = {}
    for name, (b, w) in budgets.items():
        rb, rw = rows(b), rows(w)
        s["budget"][name] = {"base": rates(rb), "published": rates(rw),
                             "accuracy_paired": paired(rb, rw, "correct")}
    w16 = rows("wsp_n32_t16k")
    s["budget"]["16k"]["published_correct_median_tokens"] = statistics.median(
        x["tokens"] for x in w16 if x["correct"])

    # Section 3.3: Qwen3.5-2B baseline (GSM8K and open prompts).
    kinds = {r["id"]: r["kind"] for r in map(json.loads, open("data/open_prompts.jsonl", encoding="utf-8"))}
    op = rows("2b_base_open_n32_t16k")
    by_kind = {}
    for kind in ("enumeration", "other"):
        sub = [x for x in op if kinds[x["id"]] == kind]
        by_kind[kind] = {"loops": f"{sum(x['repetition'] >= LOOP for x in sub)} of {len(sub)}",
                         "closed": f"{sum(x['closed_thinking'] for x in sub)} of {len(sub)}"}
    s["qwen2b_baseline"] = {"gsm8k": rates(rows("2b_base_gsm_n32_t16k")), "open_prompts": by_kind}

    # Section 3.3: loop and conclude directions (from scripts/export_directions.py).
    d = json.load(open("results/directions_2b.json"))["layers"]
    s["qwen2b_directions"] = {
        "loop_auc_layers_8_15": span(d, "loop_auc", 8, 15),
        "loop_cos_clean_layers_8_15": span(d, "loop_cos_clean", 8, 15),
        "conclude_auc_layers_4_22": span(d, "conclude_auc", 4, 22),
        "conclude_cos_neg_loop_layers_9_15": span(d, "conclude_cos_neg_loop", 9, 15),
        "conclude_cos_neg_loop_layers_20_21": span(d, "conclude_cos_neg_loop", 20, 21),
    }

    # Section 3.3: edit screens on unseen open prompts (Fig. 5).
    screens = {"first": ["ho_qwen3.5-2B-base_t6k", "ho_2b-L12-13-k6_t6k", "ho_2b-L12-13-k6-random_t6k"],
               "second": ["ho2_qwen3.5-2B-base_t6k", "ho2_2b-conc-L20-21-k3_t6k",
                          "ho2_2b-conc-L16-17-k3_t6k", "ho2_2b-conc-L20-21-k3-random_t6k"]}
    s["qwen2b_screens"] = {}
    for name, tags in screens.items():
        s["qwen2b_screens"][name] = {}
        for tag in tags:
            r = rows(tag)
            en, cl = [x for x in r if "enum" in x["id"]], [x for x in r if "clean" in x["id"]]
            s["qwen2b_screens"][name][tag] = {
                "enum_closed": f"{sum(x['closed_thinking'] for x in en)} of {len(en)}",
                "enum_loops": f"{sum(x['repetition'] >= LOOP for x in en)} of {len(en)}",
                "enum_mean_repetition": round(statistics.mean(x["repetition"] for x in en), 2),
                "clean_closed": f"{sum(x['closed_thinking'] for x in cl)} of {len(cl)}"}

    # Section 3.4: the editor agent's run (runs/run1).
    passes = {}
    for k in range(1, 13):
        try:
            p = json.load(open(f"runs/run1/passes/fwd_{k}/summary.json"))
        except FileNotFoundError:  # passes 2 and 6 were activation reads, with no summary
            continue
        passes[k] = {"base_correct": round(p["base"]["accuracy"] * p["base"]["n"]),
                     "edited_correct": round(p["edited"]["accuracy"] * p["edited"]["n"]),
                     "n": p["base"]["n"], "edited_closed": p["edited"]["closed"],
                     "minutes_per_model": round(p["base"]["seconds"] / 60, 1), **p.get("paired", {})}
    results = [json.loads(l) for l in open("runs/run1/transcript.jsonl", encoding="utf-8")]
    results = [r for r in results if r.get("type") == "result"]
    s["agent_run"] = {
        "generation_passes": passes,
        "median_minutes_per_32_problems_8k": statistics.median(p["minutes_per_model"] for p in passes.values()),
        "rounds": len(results), "turns": sum(r["num_turns"] for r in results),
        "hours": round(sum(r["duration_ms"] for r in results) / 3.6e6, 2),
        # total_cost_usd is cumulative over a resumed session: the last row's modelUsage carries the
        # first round's tokens and costs forward, and total_cost_usd equals the sum of its costUSD.
        # num_turns and duration_ms are per round, hence summed above.
        "cost_usd_list": round(results[-1]["total_cost_usd"], 2),
    }

    # Section 3.5: independent holdout check (64 unseen GSM8K problems, 8k tokens).
    base = rows("final_run1_base")
    s["holdout"] = {"base": rates(base)}
    for name in ("edited", "published"):
        r = rows(f"final_run1_{name}")
        s["holdout"][name] = {**rates(r), "strict_paired": paired(base, r, "strict_correct")}
    texts = {x["id"]: x["gen_ids"] for x in map(json.loads, open("results/texts_final_run1_edited.jsonl"))}
    closed = [x for x in rows("final_run1_edited") if x["closed_thinking"]]
    s["holdout"]["edited"]["closed_answers"] = len(closed)
    s["holdout"]["edited"]["closed_with_loop_tail"] = sum(x["repetition"] >= LOOP for x in closed)
    s["holdout"]["edited"]["closed_with_repeated_think_end"] = sum(texts[x["id"]].count(THINK_END) > 1
                                                                   for x in closed)
    f = json.load(open("results/final_run1.json"))
    cos = [c for v in f["tensors"].values() for c in v.get("abs_cos_with_published", {}).values()]
    s["holdout"]["edited"]["edit"] = {
        "layers": sorted(int(k.split(".")[3]) for k in f["tensors"]),
        "columns": sum(v["changed_entries"] for v in f["tensors"].values()) // 1024,
        "abs_cos_with_published": [min(cos), max(cos)]}

    # Consistency check: stored grades must equal the current grader's (scripts/regrade.py); expect [].
    s["grade_mismatches"] = []
    for path in sorted(glob.glob("results/loop_eval_*.jsonl")):
        for r in map(json.loads, open(path, encoding="utf-8")):
            if r.get("answer") is not None and same_answer(r["pred"], r["answer"]) != bool(r["correct"]):
                s["grade_mismatches"].append({"file": os.path.basename(path), "id": r["id"], "pred": r["pred"],
                                     "answer": r["answer"], "stored_correct": r["correct"],
                                     "closed_thinking": r["closed_thinking"]})

    with open("results/summary.json", "w") as fh:
        json.dump(s, fh, indent=1)
    for section, body in s.items():
        print(f"\n== {section}\n{json.dumps(body, indent=1)}")
    print("\nsaved results/summary.json")


if __name__ == "__main__":
    main()
