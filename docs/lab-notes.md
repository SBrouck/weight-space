# Lab notes

The full, chronological record behind the README: every table, every run, in the order it happened.


Gradient-free weight edits that remove thinking loops from small local models, on a Mac with MLX.

Builds on [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming) (Takehi, Teoh, Zhang, Khattab, 2026). In that work, GPT-6 Astra edited Qwen3.5-0.8B's weights without training and cut its thinking loops. Here the goals are:

1. **Phase 0:** reproduce and understand that edit on Apple Silicon.
2. **Phase 1:** find the loop direction by hand in Qwen3.5-2B and 4B, and measure loops outside math.
3. **Phase 2:** build an MLX edit-and-verify harness where a coding agent proposes edits, and gate each edit on dev, holdout and drift checks.

## Phase 0: what the published edit actually is

`scripts/diff_weights.py` compares `Qwen/Qwen3.5-0.8B` (revision `2fc0636`) with `RikiyaT/Qwen3.5-0.8B-weight-space-programming`.

| | Layer 15 `mlp.down_proj` | Layer 19 `mlp.down_proj` |
|---|---|---|
| Changed tensors (of 320) | 1 | 1 |
| Changed columns (MLP neurons) | 64 of 3,584 | 64 of 3,584 |
| Relative change `‖ΔW‖/‖W‖` | 0.106 | 0.081 |
| Rank-one share of ΔW on changed columns | 1.000 | 1.000 |
| corr(Δ coefficient, `u·w_j`) | −1.000 | −1.000 |
| Strength `k` in `w_j' = w_j − k (u·w_j) u` | **6.0** | **6.0** |

`|cos(u₁₅, u₁₉)| = 0.85`, so the two layers share nearly the same direction.

**Reading.**
- Each edited neuron loses its component along one residual-stream direction `u`.
- With `k = 6`, the edit is not just a removal (`k = 1`). It **flips the component and amplifies it 5×**: each of these neurons now writes against the loop direction.
- This matches the sign-inversion edits in *When Can One Neuron Fix Repetition Loops in LLMs?* (Lazaridis et al., [arXiv 2606.13705](https://arxiv.org/abs/2606.13705)). On Gemma 4 E2B, a single sign-inverted neuron was enough.

## Phase 0: loops and speed on an M5 (24 GB)

`scripts/loop_eval.py` uses GSM8K train problems, thinking mode on, and the paper's sampling (temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5). It runs batched MLX generation.

Throughput:
- about 62 tok/s on a single stream;
- about 330 tok/s at batch 8.

Two loop measures are used:
- **cut**: the answer hits the token budget before `</think>`;
- **loop**: in the last 1,024 tokens, at least half of the 16-grams repeat earlier ones. This separates a verbatim loop from a long but progressing chain of thought, even when the answer is cut.

**At a 4,096-token budget the edit barely shows** (64 problems):

| | Base | Edited |
|---|---|---|
| Cut | 92 % | 86 % |
| Accuracy | 7.8 % | 12.5 % |

The edited model's correct answers in the paper have a median of about 9,000 tokens. At 4k, long productive thinking and real loops are both cut, so the budget is too short to see the effect.

**At a 16,384-token budget the effect is clear** (32 problems, paired):

| | Base | Edited |
|---|---|---|
| Loop (repetition ≥ 0.5) | **94 %** | **19 %** |
| Mean repetition score | 0.83 | 0.30 |
| Thinking closed | 6 % | 66 % |
| Cut at budget | 94 % | 34 % |
| Accuracy (strict `\boxed{}`) | 3.1 % | 37.5 % |
| Rescued / regressed | | **11 / 0** |
| Mean tokens | 15,564 | 12,248 |

Remaining failures of the edited model, read from the saved tails:
- **Post-thinking spirals.** Five answers close `</think>`, then keep re-checking until the budget runs out.
- **Unboxed answers.** Two answers finish without `\boxed{}` (one correct, "Each top cost $5"). Strict grading counts them wrong, so accuracy is a lower bound.

Throughput falls as the context grows: 450 tok/s at 4k, 280–380 tok/s at 16k, batch 32. A 32-problem run at 16k takes 17–30 minutes.

**Takeaway for phase 1:** measure at least 16k tokens and score loops with the repetition metric, not only with the cut.

## Phase 1: baseline of Qwen3.5-2B

Same protocol, 16,384-token budget, batch 16, about 160–230 tok/s.

**GSM8K** (32 problems): the 2B loops far less than the 0.8B. Its failure on math is long thinking rather than loops.

| | 0.8B base | 0.8B edited | **2B base** |
|---|---|---|---|
| Loop | 94 % | 19 % | **19 %** |
| Cut at budget | 94 % | 34 % | 56 % |
| Accuracy | 3.1 % | 37.5 % | 46.9 % |

**Open prompts** (`data/open_prompts.jsonl`, 32 prompts, 8 in French): **the 2B loops on almost every enumeration.**

| Prompt type | Loops (repetition ≥ 0.5) | Typical length when it ends |
|---|---|---|
| Exhaustive lists of partly known facts (14, e.g. constellations, Pokémon, French départements) | **13 / 14**, repetition 0.70–0.96 | always cut at 16k |
| Advice, explanation, writing (18) | 3 / 18 (Lisbon trip, micro-entreprise steps, VAT for a child) | 1,200–3,800 tokens, repetition ≈ 0 |

The only enumeration that does not loop, the chemical elements 1–60, does not repeat (0.07) but is still cut. This matches the "doom looping" described by Lazaridis et al.: non-convergent self-correction on facts the model cannot recall.

**Phase 1 target:** remove the enumeration loop in Qwen3.5-2B without hurting the 15 prompts that already end cleanly or GSM8K accuracy. The full prompt and generated token ids are saved in `results/texts_*.jsonl`. The direction can be estimated from these loop and clean contexts without generating again.

## Phase 1: a loop direction exists, but removing it does not fix the failure

**Direction.** `scripts/capture_directions.py` re-reads the saved generations, teacher-forced. It takes positions before the loop onset ("productive") and inside the loop, in the same 19 looping samples, and calls the difference of the MLP-output means `d`. The data is split by sample; on the held-out half, the projection on `d` separates loop from productive positions with AUC 0.98–0.999 in layers 8–15. `d` also has cosine 0.90–0.95 with mean(loop) − mean(clean samples). Caveat: loop positions are always later in the text, so part of the separation may be context length.

**Edit.** `scripts/apply_edit.py`, layers 12 and 13. In each layer, the 64 neurons that push most along `d` in loop positions get `w' = w − 6 (u·w) u`, the published recipe. The control is a random direction on the same columns, rescaled to the same ‖ΔW‖.

**Screen** on unseen prompts (`data/open_holdout.jsonl`), 6,144 tokens, one sample per prompt:

| | Base | Edited | Random control |
|---|---|---|---|
| 12 enumerations: verbatim loops | 11 | 6 | 8 |
| 12 enumerations: mean repetition | 0.72 | 0.46 | 0.62 |
| 12 enumerations: **thinking closed** | **1** | **2** | **2** |
| 8 clean prompts: closed / mean tokens | 8 / 2,360 | 8 / 2,917 | 7 / 2,930 |

**Reading.** The edit changes the form of the failure, not the failure itself. Edited answers stop repeating verbatim, but they keep self-correcting on facts the model does not know ("*Wait, I am hallucinating dates. I need to stop and get the actual facts.*") and still hit the budget. This is the "doom looping" of Lazaridis et al., and it fits the source paper's limit: an edit can remove an obstacle, not add knowledge. With one sample per prompt, edited and random (6 vs 8 loops) are not distinguishable.

**Consequences.**
- The repetition score is a good loop *detector* but a poor *success* criterion. The success criterion is now "closes thinking and answers".
- Next target: amplify a "conclude" direction (positions just before `</think>` in answers that end, against positions in answers that spiral), rather than remove the loop direction. Use several samples per prompt so the control is interpretable.

### Second try: amplify a "conclude" direction

`scripts/capture_conclude.py` contrasts the 64 tokens before `</think>` in answers that end with positions inside loops. The resulting direction `c` separates the two sets on held-out samples (AUC 0.95–0.97 in layers 4–22). In layers 9–15 it is almost the opposite of the loop direction (cos(c, −u) = 0.90–0.92), so "conclude" and "do not loop" are nearly one axis there. It is more distinct in late layers (0.75 in layers 20–21).

Edit: `apply_edit.py --mode amplify`, `w' = w + 3 (c·w) c` on the 64 neurons per layer that push most along `c` when concluding. Screen on `data/open_holdout.jsonl`, 2 samples per prompt, 6,144 tokens:

| | Base | Conclude L20–21 | Conclude L16–17 | Random L20–21 |
|---|---|---|---|---|
| 24 enumeration samples: **thinking closed** | **2** | **2** | **3** | **1** |
| 24 enumeration samples: verbatim loops | 17 | 13 | 17 | 20 |
| 16 clean samples: closed | 16 | 14 | 13 | 15 |

**Reading.** No edit makes the 2B conclude on enumerations, and the clean prompts get slightly worse. Two attempts point the same way: in Qwen3.5-2B, enumeration doom-looping comes from facts the model cannot recall, not from a single-direction defect, and a rank-one weight edit does not fix it.

**Next.** Go back to a case with known ground truth. The 0.8B math loop has a published edit, so phase 2 asks: can a coding agent on a Mac rediscover it, with a harness that only gives it weights, a forward-pass budget and dev and holdout sets?

## Phase 2: can a coding agent rediscover a weight edit on a Mac?

**Harness** (`harness/`), modeled on the source paper's rules:
- **The editor** is Claude Code (Opus 5.5) run headless by `run_agent.sh`. It is never told what the published edit is.
- **What it can use:**
  - a numpy-only kernel (`kernel.py`, in a venv without MLX), to read weights and save edits;
  - `save_edit`, which checks the rules: no embedding, output head or final-norm edits, no whole-tensor scaling;
  - a budgeted forward-pass tool (`forward.py`): `generate` answers 32 problems at 8,192 tokens, base, edited or paired; `read` is a teacher-forced activation read;
  - 1,500 GSM8K dev problems.
- **What it cannot reach:** the holdout ids stay outside its workspace. Reads of the repo, the HF cache and `~/.claude` are denied, and its tools are limited to `./kpy`, `./fwd` and file tools inside the workspace.
- **Final judgment:** `finalize.py` evaluates on 64 holdout problems with a normal and a strict grade. Strict counts only an answer given after `</think>`.

**Run 1:** 12 passes, 91 turns, 3 h 15, about $5.2 at list price.

The agent's own path:
1. Ablating 48 loop-selective neurons did nothing. Its conclusion: the neurons are loop *detectors*, not drivers.
2. It rewired six of these detectors (layers 20–23) to also write a direction that raises the `</think>` logit: an "if looping, close thinking" circuit.
3. A dose sweep found runaways at S ≥ 3–4. It also tried and dropped lexical tricks ("Wait" down, `\boxed` up). The `\boxed` idea exploited a grader loophole: a `\boxed{}` inside unfinished thinking counted.
4. S = 2 won on its own holdout: 2/32 → 9/32, with the random-direction control at 3 vs 4.

Its report is in the workspace; the pilot report is in `run1.pilot_final_report.md`.

**Independent check** on 64 problems it never saw, 8,192 tokens, one sample each, paired:

| | Base | Agent edit (6 columns, L20–23) | Published edit (128 columns, L15/19) |
|---|---|---|---|
| Accuracy | 18.8 % | 25.0 % | 28.1 % |
| **Strict accuracy** (answer after `</think>`) | 15.6 % | 23.4 % | **28.1 %** |
| Strict rescued / regressed (sign test) | | 9 / 4 (p = 0.27) | 12 / 4 (p = 0.08) |
| Thinking closed | 15.6 % | 85.9 % | 40.6 % |
| Verbatim loops | 82.8 % | 60.9 % | **9.4 %** |

**Reading.**
- **The agent found a working but weaker mechanism.** Its edit is a different solution, not a rediscovery: the edited directions have |cos| 0.03–0.04 with the published one. Both edits point the right way, but with 64 problems neither gain is significant, and the agent's (+7.8 points strict) is smaller than the published one's (+12.5).
- **Its "closed" rate is inflated by runaways.** 33 of its 55 closed answers repeat `</think>` in a loop ("…drive to town and`</think>` Let's re-evaluate…"). This is the failure it saw at higher doses, and it is more frequent on unseen problems than its estimate (7/32). The published edit removes the loop itself (9 % loops), whereas the agent's edit gives the model an exit it overuses.
- **Budget matters.** Astra had 50 passes at 32k tokens; this run had 12 at 8k. The agent spent 6 of its 12 passes learning the dose and the runaway failure.

**Next.**
- Rerun with a larger budget, two samples per problem in screens, and a runaway check in `fwd` (count of `</think>`).
- Then point the harness at a fresh target with no known answer.

## Layout

```
scripts/diff_weights.py   diff two checkpoints, test the rank-one projection hypothesis
scripts/speed_test.py     one thinking answer: speed, peak memory, termination
scripts/loop_eval.py      batched loop rate, accuracy and length on GSM8K-style problems
data/gsm8k_train.jsonl    practice problems (GSM8K train, MIT licence)
results/                  summaries and per-sample results
models/                   downloaded checkpoints (git-ignored)
```

## Reproduce

```bash
uv sync
uv run hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/qwen3.5-0.8B-base
uv run hf download RikiyaT/Qwen3.5-0.8B-weight-space-programming --local-dir models/qwen3.5-0.8B-wsp
uv run python scripts/diff_weights.py models/qwen3.5-0.8B-base models/qwen3.5-0.8B-wsp
uv run python scripts/loop_eval.py models/qwen3.5-0.8B-base --n 32 --max-tokens 16384 --tag base_n32_t16k
uv run python scripts/loop_eval.py models/qwen3.5-0.8B-wsp  --n 32 --max-tokens 16384 --tag wsp_n32_t16k
```
