# Final report: Qwen3.5-0.8B weight edit for GSM8K

## Bottom line
The edit in `edit/` makes the model **leave its self-doubt loops by closing its thinking (`</think>`) and answering**. On 32 HOLDOUT problems it raised accuracy from **2/32 to 9/32** (8 rescued, 1 regressed; two-sided sign test p ≈ 0.039). On the same problems, a control edit of identical form and size writing a random direction did nothing: 3/32 against 4/32 for base (1 rescued, 2 regressed). Absolute accuracy is still low (~28%), because the base model is crippled by looping in this setup and the edit only rescues part of it. The sample is small: see the limits below.

## The setting
With thinking on and an 8,192-token limit, the base model rarely finishes. Across 7 base runs of 32 problems, accuracy was 1–6/32, about 90% of answers were cut at the limit, and 75–94% ended in a verbatim loop ("Wait, I think I see the problem…" or "Let's check if there's a typo…", repeated). Accuracy is limited by looping, not by arithmetic. Decoding is sampled: the same 32 base problems gave 2/32 and 4/32 in two passes, with no identical texts.

Split: the 1,500 ids were shuffled with seed 0. The first 750 are DEV and the last 750 are HOLDOUT (`work/dev_ids.txt`, `work/holdout_ids.txt`). HOLDOUT problems were used only in passes 3, 11 and 12.

## The edit
**Tensors**: `model.language_model.layers.{20,21,22,23}.mlp.down_proj.weight`. **Six columns** are changed: L20 n89, L21 n99, L22 n59 and n65, L23 n625 and n362. Nothing else is touched. Rebuild with `./kpy work/exit_edit.py 2 think late`.

**What it does.** Each of the six neurons is a **loop detector**. I found them by reading activations of base generations (pass 2): they are low in normal reasoning and switch on once the text starts repeating itself verbatim (AUC ≈ 0.86–0.89 each; combined score has median 1.08 inside loops and mean 0.18 elsewhere). To each neuron's output column I **added** a vector along a direction that raises the `</think>` logit:

- d ∝ (1 + w_finalnorm) ⊙ (E[`</think>`] − mean E). E is the tied embedding (= output head), and `</think>` is token 248069, identified as the token that appears exactly once in every closed generation.
- d is made orthogonal to the effect on the end-of-text and end-of-turn tokens (248044, 248046), so the model closes its *thinking* and not its *turn*. `</think>` is the token this direction promotes most.
- Scale: the six additions together have norm S = 2 when the detector score equals its mean inside loops. Each column gets S/6 divided by that neuron's mean loop activation. The added vectors have norms 0.14–0.50, which is 0.5–1.9× the original column norms.

So when the model is in a loop, the detectors now also vote for `</think>`. In normal reasoning they are mostly quiet, and the push is small.

## How I got there (12 forward passes)

| pass | set | what | base | edit | rescued / regressed |
|---|---|---|---|---|---|
| 1 | DEV a | random control: 48 neurons zeroed | 6 | 5 | 1 / 2 |
| 2 | – | read of base texts (layers 8–23) | | | |
| 3 | HOLD a | 48 loop-selective neurons zeroed | 4 | 2 | 2 / 4 |
| 4 | DEV b | `</think>` push, S=16 | 3 | 0 | 0 / 3 |
| 5 | DEV c | `</think>` push, S=4 | 1 | 4 | 4 / 1 |
| 6 | – | read of pass-5 edited texts (layers 12–23) | | | |
| 7 | DEV d | `</think>` S=3 + repeat suppressor | 6 | 3 | 1 / 4 |
| 8 | DEV e | loop detectors push "Wait" down, S=8 | 1 | 1 | 1 / 1 |
| 9 | DEV f | loop detectors push `\boxed` up, S=8 | 3 | 0 | 0 / 3 |
| 10 | DEV c | **`</think>` push, S=2** | 3 | **12** | 10 / 1 |
| 11 | HOLD b | **`</think>` push, S=2 (final)** | 2 | **9** | **8 / 1** |
| 12 | HOLD b | control: same detectors, S=2, random direction | 4 | 3 | 1 / 2 |

What each step taught me:
- **Silencing loop neurons (pass 3) does nothing.** They report the loop but don't drive it. They are, however, reliable *detectors*, which led to the reuse idea.
- **Too strong a push runs away (passes 4, 5, 9).** Once a pushed token is emitted, its repetition is itself a "loop", so the detectors push it again: `</think></think>…` or `boxedboxed…` for 8,000 tokens. Strong doses also fire on detector spikes in normal text and insert `</think>` in mid-sentence (pass 7). A suppressor built from "repeated `</think>`" neurons shortened the runaways but didn't fix the mid-sentence insertions.
- **Lexical suppression (pass 8)** cut "Wait" 3–5× but the loops simply continued in other words.
- **At S = 2 (pass 10)**, the first `</think>` usually comes *inside* the loop (median ≈ 4,500 tokens), mostly as a single token, followed by a normal answer. Closed rose from 6% to 91%, and cut fell from 94% to 9%.

## Evidence for the final edit (HOLDOUT, pass 11, paired on the same 32 problems)
- Accuracy 2/32 → 9/32. 8 rescued, 1 regressed (sign test p ≈ 0.039).
- Closed thinking 6% → 91%, cut 94% → 9%, verbatim-loop tail 88% → 53%, mean tokens 7,769 → 7,102.
- **Control (pass 12, same problems):** the same six columns with the same write size but a random direction gave 3/32 against 4/32 for base, and loops stayed at 84%. The effect is specific to the `</think>` direction.
- DEV replication (pass 10, dev_c): 3/32 → 12/32 (10 / 1). S was chosen on these problems, since S = 4 had also been run on dev_c, so this number is optimistic. HOLDOUT is the clean estimate.

## Limits
- **Small samples.** 32 HOLDOUT problems, one sampled answer each. The 95% interval on the HOLDOUT gain is wide: roughly +2 to +12 problems out of 32. p ≈ 0.04 is a single test, and some dose choice happened on DEV. I'd expect the true gain to be real but perhaps smaller than +22 points.
- **It rescues loops; it doesn't make the model reason better.** Absolute accuracy stays around 28–38%. About 7/32 HOLDOUT answers still run away (`</think>` repeated hundreds of times), and some exits produce a wrong answer.
- **It depends on the setting.** The gain comes from converting cut answers into finished ones under an 8,192-token limit. With a much larger budget, or a model that rarely loops, the effect would be smaller. With a tighter one it may be larger. One regression on HOLDOUT (a problem base solved in 1,537 tokens) shows that it can occasionally disturb a healthy generation.
- **The dose is sensitive.** S = 4 is already harmful through runaways. S = 2 was the only dose tried in the good range. A slightly lower dose might work better, untested.
- The detector neurons and the loop-onset rule come from 32 base texts, and the norm convention for the final RMSNorm (1 + w) was assumed (Qwen3-Next family), not verified.
