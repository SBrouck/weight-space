# Final report: Qwen3.5-0.8B weight edit for GSM8K

## Bottom line
**I did not find an edit that raises accuracy.** The edit left in `edit/` (48 "loop-selective" MLP neurons ablated) scored **2/32 on HOLDOUT, against 4/32 for the original model** (2 rescued, 4 regressed). It left looping unchanged. A random edit of the same size did about the same on DEV (5/32 against 6/32). Neither difference is anywhere near significant: a sign test on 2 vs 4 gives p ≈ 0.69, and on 1 vs 2, p = 1.0. I do **not** recommend this edit over the original weights. It stays in `edit/` because the task asks for the final evaluated edit to be left there. Removing the 13 files in `edit/` restores the base model.

## Budget and design
Only 3 forward passes were available (≤32 problems each), so every pass did two jobs:
1. **Pass 1**: `both` on 32 DEV problems, with a **random control edit chosen in advance** (3 random neurons per layer in layers 8–23 zeroed in `down_proj`, seed 123). This gave the base failure modes, the base texts for the read, and the control effect.
2. **Pass 2**: `read` of the 32 base texts, with no edit, layers 8–23, `spread`, 96 positions per text.
3. **Pass 3**: `both` on 32 HOLDOUT problems with the targeted edit.

Split: the 1,500 ids were shuffled with seed 0. The first 750 are DEV and the last 750 are HOLDOUT (`work/dev_ids.txt`, `work/holdout_ids.txt`). HOLDOUT was used only in pass 3.

## What the base model does wrong
With thinking on and an 8,192-token limit, the base model rarely finishes. On DEV it was 6/32 correct, with 88% cut at the limit and 75% ending in a verbatim loop. On HOLDOUT it was 4/32 correct, 94% cut and 91% looping. The loops are self-doubt cycles ("Wait, I think I see the problem…" repeated). **Accuracy here is limited by looping, not by arithmetic.**

## The edit
- **Tensors**: `model.language_model.layers.{8,9,10,11,12,13,14,15,17,18,20,22,23}.mlp.down_proj.weight`.
- **Change**: the columns of 48 neurons are set to 0, which removes those neurons' writes to the residual stream. The neuron list is in `work/target_neurons.json`.
- **Why these neurons**: for each looping text, I found the loop onset from the token ids. The onset is the earliest token after which ≥90% of tokens finish a 24-gram already seen in the text. I then compared activations inside the loop with those before it, **within the same text**, over 24 texts. The selected neurons:
  - are strongly and consistently higher inside loops (t between 12 and 26 across texts);
  - do not rise late in the 7 non-looping texts (|late − early| < 0.3 SD), which rules out neurons that merely track context length;
  - are selective: their loop mean is at least 3× their mean |activation| elsewhere.

  Offline, the ablation removes about 20% of each edited layer's MLP output inside loops and about 4% elsewhere.
- Code: `work/analyze.py`, `work/select.py`, `work/target_edit.py`.

## Evidence

| | n | accuracy | cut | loop | rescued / regressed |
|---|---|---|---|---|---|
| DEV base (pass 1) | 32 | 18.8% | 87.5% | 75% | |
| DEV random control (pass 1) | 32 | 15.6% | 84.4% | 81% | 1 / 2 |
| HOLDOUT base (pass 3) | 32 | 12.5% | 93.8% | 91% | |
| HOLDOUT targeted edit (pass 3) | 32 | 6.3% | 93.8% | 91% | 2 / 4 |

Mechanism check (`work/look3.py`): on HOLDOUT, the median loop onset was 3,164 tokens for base and 2,910 for the edit. Mean repetition was 0.832 against 0.833. **The edit did not delay or prevent loops.** Per problem, both edits mostly reshuffle *which* problems loop. Some base successes start looping and some base loops resolve. The net effect is about zero, and this edit's point estimate is slightly negative.

## What I believe, and limits
- The neurons I found are **signatures of the loop state** (they read "I am repeating"), **not causes**. Ablating them does not break the copying, which is most likely carried by attention and induction-style mechanisms.
- With 32 problems per arm and base accuracy around 15%, only a very large effect (more than about +6 problems) would have been detectable. Small real effects, good or bad, are invisible at this scale.
- A single sample per problem, and generation is very sensitive to tiny edits: even the random control changed outcomes on 3/32 problems. Paired differences of 1–3 problems are noise.
- There was no budget to iterate (sign flips, attention edits, or edits that boost the `</think>` direction). With more passes, the next step would be edits aimed at the attention layers or at the decision to close thinking, rather than at neurons that report the loop.
