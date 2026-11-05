# Notes

Your notebook. Keep it current: the next conversation starts from it.

## Constraints
- Forward-pass budget: **3 passes** total (max 32 problems / pass, 8192 max tokens). Very tight, so every pass has to do two jobs.

## Split (made before any edit)
- `work/split.py`: shuffled the 1,500 ids in `dev.jsonl` with `numpy.random.default_rng(0)`. The first 750 are DEV (`work/dev_ids.txt`) and the last 750 are HOLDOUT (`work/holdout_ids.txt`).
- Pass 1 uses the first 32 DEV ids (`work/pass1_ids.txt`). Pass 3 uses the first 32 HOLDOUT ids (`work/pass3_ids.txt`).
- HOLDOUT is used only in pass 3, to confirm.

## Plan
1. **Pass 1**: `generate --model both` on the 32 DEV problems, with a **random control edit** in place. Chosen in advance: zero the `down_proj` columns of 3 random neurons per layer in layers 8–23 (48 neurons, seed 123, `work/control_edit.py`, neurons in `work/control_neurons.json`). The pass gives (a) base accuracy and failure modes (cut/loop), (b) base texts for the read, and (c) the effect of a random edit of the same size as the targeted one.
2. **Pass 2**: drop the control edit, then `read` the base texts, so the activations are base activations. Look for MLP neurons whose activation separates failing generations (loops / cut at the token limit) from healthy ones. Build a targeted edit of the **same form and size** (48 neurons, down_proj columns changed) that suppresses them.
3. **Pass 3**: `generate --model both` on 32 HOLDOUT problems with the targeted edit.

## Log
- **Pass 1 done** (`passes/fwd_1`), 32 DEV problems:
  - base: acc 6/32 = 18.75%, closed 12.5%, cut 87.5%, loop 75%, mean 7,633 tokens.
  - random control (48 neurons zeroed): acc 5/32 = 15.6%, cut 84%, loop 81%. Paired: 1 rescued, 2 regressed. So a random edit of this size is roughly neutral and noisy.
  - Diagnosis: nearly every failure is a self-doubt loop ("Wait, I think I see the problem...") that runs to the 8,192-token limit without closing `</think>`. **The lever is the looping, not the arithmetic.**
- Dropped the control edit (`work/drop_all.py`; rebuild it with `work/control_edit.py`).
- **Pass 2 (next)**: `read --texts passes/fwd_1/texts_base.jsonl --layers 8-23 --positions spread --per 96` with no edit. Analysis: for each looping text, find the loop start from the token ids, then compare neuron activations inside the loop with those before it, in the same text. Neurons that are consistently higher in loops across texts are candidates to dampen.
- **Pass 2 done** (`passes/fwd_2/read.npz`, 3,072 rows, layers 8–23).
  - Loop onset (`work/analyze.py`, `work/onset.json`): the earliest token after which ≥90% of tokens finish a 24-gram already seen in the text. 25/32 base texts loop, with onsets between 730 and 7,769. The 7 "clean" texts include all 4 that closed, plus 3 long ones that rambled without looping.
  - Contrast within each text (loop minus pre-loop, standardized, over 24 texts): hundreds of neurons have t > 15. Loop-related MLP activity is strong and very consistent.
  - Selection (`work/select.py`, result `work/target_neurons.json`): t > 10. **Position control**: in clean texts, the late-minus-early difference must have magnitude < 0.3 SD, which rules out "context is long" neurons. **Selectivity**: loop mean ≥ 3× the mean |act| before the loop and in clean texts. 57 neurons passed, and I kept the top 48 by t·log(sel), across layers 8–23 (most in 10–14).
  - Offline check (`work/target_edit.py`): zeroing them removes ~20% of each edited layer's MLP output inside loops but only ~4% elsewhere.
- **Targeted edit saved in edit/**: down_proj columns of those 48 neurons set to 0. This is the same form and size as the control.
- **Pass 3 (last)**: `generate --model both --ids work/pass3_ids.txt` (32 HOLDOUT problems).
- **Pass 3 done** (`passes/fwd_3`): base 4/32, targeted edit 2/32 (2 rescued, 4 regressed). Cut 94% / 94% and loop 91% / 91%. Median loop onset 3,164 → 2,910; mean repetition 0.832 → 0.833 (`work/look3.py`).
- **Conclusion**: the targeted edit does not reduce looping and gives no gain. The loop-selective neurons are signatures of the loop, not drivers. Both edits just reshuffle which problems loop (noise).
- **Budget spent (3/3).** The targeted edit is left in `edit/` as the final evaluated edit, and it is NOT recommended over base. See `final_report.md`.

## Session 2: budget raised to 12 (9 passes left)
### Idea: reuse the loop detectors as an exit trigger
- The loop-selective neurons are reliable **detectors** (single-neuron AUC loop vs. non-loop ≈ 0.85–0.90, `work/detect.py`), even though ablating them does nothing. So instead of silencing them, I **add** to their down_proj columns a direction that raises the `</think>` logit. Once the model loops, it gets pushed to close its thinking and answer.
- Facts (`work/facts.py`): `</think>` = token 248069 (it appears exactly once in every closed text and never in unclosed ones). There is no lm_head, so the head is tied to `embed_tokens`. In 4–10 of ~28 cut base answers, the gold number already appears in the last 800 characters, so an exit has something to rescue.
- Detectors (`late` set): (20,89) (21,99) (22,59) (22,65) (23,625) (23,362). Each is normalized to mean 1 in loops; the combined score has loop median 1.08 and normal-text mean 0.18, but normal-text p99 is 1.37 (isolated spikes). The bet: a loop keeps the push on for thousands of tokens, while normal text spikes only on single tokens.
- Direction (`work/exit_edit.py`): d ∝ (1+w_norm) ⊙ (E[</think>] − mean E), made orthogonal to the effect on 248044/248046 (end-of-text and end-of-turn), so the model closes its thinking rather than its turn. `</think>` is the token it promotes most (lift 2.37 per unit write per 1/rms; 248044 −0.44, 248046 +0.03).
- Strength S = norm of the total write when the detector score is 1. I can't calibrate it offline (no residual reads), so I run a **dose sweep on DEV**.
- Problems: DEV ids 32–63 → `work/dev_b.txt`, 64–95 → `work/dev_c.txt`, and so on; HOLDOUT ids 32+ are kept for confirmation.
- Control for the final claim: the same detectors and the same S, writing a **random direction** (`exit_edit.py S random`).
- **Pass 4** (dev_b, S=16): base 3/32, edit 0/32. The edit emits `</think>` at about token 35 and then **runs away**: `</think></think>…` for 8,000 tokens. Far too strong.
- **Pass 5** (dev_c, S=4): base 1/32, edit 4/32 (4 rescued, 1 regressed). First `</think>` at a median of ~180 tokens; **27/32 still run away**. The 5 that did not run away gave 4 correct answers, even with exits at 84–378 tokens. So early exit is fine on GSM8K, and **the runaway is the problem**. The logit is very sensitive to this write.
- **Pass 6 (plan)**: read the pass-5 edited texts (layers 12–23, spread). Layers < 20 are unaffected by the edit, so I can find neurons that fire when the current token is `</think>` (runaway positions), versus ordinary loop positions (pass 2) and normal text. Then I make them write **−d** (a suppressor that stops a second `</think>`).
- **Pass 6 done** (`passes/fwd_6/read.npz`). `work/suppress.py`, `work/supp_check.py`:
  - Neurons (14,979) (16,175) (15,873) (19,796) fire on `</think>` in runs. Their 5th percentile there is above their 99.9th percentile in normal thinking, base loops and answer text. They are ~0 at the *first* `</think>`, ramp up over runs of 2–10, and are fully on for runs ≥ 50. So they are "repeated `</think>`" detectors.
  - The loop detectors fire at 1.25 in runaway runs, which is why the runaway feeds itself. No single neuron is quiet both in normal text and after the exit (`work/detect2.py`).
  - Suppressor direction: −(1+w)⊙(E[</think>]−Ē), with the 21 tokens it would raise projected out (lift −2.07; max other +0.30).
  - Net-push table (`work/net.py`; push in units of logit×rms). **S=3, K=8**: base loop median +6.9; base clean median +1.3 (p99 10.8); isolated `</think>` −1.7; runs 2–49 −4.2; runs ≥ 50 −7.8; answer text +1.2. Chosen for pass 7.
- `exit_edit.py S think late K` now adds the suppressor (K = norm of its total write at its level in long runs).
- **Pass 7** (dev_d, S=3 K=8): base 6/32, edit 3/32 (1 rescued, 4 regressed). Runaway runs are shorter (hundreds of `</think>`, not thousands), but `</think>` now gets **inserted in mid-sentence** ("Linda`</think>`" -> No.) by detector spikes in normal text. That corrupts the context, and the loop absorbs it. **Abandoned the `</think>` route**: injecting a special token is too disruptive, and the logit is too sensitive to the write.
### Idea 2: in loops, push down "Wait" instead
- Token ids without a tokenizer (`work/find_tokens.py`: correlation of token counts in the last 120 tokens with word counts in the last 800 characters, over 320 generations, plus embedding neighbours): "Wait" = 13428 (25k uses), 13784 (2k, a variant), " wait" = 3655. "But" = 1921 and "check" = 1716 are not targeted.
- Pushing a natural word *down* is benign: the model picks its next-best natural continuation, which should break the verbatim cycle. Overshooting is fairly safe.
- `work/wait_edit.py S`: the same 6 late detectors write v ∝ −(1+w)⊙mean(E_wait − Ē), with the 35 tokens it would raise projected out (lift ≈ −1.8 on all three; no other token above +0.30).
- **Pass 8** (dev_e, anti-Wait S=8): base 1/32, edit 1/32 (1 rescued, 1 regressed). "Wait" uses fell 3–5× per text, but **loops persisted** in other words ("Maybe the selling price is…", loop 97%). The loop attractor does not depend on a particular word. Dropped.
### Idea 3: in loops, make the model state its answer with `\boxed`
- **The extractor reads `\boxed{}` anywhere in the generation, including unfinished thinking**: cut base answers scored correct contain "I'll put \boxed{625}" inside the loop. So the model doesn't have to close its thinking, only to write its current answer in a box. This is benign LaTeX, and repeats are harmless because the last box is taken.
- Ids (`work/find_boxed.py`): "boxed" = 76367 (corr 0.95). It is preceded by 1088 (a generic " \\", 10k uses, followed by times/text/…/boxed) or 88259 (always followed by boxed), and followed by 90 "{".
- `wait_edit.py S boxed`: the same detectors **raise** {1088, 88259, 76367}, with 23 projections (lift boxed 1.14, 1088 0.38, 88259 0.32; no other token above 0.30). Control: `random_up` (3 random tokens raised).
- **Pass 9** (dev_f, boxed S=8): base 3/32, edit 0/32 (0 rescued, 3 regressed). All 32 texts run away into "boxedboxedboxed…", including ones base solved. **General lesson: any token pushed by the detectors, once emitted, is itself a repetition, so it feeds the detectors and runs away. Strong doses always do this.**
- **Pass 10 (plan)**: back to the only upward DEV signal (`</think>` S=4: 4 vs 1), at **S=2** without the suppressor, on the **same dev_c problems** for a direct dose comparison. If promising: pass 11 = the edit on hold_b, pass 12 = random_up control at the same S on hold_b.
- **Pass 10 done** (dev_c, `</think>` S=2, late detectors, no suppressor): **base 3/32, edit 12/32 (10 rescued, 1 regressed; sign test p≈0.012)**. Closed 91% (base 6%), cut 9% (base 94%), loop 50% (base 91%). The first `</think>` now usually comes inside the loop (~3,500–5,500 tokens), mostly as a single token; 4/32 still run away (n ≥ 90). S was chosen on dev_c (also used at S=4), so this is optimistic.
- **Candidate final edit: `./kpy work/exit_edit.py 2 think late`** (down_proj of layers 20–23; 6 columns changed).
- **Pass 11 (plan)**: the same edit on hold_b (HOLDOUT, never used). **Pass 12**: control = the same detectors and S=2, writing a random direction (`exit_edit.py 2 random late`), on hold_b.
- **Pass 11 done** (hold_b): **base 2/32, edit 9/32 (8 rescued, 1 regressed; sign test p≈0.039)**. Closed 91% vs 6%, cut 9% vs 94%, loop 53% vs 88%. Typical first `</think>` at ~3,500–5,800 tokens; 7/32 still run away (n ≥ 65).
- **Pass 12 done** (hold_b, random-direction control, same detectors, same S=2): base 4/32, control 3/32 (1 rescued, 2 regressed). Closed 12.5%, loop 84%. **The random direction does nothing; the gain comes from the `</think>` direction.**
- Decoding is sampled: base on hold_b gave different texts in passes 11 and 12 (0/32 identical; 2 vs 4 correct). Only paired comparisons are meaningful.
- **Budget spent (12/12). FINAL EDIT in edit/: `./kpy work/exit_edit.py 2 think late`** (rebuilt and verified after pass 12). It changes 6 down_proj columns: L20 n89, L21 n99, L22 n59/n65, L23 n625/n362. See `final_report.md`.
