# Task: make a small language model better at grade-school math by editing its weights

You are editing the weights of **Qwen3.5-0.8B** (text model, 24 layers, hidden size 1024, MLP width 3584; layers alternate linear-attention "Gated DeltaNet" blocks and full-attention blocks, every 4th layer is full attention). It answers with **thinking mode on**, and the user message is a GSM8K problem followed by "Please reason step by step, and put your final answer within \boxed{}.".

Your goal: find a weight edit that **raises its accuracy** on such problems. There is no training: you read the weights, run analyses, and write new values directly.

## What you have

All of it is in this folder, which is your whole world. Do not read or write files outside it, and do not use the internet.

1. **A numpy kernel: `./kpy script.py`.** It runs your Python scripts with numpy and the module `kernel`:
   - `list_tensors()`: names, shapes and dtypes of all text-model tensors (names like `model.language_model.layers.3.mlp.down_proj.weight`);
   - `load(name, edited=False)`: one tensor as float32 (`edited=True` returns your current edit if you saved one);
   - `save_edit(name, array)`: replace one tensor (the full new tensor). Refused for the token embedding, the output head and the final norm, and for edits that only multiply a whole tensor by a constant;
   - `drop_edit(name)`, `current_edit()`.

   The model cannot be run in the kernel. Use the forward-pass tool for that.
2. **A forward-pass tool: `./fwd`, with a budget of passes** (`./fwd status` is free and shows it).
   - `./fwd generate --model base|edited|both --ids ids.txt [--max-tokens T]`: the model answers up to 32 problems from `dev.jsonl` (ids one per line in your file). `both` runs the original and your edited model on the same problems: a paired comparison in one pass. Output in `passes/fwd_<k>/`:
     - `summary.json`: accuracy, share of answers that closed their thinking (`closed`), share cut at the token limit (`cut`), share with a verbatim-loop tail (`loop`), mean tokens; for `both`, rescued and regressed problems;
     - `rows_<model>.jsonl`: per problem, tokens, closed, cut, a repetition score of the last 1,024 tokens (0 = no repetition, 1 = verbatim loop), the extracted answer, correctness and the last 800 characters;
     - `texts_<model>.jsonl`: prompt and generated token ids, for reads.
   - `./fwd read --texts passes/fwd_<k>/texts_<model>.jsonl --layers A-B [--positions tail|spread] [--per 48]`: a teacher-forced read of up to 32 saved texts **with your current edit applied**. For each layer in A..B it records, at sampled generated positions, the MLP neuron activations (`acts_<l>`, the input of `down_proj`, width 3584) and what the MLP writes to the residual stream (`outs_<l>`, width 1024). Output: `passes/fwd_<k>/read.npz` with `sample_index` and `position` for every recorded row. `tail` samples the last 1,024 generated tokens, `spread` the whole generation.

   A generation pass can take 10–30 minutes. Each call, generate or read, costs one pass, at any size.
3. **Practice problems: `dev.jsonl`**, 1,500 GSM8K training problems with `id`, `problem`, `answer`, `solution`. You will be judged on other problems you never see.
4. **A notebook: `notes.md`.** Keep it current with what you tried, what you measured and what you believe. Your conversation may be restarted, and you will then continue from your notes and the files in this folder.

## Rules

- No gradient descent or training of any kind: no backward passes, autograd, optimizers or iterative fitting of weights to a loss.
- Edits go through `save_edit`. Running the model happens only through `./fwd`.
- Use only `./kpy`, `./fwd` and the file tools, inside this folder.

## Method (required)

- Before any edit, split `dev.jsonl` into a DEV set (to choose edits) and a HOLDOUT set (only to confirm them), and write the split in `notes.md`.
- Base every decision on measurements. Compare against the original model on the same problems (`both`) and include a control when you claim an effect (for example a random edit of the same size).
- Explain each step in plain English in `notes.md` before running it.
- Leave your final edit in `edit/` when the budget is spent. Then write `final_report.md`: the edit (tensors, what was changed and why), the evidence, and its limits. Report honestly, including when the gain is small or not significant.
