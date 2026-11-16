# One target per result. Targets marked [model] need the checkpoints in models/ and an Apple
# Silicon Mac with MLX; [long] targets generate text and take from 10 minutes to several hours.
# `make report figures test` needs neither: it reads the versioned files in results/ and runs/.

PY := uv run python
EVAL := $(PY) scripts/loop_eval.py
M08 := models/qwen3.5-0.8B-base
WSP := models/qwen3.5-0.8B-wsp
M2B := models/qwen3.5-2B-base
WS ?= $(HOME)/ws-runs/run2

.PHONY: setup kernel-env models anatomy eval-budget eval-2b directions edits-2b screens-2b \
        workspace agent finalize regrade report figures fig1 fig2 fig3 fig4 fig5 test

setup:                      ## Python environment (uv.lock)
	uv sync

kernel-env:                 ## numpy-only interpreter for the editor agent's kernel (no MLX)
	uv venv --python 3.12 harness/kernel_env
	uv pip install --python harness/kernel_env/bin/python numpy==2.5.4

models:                     ## download the three checkpoints at the revisions used here (about 7.6 GB)
	uv run hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir $(M08)
	uv run hf download RikiyaT/Qwen3.5-0.8B-weight-space-programming --revision d1671a178705bcab7644806d23c535517f35723b --local-dir $(WSP)
	uv run hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir $(M2B)

anatomy:                    ## [model] Sec. 3.1: results/edit_anatomy.json, results/wsp_directions.npz (seconds)
	$(PY) scripts/diff_weights.py $(M08) $(WSP)

eval-budget:                ## [model][long] Sec. 3.2: base vs published edit at 4k, 8k and 16k tokens
	$(EVAL) $(M08) --n 64 --max-tokens 4096 --tag base_n64_t4096
	$(EVAL) $(WSP) --n 64 --max-tokens 4096 --tag wsp_n64_t4096
	$(EVAL) $(M08) --n 32 --max-tokens 8192 --tag 08_base_n32_t8k
	$(EVAL) $(WSP) --n 32 --max-tokens 8192 --tag 08_wsp_n32_t8k
	$(EVAL) $(M08) --n 32 --max-tokens 16384 --tag base_n32_t16k
	$(EVAL) $(WSP) --n 32 --max-tokens 16384 --tag wsp_n32_t16k

eval-2b:                    ## [model][long] Sec. 3.3: Qwen3.5-2B on GSM8K and on open prompts
	$(EVAL) $(M2B) --n 32 --max-tokens 16384 --batch 16 --tag 2b_base_gsm_n32_t16k
	$(EVAL) $(M2B) --data data/open_prompts.jsonl --n 32 --max-tokens 16384 --batch 16 --tag 2b_base_open_n32_t16k

directions:                 ## [model][long] Sec. 3.3: loop and conclude directions -> results/directions_2b.json
	$(PY) scripts/capture_directions.py $(M2B) --tags 2b_base_open_n32_t16k 2b_base_gsm_n32_t16k --out results/dirs_2b.npz
	$(PY) scripts/capture_conclude.py $(M2B) --tags 2b_base_open_n32_t16k 2b_base_gsm_n32_t16k \
	    --loop-dirs results/dirs_2b.npz --out results/conclude_2b.npz
	$(MAKE) edits-2b
	$(PY) scripts/export_directions.py --edits models/2b-*

edits-2b:                   ## [model] the five hand-built 2B edits and their random controls
	$(PY) scripts/apply_edit.py $(M2B) results/dirs_2b.npz --layers 12 13 --k 6 --cols 64 --out models/2b-L12-13-k6
	$(PY) scripts/apply_edit.py $(M2B) results/dirs_2b.npz --layers 12 13 --k 6 --cols 64 --random --out models/2b-L12-13-k6-random
	$(PY) scripts/apply_edit.py $(M2B) results/conclude_2b.npz --mode amplify --layers 20 21 --k 3 --cols 64 --out models/2b-conc-L20-21-k3
	$(PY) scripts/apply_edit.py $(M2B) results/conclude_2b.npz --mode amplify --layers 16 17 --k 3 --cols 64 --out models/2b-conc-L16-17-k3
	$(PY) scripts/apply_edit.py $(M2B) results/conclude_2b.npz --mode amplify --layers 20 21 --k 3 --cols 64 --random --out models/2b-conc-L20-21-k3-random

screens-2b:                 ## [model][long] Sec. 3.3, Fig. 5: edit screens on unseen open prompts
	$(EVAL) $(M2B) --data data/open_holdout.jsonl --n 20 --max-tokens 6144 --tag ho_qwen3.5-2B-base_t6k
	$(EVAL) models/2b-L12-13-k6 --data data/open_holdout.jsonl --n 20 --max-tokens 6144 --tag ho_2b-L12-13-k6_t6k
	$(EVAL) models/2b-L12-13-k6-random --data data/open_holdout.jsonl --n 20 --max-tokens 6144 --tag ho_2b-L12-13-k6-random_t6k
	$(EVAL) $(M2B) --data data/open_holdout.jsonl --n 20 --samples 2 --max-tokens 6144 --tag ho2_qwen3.5-2B-base_t6k
	$(EVAL) models/2b-conc-L20-21-k3 --data data/open_holdout.jsonl --n 20 --samples 2 --max-tokens 6144 --tag ho2_2b-conc-L20-21-k3_t6k
	$(EVAL) models/2b-conc-L16-17-k3 --data data/open_holdout.jsonl --n 20 --samples 2 --max-tokens 6144 --tag ho2_2b-conc-L16-17-k3_t6k
	$(EVAL) models/2b-conc-L20-21-k3-random --data data/open_holdout.jsonl --n 20 --samples 2 --max-tokens 6144 --tag ho2_2b-conc-L20-21-k3-random_t6k

workspace:                  ## Sec. 3.4: a fresh editor workspace in $(WS), outside the repo
	$(PY) harness/setup_workspace.py $(WS) --base $(M08) --budget 12 --max-tokens 8192

agent:                      ## [model][long] Sec. 3.4: run the editor agent (Claude Code CLI, paid API)
	harness/run_agent.sh $(WS) 6 15

finalize:                   ## [model][long] Sec. 3.5: holdout check of the agent's edit -> results/final_<run>.json
	$(PY) harness/finalize.py $(WS) --max-tokens 8192
	$(EVAL) $(WSP) --data data/holdout_$(notdir $(WS)).jsonl --n 64 --max-tokens 8192 --tag final_$(notdir $(WS))_published

regrade:                    ## regrade stored rows with scripts/grading.py (in place)
	$(PY) scripts/regrade.py

report:                     ## every number quoted in the README -> results/summary.json
	$(PY) scripts/report.py

figures: fig1 fig2 fig3 fig4 fig5

fig1:                       ## Fig. 1, from results/edit_anatomy.json
	$(PY) scripts/make_figures.py fig1
fig2:                       ## Fig. 2, from results/loop_eval_{base,wsp,08_*}_*.jsonl
	$(PY) scripts/make_figures.py fig2
fig3:                       ## Fig. 3, from runs/run1/passes/*/summary.json
	$(PY) scripts/make_figures.py fig3
fig4:                       ## Fig. 4, from results/loop_eval_final_run1_*.jsonl
	$(PY) scripts/make_figures.py fig4
fig5:                       ## Fig. 5, from results/loop_eval_ho*_*.jsonl
	$(PY) scripts/make_figures.py fig5

test:
	uv run pytest -q tests
