"""Judge a finished run: holdout evaluation and comparison with the published edit.

1. Builds the agent's edited checkpoint from <workspace>/edit/.
2. Writes the holdout problems (from run_meta.json, never shown to the agent) to a data file and
   runs scripts/loop_eval.py on base and edited models at --max-tokens.
3. For every edited down_proj tensor, compares the agent's delta with the published edit
   (results/wsp_directions.npz): layer overlap, rank-one share and |cos| of the top direction.

Usage: uv run python harness/finalize.py ~/ws-runs/run1 [--max-tokens 16384]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()
    ws = os.path.abspath(os.path.expanduser(args.workspace))
    meta = json.load(open(ws.rstrip("/") + ".run_meta.json"))
    run = os.path.basename(ws.rstrip("/"))

    os.environ.update(WS_WORKSPACE=ws, WS_BASE_MODEL=meta["base"])
    sys.path.insert(0, HERE)
    from forward import edited_model_dir  # noqa: E402

    ckpt = edited_model_dir()
    rows = {r["id"]: r for r in map(json.loads, open(os.path.join(REPO, meta["data"])))}
    holdout = os.path.join(REPO, "data", f"holdout_{run}.jsonl")
    with open(holdout, "w") as f:
        f.writelines(json.dumps(rows[i]) + "\n" for i in meta["holdout_ids"])

    for name, model in (("base", meta["base"]), ("edited", ckpt)):
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "loop_eval.py"), model,
                        "--data", holdout, "--n", str(len(meta["holdout_ids"])),
                        "--max-tokens", str(args.max_tokens), "--batch", "32",
                        "--tag", f"final_{run}_{name}"], check=True, cwd=REPO)

    # Compare with the published Qwen3.5-0.8B edit (layers 15 and 19 down_proj).
    ref = np.load(os.path.join(REPO, "results", "wsp_directions.npz"))
    sys.path.insert(0, ws)
    os.environ["WS_BASE_SAFETENSORS"] = next(
        os.path.join(meta["base"], f) for f in os.listdir(meta["base"]) if f.endswith(".safetensors"))
    from kernel import load  # noqa: E402

    report = {"run": run, "tensors": {}}
    for f in sorted(os.listdir(os.path.join(ws, "edit"))):
        name = f[:-4]
        delta = np.load(os.path.join(ws, "edit", f)) - load(name)
        entry = {"changed_entries": int(np.count_nonzero(np.abs(delta) > 1e-6)),
                 "relative_change": float(np.linalg.norm(delta) / np.linalg.norm(load(name)))}
        if delta.ndim == 2:
            u, s, _ = np.linalg.svd(delta, full_matrices=False)
            entry["rank1_share"] = float(s[0] ** 2 / np.sum(s ** 2))
            if delta.shape[0] == 1024:
                entry["abs_cos_with_published"] = {
                    k: round(abs(float(u[:, 0] @ ref[k])), 3) for k in ref.files}
        report["tensors"][name] = entry
    for name in ("base", "edited"):
        report[name] = json.load(open(os.path.join(REPO, "results", f"loop_eval_final_{run}_{name}.json")))
    out = os.path.join(REPO, "results", f"final_{run}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
