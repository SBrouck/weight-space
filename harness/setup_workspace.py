"""Create an isolated workspace for one editor run.

The workspace holds only what the agent may see: the dev problems, the kernel module, two
wrappers (./kpy runs the agent's numpy scripts, ./fwd spends forward passes), the task and a
notes file. Holdout problem ids stay in run_meta.json next to the workspace, never inside it.

Usage: uv run python harness/setup_workspace.py ~/ws-runs/run1 --base models/qwen3.5-0.8B-base \
           --budget 12 --max-tokens 8192
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import stat
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def write_exec(path: str, body: str) -> None:
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--base", default="models/qwen3.5-0.8B-base")
    parser.add_argument("--data", default="data/gsm8k_train.jsonl")
    parser.add_argument("--dev", default="2000:3500", help="slice of the data file the agent sees")
    parser.add_argument("--holdout", default="5000:5064", help="slice kept for the final evaluation")
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    ws = os.path.abspath(os.path.expanduser(args.workspace))
    if os.path.exists(ws):
        sys.exit(f"{ws} already exists")
    os.makedirs(os.path.join(ws, "edit"))
    os.makedirs(os.path.join(ws, "passes"))
    base = os.path.abspath(os.path.join(REPO, args.base))
    shard = glob.glob(os.path.join(base, "*.safetensors"))[0]

    rows = [json.loads(l) for l in open(os.path.join(REPO, args.data))]
    d0, d1 = (int(x) for x in args.dev.split(":"))
    h0, h1 = (int(x) for x in args.holdout.split(":"))
    assert d1 <= h0 or h1 <= d0, "dev and holdout overlap"
    with open(os.path.join(ws, "dev.jsonl"), "w") as f:
        for r in rows[d0:d1]:
            f.write(json.dumps({k: r[k] for k in ("id", "problem", "answer", "solution")}) + "\n")

    shutil.copy(os.path.join(HERE, "kernel.py"), ws)
    shutil.copy(os.path.join(HERE, "task.md"), ws)
    with open(os.path.join(ws, "notes.md"), "w") as f:
        f.write("# Notes\n\nYour notebook. Keep it current: the next conversation starts from it.\n")

    kernel_py = os.path.join(HERE, "kernel_env", "bin", "python")
    write_exec(os.path.join(ws, "kpy"), f"""#!/bin/sh
# Run one of your numpy scripts in the kernel (no model execution available here).
cd "$(dirname "$0")"
export WS_BASE_SAFETENSORS="{shard}" WS_EDIT_DIR=edit PYTHONPATH=.
exec "{kernel_py}" "$@"
""")
    main_py = os.path.join(REPO, ".venv", "bin", "python")
    write_exec(os.path.join(ws, "fwd"), f"""#!/bin/sh
# Spend forward passes (generate or read). See task.md.
cd "$(dirname "$0")"
export WS_WORKSPACE="$PWD" WS_BASE_MODEL="{base}" WS_BUDGET={args.budget} WS_MAX_TOKENS={args.max_tokens}
exec "{main_py}" "{os.path.join(HERE, 'forward.py')}" "$@" 2> .fwd_stderr.log
""")

    meta = {"workspace": ws, "base": base, "data": args.data, "dev": args.dev,
            "holdout": args.holdout, "holdout_ids": [r["id"] for r in rows[h0:h1]],
            "budget": args.budget, "max_tokens": args.max_tokens}
    meta_path = ws.rstrip("/") + ".run_meta.json"
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"workspace {ws}\nmeta (holdout, outside the workspace) {meta_path}")


if __name__ == "__main__":
    main()
