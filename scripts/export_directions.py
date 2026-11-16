"""Export the per-layer statistics of the Qwen3.5-2B loop and conclude directions to JSON.

Reads the .npz files written by capture_directions.py and capture_conclude.py (git-ignored, about
1.4 MB each) and the edit_manifest.json of every hand-built 2B edit written by apply_edit.py, and
writes results/directions_2b.json, which is versioned and read by scripts/report.py.

Per layer: AUC of the loop direction u on held-out samples, cos(u, mean(loop) - mean(clean)),
AUC of the conclude direction c on held-out samples, and cos(c, -u).

Usage: uv run python scripts/export_directions.py --loop results/dirs_2b.npz \
           --conclude results/conclude_2b.npz --edits models/2b-* --out results/directions_2b.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", default="results/dirs_2b.npz")
    parser.add_argument("--conclude", default="results/conclude_2b.npz")
    parser.add_argument("--edits", nargs="*", default=[], help="edited checkpoint directories")
    parser.add_argument("--out", default="results/directions_2b.json")
    args = parser.parse_args()

    loop, conc = np.load(args.loop), np.load(args.conclude)
    n_layers = sum(k.startswith("auc_") for k in loop.files)
    layers = []
    for l in range(n_layers):
        u, c = loop[f"u_{l}"], conc[f"u_{l}"]
        layers.append({
            "layer": l,
            "loop_auc": round(float(loop[f"auc_{l}"]), 4),
            "loop_cos_clean": round(float(loop[f"cos_clean_{l}"]), 4),
            "conclude_auc": round(float(conc[f"auc_{l}"]), 4),
            "conclude_cos_neg_loop": round(float(-c @ u), 4),
        })
        print(" ".join(f"{k}={v}" for k, v in layers[-1].items()))

    edits = {}
    for d in sorted(args.edits):
        path = os.path.join(d, "edit_manifest.json")
        if os.path.exists(path):
            edits[os.path.basename(os.path.normpath(d))] = json.load(open(path))
    out = {"loop_npz": os.path.basename(args.loop), "conclude_npz": os.path.basename(args.conclude),
           "layers": layers, "edits": edits}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out} ({len(edits)} edit manifests)")


if __name__ == "__main__":
    main()
