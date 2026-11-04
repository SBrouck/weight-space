"""Write an edited checkpoint: remove (or flip) a direction u from selected down_proj columns.

For layer l with unit direction u (from capture_directions.py) and down_proj W [hidden, inter],
column j is neuron j's write vector. Neurons are ranked by how much more they push along u in
loop positions than in productive ones: (act_loop_j - act_prod_j) * (u . w_j). The top --cols
columns get  w_j' = w_j - k (u . w_j) u  (k = 1 removes the component, k = 6 flips it 5x, as in the
published Qwen3.5-0.8B edit). --mode amplify uses  w_j' = w_j + k (u . w_j) u  instead, for a
direction to strengthen (conclude_*.npz, whose neurons are ranked with act_pos - act_neg).
--random replaces u by a random unit vector (control, same columns, same ||dW||).

Usage: uv run python scripts/apply_edit.py models/qwen3.5-2B-base results/dirs_2b.npz \
           --layers 12 13 --k 6 --cols 64 --out models/2b-L12-13-k6
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import mlx.core as mx
import numpy as np

KEY = "model.language_model.layers.{}.mlp.down_proj.weight"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("dirs")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--mode", choices=["remove", "amplify"], default="remove",
                        help="remove: w - k(u.w)u; amplify: w + k(u.w)u")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dirs = np.load(args.dirs)
    rng = np.random.default_rng(args.seed)
    shard = sorted(glob.glob(os.path.join(args.base, "*.safetensors")))
    assert len(shard) == 1, "expects a single-shard checkpoint"
    weights = mx.load(shard[0])

    manifest = {"base": args.base, "dirs": args.dirs, "mode": args.mode, "k": args.k, "cols": args.cols,
                "random": args.random, "layers": {}}
    for l in args.layers:
        w = np.array(weights[KEY.format(l)].astype(mx.float32))  # [hidden, inter]
        u_loop = dirs[f"u_{l}"].astype(np.float32)
        # Loop files store act_loop/act_prod; conclude files store act_pos/act_neg.
        pos, neg = ("act_pos", "act_neg") if f"act_pos_{l}" in dirs else ("act_loop", "act_prod")
        push = (dirs[f"{pos}_{l}"] - dirs[f"{neg}_{l}"]) * (u_loop @ w)
        cols = np.argsort(-push)[: args.cols]
        k = -args.k if args.mode == "amplify" else args.k
        u = u_loop
        if args.random:
            # Same columns, random direction, k rescaled so ||dW|| matches the real edit.
            u = rng.standard_normal(u.shape).astype(np.float32)
            u /= np.linalg.norm(u)
            k = k * np.linalg.norm(u_loop @ w[:, cols]) / np.linalg.norm(u @ w[:, cols])
        proj = u @ w  # u . w_j for every neuron j
        w[:, cols] -= k * np.outer(u, proj[cols])
        weights[KEY.format(l)] = mx.array(w).astype(mx.bfloat16)
        delta_norm = float(abs(k) * np.linalg.norm(proj[cols]))
        manifest["layers"][l] = {"cols": sorted(int(c) for c in cols), "k": float(k),
                                 "delta_norm": delta_norm, "auc": float(dirs[f"auc_{l}"])}
        print(f"layer {l}: edited {len(cols)} columns, k={k:.2f}, ||dW||={delta_norm:.4f}")

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.base):
        if not f.endswith(".safetensors") and not f.startswith("."):
            src = os.path.join(args.base, f)
            if os.path.isfile(src):
                shutil.copy(src, args.out)
    mx.save_safetensors(os.path.join(args.out, os.path.basename(shard[0])), weights,
                        metadata={"format": "pt"})
    with open(os.path.join(args.out, "edit_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
