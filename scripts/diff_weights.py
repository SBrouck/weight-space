"""Diff two safetensors checkpoints and characterise each changed matrix.

For every tensor that differs, report how many entries, rows and columns changed.
For 2-D tensors, test the hypothesis that the edit removes one direction u from a
subset of columns: w_j' = w_j - c_j * u. If so, the delta restricted to the changed
columns is rank one, and c_j should track the projection u . w_j.

Usage: uv run python scripts/diff_weights.py models/base models/edited [--prefix model.language_model.]
"""

from __future__ import annotations

import argparse
import glob
import os

import mlx.core as mx
import numpy as np


def load_all(model_dir: str) -> dict[str, mx.array]:
    """Load every tensor from all safetensors shards in a directory."""
    tensors: dict[str, mx.array] = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        tensors.update(mx.load(path))
    return tensors


def to_np(a: mx.array) -> np.ndarray:
    """Convert to float32 numpy (numpy has no bfloat16)."""
    return np.array(a.astype(mx.float32))


def analyse_matrix(name: str, base: np.ndarray, edited: np.ndarray) -> np.ndarray | None:
    """Print the structure of a 2-D edit and return the recovered unit direction u."""
    delta = edited - base
    cols = np.flatnonzero(np.any(delta != 0, axis=0))
    rows = np.flatnonzero(np.any(delta != 0, axis=1))
    print(f"\n{name}  shape={base.shape}")
    print(f"  changed entries={np.count_nonzero(delta)}  rows={rows.size}  cols={cols.size}")
    print(f"  relative change ||dW||/||W|| = {np.linalg.norm(delta) / np.linalg.norm(base):.4f}")
    if cols.size == 0:
        return None

    sub = delta[:, cols]
    s = np.linalg.svd(sub, compute_uv=False)
    print(f"  delta on changed cols: top singular values {np.round(s[:4], 4)}")
    print(f"  rank-one share s1^2/sum(s^2) = {s[0] ** 2 / np.sum(s ** 2):.4f}")

    u_left, _, _ = np.linalg.svd(sub, full_matrices=False)
    u = u_left[:, 0]
    coeffs = u @ sub  # delta_j ~= u * coeffs_j
    proj = u @ base[:, cols]  # u . w_j before the edit
    ratio = -coeffs / np.where(np.abs(proj) > 1e-8, proj, np.nan)
    corr = np.corrcoef(coeffs, proj)[0, 1]
    print(f"  corr(coeff_j, u.w_j) = {corr:.4f}   (close to -1 => projection removal)")
    print(f"  removal strength -coeff_j/(u.w_j): median {np.nanmedian(ratio):.3f}  "
          f"p10 {np.nanpercentile(ratio, 10):.3f}  p90 {np.nanpercentile(ratio, 90):.3f}")
    print(f"  first changed columns (neurons): {cols[:16].tolist()}{' ...' if cols.size > 16 else ''}")
    return u


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("edited")
    parser.add_argument("--prefix", default="model.language_model.")
    args = parser.parse_args()

    base, edited = load_all(args.base), load_all(args.edited)
    keys = sorted(k for k in base if k.startswith(args.prefix))
    missing = [k for k in keys if k not in edited]
    print(f"{len(keys)} tensors under '{args.prefix}', {len(missing)} missing in edited")

    directions: dict[str, np.ndarray] = {}
    changed = []
    for k in keys:
        if k in missing:
            continue
        a, b = base[k], edited[k]
        if a.shape != b.shape or not mx.array_equal(a, b).item():
            changed.append(k)
    print(f"changed tensors: {len(changed)}")

    for k in changed:
        a, b = to_np(base[k]), to_np(edited[k])
        if a.ndim == 2:
            u = analyse_matrix(k, a, b)
            if u is not None:
                directions[k] = u
        else:
            print(f"\n{k}: {np.count_nonzero(a != b)} entries changed (not 2-D)")

    names = list(directions)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cos = abs(float(directions[names[i]] @ directions[names[j]]))
            print(f"\n|cos(u)| between\n  {names[i]}\n  {names[j]}\n  = {cos:.4f}")

    os.makedirs("results", exist_ok=True)
    np.savez("results/wsp_directions.npz", **{n.replace(".", "_"): u for n, u in directions.items()})
    print("\nsaved recovered directions to results/wsp_directions.npz")


if __name__ == "__main__":
    main()
