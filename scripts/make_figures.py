"""Render the README figures from results/ and runs/, in a light and a dark variant.

Writes figures/<name>-light.png and figures/<name>-dark.png (the README picks one with
<picture> and prefers-color-scheme). Colors: base = neutral ink, published edit = slot 1 (blue),
agent edit = slot 2 (orange), random control = slot 3 (aqua); every series is labeled directly.

Usage: uv run python scripts/make_figures.py
"""

from __future__ import annotations

import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "base": "#52514e",
              "published": "#2a78d6", "agent": "#eb6834", "control": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "base": "#c3c2b7",
             "published": "#3987e5", "agent": "#d95926", "control": "#199e70"},
}
OUT = "figures"


def style(t: dict) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["ink2"], "axes.linewidth": 0.8,
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion and its Wilson 95% interval."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def rows(tag: str) -> list[dict]:
    return [json.loads(l) for l in open(f"results/loop_eval_{tag}.jsonl")]


def save(fig, name: str, mode: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}-{mode}.png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def fig_anatomy(t: dict, mode: str) -> None:
    """Per-neuron size of the published edit in the two edited down_proj matrices."""
    base = mx.load(glob.glob("models/qwen3.5-0.8B-base/*.safetensors")[0])
    wsp = mx.load(glob.glob("models/qwen3.5-0.8B-wsp/*.safetensors")[0])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.0), sharex=True)
    for ax, layer in zip(axes, (15, 19)):
        k = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        d = np.array((wsp[k].astype(mx.float32) - base[k].astype(mx.float32)))
        col = np.linalg.norm(d, axis=0)
        idx = np.flatnonzero(col > 0)
        ax.vlines(idx, 0, col[idx], color=t["published"], linewidth=0.9)
        ax.set_ylim(0, col.max() * 1.25)
        ax.set_yticks([0, round(float(col.max()), 2)])
        ax.text(0.005, 0.86, f"layer {layer}  ·  {idx.size} of {col.size:,} neurons edited",
                transform=ax.transAxes, color=t["ink"], fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[1].set_xlabel("MLP neuron (column of down_proj)")
    fig.supylabel(r"$\|\Delta w_j\|$", color=t["ink2"], fontsize=10, x=0.02)
    fig.suptitle(r"The published edit: $w_j' = w_j - 6\,(u^\top w_j)\,u$ on 128 columns, rank one per layer",
                 color=t["ink"], fontsize=11, x=0.02, ha="left", y=1.02)
    save(fig, "fig1_edit_anatomy", mode)


def fig_budget(t: dict, mode: str) -> None:
    """Unfinished share and accuracy vs token budget, base vs published edit (0.8B, GSM8K)."""
    tags = {4096: ("base_n64_t4096", "wsp_n64_t4096"), 8192: ("08_base_n32_t8k", "08_wsp_n32_t8k"),
            16384: ("base_n32_t16k", "wsp_n32_t16k")}
    budgets = list(tags)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, metric, title in ((axes[0], "cut", "answers cut at the budget"),
                              (axes[1], "correct", "accuracy")):
        for which, name in ((0, "base"), (1, "published")):
            pts = []
            for b in budgets:
                r = rows(tags[b][which])
                pts.append(wilson(sum(bool(x[metric]) for x in r), len(r)))
            p = [x[0] for x in pts]
            ax.fill_between(budgets, [x[1] for x in pts], [x[2] for x in pts],
                            color=t[name], alpha=0.12, linewidth=0)
            ax.plot(budgets, p, color=t[name], linewidth=2, marker="o", markersize=5)
            label = "base" if name == "base" else "published edit"
            ax.annotate(label, (budgets[-1], p[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=t["ink2"], fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, ["4k", "8k", "16k"])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, loc="left", color=t["ink"])
        ax.set_xlabel("thinking budget (tokens)")
        ax.set_xlim(3400, 30000)
    fig.suptitle("The edit only shows once the budget is long enough to finish  (bands: Wilson 95%)",
                 color=t["ink"], fontsize=11, x=0.02, ha="left", y=1.04)
    save(fig, "fig2_budget", mode)


def fig_holdout(t: dict, mode: str) -> None:
    """Hidden holdout (64 problems): strict accuracy and loop rate with Wilson CIs."""
    models = [("base", "final_run1_base", "base"), ("agent", "final_run1_edited", "agent edit (6 neurons)"),
              ("published", "final_run1_published", "published edit (128 cols)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for ax, metric, title in ((axes[0], "strict_correct", "strict accuracy (answer after </think>)"),
                              (axes[1], "loop", "verbatim-loop tail")):
        for y, (key, tag, label) in enumerate(models):
            r = rows(tag)
            k = sum(bool(x[metric]) if metric != "loop" else x["repetition"] >= 0.5 for x in r)
            p, lo, hi = wilson(k, len(r))
            ax.plot([lo, hi], [y, y], color=t[key], linewidth=2, solid_capstyle="round")
            ax.plot(p, y, "o", color=t[key], markersize=8, markeredgecolor=t["surface"], markeredgewidth=2)
            ax.annotate(f"{p:.0%}", (hi, y), xytext=(6, 0), textcoords="offset points",
                        va="center", color=t["ink2"], fontsize=9)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, loc="left", color=t["ink"])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(models)), [m[2] for m in models])
    axes[0].set_ylim(-0.6, len(models) - 0.4)
    axes[0].invert_yaxis()
    fig.suptitle("Unseen holdout, 64 GSM8K problems, 8k budget  (Wilson 95%)",
                 color=t["ink"], fontsize=11, x=0.02, ha="left", y=1.05)
    save(fig, "fig4_holdout", mode)


def fig_trajectory(t: dict, mode: str) -> None:
    """The agent's 12 passes: paired base vs edited accuracy per generation pass."""
    labels = {1: "random control (48 neurons zeroed)", 3: "ablate 48 loop neurons", 4: "</think> push, S=16",
              5: "</think> push, S=4", 7: "S=3 + repeat suppressor", 8: "push 'Wait' down",
              9: r"push '\boxed' up", 10: "</think> push, S=2", 11: "S=2 on its holdout",
              12: "random direction, S=2 (control)"}
    holdout = {3, 11, 12}
    passes = []
    for k in sorted(labels):
        s = json.load(open(f"runs/run1/passes/fwd_{k}/summary.json"))
        passes.append((k, round(s["base"]["accuracy"] * 32), round(s["edited"]["accuracy"] * 32)))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for y, (k, b, e) in enumerate(passes):
        c = t["control"] if k in (1, 12) else t["agent"]
        ax.plot([b, e], [y, y], color=t["axis"], linewidth=2, zorder=1)
        ax.plot(b, y, "o", color=t["base"], markersize=6, zorder=2)
        ax.plot(e, y, "o", color=c, markersize=8, markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
        if b == e:
            ax.annotate("no change", (e, y), xytext=(9, 0), textcoords="offset points",
                        va="center", color=t["muted"], fontsize=8.5)
    ax.set_yticks(range(len(passes)),
                  [f"pass {k}{'  [holdout]' if k in holdout else ''}  ·  {labels[k]}" for k, _, _ in passes])
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 14)
    ax.set_xlabel("correct answers out of 32  (grey: base, colour: edited, same problems)")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Run 1: what the editor agent tried, pass by pass", color=t["ink"],
                 fontsize=11, x=0.02, ha="left", y=1.0)
    save(fig, "fig3_agent_trajectory", mode)


def fig_2b(t: dict, mode: str) -> None:
    """Qwen3.5-2B enumerations: share of samples that close thinking, per edit."""
    conds = [("base", "ho_qwen3.5-2B-base_t6k", "base", "1 sample"),
             ("agent", "ho_2b-L12-13-k6_t6k", "remove loop dir. (L12-13, k=6)", "1 sample"),
             ("control", "ho_2b-L12-13-k6-random_t6k", "random, same size", "1 sample"),
             ("base", "ho2_qwen3.5-2B-base_t6k", "base", "2 samples"),
             ("agent", "ho2_2b-conc-L20-21-k3_t6k", "amplify conclude (L20-21)", "2 samples"),
             ("agent", "ho2_2b-conc-L16-17-k3_t6k", "amplify conclude (L16-17)", "2 samples"),
             ("control", "ho2_2b-conc-L20-21-k3-random_t6k", "random, same size", "2 samples")]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for y, (key, tag, label, _) in enumerate(conds):
        r = [x for x in rows(tag) if "enum" in x["id"]]
        p, lo, hi = wilson(sum(x["closed_thinking"] for x in r), len(r))
        ax.plot([lo, hi], [y, y], color=t[key], linewidth=2, solid_capstyle="round")
        ax.plot(p, y, "o", color=t[key], markersize=8, markeredgecolor=t["surface"], markeredgewidth=2)
        ax.annotate(f"{sum(x['closed_thinking'] for x in r)}/{len(r)}", (hi, y), xytext=(6, 0),
                    textcoords="offset points", va="center", color=t["ink2"], fontsize=9)
    ax.set_yticks(range(len(conds)), [f"{c[2]}" for c in conds])
    ax.axhline(2.5, color=t["axis"], linewidth=0.8)
    ax.text(0.99, -0.45, "first try · 12 enumerations × 1 sample", transform=ax.get_yaxis_transform(),
            ha="right", va="center", color=t["muted"], fontsize=8.5)
    ax.text(0.99, 2.85, "second try · 12 enumerations × 2 samples", transform=ax.get_yaxis_transform(),
            ha="right", va="center", color=t["muted"], fontsize=8.5)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(len(conds) - 0.4, -0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("enumeration samples that close their thinking  (Wilson 95%)")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Qwen3.5-2B: no rank-one edit makes it finish lists it cannot recall",
                 color=t["ink"], fontsize=11, x=0.02, ha="left", y=1.0)
    save(fig, "fig5_2b_negative", mode)


def main() -> None:
    for mode, t in THEMES.items():
        style(t)
        fig_anatomy(t, mode)
        fig_budget(t, mode)
        fig_trajectory(t, mode)
        fig_holdout(t, mode)
        fig_2b(t, mode)
    print(sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
