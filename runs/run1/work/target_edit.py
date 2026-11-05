# Targeted edit: zero the down_proj columns of the 48 loop-selective neurons (same form/size as control).
# Also reports, from the pass-2 read, the share of MLP output removed inside loops vs elsewhere.
import json, numpy as np
from kernel import load, save_edit, current_edit

R = np.load('passes/fwd_2/read.npz')
onset = {int(k): v for k, v in json.load(open('work/onset.json')).items()}
si, pos = R['sample_index'], R['position']
inloop = np.array([onset[s] is not None and p >= onset[s] for s, p in zip(si, pos)])
chosen = {int(k): v for k, v in json.load(open('work/target_neurons.json')).items()}
for l, idx in sorted(chosen.items()):
    name = f'model.language_model.layers.{l}.mlp.down_proj.weight'
    W = load(name)
    removed = R[f'acts_{l}'][:, idx] @ W[:, idx].T          # contribution the edit removes
    out = R[f'outs_{l}']
    ratio = np.linalg.norm(removed, axis=1) / (np.linalg.norm(out, axis=1) + 1e-6)
    print(l, idx, 'removed/|mlp out|  in-loop %.3f  outside %.3f' % (ratio[inloop].mean(), ratio[~inloop].mean()))
    W[:, idx] = 0.0
    save_edit(name, W)
print(current_edit())
