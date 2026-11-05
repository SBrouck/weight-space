# Control: zero the down_proj columns of 3 random neurons per layer, layers 8-23 (48 neurons).
import json, numpy as np
from kernel import list_tensors, load, save_edit, current_edit

t = list_tensors()
print([k for k in t if 'layers.3.' in k])
rng = np.random.default_rng(123)
chosen = {}
for l in range(8, 24):
    name = f'model.language_model.layers.{l}.mlp.down_proj.weight'
    W = load(name)
    idx = sorted(rng.choice(W.shape[1], 3, replace=False).tolist())
    W[:, idx] = 0.0
    save_edit(name, W)
    chosen[l] = idx
json.dump(chosen, open('work/control_neurons.json', 'w'))
print(W.shape, chosen, current_edit())
