# "Loop -> no Wait" edit: late-layer loop detectors write a direction that lowers the Wait-family logits.
# usage: ./kpy work/wait_edit.py <S> [wait|random]
# S = norm of the total write when the combined detector score is 1 (its mean inside loops).
import json, sys, numpy as np
from kernel import load, save_edit, drop_edit, current_edit

S = float(sys.argv[1])
KIND = sys.argv[2] if len(sys.argv) > 2 else 'wait'
WAIT = [13428, 13784, 3655]
DETS = [(20, 89), (21, 99), (22, 59), (22, 65), (23, 625), (23, 362)]

for n in current_edit():
    drop_edit(n)
R = np.load('passes/fwd_2/read.npz')
onset = {int(k): v for k, v in json.load(open('work/onset.json')).items()}
inloop = np.array([onset[s] is not None and p >= onset[s] for s, p in zip(R['sample_index'], R['position'])])

E = load('model.language_model.embed_tokens.weight')
g = 1.0 + load('model.language_model.norm.weight')
Em = E - E.mean(0)
BOXED = [1088, 88259, 76367]          # " \\" (generic backslash), a boxed-only prefix, "boxed"
sign = -1.0                            # lower the targets
if KIND == 'wait':
    targets = WAIT
elif KIND == 'boxed':                  # raise "\boxed" so a looping model states its current answer
    targets, sign = BOXED, 1.0
elif KIND == 'random_up':              # control for 'boxed': three random mid-frequency tokens, raised
    targets, sign = [int(x) for x in np.random.default_rng(11).choice(np.arange(1000, 20000), 3, replace=False)], 1.0
else:                       # control: three random mid-frequency tokens instead of the Wait family
    targets = [int(x) for x in np.random.default_rng(11).choice(np.arange(1000, 20000), 3, replace=False)]
v = sign * (g * Em[targets].mean(0))
for it in range(60):        # do not raise any other token by more than 0.3 per unit write
    lift = Em @ (g * v / np.linalg.norm(v))
    lift[targets] = -1e9
    j = int(np.argmax(lift))
    if lift[j] < 0.3:
        break
    u = g * (g * Em[j])
    v = v - (v @ u) / (u @ u) * u
v = v / np.linalg.norm(v)
lift = Em @ (g * v)
print('targets', targets, 'lift', np.round(lift[targets], 2), 'median %.2f' % np.median(lift),
      'max other %.2f' % np.delete(lift, targets).max(), 'projections', it)

by_layer = {}
for l, k in DETS:
    by_layer.setdefault(l, []).append(k)
for l, ks in by_layer.items():
    name = f'model.language_model.layers.{l}.mlp.down_proj.weight'
    W = load(name)
    for k in ks:
        W[:, k] += S / (len(DETS) * R[f'acts_{l}'][inloop, k].mean()) * v
    save_edit(name, W)
json.dump(dict(S=S, kind=KIND, targets=targets, dets=DETS), open('work/wait_edit_last.json', 'w'))
print('saved', current_edit())
