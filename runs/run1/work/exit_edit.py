# "Loop -> exit" edit: late-layer loop-detector neurons also write a direction that raises the </think> logit.
# usage: ./kpy work/exit_edit.py <strength> [think|random] [detector-set]
# strength = norm of the total write when the combined detector score is 1 (its mean inside loops).
import json, sys, numpy as np
from kernel import load, save_edit, drop_edit, current_edit

S = float(sys.argv[1])
KIND = sys.argv[2] if len(sys.argv) > 2 else 'think'
SET = sys.argv[3] if len(sys.argv) > 3 else 'late'
THINK = 248069
DETS = {
    'late': [(20, 89), (21, 99), (22, 59), (22, 65), (23, 625), (23, 362)],
    'wide': [(17, 86), (17, 20), (18, 106), (18, 29), (19, 119), (20, 89), (21, 99), (22, 59), (22, 65), (23, 625), (23, 362)],
}[SET]

for n in current_edit():
    drop_edit(n)

R = np.load('passes/fwd_2/read.npz')
onset = {int(k): v for k, v in json.load(open('work/onset.json')).items()}
si, pos = R['sample_index'], R['position']
inloop = np.array([onset[s] is not None and p >= onset[s] for s, p in zip(si, pos)])
clean = np.array([onset[s] is None or p < onset[s] - 50 for s, p in zip(si, pos)])

E = load('model.language_model.embed_tokens.weight')
g = 1.0 + load('model.language_model.norm.weight')       # zero-centred RMSNorm convention (Qwen3-Next family)
if KIND == 'think':
    d = g * (E[THINK] - E.mean(0))
    for j in (248044, 248046):                            # end-of-text / end-of-turn: do not promote them
        v = g * (g * (E[j] - E.mean(0)))
        d = d - (d @ v) / (v @ v) * v
else:                                                    # control: random direction, same norm, same detectors
    d = np.random.default_rng(7).standard_normal(E.shape[1]).astype(np.float32)
d = d / np.linalg.norm(d)
lift = E @ (g * d)                                       # logit change per unit write (before 1/rms)
top = np.argsort(-lift)[:8]
print('most promoted ids', top.tolist(), 'lift(</think>) %.2f, rank %d, median %.3f' % (lift[THINK], int((lift > lift[THINK]).sum()), np.median(lift)))

score = 0
for l, k in DETS:
    score = score + R[f'acts_{l}'][:, k] / R[f'acts_{l}'][inloop, k].mean()
score = score / len(DETS)
print('score loop median %.2f p10 %.2f | clean mean %.2f p99 %.2f' % (np.median(score[inloop]), np.percentile(score[inloop], 10), score[clean].mean(), np.percentile(score[clean], 99)))

by_layer = {}
for l, k in DETS:
    by_layer.setdefault(l, []).append(k)
for l, ks in by_layer.items():
    name = f'model.language_model.layers.{l}.mlp.down_proj.weight'
    W = load(name)
    for k in ks:
        c = S / (len(DETS) * R[f'acts_{l}'][inloop, k].mean())
        W[:, k] += c * d
    save_edit(name, W)
# Suppressor: neurons that fire on repeated </think> (found in pass 6) write a direction that lowers </think>.
# K = norm of their total write at their mean level inside long </think> runs.
K = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
SUP = [(14, 979), (16, 175), (15, 873), (19, 796)]
if K > 0:
    R6 = np.load('passes/fwd_6/read.npz')
    texts = [json.loads(l) for l in open('passes/fwd_5/texts_edited.jsonl')]
    gen = [t[[k for k in t if k not in ('id', 'prompt_ids')][0]] for t in texts]
    run = np.array([next(n for n in range(10**6) if p - n < 0 or gen[s][p - n] != THINK)
                    for s, p in zip(R6['sample_index'], R6['position'])])
    Em = E - E.mean(0)
    v = -(g * Em[THINK])
    for _ in range(30):                                   # stop it from raising any other token
        lift = Em @ (g * v / np.linalg.norm(v))
        j = int(np.argmax(lift))
        if lift[j] < 0.3:
            break
        u = g * (g * Em[j])
        v = v - (v @ u) / (u @ u) * u
    v = v / np.linalg.norm(v)
    if KIND != 'think':
        v = -d                                            # control: same detectors, same random axis, opposite sign
    for l, k in SUP:
        name = f'model.language_model.layers.{l}.mlp.down_proj.weight'
        W = load(name, edited=True)
        W[:, k] += K / (len(SUP) * R6[f'acts_{l}'][run >= 50, k].mean()) * v
        save_edit(name, W)
json.dump(dict(S=S, kind=KIND, set=SET, dets=DETS, K=K, sup=SUP), open('work/exit_edit_last.json', 'w'))
print('saved', current_edit())
