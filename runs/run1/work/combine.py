# Combined late-layer loop detector + </think> direction through the final norm.
import json, numpy as np
from kernel import load

R = np.load('passes/fwd_2/read.npz')
onset = {int(k): v for k, v in json.load(open('work/onset.json')).items()}
si, pos = R['sample_index'], R['position']
inloop = np.array([onset[s] is not None and p >= onset[s] for s, p in zip(si, pos)])
clean = np.array([onset[s] is None or p < onset[s] - 50 for s, p in zip(si, pos)])

DET = [(20, 89), (21, 99), (22, 59), (22, 65), (23, 22), (23, 625), (23, 362), (17, 86), (17, 20), (18, 106), (18, 29), (19, 119)]
score = 0
for l, k in DET:
    a = R[f'acts_{l}'][:, k]
    score = score + a / a[inloop].mean()
score = score / len(DET)            # ~1 inside loops by construction
lo, cl = score[inloop], score[clean]
print('combined: loop mean %.2f median %.2f p10 %.2f | clean mean %.2f p90 %.2f p99 %.2f p99.9 %.2f max %.2f' % (
    lo.mean(), np.median(lo), np.percentile(lo, 10), cl.mean(), np.percentile(cl, 90), np.percentile(cl, 99), np.percentile(cl, 99.9), cl.max()))
for name, s in [('late only (20-23)', [d for d in DET if d[0] >= 20])]:
    sc = sum(R[f'acts_{l}'][:, k] / R[f'acts_{l}'][inloop, k].mean() for l, k in s) / len(s)
    print(name, 'loop p10 %.2f median %.2f | clean p99 %.2f p99.9 %.2f' % (np.percentile(sc[inloop], 10), np.median(sc[inloop]), np.percentile(sc[clean], 99), np.percentile(sc[clean], 99.9)))
# per-text: mean score before loop vs in loop, and in clean texts by position
for i in sorted(onset):
    m = si == i
    s, p = score[m], pos[m]
    if onset[i] is None:
        print('clean text', i, 'max %.2f mean %.2f' % (s.max(), s.mean()))

print('input_layernorm.0 mean %.3f' % load('model.language_model.layers.0.input_layernorm.weight').mean(),
      'q_norm.3 mean %.3f' % load('model.language_model.layers.3.self_attn.q_norm.weight').mean())
E = load('model.language_model.embed_tokens.weight')
print('E', E.shape, '|E_</think>| %.3f mean |E| %.3f' % (np.linalg.norm(E[248069]), np.linalg.norm(E, axis=1).mean()))
