# Late-layer loop detectors (per-neuron AUC, loop vs non-loop positions) and the </think> write direction.
import json, numpy as np
from kernel import load

R = np.load('passes/fwd_2/read.npz')
onset = {int(k): v for k, v in json.load(open('work/onset.json')).items()}
si, pos = R['sample_index'], R['position']
inloop = np.array([onset[s] is not None and p >= onset[s] for s, p in zip(si, pos)])
clean = np.array([onset[s] is None or p < onset[s] - 50 for s, p in zip(si, pos)])
print('loop rows', inloop.sum(), 'clean rows', clean.sum())

def auc(x, y):  # P(x > y) via ranks
    r = np.argsort(np.argsort(np.concatenate([x, y]))) + 1
    return (r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))

out = {}
for l in range(16, 24):
    A = R[f'acts_{l}']
    lo, cl = A[inloop], A[clean]
    a = np.array([auc(lo[:, k], cl[:, k]) for k in range(A.shape[1])])
    top = np.argsort(-a)[:6]
    print(l, [(int(k), round(float(a[k]), 3), round(float(lo[:, k].mean()), 2), round(float(cl[:, k].mean()), 2), round(float(np.percentile(cl[:, k], 99)), 2)) for k in top])
    out[l] = a
np.savez('work/auc_late.npz', **{str(l): v for l, v in out.items()})
O = R['outs_23']
print('|outs_23| loop %.2f clean %.2f' % (np.linalg.norm(O[inloop], axis=1).mean(), np.linalg.norm(O[clean], axis=1).mean()))

E = load('model.language_model.embed_tokens.weight')
g = load('model.language_model.norm.weight')
print('final norm weight mean %.3f min %.3f max %.3f' % (g.mean(), g.min(), g.max()))
