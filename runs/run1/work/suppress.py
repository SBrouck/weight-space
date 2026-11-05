# Find neurons (layers 12-19, unaffected by the exit edit) that fire when the current token is </think>.
import json, numpy as np

THINK = 248069
R6 = np.load('passes/fwd_6/read.npz')
R2 = np.load('passes/fwd_2/read.npz')
texts = [json.loads(l) for l in open('passes/fwd_5/texts_edited.jsonl')]
gen = [t[[k for k in t if k not in ('id', 'prompt_ids')][0]] for t in texts]
si, pos = R6['sample_index'], R6['position']
cur = np.array([gen[s][p] for s, p in zip(si, pos)])
first = np.array([gen[s].index(THINK) if THINK in gen[s] else 10**9 for s in si])
isT = cur == THINK
firstT = isT & (pos == first)
before = pos < first
onset = {int(k): v for k, v in json.load(open('work/onset.json')).items()}
loop2 = np.array([onset[s] is not None and p >= onset[s] for s, p in zip(R2['sample_index'], R2['position'])])
print('think-token rows', isT.sum(), 'first-think rows', firstT.sum(), 'before rows', before.sum(), 'base loop rows', loop2.sum())
# after first </think>, positions that are not </think> (answer text in the non-runaway texts)
ansT = (~isT) & (pos > first)
print('answer (non-think) rows after exit', ansT.sum())

best = []
for l in range(12, 20):
    A6, A2 = R6[f'acts_{l}'], R2[f'acts_{l}']
    mT = A6[isT].mean(0)
    other = np.concatenate([A6[before], A2[loop2], A6[ansT]])
    mo = np.abs(other).mean(0)
    p999 = np.percentile(other, 99.9, axis=0)
    minT = np.percentile(A6[isT], 5, axis=0)
    sep = minT / (p999 + 1e-3)
    for k in np.argsort(-(mT / (mo + 1e-2)))[:5]:
        best.append((l, int(k), float(mT[k]), float(mo[k]), float(p999[k]), float(minT[k]), float(sep[k])))
best.sort(key=lambda r: -r[6])
for r in best[:15]:
    print('L%d n%d  meanT %.2f  mean|other| %.3f  other p99.9 %.2f  T p5 %.2f  sep %.2f' % r)
json.dump(best[:15], open('work/suppress_cands.json', 'w'))
