# Loop detectors that stay quiet on </think> runs and on post-exit answer text (pass 6), layers 16-23.
import json, numpy as np

THINK = 248069
R2 = np.load('passes/fwd_2/read.npz'); R6 = np.load('passes/fwd_6/read.npz')
on = {int(k): v for k, v in json.load(open('work/onset.json')).items()}
s2, p2 = R2['sample_index'], R2['position']
loop = np.array([on[s] is not None and p >= on[s] for s, p in zip(s2, p2)])
clean = np.array([on[s] is None or p < on[s] - 50 for s, p in zip(s2, p2)])
texts = [json.loads(l) for l in open('passes/fwd_5/texts_edited.jsonl')]
gen = [t[[k for k in t if k not in ('id', 'prompt_ids')][0]] for t in texts]
s6, p6 = R6['sample_index'], R6['position']
cur = np.array([gen[s][p] for s, p in zip(s6, p6)])
first = np.array([gen[s].index(THINK) for s in s6])
after = p6 >= first          # everything after the exit: </think> runs and answer text

rows = []
for l in range(16, 24):
    L, C, A = R2[f'acts_{l}'][loop], R2[f'acts_{l}'][clean], R6[f'acts_{l}'][after]
    mL = L.mean(0)
    # loop mean relative to the 99th percentile of the "must stay quiet" sets
    q = np.maximum(np.percentile(C, 99, axis=0), np.percentile(A, 99, axis=0))
    r = mL / (np.maximum(q, 0) + 0.05)
    for k in np.argsort(-r)[:6]:
        rows.append((l, int(k), float(r[k]), float(mL[k]), float(np.median(L[:, k])), float(C[:, k].mean()), float(A[:, k].mean()), float(q[k])))
rows.sort(key=lambda x: -x[2])
for x in rows[:20]:
    print('L%d n%d ratio %.2f  loop mean %.2f med %.2f | clean mean %.2f  after-exit mean %.2f | quiet p99 %.2f' % x)
json.dump(rows[:20], open('work/detect2.json', 'w'))
