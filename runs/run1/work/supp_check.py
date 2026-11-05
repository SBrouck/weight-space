import json, numpy as np
from kernel import load
THINK = 248069
R6 = np.load('passes/fwd_6/read.npz')
texts = [json.loads(l) for l in open('passes/fwd_5/texts_edited.jsonl')]
gen = [t[[k for k in t if k not in ('id', 'prompt_ids')][0]] for t in texts]
si, pos = R6['sample_index'], R6['position']
cur = np.array([gen[s][p] for s, p in zip(si, pos)])
first = np.array([gen[s].index(THINK) for s in si])
SUP = [(14, 979), (16, 175), (15, 873), (19, 796)]
for l, k in SUP:
    a = R6[f'acts_{l}'][:, k]
    print(l, k, 'first-</think> rows', np.round(a[(cur == THINK) & (pos == first)], 2), 'next-after-first', np.round(a[pos == first + 1], 2))
# detector score on </think> rows in the edited model
on = {int(k): v for k, v in json.load(open('work/onset.json')).items()}
R2 = np.load('passes/fwd_2/read.npz')
inl = np.array([on[s] is not None and p >= on[s] for s, p in zip(R2['sample_index'], R2['position'])])
D = [(20, 89), (21, 99), (22, 59), (22, 65), (23, 625), (23, 362)]
sc = sum(R6[f'acts_{l}'][:, k] / R2[f'acts_{l}'][inl, k].mean() for l, k in D) / len(D)
print('detector score: </think> rows mean %.2f p90 %.2f | before-exit mean %.2f' % (sc[cur == THINK].mean(), np.percentile(sc[cur == THINK], 90), sc[pos < first].mean()))
# suppressor direction: lowers </think> without raising other tokens much
E = load('model.language_model.embed_tokens.weight'); g = 1 + load('model.language_model.norm.weight')
Em = E - E.mean(0)
v = -(g * Em[THINK])
for it in range(30):
    lift = Em @ (g * v / np.linalg.norm(v))
    j = int(np.argmax(lift))
    if lift[j] < 0.3:
        break
    u = g * (g * Em[j]); v = v - (v @ u) / (u @ u) * u
lift = Em @ (g * v / np.linalg.norm(v))
print('suppressor: </think> lift %.2f, max other %.2f (id %d), after %d projections' % (lift[THINK], np.delete(lift, THINK).max(), int(np.argmax(np.delete(lift, THINK))), it))
