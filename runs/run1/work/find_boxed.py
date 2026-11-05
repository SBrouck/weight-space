# Token id of "boxed" (same correlation trick as find_tokens.py) and its usual neighbours.
import json, glob, numpy as np
from collections import Counter

docs = []
for f in sorted(glob.glob('passes/fwd_*/rows_*.jsonl')):
    for r, t in zip(map(json.loads, open(f)), map(json.loads, open(f.replace('rows_', 'texts_')))):
        g = t[[k for k in t if k not in ('id', 'prompt_ids')][0]]
        docs.append((r['tail'], Counter(g[-120:]), g))
vocab = sorted(set().union(*[c for _, c, _ in docs]))
M = np.array([[c[v] for v in vocab] for _, c, _ in docs], float)
y = np.array([t[-500:].count('boxed') for t, _, _ in docs], float)
print('boxed in tails', y.sum())
Mc, yc = M - M.mean(0), y - y.mean()
corr = (Mc * yc[:, None]).sum(0) / (np.sqrt((Mc ** 2).sum(0) * (yc ** 2).sum()) + 1e-9)
top = np.argsort(-corr)[:6]
print([(vocab[i], round(float(corr[i]), 2), int(M[:, i].sum())) for i in top])
for b in [vocab[i] for i in top[:3]]:
    pre, post = Counter(), Counter()
    for _, _, g in docs:
        for i in range(1, len(g) - 1):
            if g[i] == b:
                pre[g[i - 1]] += 1
                post[g[i + 1]] += 1
    print(b, 'preceded by', pre.most_common(5), 'followed by', post.most_common(3))
