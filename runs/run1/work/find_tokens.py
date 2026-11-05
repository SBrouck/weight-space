# Find token ids for reflection words (no tokenizer): correlate per-text token counts in the last 120 tokens
# with word counts in the last 800 characters ('tail'), across all saved generations.
import json, glob, re, numpy as np
from collections import Counter

docs = []
for f in sorted(glob.glob('passes/fwd_*/rows_*.jsonl')):
    tf = f.replace('rows_', 'texts_')
    for r, t in zip(map(json.loads, open(f)), map(json.loads, open(tf))):
        g = t[[k for k in t if k not in ('id', 'prompt_ids')][0]]
        docs.append((r['tail'], Counter(g[-120:]), Counter(g)))
print('docs', len(docs))
vocab = sorted(set().union(*[c for _, c, _ in docs]))
M = np.array([[c[v] for v in vocab] for _, c, _ in docs], float)
for word in [r'\bWait\b', r'\bwait\b', r'\bHmm\b', r'\bAlternatively\b', r'\bBut\b', r'\bActually\b', r'\bcheck\b']:
    y = np.array([len(re.findall(word, t[-500:])) for t, _, _ in docs], float)
    if y.std() == 0:
        print(word, 'absent'); continue
    Mc = M - M.mean(0); yc = y - y.mean()
    corr = (Mc * yc[:, None]).sum(0) / (np.sqrt((Mc ** 2).sum(0) * (yc ** 2).sum()) + 1e-9)
    top = np.argsort(-corr)[:4]
    print('%-18s total %d  ' % (word, y.sum()), [(vocab[i], round(float(corr[i]), 2), int(M[:, i].sum())) for i in top])
