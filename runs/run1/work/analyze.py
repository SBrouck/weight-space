# Loop-onset contrast: for each looping text, compare neuron activations in the loop vs before it.
import json, numpy as np

texts = [json.loads(l) for l in open('passes/fwd_1/texts_base.jsonl')]
rows = [json.loads(l) for l in open('passes/fwd_1/rows_base.jsonl')]
R = np.load('passes/fwd_2/read.npz')
print(sorted(R.files)[:6], len(R.files))
si, pos = R['sample_index'], R['position']
print('rows', len(si), 'pos range', pos.min(), pos.max())

N = 24  # n-gram length for "this token continues an earlier verbatim stretch"
onset = {}
for i, t in enumerate(texts):
    g = t['generated_ids'] if 'generated_ids' in t else t[[k for k in t if k != 'prompt_ids' and isinstance(t[k], list)][0]]
    seen, rep = set(), np.zeros(len(g), bool)
    for j in range(N, len(g) + 1):
        k = tuple(g[j - N:j])
        if k in seen:
            rep[j - 1] = True
        seen.add(k)
    # onset = earliest t such that >=90% of tokens from t to the end are repeats
    suf = np.cumsum(rep[::-1])[::-1] / np.arange(len(g), 0, -1)
    ok = np.where(suf >= 0.9)[0]
    onset[i] = int(ok[0]) if len(ok) and len(g) - ok[0] >= 300 else None
    print(i, rows[i]['id'], len(g), rows[i]['repetition'], rows[i]['correct'], 'onset', onset[i])
json.dump(onset, open('work/onset.json', 'w'))

# Position in the read: is it relative to the generation? check max vs generation lengths
loopers = [i for i in onset if onset[i] is not None and onset[i] > 500]
print('loopers with >500 clean tokens before onset:', len(loopers))

res = {}
for l in range(8, 24):
    A = R[f'acts_{l}']
    D, S = [], []
    for i in loopers:
        m = si == i
        p = pos[m]
        a = A[m]
        inl, pre = a[p >= onset[i]], a[p < onset[i] - 50]
        if len(inl) < 5 or len(pre) < 5:
            continue
        D.append(inl.mean(0) - pre.mean(0))
        S.append(np.sqrt(0.5 * (inl.var(0) + pre.var(0))) + 1e-4)
    D, S = np.array(D), np.array(S)
    eff = D / S                                  # per-text standardized difference
    t = eff.mean(0) / (eff.std(0, ddof=1) / np.sqrt(len(eff)) + 1e-6)
    res[l] = dict(meanD=D.mean(0), eff=eff.mean(0), t=t, frac_pos=(D > 0).mean(0))
    o = np.argsort(-t)[:5]
    print(l, 'n texts', len(D), 'top +t', [(int(k), round(float(t[k]), 1), round(float(eff.mean(0)[k]), 2), round(float(D.mean(0)[k]), 3)) for k in o])
np.savez('work/contrast.npz', **{f'{k}_{l}': v for l, d in res.items() for k, v in d.items()})
