# Per-problem comparison for pass 3, plus loop onset (same rule as analyze.py) base vs edited.
import json, numpy as np

def onset(g, N=24):
    seen, rep = set(), np.zeros(len(g), bool)
    for j in range(N, len(g) + 1):
        k = tuple(g[j - N:j])
        rep[j - 1] = k in seen
        seen.add(k)
    suf = np.cumsum(rep[::-1])[::-1] / np.arange(len(g), 0, -1)
    ok = np.where(suf >= 0.9)[0]
    return int(ok[0]) if len(ok) and len(g) - ok[0] >= 300 else None

for d in ['passes/fwd_1', 'passes/fwd_3']:
    rb = [json.loads(l) for l in open(f'{d}/rows_base.jsonl')]
    re_ = [json.loads(l) for l in open(f'{d}/rows_edited.jsonl')]
    tb = [json.loads(l) for l in open(f'{d}/texts_base.jsonl')]
    te = [json.loads(l) for l in open(f'{d}/texts_edited.jsonl')]
    gen = lambda t: t[[k for k in t if k not in ('id', 'prompt_ids')][0]]
    ob = [onset(gen(t)) for t in tb]
    oe = [onset(gen(t)) for t in te]
    print(d)
    for a, b, x, y in zip(rb, re_, ob, oe):
        print(' ', a['id'], 'base', a['correct'], a['tokens'], a['repetition'], x, '| edit', b['correct'], b['tokens'], b['repetition'], y)
    both = [(x, y) for x, y in zip(ob, oe) if x is not None and y is not None]
    print('  median onset base %d edit %d (n=%d)' % (np.median([x for x, _ in both]), np.median([y for _, y in both]), len(both)))
    print('  mean repetition base %.3f edit %.3f' % (np.mean([r['repetition'] for r in rb]), np.mean([r['repetition'] for r in re_])))
