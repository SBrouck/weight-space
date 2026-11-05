# Facts for an "exit the loop" edit: </think> id, head tying, gold answer present in loop tails.
import json, re, numpy as np
from collections import Counter
from kernel import list_tensors, load

t = list_tensors()
print([k for k in t if 'layers' not in k])
probs = {json.loads(l)['id']: json.loads(l) for l in open('dev.jsonl')}
for d in ['passes/fwd_1', 'passes/fwd_3']:
    for m in ['base', 'edited']:
        rows = [json.loads(l) for l in open(f'{d}/rows_{m}.jsonl')]
        texts = [json.loads(l) for l in open(f'{d}/texts_{m}.jsonl')]
        gen = [x[[k for k in x if k not in ('id', 'prompt_ids')][0]] for x in texts]
        closed = [g for g, r in zip(gen, rows) if r['closed']]
        unclosed = [g for g, r in zip(gen, rows) if not r['closed']]
        if closed:
            once = set.intersection(*[{k for k, c in Counter(g).items() if c == 1} for g in closed])
            never = set().union(*map(set, unclosed))
            print(d, m, 'candidate </think> ids:', sorted(once - never)[:20])
        n_in = 0
        for r in rows:
            if r['cut']:
                gold = probs[r['id']]['answer'].replace(',', '')
                if re.search(r'(?<![\d.])' + re.escape(gold) + r'(?![\d])', r['tail'].replace(',', '')):
                    n_in += 1
        print(d, m, 'cut answers whose last 800 chars contain the gold number:', n_in, '/', sum(r['cut'] for r in rows))
