import json, numpy as np
rows = [json.loads(l) for l in open('dev.jsonl')]
ids = [r['id'] for r in rows]
rng = np.random.default_rng(0)
p = rng.permutation(len(ids))
dev = [ids[i] for i in p[:750]]
hold = [ids[i] for i in p[750:]]
open('work/dev_ids.txt', 'w').write('\n'.join(dev) + '\n')
open('work/holdout_ids.txt', 'w').write('\n'.join(hold) + '\n')
open('work/pass1_ids.txt', 'w').write('\n'.join(dev[:32]) + '\n')
open('work/pass3_ids.txt', 'w').write('\n'.join(hold[:32]) + '\n')
print(len(dev), len(hold), dev[:3], hold[:3])
