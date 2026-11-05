# Net </think> push (logit units x rms) = S*score*2.37 - K*sup*2.07, per position type.
import json, numpy as np
T = 248069
R2 = np.load('passes/fwd_2/read.npz'); R6 = np.load('passes/fwd_6/read.npz')
on = {int(k): v for k, v in json.load(open('work/onset.json')).items()}
s2, p2 = R2['sample_index'], R2['position']
loop = np.array([on[s] is not None and p >= on[s] for s, p in zip(s2, p2)])
clean = ~loop
texts = [json.loads(l) for l in open('passes/fwd_5/texts_edited.jsonl')]
gen = [t[[k for k in t if k not in ('id', 'prompt_ids')][0]] for t in texts]
s6, p6 = R6['sample_index'], R6['position']
cur = np.array([gen[s][p] for s, p in zip(s6, p6)])
first = np.array([gen[s].index(T) for s in s6])
run = np.array([next(n for n in range(10**6) if p - n < 0 or gen[s][p - n] != T) for s, p in zip(s6, p6)])
DET = [(20, 89), (21, 99), (22, 59), (22, 65), (23, 625), (23, 362)]
SUP = [(14, 979), (16, 175), (15, 873), (19, 796)]
mD = {d: R2[f'acts_{d[0]}'][loop, d[1]].mean() for d in DET}
runT = run >= 50
mS = {d: R6[f'acts_{d[0]}'][runT, d[1]].mean() for d in SUP}
def sc(R, m):
    return sum(R[f'acts_{l}'][m, k] / mD[(l, k)] for l, k in DET) / len(DET)
def sp(R, m):
    return sum(R[f'acts_{l}'][m, k] / mS[(l, k)] for l, k in SUP) / len(SUP)
groups = {'base clean': (R2, clean), 'base loop': (R2, loop),
          'edit: before exit': (R6, p6 < first), 'edit: isolated </think>': (R6, (cur == T) & (run == 1)),
          'edit: </think> run 2-49': (R6, (run >= 2) & (run < 50)), 'edit: </think> run>=50': (R6, runT),
          'edit: answer text': (R6, (cur != T) & (p6 > first))}
for name, (R, m) in groups.items():
    a, b = sc(R, m), sp(R, m)
    print('%-26s n=%4d  det %.2f (p99 %.2f)  sup %.2f (p99 %.2f)' % (name, m.sum(), a.mean(), np.percentile(a, 99), b.mean(), np.percentile(b, 99)))
for S, K in [(3, 0), (3, 4), (3, 8), (2, 4), (2.5, 6)]:
    out = []
    for name, (R, m) in groups.items():
        net = S * sc(R, m) * 2.37 - K * sp(R, m) * 2.07
        out.append('%s %.1f/%.1f' % (name.split(':')[-1].strip()[:10], np.median(net), np.percentile(net, 99)))
    print('S=%g K=%g  median/p99:' % (S, K), ' | '.join(out))
