# Pick 48 loop-selective neurons (layers 8-23), controlling for position (context length).
import json, numpy as np

R = np.load('passes/fwd_2/read.npz')
C = np.load('work/contrast.npz')
onset = {int(k): v for k, v in json.load(open('work/onset.json')).items()}
si, pos = R['sample_index'], R['position']
loopers = [i for i in onset if onset[i] is not None and onset[i] > 500]
clean = [i for i in onset if onset[i] is None]          # texts with no loop
print('clean texts', clean)

cands = []
for l in range(8, 24):
    A = R[f'acts_{l}']
    t = C[f't_{l}']
    # position control: in clean texts, late (>=3000) minus early (<2000), standardized
    pc = []
    for i in clean:
        m = si == i
        p, a = pos[m], A[m]
        late, early = a[p >= 3000], a[p < 2000]
        if len(late) >= 5 and len(early) >= 5:
            pc.append((late.mean(0) - early.mean(0)) / (np.sqrt(0.5 * (late.var(0) + early.var(0))) + 1e-4))
    pc = np.mean(pc, 0)
    # selectivity: mean activation in loop vs before loop, pooled over loopers
    inl = np.concatenate([A[(si == i) & (pos >= onset[i])] for i in loopers])
    pre = np.concatenate([A[(si == i) & (pos < onset[i] - 50)] for i in loopers])
    cl = np.concatenate([A[si == i] for i in clean])
    mi, mp, mc = inl.mean(0), np.abs(pre).mean(0), np.abs(cl).mean(0)
    sel = mi / (np.maximum(mp, mc) + 1e-3)
    for k in range(A.shape[1]):
        cands.append((l, k, float(t[k]), float(pc[k]), float(sel[k]), float(mi[k]), float(mp[k]), float(mc[k])))
    print(l, 'n clean w/ late part', 'pos-ctrl ok')

cands = np.array(cands)
ok = (cands[:, 2] > 10) & (np.abs(cands[:, 3]) < 0.3) & (cands[:, 4] > 3) & (cands[:, 5] > 0)
print('passing filters:', ok.sum())
sub = cands[ok]
sub = sub[np.argsort(-(sub[:, 2] * np.log(sub[:, 4])))][:48]
for r in sub:
    print(int(r[0]), int(r[1]), 't=%.1f posctrl=%.2f sel=%.1f loop=%.3f pre=%.3f clean=%.3f' % tuple(r[2:]))
chosen = {}
for r in sub:
    chosen.setdefault(int(r[0]), []).append(int(r[1]))
json.dump(chosen, open('work/target_neurons.json', 'w'))
print({l: len(v) for l, v in sorted(chosen.items())})
