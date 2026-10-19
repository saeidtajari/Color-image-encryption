"""Summary of E1 over all keys (results/sa_e1_0.json, sa_e1_1.json): the three lag-0 pairs (y,x), (z,x), (z,y)
duplicate (x,y), (x,z), (y,z) and are left out, so that 75 histograms per key are counted."""
import json, numpy as np
from scipy.stats import kstest
dup = {'y_i,x_i+0', 'z_i,x_i+0', 'z_i,y_i+0'}
ps, zs, nk = [], [], 0
for s in (0, 1):
    for k in json.load(open(f'results/sa_e1_{s}.json')):
        nk += 1
        for r in k['res']:
            if r['pair'] in dup: continue
            ps.append(r['chi_p']); zs.append(r['max_z']); N = r['n']
ps = np.array(ps); zs = np.array(zs)
out = dict(n_hist=len(ps) // nk, keys=nk, N=N, ks_p=float(kstest(ps, 'uniform').pvalue), n_p05=int((ps < .05).sum()),
           n_p01=int((ps < .01).sum()), n_p001=int((ps < .001).sum()), total=len(ps), min_p=float(ps.min()),
           max_z=float(zs.max()), median_max_z=float(np.median(zs)), source='chaotic layer of version 2 (Eq. (10)), anchor off')
print(out); json.dump(out, open('results/sa_e1_summary.json', 'w'), indent=1)
