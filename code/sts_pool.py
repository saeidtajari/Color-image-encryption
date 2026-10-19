"""Pooled analysis of the per-sequence p-values written by the NIST suite (run directories run_<name>).
Usage: python sts_pool.py /path/to/sts   (directory that contains run_c, run_c2, run_c3, run_r1, run_r2)"""
import numpy as np, json, sys
from scipy.stats import chisquare
base=sys.argv[1] if len(sys.argv)>1 else '/tmp/sts'   # STS_WORK of sts_run.sh
def unif(p): return float(chisquare(np.histogram(p,bins=10,range=(0,1))[0]).pvalue)
def NO(run): return np.loadtxt(f'{base}/run_{run}/experiments/AlgorithmTesting/NonOverlappingTemplate/results.txt').reshape(148,200)
single=['Frequency','BlockFrequency','Runs','LongestRun','Rank','FFT','OverlappingTemplate','Universal','ApproximateEntropy','LinearComplexity']
out={}
for name,runs in (('cipher',['c','c2','c3']),('control',['r1','r2'])):
    P=np.concatenate([NO(r) for r in runs],axis=1); u=np.array([unif(P[i]) for i in range(148)])
    s={t:unif(np.concatenate([np.loadtxt(f'{base}/run_{r}/experiments/AlgorithmTesting/{t}/results.txt') for r in runs])) for t in single}
    out[name]=dict(nseq=200*len(runs),no_min=float(u.min()),no_lt01=int((u<0.01).sum()),no_lt001=int((u<0.001).sum()),single_min=float(min(s.values())))
json.dump(out,open('results/sts_pooled.json','w'),indent=1); print(out)
