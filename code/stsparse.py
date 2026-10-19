"""Parser for finalAnalysisReport.txt of the NIST SP 800-22 reference suite (STS 2.1.x)."""
import re, math
from scipy.stats import binom
ORDER=['Frequency','BlockFrequency','CumulativeSums','Runs','LongestRun','Rank','FFT','NonOverlappingTemplate',
       'OverlappingTemplate','Universal','ApproximateEntropy','RandomExcursions','RandomExcursionsVariant','Serial','LinearComplexity']
LABEL={'Frequency':'Frequency (monobit)','BlockFrequency':'Block Frequency','CumulativeSums':'Cumulative Sums','Runs':'Runs',
 'LongestRun':'Longest Run of Ones','Rank':'Rank','FFT':'Discrete Fourier Transform','NonOverlappingTemplate':'Non-overlapping Template',
 'OverlappingTemplate':'Overlapping Template','Universal':'Maurer’s Universal','ApproximateEntropy':'Approximate Entropy',
 'RandomExcursions':'Random Excursions','RandomExcursionsVariant':'Random Excursions Variant','Serial':'Serial','LinearComplexity':'Linear Complexity'}
def parse(path):
    res={}
    rx=re.compile(r'^\s*((?:\d+\s+){10})(\S+)\s+(?:\*\s+)?(\d+)/(\d+)\s*(?:\*\s*)?(\w+)\s*$')
    for line in open(path):
        m=rx.match(line)
        if not m: continue
        c=[int(v) for v in m.group(1).split()]
        pv=None if m.group(2).startswith('-') else float(m.group(2).rstrip('*'))
        res.setdefault(m.group(5),[]).append(dict(c=c,pv=pv,k=int(m.group(3)),n=int(m.group(4))))
    return res
def lo(n,p=0.99): return p-3*math.sqrt(p*(1-p)/n)
def summarize(res):
    out={}
    for t in ORDER:
        rows=res.get(t,[])
        if not rows: out[t]=None; continue
        props=[r['k']/r['n'] for r in rows]
        outside=sum(1 for r in rows if r['k']/r['n']<lo(r['n']))
        expect=sum(float(binom.cdf(math.ceil(lo(r['n'])*r['n'])-1,r['n'],0.99)) for r in rows)
        pvs=[r['pv'] for r in rows if r['pv'] is not None]
        out[t]=dict(nsub=len(rows),minprop=min(props),maxprop=max(props),outside=outside,expect=expect,minpu=min(pvs) if pvs else None,
                    nout_pu=sum(1 for v in pvs if v<0.0001),nmin=min(r['n'] for r in rows),nmax=max(r['n'] for r in rows))
    return out
