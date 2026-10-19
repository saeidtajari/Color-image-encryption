"""E7: guess-and-determine state-recovery attack on the key-stream bytes (oracle model).
The attacker sees the low bytes b_t = floor(1e14 v_t) mod 256 of the three coordinates for
consecutive iterates t = 0..m, knows the map and the parameters (gamma, r, mu) -- both are
advantages that the attacker does NOT have in the cipher -- and must find the state v_0.
Reduced-search model: v_0 is known except for the u lowest mantissa bits of each coordinate.
The attack enumerates x- and y-candidates and finds z-candidates by a table on the low byte of the
branch output (relation modulo 256, carries 0..7, 0..21, 0..16); every hit is verified by
iterating the true map."""
import sys, json, time, math
import numpy as np, numba as nb
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..')
import cipher as C
from cipher import psin
A = np.array([[3, 1, 4], [8, 3, 11], [6, 2, 9]], dtype=np.int64)
R = np.array([7, 21, 16], dtype=np.int64)       # row sums minus 1 = largest carry

@nb.njit(cache=True)
def fx(x, gam, r): return r * psin(x) if x < 0.5 else gam * x * (1.0 - x)
@nb.njit(cache=True)
def fy(y, r, mu): return mu * y if y < 0.5 else r * psin(1.0 - y)
@nb.njit(cache=True)
def fz(z, gam, mu): return gam * z * (1.0 - z) if z < 0.5 else mu * (1.0 - z)
@nb.njit(cache=True)
def lob(v): return int(math.floor(v * 1e14)) % 256
@nb.njit(cache=True)
def stepn(x, y, z, gam, r, mu):
    xx = fx(x, gam, r); yy = fy(y, r, mu); zz = fz(z, gam, mu)
    return ((3.0 * xx + 1.0 * yy + 4.0 * zz) % 1.0, (8.0 * xx + 3.0 * yy + 11.0 * zz) % 1.0,
            (6.0 * xx + 2.0 * yy + 9.0 * zz) % 1.0)

@nb.njit(cache=True)
def cand(mbase, ulp, u, obs, which, p1, p2):
    """all v = (mbase+j)*ulp, j<2^u, with low byte == obs; returns v and lo of branch output"""
    n = 1 << u
    vs = np.empty(n); lo = np.empty(n, dtype=np.int64); k = 0
    for j in range(n):
        v = (mbase + j) * ulp
        if lob(v) == obs:
            vs[k] = v
            if which == 0: b = fx(v, p1, p2)
            elif which == 1: b = fy(v, p1, p2)
            else: b = fz(v, p1, p2)
            lo[k] = int(math.floor(b * 1e14)) % 256
            k += 1
    return vs[:k].copy(), lo[:k].copy()

@nb.njit(cache=True)
def allowed_table(b1):
    T = np.zeros((256, 256, 256), dtype=np.bool_)
    for a in range(256):
        for b in range(256):
            for c in range(256):
                ok = True
                for i in range(3):
                    s = (b1[i] - (A[i, 0] * a + A[i, 1] * b + A[i, 2] * c)) % 256
                    if s > R[i]: ok = False; break
                T[a, b, c] = ok
    return T

@nb.njit(cache=True)
def attack(vx, lx, vy, ly, zbuck, zoff, zv, T, gam, r, mu, B, m, maxfound, tN):
    n_rel = 0; n_other = 0; found = np.empty((maxfound, 3)); nf = 0
    for i in range(vx.shape[0]):
        for j in range(vy.shape[0]):
            a = lx[i]; b = ly[j]
            for c in range(256):
                if not T[a, b, c]: continue
                for q in range(zoff[c], zoff[c + 1]):
                    n_rel += 1
                    if not (int(math.floor(vx[i]*1e14)) == tN[0] and int(math.floor(vy[j]*1e14)) == tN[1] and int(math.floor(zv[q]*1e14)) == tN[2]):
                        n_other += 1
                    x = vx[i]; y = vy[j]; z = zv[q]
                    ok = True
                    for t in range(1, m + 1):
                        x, y, z = stepn(x, y, z, gam, r, mu)
                        if lob(x) != B[t, 0] or lob(y) != B[t, 1] or lob(z) != B[t, 2]:
                            ok = False; break
                    if ok and nf < maxfound:
                        found[nf, 0] = vx[i]; found[nf, 1] = vy[j]; found[nf, 2] = zv[q]; nf += 1
    return n_rel, n_other, found[:nf]

def run(u, seed, m=14):
    rng = np.random.default_rng(seed)
    gam, r, mu = C.KEY[3:6]
    x0, y0, z0 = rng.random(3)
    X = np.empty(1000); Y = np.empty(1000); Z = np.empty(1000)
    st = C.hybrid_steps(x0, y0, z0, gam, r, mu, 1000, X, Y, Z, 0)
    s = list(st); B = np.zeros((m + 1, 3), dtype=np.int64); S = [tuple(s)]
    for t in range(m + 1):
        B[t] = [lob(s[0]), lob(s[1]), lob(s[2])]
        s = list(stepn(s[0], s[1], s[2], gam, r, mu))
    true = st
    lists = []
    params = ((gam, r), (r, mu), (gam, mu))
    for c in range(3):
        v = true[c]; ulp = float(np.spacing(v)); mm = int(round(v / ulp))
        mb = (mm >> u) << u
        lists.append(cand(mb, ulp, u, B[0, c], c, *params[c]))
    (vx, lx), (vy, ly), (vz, lz) = lists
    order = np.argsort(lz, kind='stable'); zv = vz[order]; lzs = lz[order]
    zoff = np.searchsorted(lzs, np.arange(257)).astype(np.int64)
    T = allowed_table(B[1])
    t0 = time.time()
    tN = np.array([int(math.floor(true[c]*1e14)) for c in range(3)], dtype=np.int64)
    n_rel, n_other, found = attack(vx, lx, vy, ly, None, zoff, zv, T, gam, r, mu, B, m, 100000, tN)
    dt = time.time() - t0
    ok = any(tuple(f) == tuple(true) for f in found)
    nx, ny, nz = len(vx), len(vy), len(vz)
    return dict(u=u, seed=seed, nx=nx, ny=ny, nz=nz, triples=nx * ny * nz, pairs=nx * ny,
                n_rel=int(n_rel), n_other=int(n_other), pred_rel=nx * ny * nz * 2992 / 2 ** 24,
                found=len(found), true_found=bool(ok), sec=dt)

if __name__ == '__main__':
    us = [int(a) for a in sys.argv[1].split(',')]
    nseed = int(sys.argv[2])
    out = []
    for u in us:
        for sd in range(nseed):
            rr = run(u, 1000 + sd); out.append(rr); print(rr, flush=True)
    json.dump(out, open(sys.argv[3], 'w'))
