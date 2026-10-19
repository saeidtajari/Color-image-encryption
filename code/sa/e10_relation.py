"""E10: is there still a relation modulo 256 between consecutive key-stream bytes?  (Section 3.18, Table 17)

Variants of the byte extraction (the map is the same in all of them):
  v1      : floor(1e14 v) mod 256 of the raw state (first version, no filter, no anchor)
  scale   : floor((1e14+1) v) mod 256 (a scale not divisible by 256; shown only to explain why the filter is needed)
  filter  : chaotic layer of version 2: floor(1e14 g(s)) mod 256 with g of Eq. (10), no anchor
  v2      : version 2 as used by the cipher: the filtered value XOR the 47-bit ChaCha20 word (Eq. (11))
Tests (the attacker knows the map, the parameters and the exact state s_n -- far more than in the cipher):
 (a) relation pass rate: o_{n+1,k} - sum_j A_kj beta_j(s_n) mod 256 in [0, R_k] for k = 1..3, beta_j = low byte of
     floor(1e14 f_j) of the branch output of coordinate j (the relation used by the attack of Table S16).
     Chance level for a structure-free byte: 8*22*17/2^24 = 1.78e-4.
 (b) separability: D = G(x',y',z) - G(x',y,z) - G(x,y',z) + G(x,y,z) mod 256 for the next byte G_k; a relation of the form
     'sum of single-coordinate terms + small carry' (whatever the coefficients) concentrates D; total-variation distance
     of D from uniform and chi-square p-value, 3 coordinate pairs x 3 outputs.
 (c) prediction: fraction of the next observable bytes predicted correctly from the exact state (chance 1/256).
Usage: python3 sa/e10_relation.py N1 N2"""
import sys, os, json, math, numpy as np
from numba import njit
from scipy.stats import chi2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import cipher as C
from cipher import psin
gam, r, mu = C.KEY[3:6]
A = np.array([[3, 1, 4], [8, 3, 11], [6, 2, 9]], dtype=np.int64)
R = np.array([7, 21, 16], dtype=np.int64)
KIND = {'v1': 0, 'scale': 1, 'filter': 2, 'v2': 2}


@njit(cache=True)
def F(x, y, z, gam, r, mu):
    xx = r * psin(x) if x < 0.5 else gam * x * (1.0 - x)
    yy = mu * y if y < 0.5 else r * psin(1.0 - y)
    zz = gam * z * (1.0 - z) if z < 0.5 else mu * (1.0 - z)
    return xx, yy, zz


@njit(cache=True)
def step(x, y, z, gam, r, mu):
    xx, yy, zz = F(x, y, z, gam, r, mu)
    return ((3.0 * xx + 1.0 * yy + 4.0 * zz) % 1.0, (8.0 * xx + 3.0 * yy + 11.0 * zz) % 1.0, (6.0 * xx + 2.0 * yy + 9.0 * zz) % 1.0)


@njit(cache=True)
def qvals(x, y, z, kind):
    out = np.empty(3, dtype=np.int64)
    if kind == 2:
        out[0] = int(math.floor(((x + y * z) % 1.0) * 1e14)); out[1] = int(math.floor(((y + z * x) % 1.0) * 1e14)); out[2] = int(math.floor(((z + x * y) % 1.0) * 1e14))
    elif kind == 1:
        s = 1e14 + 1.0
        out[0] = int(math.floor(x * s)); out[1] = int(math.floor(y * s)); out[2] = int(math.floor(z * s))
    else:
        out[0] = int(math.floor(x * 1e14)); out[1] = int(math.floor(y * 1e14)); out[2] = int(math.floor(z * 1e14))
    return out


@njit(cache=True)
def G(x, y, z, kind, gam, r, mu):
    a, b, c = step(x, y, z, gam, r, mu)
    return qvals(a, b, c, kind) % 256


@njit(cache=True)
def relation(n, seed, kind, W, gam, r, mu, A, R):
    """W: (n, 3) anchor bytes (low byte of the 47-bit words) or None-like zeros when not anchored."""
    np.random.seed(seed)
    ok = 0; hit = 0
    for i in range(n):
        x = np.random.random(); y = np.random.random(); z = np.random.random()
        g = G(x, y, z, kind, gam, r, mu)
        o = g.copy()
        for k in range(3): o[k] = o[k] ^ W[i, k]
        fx, fy, fz = F(x, y, z, gam, r, mu)
        b0 = int(math.floor(fx * 1e14)) % 256; b1 = int(math.floor(fy * 1e14)) % 256; b2 = int(math.floor(fz * 1e14)) % 256
        good = True
        for k in range(3):
            s = (o[k] - (A[k, 0] * b0 + A[k, 1] * b1 + A[k, 2] * b2)) % 256
            if s > R[k]: good = False
        if good: ok += 1
        for k in range(3):
            if g[k] == o[k]: hit += 1                                    # byte predicted from the exact state
    return ok, hit


@njit(cache=True)
def mixed(n, seed, kind, a, b, gam, r, mu):
    np.random.seed(seed)
    H = np.zeros((3, 256), dtype=np.int64)
    for i in range(n):
        s = np.random.random(3); t = np.random.random(3)
        s10 = s.copy(); s10[a] = t[a]
        s01 = s.copy(); s01[b] = t[b]
        s11 = s.copy(); s11[a] = t[a]; s11[b] = t[b]
        g00 = G(s[0], s[1], s[2], kind, gam, r, mu); g10 = G(s10[0], s10[1], s10[2], kind, gam, r, mu)
        g01 = G(s01[0], s01[1], s01[2], kind, gam, r, mu); g11 = G(s11[0], s11[1], s11[2], kind, gam, r, mu)
        for k in range(3):
            H[k, (g11[k] - g10[k] - g01[k] + g00[k]) % 256] += 1
    return H


def tv(h):
    p = h / h.sum(); return float(0.5 * np.abs(p - 1 / 256).sum())


if __name__ == '__main__':
    N1 = int(float(sys.argv[1])); N2 = int(float(sys.argv[2]))
    # consistency with the cipher's own quantization
    X = np.random.default_rng(0).random(1000); Y = np.random.default_rng(1).random(1000); Z = np.random.default_rng(2).random(1000)
    Qc = C.quantize(X, Y, Z, True)
    assert all((qvals(X[i], Y[i], Z[i], 2) == Qc[:, i]).all() for i in range(1000))
    res = {'chance_relation': 8 * 22 * 17 / 2 ** 24, 'chance_prediction': 1 / 256, 'N1': N1, 'N2': N2}
    rng = np.random.default_rng(0)
    nulls = [tv(np.bincount(rng.integers(0, 256, N2), minlength=256).astype(float)) for _ in range(50)]
    res['tv_random_mean'] = float(np.mean(nulls)); res['tv_random_q95'] = float(np.quantile(nulls, .95))
    zeros = np.zeros((N1, 3), dtype=np.int64)
    # anchor bytes: low bytes of real ChaCha20 words of the cipher (one message key, round 1)
    ks = C.stream_key(C.KEY, bytes(range(16)), b'\x00' * 32)
    Wb = (C.anchor_words(ks, 1, N1) % 256).T.copy()
    for var in ('v1', 'scale', 'filter', 'v2'):
        kind = KIND[var]
        ok, hit = relation(N1, 1, kind, Wb if var == 'v2' else zeros, gam, r, mu, A, R)
        d = dict(relation_pass=ok / N1, predict_byte=hit / (3 * N1))
        if var != 'v2':
            sep = {}
            for (a, b) in ((0, 1), (0, 2), (1, 2)):
                H = mixed(N2, 7 + a + b, kind, a, b, gam, r, mu)
                for k in range(3):
                    h = H[k].astype(float); e = h.sum() / 256
                    sep['xyz'[a] + 'xyz'[b] + '->' + 'xyz'[k]] = dict(tv=tv(h), chi_p=float(chi2.sf(((h - e) ** 2 / e).sum(), 255)))
            d['separability'] = sep; d['max_tv'] = max(v['tv'] for v in sep.values()); d['min_chi_p'] = min(v['chi_p'] for v in sep.values())
        res[var] = d
        print(var, {k: v for k, v in d.items() if k != 'separability'}, flush=True)
    json.dump(res, open('results/sa_e10.json', 'w'), indent=1)
