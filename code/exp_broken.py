"""Evaluation protocol applied to a published scheme that was broken by a chosen-plaintext attack.

The schemes are reconstructions of their structural cores from the descriptions in the attack papers;
details that are not specified there are replaced by the choices marked ASSUMPTION below.  The tested
properties (only 252 possible hash-derived initial values; no chaining between pixels) are structural and do not
depend on these choices.

  W: Wang et al., Multimed. Tools Appl. 76 (2017) 6229 (NEIEA-BCD), as described by Zhao, Shi & Ding,
     Entropy 27 (2025) 40 (MD5-dependent logistic initial value, row-wise DNA operations with a PWLCM
     key image, rotation; no chaining between pixels).

Output: results/broken.json
"""
import numpy as np, json, hashlib, time
from PIL import Image
from numba import njit
import cipher as C
from metrics import *

NAMES = ['Lena', 'Baboon', 'Peppers', 'Fruits']
IM = {n: np.array(Image.open(f'images/{n}.png')) for n in NAMES}
IM5 = {n: np.array(Image.open(f'images512/{n}.png')) for n in NAMES}
DISCARD = 1000      # ASSUMPTION: transient iterations discarded (not given in the attack paper)

# ---------------------------------------------------------------- scheme W (Wang et al. 2017)
W_KEY = dict(mu=3.99999999, p=0.256789, y0=0.3)   # mu and p as in the attack paper; y0 ASSUMPTION
W_ROUNDS = 2                                       # ASSUMPTION: number of repetitions not stated

def w_x0(img):
    d = hashlib.md5(img.tobytes()).digest()
    v = 0
    for k in range(4): v ^= int.from_bytes(d[4 * k:4 * k + 4], 'big')
    v %= 256
    if v in (0, 255): v = 1 if v == 0 else 254    # ASSUMPTION: x0 must not be 0 or 1
    return v / 255.0

@njit(cache=True)
def logistic(mu, x, n):
    out = np.empty(n)
    for i in range(n):
        x = mu * x * (1 - x); out[i] = x
    return out

@njit(cache=True)
def pwlcm(p, y, n):
    out = np.empty(n)
    for i in range(n):
        t = y if y < 0.5 else 1 - y
        y = t / p if t < p else (t - p) / (0.5 - p)
        if y >= 1.0: y = 0.9999999999
        out[i] = y
    return out

ENC, DEC = C.ENC, C.DEC   # the eight complementary DNA rules of the paper (Table 1)

def to_bases(b, rule):          # bytes (rows x W) -> bases (rows x W x 4), rule per row
    codes = np.stack([(b >> s) & 3 for s in (6, 4, 2, 0)], -1)
    return DEC[rule[:, None, None], codes]

def from_bases(B, rule):
    codes = ENC[rule[:, None, None], B]
    return ((codes[..., 0] << 6) | (codes[..., 1] << 4) | (codes[..., 2] << 2) | codes[..., 3]).astype(np.uint8)

def enc_W(img, x0=None):
    H, W, _ = img.shape
    x0 = w_x0(img) if x0 is None else x0
    lg = logistic(W_KEY['mu'], x0, DISCARD + W_ROUNDS * 3 * H * 4)[DISCARD:]
    kimg = (np.floor(pwlcm(W_KEY['p'], W_KEY['y0'], DISCARD + W_ROUNDS * 3 * H * W)[DISCARD:] * 256) % 256).astype(np.int64)
    cur = img.astype(np.int64).copy(); li = 0; ki = 0
    for t in range(W_ROUNDS):
        nxt = np.empty_like(cur)
        for c in range(3):
            P = cur[:, :, c]; K = kimg[ki:ki + H * W].reshape(H, W); ki += H * W
            v = lg[li:li + 4 * H].reshape(H, 4); li += 4 * H
            rp = np.floor(v[:, 0] * 8).astype(np.int64)      # Eq. (5): rule per row (line)
            rk = np.floor(v[:, 1] * 8).astype(np.int64)
            op = np.floor(v[:, 2] * 3).astype(np.int64)      # Eq. (6): XOR / addition / subtraction per row
            ro = np.floor(v[:, 3] * 8).astype(np.int64)      # random decoding rule
            a = ENC[0][to_bases(P, rp)]; b = ENC[0][to_bases(K, rk)]   # base -> 2-bit value (rule 1 arithmetic)
            r = np.where(op[:, None, None] == 0, a ^ b, np.where(op[:, None, None] == 1, (a + b) % 4, (a - b) % 4))
            nxt[:, :, c] = from_bases(DEC[0][r], ro)
        cur = np.rot90(nxt, 1, axes=(0, 1)).copy()           # 90 degrees counter-clockwise
    return cur.astype(np.uint8)

# ---------------------------------------------------------------- proposed cipher
NON = b'B' * 16
def enc_P(img, digest=None): return C.encrypt(img, C.KEY, nonce=NON, digest=digest)[0]

# ---------------------------------------------------------------- protocol
def stats(encf):
    ent = []; cor = []; chi = 0; loc = []
    for n in NAMES:
        c = encf(IM[n]); ent.append(entropy(c))
        cor.append(np.mean(np.abs([corr_full(c[:, :, k]) for k in range(3)])))
        chi += sum(chi2_p(c[:, :, k])[1] > 0.05 for k in range(3))
        rl = np.random.default_rng(5); loc.append(np.mean([local_entropy(c[:, :, k], rl) for k in range(3)]))
    return dict(entropy=float(np.mean(ent)), mean_abs_corr=float(np.mean(cor)), chi_pass=int(chi), local_entropy=float(np.mean(loc)))

def diff(encf_pair, trials=25):
    """encf_pair(im, im2) -> (c1, c2).  One-value +1 changes at random positions."""
    cr = crit(65536, 0.05); res = []
    for n in NAMES:
        rng = np.random.default_rng(777 + NAMES.index(n)); im = IM[n]
        for _ in range(trials):
            im2 = im.copy(); i, j, k = rng.integers(0, 256), rng.integers(0, 256), rng.integers(0, 3)
            im2[i, j, k] = (int(im2[i, j, k]) + 1) % 256
            a, b = encf_pair(im, im2); res.append(np.concatenate(npcr_uaci(a, b)))
    res = np.array(res)
    return dict(npcr=float(res[:, :3].mean()), uaci=float(res[:, 3:].mean()),
                npcr_min=float(res[:, :3].min()),
                npcr_pass=float(np.mean(res[:, :3] >= cr['npcr_min'])),
                uaci_pass=float(np.mean((res[:, 3:] >= cr['uaci_lo']) & (res[:, 3:] <= cr['uaci_hi']))))

rng = np.random.default_rng(2024); CROPS = []        # same 100 crops as the ablation study
for n in NAMES:
    for _ in range(25):
        i, j = rng.integers(0, 257, 2); CROPS.append(IM5[n][i:i + 256, j:j + 256].copy())

def structure(encf_fixed):
    D = []
    for im in CROPS:
        im2 = im.copy(); im2[128, 128, 0] = (int(im2[128, 128, 0]) + 1) % 256
        D.append(encf_fixed(im) ^ encf_fixed(im2))
    D = np.array(D); r2 = np.random.default_rng(1); m = []
    for _ in range(1500):
        i, j = r2.choice(len(D), 2, replace=False); m.append(np.mean(D[i] == D[j]))
    return dict(diff_match=float(np.mean(m)), zero_frac=float(np.mean(D == 0)))

def additivity(encf_fixed, n=20):
    """Fraction of positions with E(P1)+E(P2) == E(P1+P2)+E(0) (mod 256), identical key streams.
    1/256 for a cipher that is not affine; 1 for an affine one."""
    r = np.random.default_rng(11); f = []; Z = np.zeros((256, 256, 3), np.uint8); EZ = encf_fixed(Z).astype(np.int64)
    for _ in range(n):
        a = r.integers(0, 256, (256, 256, 3), dtype=np.uint8); b = r.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        s = ((a.astype(np.int64) + b) % 256).astype(np.uint8)
        lhs = (encf_fixed(a).astype(np.int64) + encf_fixed(b)) % 256; rhs = (encf_fixed(s).astype(np.int64) + EZ) % 256
        f.append(np.mean(lhs == rhs))
    return float(np.mean(f))

R = {}
t0 = time.time()
# W: standard test (MD5 recomputed) and with the MD5-derived initial value held fixed
fixW = lambda x0: (lambda im: enc_W(im, x0))
R['W'] = dict(stats=stats(enc_W), diff_standard=diff(lambda a, b: (enc_W(a), enc_W(b))),
              diff_fixed=diff(lambda a, b: (enc_W(a), enc_W(b, w_x0(a)))),
              structure=structure(fixW(w_x0(CROPS[0]))), additivity=additivity(fixW(0.3)))
r = np.random.default_rng(3)
R['W']['x0_distinct_values_of_2000_random_images'] = len({w_x0(r.integers(0, 256, (16, 16, 3), dtype=np.uint8)) for _ in range(2000)})
print('W', R['W'], time.time() - t0, flush=True)
# proposed cipher (same nonce; digest recomputed, and digest held fixed)
Hfix = C.image_digest(CROPS[0])
R['P'] = dict(stats=stats(enc_P), diff_standard=diff(lambda a, b: (enc_P(a), enc_P(b))),
              diff_fixed=diff(lambda a, b: (enc_P(a), enc_P(b, digest=C.image_digest(a)))),
              structure=structure(lambda im: enc_P(im, Hfix)), additivity=additivity(lambda im: enc_P(im, Hfix), n=5))
print('P', R['P'], time.time() - t0, flush=True)
json.dump(R, open('results/broken.json', 'w'), indent=1)
