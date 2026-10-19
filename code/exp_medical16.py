"""16-bit grey-level medical images (DICOM): the stored 16-bit sample words are encrypted as an image with two byte
planes (format f = 2; Section 2.2).  Six DICOM files of the pydicom test-data collection (SHA-256 checked against the
hashes shipped with pydicom); the header is not encrypted.  Writes results/medical16.json.
Reference values for random 16-bit images are computed exactly (F = 65535)."""
import numpy as np, json, os, sys, time, hashlib, math
from scipy.stats import norm, chi2
import cipher as C
from metrics import entropy, corr_full

F16 = 65535
FILES = [('693_UNCI.dcm', 'CT'), ('MR2_UNCI.dcm', 'MR'), ('MR-SIEMENS-DICOM-WithOverlays.dcm', 'MR'),
         ('RG1_UNCI.dcm', 'CR'), ('RG3_UNCI.dcm', 'CR'), ('JPEG2000_UNC.dcm', 'NM')]
DIR = os.environ.get('DICOM_DIR', 'images_dicom')


def load(fn):
    import pydicom
    ds = pydicom.dcmread(os.path.join(DIR, fn))
    a = ds.pixel_array
    if a.ndim == 3: a = a[0]
    a = np.ascontiguousarray(a)
    words = a.view(np.uint16) if a.dtype == np.int16 else a.astype(np.uint16)   # stored 16-bit words (two's complement if signed)
    info = dict(modality=str(ds.get('Modality')), rows=int(ds.Rows), cols=int(ds.Columns), bits_stored=int(ds.BitsStored),
                signed=int(ds.PixelRepresentation), sha256=hashlib.sha256(open(os.path.join(DIR, fn), 'rb').read()).hexdigest())
    return words, info


def entropy16(a):
    h = np.bincount(a.reshape(-1), minlength=65536).astype(float); p = h / h.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def ideal_entropy16(n, reps=20, seed=1):
    rng = np.random.default_rng(seed)
    return float(np.mean([entropy16(rng.integers(0, 65536, n).astype(np.uint16)) for _ in range(reps)]))


def crit16(L, alpha=0.05):
    """Critical values of Wu et al. [67] for F = 65535: NPCR >= (F - z_a sqrt(F/L))/(F+1); UACI in mu -/+ z_{a/2} sigma."""
    npcr_min = (F16 - norm.ppf(1 - alpha) * math.sqrt(F16 / L)) / (F16 + 1)
    m1 = F16 * (F16 + 2) / (3 * (F16 + 1))                 # E|a-b|
    m2 = F16 * (F16 + 2) / 6                               # E(a-b)^2
    mu = m1 / F16; sd = math.sqrt(m2 - m1 * m1) / F16 / math.sqrt(L)
    z = norm.ppf(1 - alpha / 2)
    return dict(npcr_exp=100 * F16 / (F16 + 1), npcr_min=100 * npcr_min, uaci_exp=100 * mu, uaci_lo=100 * (mu - z * sd), uaci_hi=100 * (mu + z * sd))


def npcr_uaci16(a, b):
    a = a.astype(np.int64); b = b.astype(np.int64)
    return 100 * float(np.mean(a != b)), 100 * float(np.mean(np.abs(a - b)) / F16)


def chi16(a):
    h = np.bincount(a.reshape(-1), minlength=65536).astype(float); e = h.sum() / 65536
    x = float(((h - e) ** 2 / e).sum()); return x, float(chi2.sf(x, 65535))


def plane_chi(a):
    out = []
    for v in ((a >> 8).astype(np.uint8), (a & 255).astype(np.uint8)):
        h = np.bincount(v.reshape(-1), minlength=256).astype(float); e = h.sum() / 256
        out.append(float(chi2.sf(((h - e) ** 2 / e).sum(), 255)))
    return out


if __name__ == '__main__':
    K = C.KEY; R = {}
    for fn, mod in FILES:
        t0 = time.time()
        a, info = load(fn); M, N = a.shape; L = M * N
        nonce = hashlib.sha256(fn.encode()).digest()[:16]
        c, meta = C.encrypt(a, K, nonce=nonce)
        p, tk, hk = C.decrypt(c, meta, K)
        r = dict(info, file=fn, exact=bool(np.array_equal(p, a) and tk and hk),
                 plain_entropy16=entropy16(a), cipher_entropy16=entropy16(c), ideal_entropy16=ideal_entropy16(L),
                 plain_corr=[float(v) for v in corr_full(a.astype(np.float64))], cipher_corr=[float(v) for v in corr_full(c.astype(np.float64))],
                 cipher_plane_entropy=[entropy((c >> 8).astype(np.uint8)), entropy((c & 255).astype(np.uint8))],
                 chi16=chi16(c), plane_chi_p=plane_chi(c), distinct_plain=int(len(np.unique(a))))
        H = C.image_digest(a); rng = np.random.default_rng(int.from_bytes(nonce[:4], 'big'))
        ntr = 10 if L <= 600000 else 4
        res = {'complete': [], 'fixed': []}
        for t in range(ntr):
            i, j = int(rng.integers(0, M)), int(rng.integers(0, N))
            a2 = a.copy(); a2[i, j] = np.uint16((int(a2[i, j]) + 1) % 65536)
            for mode in ('complete', 'fixed'):
                c2, _ = C.encrypt(a2, K, nonce=nonce, digest=(H if mode == 'fixed' else None))
                res[mode].append(npcr_uaci16(c, c2))
        cr = crit16(L)
        for mode, v in res.items():
            v = np.array(v)
            r[mode] = dict(trials=ntr, npcr=float(v[:, 0].mean()), uaci=float(v[:, 1].mean()), npcr_min=float(v[:, 0].min()),
                           npcr_pass=float(np.mean(v[:, 0] >= cr['npcr_min'])), uaci_pass=float(np.mean((v[:, 1] >= cr['uaci_lo']) & (v[:, 1] <= cr['uaci_hi']))))
        r['crit'] = cr; r['sec'] = time.time() - t0
        R[fn] = r
        print(fn, mod, M, N, info['bits_stored'], 'H', round(r['plain_entropy16'], 4), round(r['cipher_entropy16'], 4), round(r['ideal_entropy16'], 4),
              'npcr', round(r['fixed']['npcr'], 4), 'uaci', round(r['fixed']['uaci'], 4), r['exact'], round(r['sec']), flush=True)
        json.dump(R, open('results/medical16.json', 'w'), indent=1)
