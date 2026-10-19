"""Original color files of the USC-SIPI image database (volume "Miscellaneous"), read from images_sipi/; the paper uses
4.1.01, 4.1.08 (256x256) and 4.2.01, 4.2.07 (512x512) (TIFF files as distributed; not redistributed with the code).
Entropy, adjacent-pixel correlation, chi-square, NPCR and UACI (30 one-value changes, complete algorithm and
digest held fixed), exact recovery.  Writes results/sipi.json."""
import numpy as np, json, os, glob, hashlib, time
from PIL import Image
import cipher as C
from metrics import entropy, corr_full, npcr_uaci, chi2_p, crit

DIR = os.environ.get('SIPI_DIR', 'images_sipi')
K = C.KEY; R = {}
files = sorted(glob.glob(os.path.join(DIR, '4.1.0*.tif*')) + glob.glob(os.path.join(DIR, '4.2.0*.tif*')))
for fn in files:
    t0 = time.time()
    raw = open(fn, 'rb').read()
    im = Image.open(fn)
    a = np.array(im.convert('RGB'))
    name = os.path.basename(fn)
    M, N = a.shape[:2]; L = M * N
    nonce = hashlib.sha256(name.encode()).digest()[:16]
    c, meta = C.encrypt(a, K, nonce=nonce)
    p, tk, hk = C.decrypt(c, meta, K)
    H = C.image_digest(a); rng = np.random.default_rng(int.from_bytes(nonce[:4], 'big'))
    cr = crit(L, 0.05); res = {'complete': [], 'fixed': []}
    for t in range(30):
        i, j, ch = int(rng.integers(0, M)), int(rng.integers(0, N)), int(rng.integers(0, 3))
        a2 = a.copy(); a2[i, j, ch] = (int(a2[i, j, ch]) + 1) % 256
        for mode in ('complete', 'fixed'):
            c2, _ = C.encrypt(a2, K, nonce=nonce, digest=(H if mode == 'fixed' else None))
            res[mode].append(np.concatenate(npcr_uaci(c, c2)))
    r = dict(file=name, mode=im.mode, size=f'{M}×{N}', sha256=hashlib.sha256(raw).hexdigest(), exact=bool(np.array_equal(p, a) and tk and hk),
             plain_entropy=entropy(a), cipher_entropy=entropy(c), cipher_entropy_ch=[entropy(c[:, :, k]) for k in range(3)],
             plain_corr_max=float(max(abs(v) for k in range(3) for v in corr_full(a[:, :, k]))),
             cipher_corr_max=float(max(abs(v) for k in range(3) for v in corr_full(c[:, :, k]))),
             chi_p=[chi2_p(c[:, :, k])[1] for k in range(3)])
    for mode, v in res.items():
        v = np.array(v)
        r[mode] = dict(npcr=float(v[:, :3].mean()), uaci=float(v[:, 3:].mean()), npcr_pass=float(np.mean(v[:, :3] >= cr['npcr_min'])),
                       uaci_pass=float(np.mean((v[:, 3:] >= cr['uaci_lo']) & (v[:, 3:] <= cr['uaci_hi']))))
    r['sec'] = time.time() - t0
    R[name] = r; print(name, r['size'], round(r['cipher_entropy'], 4), r['fixed'], r['exact'], flush=True)
    json.dump(R, open('results/sipi.json', 'w'), indent=1)
