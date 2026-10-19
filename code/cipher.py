"""Colour image cipher based on a hybrid 3D chaotic map and DNA coding (version 2).
Reference implementation (IEEE-754 double precision, numba-compiled).

Version 2 adds, with respect to the first version: (i) a nonlinear cross-coordinate output filter
of the chaotic states (Eq. (10)), which removes the linear relation modulo 256 between consecutive
key-stream bytes; (ii) a cryptographic anchor: every quantized chaotic value is combined by XOR with
a 47-bit word of ChaCha20 under a message key K_s = HKDF(n, K, 'stream' || H) (Eq. (11)); (iii) four
rounds instead of three; (iv) grey-level images of 8 and 16 bits (one and two byte planes)."""
import hashlib, hmac, os, struct, math
from fractions import Fraction
import numpy as np
from numba import njit

# ---- 3D Arnold cat matrix (det = 1) ---------------------------------------
A11, A12, A13 = 3.0, 1.0, 4.0
A21, A22, A23 = 8.0, 3.0, 11.0
A31, A32, A33 = 6.0, 2.0, 9.0

N_TRANSIENT = 1000
SCALE = 1e14          # decimal scaling of the quantization, Eq. (10)
ROUNDS = 4            # number of rounds
MASK47 = (1 << 47) - 1
MAX_PIXELS = 1 << 30  # M*N limit (ChaCha20 block counter of one round)

# ---- DNA rules (Table 1). Base index: A=0, C=1, G=2, T=3 -------------------
# code assigned to (A, T, C, G) for rules 1..8
_RULES = [(0, 3, 1, 2), (0, 3, 2, 1), (3, 0, 1, 2), (3, 0, 2, 1),
          (2, 1, 0, 3), (2, 1, 3, 0), (1, 2, 0, 3), (1, 2, 3, 0)]
ENC = np.zeros((8, 4), dtype=np.int64)   # ENC[rule, base] -> 2-bit code
DEC = np.zeros((8, 4), dtype=np.int64)   # DEC[rule, code] -> base
for _r, (ca, ct, cc, cg) in enumerate(_RULES):
    for base, code in zip((0, 3, 1, 2), (ca, ct, cc, cg)):   # A,T,C,G
        ENC[_r, base] = code
        DEC[_r, code] = base


# ---- sin(pi t) for t in [0, 0.5] by a Taylor polynomial of degree 27 ----------
# Only additions and multiplications are used, so every IEEE-754 binary64 platform
# (without fused multiply-add and extended precision) gives bit-identical results.
# Truncation error < 1e-25, i.e. far below the double-precision resolution.
_SC = np.array([float(Fraction((-1) ** k, math.factorial(2 * k + 1))) for k in range(14)])
_PI = math.pi


@njit(cache=True)
def psin(t):
    u = _PI * t
    u2 = u * u
    acc = _SC[13]
    for k in range(12, -1, -1):
        acc = acc * u2 + _SC[k]
    return u * acc


@njit(cache=True)
def hybrid_steps(x, y, z, gam, r, mu, n, X, Y, Z, mode=0):
    """Iterate the proposed hybrid 3D map n times; write states into X, Y, Z.
    mode 0: proposed map; 1: three uncoupled Logistic maps; 2: hybrid branches without
    Arnold coupling; 3: Arnold coupling of Logistic maps only (modes 1-3: ablation)."""
    for i in range(n):
        if mode == 0 or mode == 2:
            if x < 0.5:
                xx = r * psin(x)
            else:
                xx = gam * x * (1.0 - x)
            if y < 0.5:
                yy = mu * y                     # Tent branch (y < 0.5)
            else:
                yy = r * psin(1.0 - y)          # Sine branch (y >= 0.5), sin(pi y) = sin(pi (1-y))
            if z < 0.5:
                zz = gam * z * (1.0 - z)
            else:
                zz = mu * (1.0 - z)
        else:
            xx = gam * x * (1.0 - x); yy = gam * y * (1.0 - y); zz = gam * z * (1.0 - z)
        if mode == 0 or mode == 3:
            x = (A11 * xx + A12 * yy + A13 * zz) % 1.0
            y = (A21 * xx + A22 * yy + A23 * zz) % 1.0
            z = (A31 * xx + A32 * yy + A33 * zz) % 1.0
        else:
            x = xx % 1.0; y = yy % 1.0; z = zz % 1.0
        X[i] = x; Y[i] = y; Z[i] = z
    return x, y, z


def hybrid_steps_py(x, y, z, gam, r, mu, n):
    """Pure-Python (no Numba, no libm) reference of mode 0 for bit-equality tests."""
    def ps(t):
        u = _PI * t; u2 = u * u; acc = float(_SC[13])
        for k in range(12, -1, -1):
            acc = acc * u2 + float(_SC[k])
        return u * acc
    out = []
    for _ in range(n):
        xx = r * ps(x) if x < 0.5 else gam * x * (1.0 - x)
        yy = mu * y if y < 0.5 else r * ps(1.0 - y)
        zz = gam * z * (1.0 - z) if z < 0.5 else mu * (1.0 - z)
        x = (A11 * xx + A12 * yy + A13 * zz) % 1.0
        y = (A21 * xx + A22 * yy + A23 * zz) % 1.0
        z = (A31 * xx + A32 * yy + A33 * zz) % 1.0
        out.append((x, y, z))
    return out


@njit(cache=True)
def dna_chain(p, ra, rb, kk, iv, forward, enc, dec, sbox, sinv, inverse, use_dna, use_sbox):
    """One chained pass over the sequence p (values 0..255 held in int64).
    Encryption of element i:   s_i = DNAadd_a( p_i , DNAenc_b(k_i) )
                               c_i = S[ (s_i + c_prev) mod 256 ]
    inverse=True performs the exact inverse (c_prev is the previous cipher value).
    use_dna=False replaces the DNA addition by addition mod 256 and use_sbox=False
    removes the substitution box (both only for the ablation study)."""
    n = p.shape[0]
    out = np.empty(n, dtype=np.int64)
    prev = iv
    for t in range(n):
        i = t if forward else n - 1 - t
        a = ra[i]; b = rb[i]
        v = p[i]
        if not inverse:
            if use_dna:
                res = 0
                for j in range(4):
                    sh = 6 - 2 * j
                    bv = dec[a, (v >> sh) & 3]
                    bm = dec[b, (kk[i] >> sh) & 3]
                    res |= enc[a, (bv + bm) & 3] << sh
            else:
                res = (v + kk[i]) & 255
            if use_sbox:
                c = sbox[(res + prev) & 255]
            else:
                c = (res + prev) & 255
            out[i] = c
            prev = c
        else:
            if use_sbox:
                u = (sinv[v] - prev) & 255
            else:
                u = (v - prev) & 255
            if use_dna:
                res = 0
                for j in range(4):
                    sh = 6 - 2 * j
                    bv = dec[a, (u >> sh) & 3]
                    bm = dec[b, (kk[i] >> sh) & 3]
                    res |= enc[a, (bv - bm) & 3] << sh
            else:
                res = (u - kk[i]) & 255
            out[i] = res
            prev = v
    return out


# ---- output filter and quantization (Eq. (10)) -------------------------------------
@njit(cache=True)
def quantize(X, Y, Z, filt):
    """Q[c, i] = floor(1e14 * g_c(s_i)) with the cross-coordinate filter
    g_x = (x + y*z) mod 1, g_y = (y + z*x) mod 1, g_z = (z + x*y) mod 1   (binary64, rounded products).
    filt=False gives floor(1e14 * v) of the raw state (version 1; ablation only)."""
    n = X.shape[0]
    Q = np.empty((3, n), dtype=np.int64)
    for i in range(n):
        x = X[i]; y = Y[i]; z = Z[i]
        if filt:
            gx = (x + y * z) % 1.0
            gy = (y + z * x) % 1.0
            gz = (z + x * y) % 1.0
        else:
            gx = x; gy = y; gz = z
        Q[0, i] = int(math.floor(gx * SCALE))
        Q[1, i] = int(math.floor(gy * SCALE))
        Q[2, i] = int(math.floor(gz * SCALE))
    return Q


def quant(v):
    """floor(1e14 * v) of raw states (used only by analysis scripts)."""
    return np.floor(v * SCALE).astype(np.int64)


# ---- cryptographic anchor (Eq. (11)) -------------------------------------------------
def chacha20_stream(key32, nonce12, nbytes):
    """ChaCha20 key stream of RFC 8439 (block counter starting at 0)."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    enc = Cipher(algorithms.ChaCha20(key32, b'\x00\x00\x00\x00' + nonce12), mode=None).encryptor()
    return enc.update(b'\x00' * nbytes)


def anchor_words(ks, rnd, n_iter):
    """47-bit words W[c, i] of round rnd (1, 2, ...): word j = 3i + c is bytes 8j..8j+7 of the
    ChaCha20 key stream (key K_s, nonce = LE32(rnd) || 0^8), read little-endian, masked to 47 bits."""
    raw = chacha20_stream(ks, struct.pack('<I', rnd) + bytes(8), 24 * n_iter)
    w = np.frombuffer(raw, dtype='<u8') & np.uint64(MASK47)
    return np.ascontiguousarray(w.astype(np.int64).reshape(n_iter, 3).T)


def _kb(key):
    return struct.pack('>6d', *key)


# ---- image formats ---------------------------------------------------------------
# format code f = number of byte planes P: 1 = 8-bit grey, 2 = 16-bit grey (high byte, low byte), 3 = 8-bit RGB
def image_format(img):
    if img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
        return 3
    if img.ndim == 2 and img.dtype == np.uint16:
        return 2
    if img.ndim == 2 and img.dtype == np.uint8:
        return 1
    raise ValueError('the image must be uint8 (M,N,3), uint16 (M,N) or uint8 (M,N)')


def to_planes(img):
    """(P, L) uint8 byte planes; for 16-bit images plane 0 is the high byte."""
    f = image_format(img)
    if f == 3:
        return np.ascontiguousarray(np.transpose(img, (2, 0, 1)).reshape(3, -1))
    if f == 2:
        v = img.reshape(-1)
        return np.stack([(v >> 8).astype(np.uint8), (v & 255).astype(np.uint8)])
    return img.reshape(1, -1).copy()


def from_planes(planes, shape, f):
    M, N = shape
    if f == 3:
        return np.transpose(planes.reshape(3, M, N), (1, 2, 0)).astype(np.uint8).copy()
    if f == 2:
        return ((planes[0].astype(np.uint16) << 8) | planes[1].astype(np.uint16)).reshape(M, N)
    return planes[0].reshape(M, N).astype(np.uint8)


def image_bytes(img):
    """Sample bytes in row-major order, planes interleaved (RGB; 16-bit samples big-endian)."""
    f = image_format(img)
    if f == 2:
        return img.astype('>u2').tobytes()
    return np.ascontiguousarray(img, dtype=np.uint8).tobytes()


def image_digest(img):
    """H = SHA-256(M || N || f || sample bytes), M and N 16-bit big-endian, f one byte."""
    f = image_format(img)
    h = hashlib.sha256()
    h.update(struct.pack('>HHB', img.shape[0], img.shape[1], f))
    h.update(image_bytes(img))
    return h.digest()


def hkdf_extract(salt, ikm):
    """HKDF-Extract of RFC 5869 with HMAC-SHA-256."""
    return hmac.new(salt if len(salt) else b'\x00' * 32, ikm, hashlib.sha256).digest()


def hkdf_expand(prk, info, length):
    """HKDF-Expand of RFC 5869 with HMAC-SHA-256 (length <= 255*32)."""
    okm, t = b'', b''
    for i in range(1, -(-length // 32) + 1):
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
    return okm[:length]


def seed_digest(digest, nonce):
    """E = HKDF-SHA-256(salt = nonce, IKM = H, info = 'init', 24 bytes): the three 64-bit
    words that shift the initial values of the map (Eq. 9).  Not secret."""
    return hkdf_expand(hkdf_extract(nonce, digest), b'init', 24)


def derive_initial(key, E):
    """Message-dependent initial values: x0' = x0 + u1 (mod 1), ... (Eq. 9)."""
    x0, y0, z0 = key[:3]
    u = [int.from_bytes(E[8 * i: 8 * i + 8], 'big') / 2.0 ** 64 for i in range(3)]
    out = []
    for v, d in zip((x0, y0, z0), u):
        w = (v + d) % 1.0
        if min(w, 1.0 - w) < 1e-3:
            w = (w + 0.25) % 1.0
        out.append(w)
    return out


def _prk(key, nonce):
    return hkdf_extract(nonce, _kb(key))


def key_mask(key, nonce):
    """Mask of the digest: HKDF-SHA-256(salt = nonce, IKM = key bytes, info = 'mask', 32 bytes)."""
    return hkdf_expand(_prk(key, nonce), b'mask', 32)


def stream_key(key, nonce, digest):
    """K_s = HKDF-SHA-256(salt = nonce, IKM = key bytes, info = 'stream' || H, 32 bytes)."""
    return hkdf_expand(_prk(key, nonce), b'stream' + digest, 32)


def mac_tag(key, nonce, hstar, shape, f, cipher_bytes):
    """HMAC-SHA-256 under K_mac = HKDF-SHA-256(salt = nonce, IKM = key bytes, info = 'mac')
    over n || H* || M || N || f || C."""
    k = hkdf_expand(_prk(key, nonce), b'mac', 32)
    m = hmac.new(k, nonce + hstar + struct.pack('>HHB', shape[0], shape[1], f) + cipher_bytes, hashlib.sha256)
    return m.digest()


def xor_bytes(a, b):
    return bytes(i ^ j for i, j in zip(a, b))


class RoundStreams:
    pass


DEFAULT_CFG = dict(dna=True, sbox=True, mapmode=0, filt=True, anchor=True)


def _rank(q):
    """Algorithm 1: rank of every element in ascending order, ties broken by the index."""
    order = np.argsort(q, kind='stable')
    tp = np.empty(q.shape[0], dtype=np.int64)
    tp[order] = np.arange(q.shape[0])
    return tp


def round_values(state, params, L, rnd, ks, cfg):
    """Iterate the map 8L+514 times and return Q' (3, 8L+514): filtered, quantized and anchored values."""
    gam, r, mu = params
    n = 8 * L + 514
    X = np.empty(n); Y = np.empty(n); Z = np.empty(n)
    st = hybrid_steps(state[0], state[1], state[2], gam, r, mu, n, X, Y, Z, cfg.get('mapmode', 0))
    Q = quantize(X, Y, Z, cfg.get('filt', True))
    del X, Y, Z
    if cfg.get('anchor', True):
        # the same words as anchor_words(ks, rnd, n), generated in chunks to bound the memory
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
        enc = Cipher(algorithms.ChaCha20(ks, b'\x00\x00\x00\x00' + struct.pack('<I', rnd) + bytes(8)), mode=None).encryptor()
        CH = 1 << 20
        for a in range(0, n, CH):
            m = min(CH, n - a)
            w = np.frombuffer(enc.update(b'\x00' * (24 * m)), dtype='<u8').reshape(m, 3) & np.uint64(MASK47)
            Q[:, a:a + m] ^= w.T.astype(np.int64)
    return Q, st


def gen_round(state, params, L, P, rnd, ks, cfg):
    """Key material of one round (Table 2) for P byte planes."""
    Q, st = round_values(state, params, L, rnd, ks, cfg)
    blk = [Q[:, b * L:(b + 1) * L] for b in range(8)]
    rs = RoundStreams()
    rs.TP = np.stack([_rank(blk[0][c]) for c in range(P)])                    # block 1
    flat = lambda b: np.ascontiguousarray(blk[b][:P].reshape(-1))              # planes concatenated
    rs.RA1 = flat(1) % 8; rs.RB1 = flat(2) % 8; rs.KD1 = flat(3) % 256           # blocks 2-4
    rs.RA2 = flat(4) % 8; rs.RB2 = flat(5) % 8; rs.KD2 = flat(6) % 256           # blocks 5-7
    rs.K3 = blk[7][:P] % 256                                                    # block 8
    tail = Q[0, 8 * L:]                                                         # block 9 (x only)
    rs.IV1 = int(tail[0] % 256); rs.IV2 = int(tail[1] % 256)
    rs.S1 = _rank(tail[2:258]); rs.S2 = _rank(tail[258:514])
    rs.S1i = np.argsort(rs.S1); rs.S2i = np.argsort(rs.S2)
    return rs, st


def _setup(key, nonce, H, L, P, rounds=ROUNDS, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    mm = cfg.get('mapmode', 0)
    x0, y0, z0 = derive_initial(key, seed_digest(H, nonce))
    params = key[3:6]
    X = np.empty(N_TRANSIENT); Y = np.empty(N_TRANSIENT); Z = np.empty(N_TRANSIENT)
    st = hybrid_steps(x0, y0, z0, params[0], params[1], params[2], N_TRANSIENT, X, Y, Z, mm)
    ks = stream_key(key, nonce, H) if cfg.get('anchor', True) else None
    streams = []
    for t in range(1, rounds + 1):
        rs, st = gen_round(st, params, L, P, t, ks, cfg)
        streams.append(rs)
    return streams


def _round_forward(planes, rs, L, cfg):
    d_, s_ = cfg.get('dna', True), cfg.get('sbox', True)
    P = planes.shape[0]
    conf = np.empty_like(planes)
    for c in range(P):
        conf[c][rs.TP[c]] = planes[c]
    seq = conf.reshape(-1).astype(np.int64)
    d = dna_chain(seq, rs.RA1, rs.RB1, rs.KD1, rs.IV1, True, ENC, DEC, rs.S1, rs.S1i, False, d_, s_)
    d = dna_chain(d, rs.RA2, rs.RB2, rs.KD2, rs.IV2, False, ENC, DEC, rs.S2, rs.S2i, False, d_, s_)
    d = d.reshape(P, L) ^ rs.K3
    return d.astype(np.uint8)


def _round_inverse(c_planes, rs, L, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    d_, s_ = cfg.get('dna', True), cfg.get('sbox', True)
    P = c_planes.shape[0]
    d = (c_planes.astype(np.int64) ^ rs.K3).reshape(-1)
    d = dna_chain(d, rs.RA2, rs.RB2, rs.KD2, rs.IV2, False, ENC, DEC, rs.S2, rs.S2i, True, d_, s_)
    d = dna_chain(d, rs.RA1, rs.RB1, rs.KD1, rs.IV1, True, ENC, DEC, rs.S1, rs.S1i, True, d_, s_)
    conf = d.reshape(P, L)
    out = np.empty_like(conf)
    for c in range(P):
        out[c] = conf[c][rs.TP[c]]
    return out.astype(np.uint8)


KEY_RANGES = ((0, 1), (0, 1), (0, 1), (3.7, 4.0), (0.9, 1.0), (1.8, 2.0))


def check_key(key):
    if len(key) != 6:
        raise ValueError('the key has six parameters')
    for v, (lo_, hi_) in zip(key, KEY_RANGES):
        if not (lo_ <= v <= hi_) or v != v:
            raise ValueError('key parameter outside the permitted range')


def check_image(img):
    image_format(img)
    if not (1 <= img.shape[0] <= 65535 and 1 <= img.shape[1] <= 65535):
        raise ValueError('image size outside 1..65535')
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise ValueError('more than 2^30 pixels')


def encrypt(img, key, rounds=ROUNDS, nonce=None, digest=None, cfg=None, trace=None):
    """img: uint8 (M,N,3) RGB, uint16 (M,N) or uint8 (M,N) grey.  Returns (cipher image, meta),
    meta = (nonce, H*, tag).  nonce: 16 bytes, fresh random by default (pass one for reproducible
    experiments).  digest: override of H (only for the experiments that hold H fixed).
    trace: optional list that receives the planes after every round (figures only)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    check_key(key); check_image(img)
    f = image_format(img)
    M, N = img.shape[:2]
    L = M * N
    if nonce is None:
        nonce = os.urandom(16)
    if len(nonce) != 16:
        raise ValueError('the nonce has 16 bytes')
    H = image_digest(img) if digest is None else digest
    cur = to_planes(img)
    for rs in _setup(key, nonce, H, L, f, rounds, cfg):
        cur = _round_forward(cur, rs, L, cfg)
        if trace is not None:
            trace.append(from_planes(cur, (M, N), f))
    cipher = from_planes(cur, (M, N), f)
    hstar = xor_bytes(H, key_mask(key, nonce))
    tag = mac_tag(key, nonce, hstar, (M, N), f, image_bytes(cipher))
    return cipher, (nonce, hstar, tag)


def decrypt(cipher, meta, key, rounds=ROUNDS, check_tag=True, cfg=None):
    """Returns (plain image or None, tag_ok, hash_ok).  With check_tag=True a modified
    cipher image or a wrong key is rejected before any decryption is done."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    check_key(key); check_image(cipher)
    nonce, hstar, tag = meta
    if len(nonce) != 16 or len(hstar) != 32 or len(tag) != 32:
        raise ValueError('malformed metadata')
    f = image_format(cipher)
    M, N = cipher.shape[:2]
    L = M * N
    tag_ok = hmac.compare_digest(tag, mac_tag(key, nonce, hstar, (M, N), f, image_bytes(cipher)))
    if check_tag and not tag_ok:
        return None, False, False
    H = xor_bytes(hstar, key_mask(key, nonce))
    streams = _setup(key, nonce, H, L, f, rounds, cfg)
    cur = to_planes(cipher)
    for rs in reversed(streams):
        cur = _round_inverse(cur, rs, L, cfg)
    plain = from_planes(cur, (M, N), f)
    return plain, tag_ok, image_digest(plain) == H


# default secret key used in the experiments
KEY = (0.31415926535897, 0.27182818284590, 0.57721566490153, 3.99, 0.999, 1.99)
