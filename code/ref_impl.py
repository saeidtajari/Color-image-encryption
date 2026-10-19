"""Independent, slow re-implementation of the cipher (version 2) written from the text of the paper
(Sections 2.1.5, 2.2, Tables 1-2, Algorithms 1-5, Eqs. (5)-(16)).  It shares no code with cipher.py
(only hashlib/hmac/struct/math from the standard library); ChaCha20 is implemented here from RFC 8439."""
import hashlib, hmac, math, struct
from fractions import Fraction

# Table 1: bit pairs of the bases (A, T, C, G) for the rules 1..8
TABLE2 = {'A': ['00', '00', '11', '11', '10', '10', '01', '01'],
          'T': ['11', '11', '00', '00', '01', '01', '10', '10'],
          'C': ['01', '10', '01', '10', '00', '11', '00', '11'],
          'G': ['10', '01', '10', '01', '11', '00', '11', '00']}
NUM = {'A': 0, 'C': 1, 'G': 2, 'T': 3}             # bases ordered A<C<G<T, numbered 0..3
def dec(rule, pair):                                # bit pair -> base number under rule (1..8)
    for b in 'ATCG':
        if TABLE2[b][rule - 1] == pair: return NUM[b]
def enc(rule, num):                                 # base number -> bit pair
    for b in 'ATCG':
        if NUM[b] == num: return TABLE2[b][rule - 1]

# sin(pi t): Taylor polynomial of degree 27 in u = pi t, Horner in u^2, coefficients (-1)^k/(2k+1)!
C = [float(Fraction((-1) ** k, math.factorial(2 * k + 1))) for k in range(14)]
def psin(t):
    u = math.pi * t; u2 = u * u; acc = C[13]
    for k in range(12, -1, -1): acc = acc * u2 + C[k]
    return u * acc

def step(s, g, r, m):
    x, y, z = s
    xx = r * psin(x) if x < 0.5 else (g * x) * (1.0 - x)           # Eq. (5)
    yy = m * y if y < 0.5 else r * psin(1.0 - y)                   # Eq. (6)  (sin(pi y) = sin(pi (1-y)))
    zz = (g * z) * (1.0 - z) if z < 0.5 else m * (1.0 - z)         # Eq. (7)
    return ((3.0 * xx + 1.0 * yy + 4.0 * zz) % 1.0,                # Eq. (8)
            (8.0 * xx + 3.0 * yy + 11.0 * zz) % 1.0,
            (6.0 * xx + 2.0 * yy + 9.0 * zz) % 1.0)

def hkdf(salt, ikm, info, n):
    prk = hmac.new(salt, ikm, hashlib.sha256).digest(); out = b''; t = b''; i = 1
    while len(out) < n:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest(); out += t; i += 1
    return out[:n]

# ---- ChaCha20 (RFC 8439, Section 2.3 and 2.4) -----------------------------------------
def _rotl(v, c): return ((v << c) & 0xffffffff) | (v >> (32 - c))
def _qr(s, a, b, c, d):
    s[a] = (s[a] + s[b]) & 0xffffffff; s[d] = _rotl(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & 0xffffffff; s[b] = _rotl(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b]) & 0xffffffff; s[d] = _rotl(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & 0xffffffff; s[b] = _rotl(s[b] ^ s[c], 7)
def chacha20_block(key, counter, nonce):
    st = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574] + list(struct.unpack('<8I', key)) + \
         [counter] + list(struct.unpack('<3I', nonce))
    w = st[:]
    for _ in range(10):
        _qr(w, 0, 4, 8, 12); _qr(w, 1, 5, 9, 13); _qr(w, 2, 6, 10, 14); _qr(w, 3, 7, 11, 15)
        _qr(w, 0, 5, 10, 15); _qr(w, 1, 6, 11, 12); _qr(w, 2, 7, 8, 13); _qr(w, 3, 4, 9, 14)
    return struct.pack('<16I', *[(a + b) & 0xffffffff for a, b in zip(w, st)])
def chacha20_stream(key, nonce, nbytes, counter=0):
    out = bytearray()
    while len(out) < nbytes:
        out += chacha20_block(key, counter, nonce); counter += 1
    return bytes(out[:nbytes])

def encrypt(img, key, nonce, rounds=4, fmt=3):
    """img: list of rows; a row is a list of pixels; a pixel is [R,G,B] (fmt 3), [v] with v < 2^16 (fmt 2)
    or [v] with v < 256 (fmt 1).  Returns (cipher image in the same layout, H*, tag)."""
    M, N = len(img), len(img[0]); L = M * N; P = fmt
    kb = struct.pack('>6d', *key)
    # sample bytes, row-major, planes interleaved; 16-bit samples big-endian
    if fmt == 2:
        planes_img = [[[v >> 8, v & 255] for v in (px[0] for px in row)] for row in img]
    else:
        planes_img = img
    sample = bytes(v for row in planes_img for px in row for v in px)
    H = hashlib.sha256(struct.pack('>HHB', M, N, fmt) + sample).digest()
    E = hkdf(nonce, H, b'init', 24)                                 # Section 2.2.1
    Ks = hkdf(nonce, kb, b'stream' + H, 32)                          # Eq. (11)
    x0, y0, z0, g, r, m = key
    st = []
    for v, i in zip((x0, y0, z0), range(3)):
        u = int.from_bytes(E[8 * i:8 * i + 8], 'big') / 2.0 ** 64
        w = (v + u) % 1.0
        if min(w, 1.0 - w) < 1e-3: w = (w + 0.25) % 1.0
        st.append(w)
    st = tuple(st)
    for _ in range(1000): st = step(st, g, r, m)
    ch = [[planes_img[i][j][c] for i in range(M) for j in range(N)] for c in range(P)]   # row-major flatten
    for rd in range(1, rounds + 1):
        n = 8 * L + 514
        ksb = chacha20_stream(Ks, struct.pack('<I', rd) + bytes(8), 24 * n)
        X = [[], [], []]
        for i in range(n):
            st = step(st, g, r, m)
            x, y, z = st
            gv = ((x + y * z) % 1.0, (y + z * x) % 1.0, (z + x * y) % 1.0)     # Eq. (10), filter
            for c in range(3):
                q = math.floor(gv[c] * 1e14)                                    # Eq. (10), quantization
                j = 3 * i + c
                wd = int.from_bytes(ksb[8 * j:8 * j + 8], 'little') & ((1 << 47) - 1)
                X[c].append(q ^ wd)                                             # Eq. (11), anchor
        blk = [[X[c][b * L:(b + 1) * L] for c in range(3)] for b in range(8)]
        flat = lambda b: [v for c in range(P) for v in blk[b][c]]               # planes in order
        TP = []
        for c in range(P):                                                      # block 1, Algorithm 1
            order = sorted(range(L), key=lambda i: (blk[0][c][i], i)); tp = [0] * L
            for jj, i in enumerate(order): tp[i] = jj
            TP.append(tp)
        conf = []
        for c in range(P):
            out = [0] * L
            for i in range(L): out[TP[c][i]] = ch[c][i]
            conf.append(out)
        D = [v for c in range(P) for v in conf[c]]
        da1 = [v % 8 + 1 for v in flat(1)]; db1 = [v % 8 + 1 for v in flat(2)]; k1 = [v % 256 for v in flat(3)]
        da2 = [v % 8 + 1 for v in flat(4)]; db2 = [v % 8 + 1 for v in flat(5)]; k2 = [v % 256 for v in flat(6)]
        K3 = [[v % 256 for v in blk[7][c]] for c in range(P)]
        tail = X[0][8 * L:]                                                     # block 9 (x only)
        iv1, iv2 = tail[0] % 256, tail[1] % 256
        def sbox(vals):
            order = sorted(range(256), key=lambda i: (vals[i], i)); s = [0] * 256
            for jj, i in enumerate(order): s[i] = jj
            return s
        S1, S2 = sbox(tail[2:258]), sbox(tail[258:514])
        def passf(D, da, db, kk, iv, S, forward):
            out = [0] * len(D); prev = iv
            idx = range(len(D)) if forward else range(len(D) - 1, -1, -1)
            for i in idx:
                pb = format(D[i], '08b'); kbits = format(kk[i], '08b'); e = ''
                for jj in range(4):
                    a = dec(da[i], pb[2 * jj:2 * jj + 2]); b = dec(db[i], kbits[2 * jj:2 * jj + 2])
                    e += enc(da[i], (a + b) % 4)
                c = S[(int(e, 2) + prev) % 256]; out[i] = c; prev = c
            return out
        D = passf(D, da1, db1, k1, iv1, S1, True)
        D = passf(D, da2, db2, k2, iv2, S2, False)
        ch = [[D[c * L + i] ^ K3[c][i] for i in range(L)] for c in range(P)]
    cplanes = [[[ch[c][i * N + j] for c in range(P)] for j in range(N)] for i in range(M)]
    cbytes = bytes(v for row in cplanes for px in row for v in px)
    if fmt == 2:
        cipher = [[[(px[0] << 8) | px[1]] for px in row] for row in cplanes]
    else:
        cipher = cplanes
    prk = hmac.new(nonce, kb, hashlib.sha256).digest()
    expand = lambda info: hmac.new(prk, info + b'\x01', hashlib.sha256).digest()
    hstar = bytes(a ^ b for a, b in zip(H, expand(b'mask')))
    tag = hmac.new(expand(b'mac'), nonce + hstar + struct.pack('>HHB', M, N, fmt) + cbytes, hashlib.sha256).digest()
    return cipher, hstar, tag
