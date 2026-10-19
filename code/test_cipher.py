"""Unit tests: python -m pytest -q test_cipher.py"""
import hashlib, struct, numpy as np, pytest
import cipher as C

KEY = C.KEY
NONCE = bytes(range(16))
IMG = (np.arange(16 * 16 * 3, dtype=np.int64) * 7 % 256).astype(np.uint8).reshape(16, 16, 3)
G16 = (np.arange(12 * 10, dtype=np.int64) * 977 % 65536).astype(np.uint16).reshape(12, 10)


def test_hkdf_rfc5869_case1():
    ikm = bytes.fromhex('0b' * 22); salt = bytes.fromhex('000102030405060708090a0b0c'); info = bytes.fromhex('f0f1f2f3f4f5f6f7f8f9')
    prk = C.hkdf_extract(salt, ikm)
    assert prk.hex() == '077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5'
    assert C.hkdf_expand(prk, info, 42).hex() == '3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865'


def test_chacha20_rfc8439_block():
    """RFC 8439, Section 2.3.2 (block counter 1), for both implementations."""
    import ref_impl as R
    key = bytes(range(32)); nonce = bytes.fromhex('000000090000004a00000000')
    ref = ('10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e'
           'd2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e')
    assert C.chacha20_stream(key, nonce, 128)[64:].hex() == ref
    assert R.chacha20_block(key, 1, nonce).hex() == ref


def test_map_vector():
    X = np.empty(3); Y = np.empty(3); Z = np.empty(3)
    C.hybrid_steps(0.3, 0.2, 0.6, 3.99, 0.999, 1.99, 3, X, Y, Z, 0)
    got = [struct.pack('>d', v).hex() for v in (X[0], Y[0], Z[0], X[2], Y[2], Z[2])]
    assert got == ['3f7b21b23eb6b000', '3fda9a3c6b3b5300', '3fe9e55bc6ee9100', '3fed114e8e312ac8', '3fc1f22aed430700', '3fdc6c7053c2d780']


def test_output_filter_vector():
    """Eq. (10) for the state (x1, y1, z1) of the map vector."""
    X = np.empty(1); Y = np.empty(1); Z = np.empty(1)
    C.hybrid_steps(0.3, 0.2, 0.6, 3.99, 0.999, 1.99, 1, X, Y, Z, 0)
    Q = C.quantize(X, Y, Z, True)[:, 0]
    x, y, z = X[0], Y[0], Z[0]
    exp = [int(np.floor(((x + y * z) % 1.0) * 1e14)), int(np.floor(((y + z * x) % 1.0) * 1e14)),
           int(np.floor(((z + x * y) % 1.0) * 1e14))]
    assert list(Q) == exp == [TEST_Q1[0], TEST_Q1[1], TEST_Q1[2]]


TEST_Q1 = (34299898996344, 42102422198342, 81200119321455)   # Table 11 (computed with ref_impl.py)


VEC_RGB = ('c223972fa025e650bfb29bf8a9f46455b9ce84ed743c299f4e0a71bd3b8423bc', '59ec165882676ed111d40ff3eb6c3e6e64da312a1658c7d815c10f527d6d54f6', 'd0f651fa033d1e21f84ee00eb9b4dce861ae61f15fd9dcd5e2d875870ae31523')   # Table 11: SHA-256 of cipher bytes, H*, tag
VEC_G16 = ('3cf1086d8a2afba4adeb84352ece6a36c4c38a23e1e721e75f16302291bfccc9', '3625ca5e8505c0e68e92d2f529e2e075e847c7f1e1def0880c1bf1d29c69718b', 'e6f9631037790157d2db39f395d830975bfb777b4c95939b3c1a399744081fe6')


def test_encryption_vector():
    c, (n, hs, tag) = C.encrypt(IMG, KEY, nonce=NONCE)
    assert hashlib.sha256(c.tobytes()).hexdigest() == VEC_RGB[0]
    assert hs.hex() == VEC_RGB[1]
    assert tag.hex() == VEC_RGB[2]


def test_encryption_vector_16bit():
    c, (n, hs, tag) = C.encrypt(G16, KEY, nonce=NONCE)
    assert hashlib.sha256(c.astype('>u2').tobytes()).hexdigest() == VEC_G16[0]
    assert hs.hex() == VEC_G16[1]
    assert tag.hex() == VEC_G16[2]


@pytest.mark.parametrize('shape', [(1, 1), (1, 7), (5, 3), (17, 13), (32, 32)])
def test_roundtrip_shapes(shape):
    rng = np.random.default_rng(shape[0] * 100 + shape[1])
    im = rng.integers(0, 256, shape + (3,), dtype=np.uint8)
    c, meta = C.encrypt(im, KEY)
    p, tag_ok, hash_ok = C.decrypt(c, meta, KEY)
    assert tag_ok and hash_ok and np.array_equal(p, im)


@pytest.mark.parametrize('dtype,shape', [(np.uint16, (1, 1)), (np.uint16, (19, 23)), (np.uint8, (1, 5)), (np.uint8, (21, 17))])
def test_roundtrip_grey(dtype, shape):
    rng = np.random.default_rng(shape[0] * 7 + shape[1])
    im = rng.integers(0, np.iinfo(dtype).max + 1, shape).astype(dtype)
    c, meta = C.encrypt(im, KEY)
    p, tag_ok, hash_ok = C.decrypt(c, meta, KEY)
    assert c.dtype == dtype and c.shape == shape
    assert tag_ok and hash_ok and np.array_equal(p, im)


def test_format_bound_to_tag():
    """A 16-bit cipher image presented as an 8-bit image of the same bytes is rejected."""
    c, meta = C.encrypt(G16, KEY)
    as8 = c.astype('>u2').view(np.uint8).reshape(12, 20)
    assert C.decrypt(as8, meta, KEY)[0] is None


def test_special_images():
    for v in (0, 255):
        im = np.full((16, 16, 3), v, np.uint8)
        c, meta = C.encrypt(im, KEY)
        p, t, h = C.decrypt(c, meta, KEY)
        assert t and h and np.array_equal(p, im)


def test_probabilistic():
    c1, m1 = C.encrypt(IMG, KEY); c2, m2 = C.encrypt(IMG, KEY)
    assert m1[0] != m2[0] and not np.array_equal(c1, c2) and m1[1] != m2[1]


def test_same_nonce_deterministic():
    c1, m1 = C.encrypt(IMG, KEY, nonce=NONCE); c2, m2 = C.encrypt(IMG, KEY, nonce=NONCE)
    assert np.array_equal(c1, c2) and m1 == m2


def test_wrong_key_rejected():
    c, meta = C.encrypt(IMG, KEY)
    k2 = list(KEY); k2[0] += 1e-13
    p, tag_ok, hash_ok = C.decrypt(c, meta, tuple(k2))
    assert p is None and not tag_ok


def test_tamper_cipher_nonce_hstar_tag():
    c, (n, hs, tag) = C.encrypt(IMG, KEY)
    c2 = c.copy(); c2[3, 4, 1] ^= 1
    assert C.decrypt(c2, (n, hs, tag), KEY)[0] is None
    assert C.decrypt(c, (bytes([n[0] ^ 1]) + n[1:], hs, tag), KEY)[0] is None
    assert C.decrypt(c, (n, bytes([hs[0] ^ 1]) + hs[1:], tag), KEY)[0] is None
    assert C.decrypt(c, (n, hs, bytes([tag[0] ^ 1]) + tag[1:]), KEY)[0] is None


def test_tamper_dimensions():
    c, meta = C.encrypt(IMG, KEY)
    flat = c.reshape(8, 32, 3)             # same bytes, different size
    assert C.decrypt(flat, meta, KEY)[0] is None
    assert C.decrypt(c[:, :15], meta, KEY)[0] is None


def test_invalid_inputs():
    with pytest.raises(ValueError):
        C.encrypt(IMG.astype(np.int32), KEY)
    with pytest.raises(ValueError):
        C.encrypt(IMG[:, :, :2], KEY)
    with pytest.raises(ValueError):
        C.encrypt(IMG, (1.5,) + KEY[1:])
    with pytest.raises(ValueError):
        C.encrypt(IMG, KEY[:5])
    c, (n, hs, tag) = C.encrypt(IMG, KEY)
    with pytest.raises(ValueError):
        C.decrypt(c, (n[:8], hs, tag), KEY)


def test_nonce_changes_keystream_not_only_mask():
    c1, _ = C.encrypt(IMG, KEY, nonce=NONCE); c2, _ = C.encrypt(IMG, KEY, nonce=bytes(16))
    assert np.mean(c1 == c2) < 0.02


def test_one_pixel_change_diffuses():
    im2 = IMG.copy(); im2[0, 0, 0] ^= 1
    c1, _ = C.encrypt(IMG, KEY, nonce=NONCE); c2, _ = C.encrypt(im2, KEY, nonce=NONCE)
    assert np.mean(c1 != c2) > 0.95


def test_anchor_and_filter_change_key_material():
    """The chaos-only variants (no anchor, no filter) give different cipher images."""
    c0, _ = C.encrypt(IMG, KEY, nonce=NONCE)
    c1, _ = C.encrypt(IMG, KEY, nonce=NONCE, cfg=dict(anchor=False))
    c2, _ = C.encrypt(IMG, KEY, nonce=NONCE, cfg=dict(filt=False))
    assert np.mean(c0 == c1) < 0.02 and np.mean(c0 == c2) < 0.02 and np.mean(c1 == c2) < 0.02


@pytest.mark.parametrize('shape', [(1, 1), (3, 5), (16, 16), (20, 12), (40, 40)])
def test_independent_reference_implementation(shape):
    """ref_impl.py was written from the text of the paper and shares no code with cipher.py."""
    import ref_impl as R
    rng = np.random.default_rng(shape[0] * 31 + shape[1])
    im = rng.integers(0, 256, shape + (3,), dtype=np.uint8)
    rc, rh, rt = R.encrypt(im.tolist(), KEY, NONCE)
    c, (n, hs, tag) = C.encrypt(im, KEY, nonce=NONCE)
    assert np.array_equal(np.array(rc, dtype=np.uint8), c) and rh == hs and rt == tag


@pytest.mark.parametrize('fmt,shape', [(2, (1, 1)), (2, (7, 9)), (1, (5, 8))])
def test_independent_reference_grey(fmt, shape):
    import ref_impl as R
    rng = np.random.default_rng(fmt * 1000 + shape[0])
    dt = np.uint16 if fmt == 2 else np.uint8
    im = rng.integers(0, np.iinfo(dt).max + 1, shape).astype(dt)
    rc, rh, rt = R.encrypt([[[int(v)] for v in row] for row in im], KEY, NONCE, fmt=fmt)
    c, (n, hs, tag) = C.encrypt(im, KEY, nonce=NONCE)
    assert np.array_equal(np.array(rc).reshape(shape).astype(dt), c) and rh == hs and rt == tag


def test_independent_reference_random_keys():
    import ref_impl as R
    rng = np.random.default_rng(9)
    for _ in range(5):
        key = (*rng.random(3), rng.uniform(3.7, 4), rng.uniform(.9, 1), rng.uniform(1.8, 2))
        nonce = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
        im = rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)
        rc, rh, rt = R.encrypt(im.tolist(), tuple(key), nonce)
        c, (n, hs, tag) = C.encrypt(im, tuple(key), nonce=nonce)
        assert np.array_equal(np.array(rc, dtype=np.uint8), c) and rh == hs and rt == tag
