"""Collects the numbers quoted in the manuscript and the supplementary material from results/*.json
into results/values.json (formatted strings), so that the text and the tables are generated from the results."""
import json, os, math, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stsparse as SP
R = lambda f: json.load(open(os.path.join('results', f)))
NAMES = ['Lena', 'Baboon', 'Peppers', 'Fruits']
V = {}
f2 = lambda x: f'{x:.2f}'; f3 = lambda x: f'{x:.3f}'; f4 = lambda x: f'{x:.4f}'


def sgn4(x):
    return f'{x:.4f}'.replace('-', '-')


main = R('main.json')
# ---- Table 4 / 5 / correlations ------------------------------------------------------------------
corr_all = []
for n in NAMES:
    b = main['b256'][n]
    for c in range(3):
        for d in range(3):
            V[f't4_{n}_{c}_plain_{d}'] = f4(b['corr_plain'][c][d])
            V[f't4_{n}_{c}_cipher_{d}'] = f4(b['corr_cipher'][c][d])
            corr_all.append(abs(b['corr_cipher'][c][d]))
    for c in range(3):
        V[f't5_{n}_plain_{c}'] = f4(b['plain_ch'][c]); V[f't5_{n}_cipher_{c}'] = f4(b['cipher_ch'][c])
    V[f't5_{n}_plain_all'] = f4(b['plain_all']); V[f't5_{n}_cipher_all'] = f4(b['cipher_all'])
V['corr_cipher_maxabs'] = f4(max(corr_all)); V['corr_cipher_maxabs_sd'] = f'{max(corr_all) / 0.0039:.1f}'
V['ideal_H_65536'] = f4(main['ideal_entropy_65536']); V['ideal_H_196608'] = f4(main['ideal_entropy_196608'])
V['ideal_H_262144'] = f4(main['ideal_entropy_262144'])
ln = main['local_null']; V['local_null_mean'] = f4(ln[0]); V['local_null_lo'] = f4(ln[1]); V['local_null_hi'] = f4(ln[2])
V['local_null_sd'] = f4(main['local_null_sd'])
loc = R('local.json')
V['local_pass_avg'] = f'{100 * np.mean([loc[n][c]["pass"] for n in NAMES for c in range(3)]):.1f}'
for n in NAMES:
    for c in range(3):
        V[f's2_{n}_{c}'] = f'{loc[n][c]["mean"]:.4f} ± {loc[n][c]["sd"]:.4f} ({100 * loc[n][c]["pass"]:.0f}%)'
V['local_plain_min'] = f2(min(min(main['b256'][n]['local_plain']) for n in NAMES))
V['local_plain_max'] = f2(max(max(main['b256'][n]['local_plain']) for n in NAMES))

# ---- Table 6, S3, S4 ------------------------------------------------------------------------------
crit = main['crit']['0.05']
for n in NAMES:
    for mode, key in (('full', 'c'), ('fixed', 'f')):
        r = main['npcr'][f'{n}|{mode}']
        V[f't6_{n}_{key}_npcr'] = f4(np.mean(r['mean'][:3])); V[f't6_{n}_{key}_uaci'] = f4(np.mean(r['mean'][3:]))
        V[f't6_{n}_{key}_npass'] = f'{100 * r["pass_npcr"]:.0f}'; V[f't6_{n}_{key}_upass'] = f'{100 * r["pass_uaci"]:.0f}'
    r = main['npcr'][f'{n}|full']
    for c in range(3):
        V[f's3_{n}_{c}'] = f'{r["mean"][c]:.4f} ± {r["std"][c]:.4f}'
        V[f's4_{n}_{c}'] = f'{r["mean"][3 + c]:.4f} ± {r["std"][3 + c]:.4f}'
first = [x for n in NAMES for x in main['npcr'][f'{n}|fixed']['first'][:3]]
last = [x for n in NAMES for x in main['npcr'][f'{n}|fixed']['last'][:3]]
V['first_range'] = f'{min(first):.2f}–{max(first):.2f}'; V['last_range'] = f'{min(last):.2f}–{max(last):.2f}'
V['fresh_mod'] = f2(np.mean([np.mean(main['fresh'][n]['modified'][:3]) for n in NAMES]))
V['fresh_same'] = f2(np.mean([np.mean(main['fresh'][n]['identical_image'][:3]) for n in NAMES]))
# 512 (Table S5 and Table 13)
for n in NAMES:
    b = main['b512'][n]
    V[f's5_{n}_H'] = f4(b['cipher_all'])
    V[f's5_{n}_maxrho'] = f4(max(abs(v) for row in b['corr_cipher'] for v in row))
    V[f's5_{n}_chi'] = f'{sum(p > 0.05 for _, p in b["chi2"])} / 3'
    for mode, key in (('full', 'c'), ('fixed', 'f')):
        r = main['npcr512'][f'{n}|{mode}']
        V[f's5_{n}_{key}_npcr'] = f4(np.mean(r['mean'][:3])); V[f's5_{n}_{key}_uaci'] = f4(np.mean(r['mean'][3:]))
    V[f't13_{n}_512_H'] = f4(np.mean(b['cipher_ch'])); V[f't13_{n}_512_rho'] = f4(np.mean([abs(v) for row in b['corr_cipher'] for v in row]))
    r = main['npcr512'][f'{n}|full']; V[f't13_{n}_512_npcr'] = f4(np.mean(r['mean'][:3])); V[f't13_{n}_512_uaci'] = f4(np.mean(r['mean'][3:]))
    b = main['b256'][n]
    V[f't13_{n}_256_H'] = f4(np.mean(b['cipher_ch'])); V[f't13_{n}_256_rho'] = f4(np.mean([abs(v) for row in b['corr_cipher'] for v in row]))
    r = main['npcr'][f'{n}|full']; V[f't13_{n}_256_npcr'] = f4(np.mean(r['mean'][:3])); V[f't13_{n}_256_uaci'] = f4(np.mean(r['mean'][3:]))
V['H512_all'] = sorted(set(V[f's5_{n}_H'] for n in NAMES))

# ---- special images (3.6) -----------------------------------------------------------------------
sp = main['special']
V['black_H'] = f4(sp['black']['entropy']); V['black_Hch'] = f4(np.mean(sp['black']['entropy_ch']))
V['white_H'] = f4(sp['white']['entropy']); V['white_Hch'] = f4(np.mean(sp['white']['entropy_ch']))
V['bw_npcr'] = f4(np.mean(sp['black_vs_white_same_nonce']['npcr'])); V['bw_uaci'] = f4(np.mean(sp['black_vs_white_same_nonce']['uaci']))
V['special_ok'] = sp['black']['ok'] and sp['white']['ok']

# ---- key sensitivity (Table 7, S7) -------------------------------------------------------------
names6 = ['x0', 'y0', 'z0', 'gamma', 'r', 'mu', 'all']
for n in NAMES:
    dn, cc, tag_rej, hash_fail = [], [], 0, 0
    for i, k in enumerate(names6):
        r = main['keysens'][n][f'{k}|1e-13']
        V[f't7_{i + 1}_{n}_npcr'] = f4(np.mean(r['npcr'])); V[f't7_{i + 1}_{n}_uaci'] = f4(np.mean(r['uaci']))
        dn.append(np.mean(r['dec_npcr'])); cc.append(r['dec_corr']); tag_rej += (not r['tag_ok']); hash_fail += (not r['dec_hash_ok'])
    V[f's7_{n}_npcr'] = f'{min(dn):.2f} – {max(dn):.2f}'; V[f's7_{n}_corr'] = f'{min(cc):.4f} – {max(cc):.4f}'
    V[f's7_{n}_tag'] = f'{tag_rej} of 7'; V[f's7_{n}_hash'] = f'{hash_fail} of 7'
ex = R('extra_new.json')
V['ks_rand_min'] = f2(ex['keysens_random13']['min']); V['ks_rand_max'] = f2(ex['keysens_random13']['max']); V['ks_rand_mean'] = f2(ex['keysens_random13']['mean'])
ab = ex['absorb']; V['abs15'] = ab['1e-15']['unresolved']; V['abs14'] = ab['1e-14']['unresolved']; V['abs13'] = ab['1e-13']['unresolved']; V['abs12'] = ab['1e-12']['unresolved']
V['abs_params'] = [ab[k]['per_param'] for k in ab]
ck = ex['chi2_keys']
V['chi_keys_005'] = f'{100 * ck["pass_005"]:.1f}'; V['chi_keys_001'] = f'{100 * ck["pass_001"]:.1f}'; V['chi_keys_bins'] = ', '.join(map(str, ck['p_hist']))
V['ent_keys'] = f'{ck["mean_entropy"]:.4f} ± {ck["std_entropy"]:.4f}'

# ---- chi-square (S8) ------------------------------------------------------------------------------
npass = 0
for n in NAMES:
    for c in range(3):
        x, p = main['b256'][n]['chi2'][c]
        V[f's8_{n}_{c}'] = f'{x:.2f} ({p:.3f})'; npass += p > 0.05
    V[f's8_{n}_res'] = 'Pass' if all(p > 0.05 for _, p in main['b256'][n]['chi2']) else 'Fail ({} of 3)'.format(sum(p <= 0.05 for _, p in main['b256'][n]['chi2']))
V['chi_npass'] = npass

# ---- reuse attacks (S9), hstar, damage ---------------------------------------------------------
for n in NAMES:
    a = main['attack'][n]
    V[f's9_{n}'] = (f2(np.mean(a['cancel_npcr'])), f4(a['cancel_corr']), f2(np.mean(a['reuse_npcr'])), f4(a['reuse_corr']))
V['hstar_distinct'] = main['hstar']['distinct']; V['hstar_ham'] = f'{main["hstar"]["mean_hamming_between_consecutive"]:.1f}'
V['hstar_bits'] = f3(main['hstar']['bit_freq_mean'])
V['damage_npcr'] = f'{max(np.mean(main["damage"][n]["npcr_unchecked"]) for n in NAMES):.2f}'
V['damage_rej'] = sum(main['damage'][n]['rejected_with_tag'] for n in NAMES)
for n in ('Lena', 'Baboon'):
    for k, lab in (('salt_pepper_0.1%', 'sp01'), ('salt_pepper_1%', 'sp1'), ('crop_1/16_zero', 'crop')):
        r = main['robust'][n][k]; V[f's11_{n}_{lab}'] = (f'{r["psnr"]:.1f}', f'{r["npcr"]:.1f}', 'Rejected' if not r['tag_ok'] else 'Accepted')

# ---- S-boxes ------------------------------------------------------------------------------------
sb = R('sbox.json')
for k, lab in (('ours', 'ours'), ('random_perm', 'rand'), ('chaos_only', 'chaos')):
    s = sb[k]; V[f's10_{lab}'] = (f'{s["nl_mean"]:.2f} ({s["nl_min"]}–{s["nl_max"]})', f'{s["du_mean"]:.2f} ({s["du_min"]}–{s["du_max"]})')

# ---- key-stream statistics (E1), relation (E10), param sensitivity (E7c), port3 --------------------
e1 = R('sa_e1_summary.json')
V['e1'] = e1; V['e1_maxz'] = f2(e1['max_z']); V['e1_medz'] = f2(e1['median_max_z']); V['e1_ksp'] = f2(e1['ks_p'])
e10 = R('sa_e10.json'); V['e10'] = e10
e7c = R('sa_e7c.json'); V['e7c'] = e7c
p3 = R('port3.json'); V['port3'] = p3

# ---- rounds (E4, E6, E6b, E9) ---------------------------------------------------------------------
e4 = R('sa_e4.json'); V['e4'] = e4
V['e4_1_npcr'] = f2(np.mean([v['npcr'] for v in e4['1']['positions'].values()]))
V['e4_1_range'] = f'{min(v["npcr"] for v in e4["1"]["positions"].values()):.2f}–{max(v["npcr"] for v in e4["1"]["positions"].values()):.2f}'
for r in ('2', '3', '4', '5'):
    V[f'e4_{r}_npcr'] = f2(np.mean([v['npcr'] for v in e4[r]['positions'].values()]))
    V[f'e4_{r}_dm'] = f4(e4[r]['diff_match'])
e6 = R('sa_e6.json'); V['e6'] = e6
e6b = R('sa_e6b.json'); V['e6b'] = e6b
e9 = R('sa_e9.json'); V['e9'] = e9
json.dump(V, open('results/values.json', 'w'), indent=1, default=str)
print(len(V), 'values')

# ---- ablation (Table 10) --------------------------------------------------------------------------
ab = R('ablation.json'); V['ablation'] = ab
# ---- broken scheme (Table 15) ----------------------------------------------------------------------
V['broken'] = R('broken.json')
# ---- standard ciphers (Table 12) -------------------------------------------------------------------
V['aes2'] = R('aes2.json')
V['t12_fixed_npcr'] = f4(np.mean([np.mean(main['npcr'][f'{n}|fixed']['mean'][:3]) for n in NAMES]))
V['t12_fixed_uaci'] = f4(np.mean([np.mean(main['npcr'][f'{n}|fixed']['mean'][3:]) for n in NAMES]))
# ---- optional results ------------------------------------------------------------------------------
for f, k in (('multi.json', 'multi'), ('sa_e8.json', 'e8'), ('medical16.json', 'med'), ('sipi.json', 'sipi'), ('breakdown.json', 'breakdown'),
             ('sa_e5_1x1.json', 'e5_1'), ('sa_e5_1x2.json', 'e5_2'), ('sts_pooled.json', 'sts_pooled')):
    if os.path.exists(os.path.join('results', f)):
        d = R(f)
        if k in ('e5_1', 'e5_2'): d = {a: b for a, b in d.items() if a != 'res'}
        V[k] = d
for name in ('x', 'y', 'z', 'a', 'c', 'c2', 'c3', 'r1', 'r2', 'chk_pi', 'chk_e', 'chk_sqrt2', 'chk_sqrt3'):
    fn = os.path.join('results', f'sts_{name}.txt')
    if os.path.exists(fn):
        V[f'sts_{name}'] = SP.summarize(SP.parse(fn))
json.dump(V, open('results/values.json', 'w'), indent=1, default=str)
print(len(V), 'values (final)')

# ---- test vectors (Table 11), from the unit tests --------------------------------------------------
import re, ast
src = open('test_cipher.py').read()
V['vec_rgb'] = ast.literal_eval(re.search(r'VEC_RGB = (\(.*?\))', src).group(1))
V['vec_g16'] = ast.literal_eval(re.search(r'VEC_G16 = (\(.*?\))', src).group(1))
V['vec_q1'] = ast.literal_eval(re.search(r'TEST_Q1 = (\(.*?\))', src).group(1))
import subprocess
_out = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q', 'test_cipher.py'], capture_output=True, text=True).stdout
V['n_tests'] = int(re.search(r'(\d+) tests? collected', _out).group(1))
# ---- Table 13: inside the random region? -----------------------------------------------------------
for n in NAMES:
    for size, key in (('256', '0.05'), ('512', '0.05')):
        cr = main['crit'][key] if size == '256' else main['crit512'][key]
        npcr = float(V[f't13_{n}_{size}_npcr']); uaci = float(V[f't13_{n}_{size}_uaci'])
        V[f't13_{n}_{size}_inside'] = ('✓' if npcr >= cr['npcr_min'] else '✗') + ' / ' + ('✓' if cr['uaci_lo'] <= uaci <= cr['uaci_hi'] else '✗')
if os.path.exists('results/sa_e9_check.json'):
    V['e9_check'] = R('sa_e9_check.json')
json.dump(V, open('results/values.json', 'w'), indent=1, default=str)
print('vectors', V['vec_rgb'][0][:8], V['n_tests'])

# ---- ten further images (Section 3.17, Table S15) ---------------------------------------------------
if 'multi' in V:
    mu = V['multi']; rowsS15 = []
    for n, r in mu.items():
        rowsS15.append([n, f'{r["plain_entropy"]:.2f}', f'{np.mean(r["ent"]):.4f} ± {np.std(r["ent"]):.4f}', f'{np.mean(r["corr"]):.4f}',
                        f'{100 * np.mean(r["chi"]):.0f}', f'{np.mean(r["loc"]):.4f}', f'{np.mean(r["npcr"]):.4f} ± {np.std(r["npcr"]):.4f}',
                        f'{np.mean(r["uaci"]):.3f} ± {np.std(r["uaci"]):.3f}', f'{100 * np.mean(r["np_pass"]):.0f} / {100 * np.mean(r["ua_pass"]):.0f}'])
    V['s15_rows'] = rowsS15
    ents = [np.mean(r['ent']) for r in mu.values()]
    V['multi_sum'] = dict(ent_min=f'{min(ents):.4f}', ent_max=f'{max(ents):.4f}',
                          chi=f'{100 * np.mean([x for r in mu.values() for x in r["chi"]]):.1f}',
                          np=f'{100 * np.mean([x for r in mu.values() for x in r["np_pass"]]):.1f}',
                          ua=f'{100 * np.mean([x for r in mu.values() for x in r["ua_pass"]]):.1f}',
                          ok=all(r['ok'] for r in mu.values()), n_ok=sum(len(r['ent']) for r in mu.values() if r['ok']),
                          checker=f'{mu["Checkerboard"]["plain_entropy"]:.2f}', colorwheel=f'{mu["Colorwheel"]["plain_entropy"]:.2f}')
# ---- fifty images (Table 16) -----------------------------------------------------------------------
if 'e8' in V and isinstance(V['e8'], list) and len(V['e8']) == 150:
    e8 = V['e8']
    groups = [('standard 256', 'Standard images, 256×256'), ('other 256', 'Other natural, medical and technical images, 256×256'),
              ('size 64 to 1024', 'Sizes 64×64 to 1024×1024'), ('synthetic', 'Synthetic images, 256×256'), (None, 'All')]
    rows16 = []; summ = {}
    for g, lab in groups:
        rs = [r for r in e8 if g is None or r['group'] == g]
        ent = [r['ent'] for r in rs]
        rows16.append([f'{min(ent):.5f}–{max(ent):.5f}', f'{max(r["rho_max"] for r in rs):.4f}',
                       f'{100 * np.mean([p > 0.05 for r in rs for p in r["chi_p"]]):.1f}',
                       f'{100 * np.mean([x for r in rs for x in r["npcr_ok"]]):.1f}', f'{100 * np.mean([x for r in rs for x in r["uaci_ok"]]):.1f}',
                       f'{sum(r["exact"] for r in rs)} of {len(rs)}'])
    V['e8_table'] = rows16
    allp = [p for r in e8 for p in r['chi_p']]
    # ideal entropy of random data by size (all channel values), 50 samples each
    rng_ = np.random.default_rng(8)
    def Hn(n):
        h = np.bincount(rng_.integers(0, 256, n), minlength=256) / n; h = h[h > 0]; return float(-(h * np.log2(h)).sum())
    ideal = {s: float(np.mean([Hn(3 * s * s) for _ in range(50)])) for s in (64, 128, 256, 512, 1024)}
    dev = max(abs(r['ent'] - ideal[r['M']]) for r in e8 if r['M'] == r['N'])
    V['e8_sum'] = dict(chi=rows16[-1][2], np=rows16[-1][3], ua=rows16[-1][4], npcr=f'{np.mean([x for r in e8 for x in r["npcr"]]):.2f}',
                       uaci=f'{np.mean([x for r in e8 for x in r["uaci"]]):.2f}', minp=f'{min(allp):.4f}', n001=int(sum(p < 0.001 for p in allp)),
                       dev=f'{dev:.4f}', ideal={k: (f'{v:.5f}' if k == 1024 else f'{v:.4f}') for k, v in ideal.items()},
                       exact=sum(r['exact'] for r in e8))
# ---- 16-bit medical images --------------------------------------------------------------------------
if 'med' in V:
    V['med_rows'] = []
    for fn, r in V['med'].items():
        V['med_rows'].append(dict(file=fn, mod=r['modality'], size=f'{r["rows"]}×{r["cols"]}', bits=r['bits_stored'],
                                  pH=f'{r["plain_entropy16"]:.2f}', cH=f'{r["cipher_entropy16"]:.4f}', iH=f'{r["ideal_entropy16"]:.4f}',
                                  pc=f'{max(abs(x) for x in r["plain_corr"]):.4f}', cc=f'{max(abs(x) for x in r["cipher_corr"]):.4f}',
                                  chi=f'{r["chi16"][1]:.3f}', pchi=f'{min(r["plane_chi_p"]):.3f}',
                                  npcr=f'{r["fixed"]["npcr"]:.4f}', uaci=f'{r["fixed"]["uaci"]:.4f}', npcr_c=f'{r["complete"]["npcr"]:.4f}',
                                  uaci_c=f'{r["complete"]["uaci"]:.4f}', ntr=r['fixed']['trials'],
                                  npass=f'{100 * r["fixed"]["npcr_pass"]:.0f}', upass=f'{100 * r["fixed"]["uaci_pass"]:.0f}',
                                  exact=r['exact'], crit=r['crit']))
json.dump(V, open('results/values.json', 'w'), indent=1, default=str)
print('extra done', 'multi_sum' in V, 'e8_table' in V, 'med_rows' in V)
# ---- USC-SIPI originals --------------------------------------------------------------------------------
if 'sipi' in V and V['sipi']:
    rows = []
    for fn, r in V['sipi'].items():
        rows.append(dict(file=fn.rsplit('.', 1)[0], size=r['size'], pH=f'{r["plain_entropy"]:.4f}', cH=f'{r["cipher_entropy"]:.4f}',
                         pc=f'{r["plain_corr_max"]:.4f}', cc=f'{r["cipher_corr_max"]:.4f}', chi=f'{sum(p > 0.05 for p in r["chi_p"])} / 3',
                         npcr_c=f'{r["complete"]["npcr"]:.4f}', uaci_c=f'{r["complete"]["uaci"]:.4f}', npcr_f=f'{r["fixed"]["npcr"]:.4f}',
                         uaci_f=f'{r["fixed"]["uaci"]:.4f}', exact=r['exact']))
    V['sipi_rows'] = rows
    s256 = [r for r in V['sipi'].values() if r['size'] == '256×256']; s512 = [r for r in V['sipi'].values() if r['size'] == '512×512']
    rng_ = lambda L: f'{min(x["cipher_entropy"] for x in L):.4f}–{max(x["cipher_entropy"] for x in L):.4f}' if L else '—'
    V['sipi_sum'] = dict(n256=len(s256), n512=len(s512), H256=rng_(s256), H512=rng_(s512),
                         rho=f'{max(r["cipher_corr_max"] for r in V["sipi"].values()):.4f}',
                         chi=sum(p > 0.05 for r in V['sipi'].values() for p in r['chi_p']),
                         **{f'{m}_{k}': f'{100 * np.mean([r[m][k] for r in V["sipi"].values()]):.1f}'
                            for m in ('complete', 'fixed') for k in ('npcr_pass', 'uaci_pass')})
    json.dump(V, open('results/values.json', 'w'), indent=1, default=str)
# ---- timing (Section 3.14, Tables S13, S14) ------------------------------------------------------------
if 'breakdown' in V and main.get('timing_quiet'):
    bd = V['breakdown']; t256 = main['time_256']; ts = main['time_scaling']
    enc = bd['encrypt_total'][0] * 1000
    tm = dict(enc256=float(np.mean([t256['enc'][n][0] for n in NAMES])), enc512=ts['512'][0], dec512=ts['512'][1],
              enc1024=ts['1024'][0], dec1024=ts['1024'][1])
    tm['map_share'] = 100 * bd['map_and_key_material_4_rounds'][0] / bd['encrypt_total'][0]
    tm['anchor_ms'] = bd['anchor_words_4_rounds'][0] * 1000; tm['anchor_share'] = 100 * bd['anchor_words_4_rounds'][0] / bd['encrypt_total'][0]
    cms = (bd['digest_sha256'][0] + bd['hkdf_init_mask_mac_stream'][0] + bd['hmac_tag'][0]) * 1000
    tm['crypto_ms'] = cms; tm['crypto_share'] = 100 * cms / enc; tm['reject_ms'] = bd['decrypt_tag_reject'][0] * 1000
    tm['rounds_share'] = 100 * bd['confusion_diffusion_cipher_4_rounds'][0] / bd['encrypt_total'][0]
    A = V['aes2']; ratio = A['Proposed, same nonce']['time'] / min(A[k]['time'] for k in ('AES-128 CTR, fresh nonce', 'ChaCha20, fresh nonce'))
    tm['ratio'] = ratio; tm['orders'] = f'{np.log10(ratio):.1f}'
    V['timing'] = tm
    V['s13_rows'] = [[f'{1000 * t256["enc"][n][0]:.0f} ± {1000 * t256["enc"][n][1]:.0f}', f'{1000 * t256["dec"][n][0]:.0f} ± {1000 * t256["dec"][n][1]:.0f}'] for n in NAMES]
    steps = [('SHA-256 of the plain image', 'digest_sha256'), ('HKDF: shift, mask, message key of the anchor and MAC key', 'hkdf_init_mask_mac_stream'),
             ('Chaotic map, output filter, anchor and key material, four rounds (iterations, quantization, ChaCha20 words, sorting)', 'map_and_key_material_4_rounds'),
             ('Confusion, diffusion and cipher phases, four rounds', 'confusion_diffusion_cipher_4_rounds'), ('HMAC-SHA-256 tag', 'hmac_tag')]
    rows14 = []; ssum = 0
    for lab, k in steps:
        m, s = bd[k]; ssum += m
        rows14.append([lab, f'{1000 * m:.2f} ± {1000 * s:.2f}', f'{100 * m / bd["encrypt_total"][0]:.1f}'])
    rows14.append(['Sum of the five steps above', f'{1000 * ssum:.2f}', f'{100 * ssum / bd["encrypt_total"][0]:.1f}'])
    rows14.append(['Encryption, total', f'{1000 * bd["encrypt_total"][0]:.2f} ± {1000 * bd["encrypt_total"][1]:.2f}', '100.0'])
    rows14.append(['Total minus sum of the steps (mean ± s.d. of the paired differences)', f'{1000 * bd["other"][0]:.2f} ± {1000 * bd["other"][1]:.2f}', f'{100 * bd["other"][0] / bd["encrypt_total"][0]:.1f}'])
    rows14.append(['ChaCha20 words of the anchor, four rounds (part of the third step)', f'{1000 * bd["anchor_words_4_rounds"][0]:.2f} ± {1000 * bd["anchor_words_4_rounds"][1]:.2f}', f'{100 * bd["anchor_words_4_rounds"][0] / bd["encrypt_total"][0]:.1f}'])
    rows14.append(['Decryption, total (including tag check)', f'{1000 * bd["decrypt_total"][0]:.2f} ± {1000 * bd["decrypt_total"][1]:.2f}', f'{100 * bd["decrypt_total"][0] / bd["encrypt_total"][0]:.1f}'])
    rows14.append(['Decryption of a message with a wrong tag (rejected)', f'{1000 * bd["decrypt_tag_reject"][0]:.2f} ± {1000 * bd["decrypt_tag_reject"][1]:.2f}', f'{100 * bd["decrypt_tag_reject"][0] / bd["encrypt_total"][0]:.1f}'])
    V['s14_rows'] = rows14
    import platform, numba
    V['s13_sw'] = f'Python {platform.python_version()} with NumPy {np.__version__} and Numba {numba.__version__}'
    json.dump(V, open('results/values.json', 'w'), indent=1, default=str)
    print('timing collected')
for d in ('pi', 'e', 'sqrt2', 'sqrt3'):
    if V.get(f'sts_chk_{d}'):
        V[f'sts_chklow_{d}'] = sum(v['outside'] for v in V[f'sts_chk_{d}'].values() if v)
json.dump(V, open('results/values.json', 'w'), indent=1, default=str)
