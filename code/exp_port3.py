"""Bit-exactness of the map together with the output filter and quantization (Eq. (10)) across languages and
toolchains: compares the int64 values q written by port3/build_run.sh with Numba and with pure CPython."""
import numpy as np, json, os, sys, math, subprocess, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cipher as C
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8500000
X = np.empty(N); Y = np.empty(N); Z = np.empty(N)
C.hybrid_steps(0.3, 0.2, 0.6, 3.99, 0.999, 1.99, N, X, Y, Z, 0)
ref = C.quantize(X, Y, Z, True).T.copy()                    # (N, 3)
py = []
for (x, y, z) in C.hybrid_steps_py(0.3, 0.2, 0.6, 3.99, 0.999, 1.99, N):
    py.append((math.floor(((x + y * z) % 1.0) * 1e14), math.floor(((y + z * x) % 1.0) * 1e14), math.floor(((z + x * y) % 1.0) * 1e14)))
py = np.array(py, dtype=np.int64)
R = {'iterations': N, 'cpython_vs_numba_equal': bool(np.array_equal(py, ref))}
for f in sorted(glob.glob('port3/out_*.bin')):
    a = np.fromfile(f, dtype='<i8').reshape(-1, 3)
    d = np.nonzero(np.any(a != ref, axis=1))[0]
    R[os.path.basename(f)[4:-4]] = dict(bit_equal=bool(a.shape == ref.shape and len(d) == 0), first_divergence=int(d[0]) + 1 if len(d) else None)
def ver(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.splitlines()[0]
    except Exception: return None
R['versions'] = {k: ver(v) for k, v in dict(gcc='gcc --version', clang='clang --version', go='go version', rustc='rustc --version',
                 java='java -version 2>&1 | grep -v JAVA_TOOL', node='node --version', qemu='qemu-aarch64-static --version',
                 gcc_a64='aarch64-linux-gnu-gcc --version', gcc_arm32='arm-linux-gnueabihf-gcc --version').items()}
json.dump(R, open('results/port3.json', 'w'), indent=1); print(json.dumps(R, indent=1))
