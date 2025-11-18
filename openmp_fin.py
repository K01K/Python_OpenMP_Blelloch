import numpy as np
import time
from numba import njit, prange

def now():
    return time.perf_counter()

@njit
def next_pow2(n):
    """Smallest power of two >= n (Numba-compatible)."""
    m = 1
    while m < n:
        m *= 2
    return m

@njit(parallel=True)
def parallel_sum(a):
    n = len(a)
    tmp = np.zeros(n, dtype=a.dtype)
    for i in prange(n):
        tmp[i] = a[i]
    total = tmp.sum()
    return total

@njit(parallel=True)
def parallel_min(a):
    n = len(a)
    nb = 64
    block_size = (n + nb - 1) // nb
    block_mins = np.full(nb, np.inf, dtype=a.dtype)
    for b in prange(nb):
        start = b * block_size
        end = min(start + block_size, n)
        if end > start:
            local_min = a[start]
            for i in range(start+1, end):
                if a[i] < local_min:
                    local_min = a[i]
            block_mins[b] = local_min
    return np.min(block_mins)

@njit(parallel=True)
def parallel_max(a):
    n = len(a)
    nb = 64
    block_size = (n + nb - 1) // nb
    block_maxs = np.full(nb, -np.inf, dtype=a.dtype)
    for b in prange(nb):
        start = b * block_size
        end = min(start + block_size, n)
        if end > start:
            local_max = a[start]
            for i in range(start+1, end):
                if a[i] > local_max:
                    local_max = a[i]
            block_maxs[b] = local_max
    return np.max(block_maxs)

@njit
def blelloch_scan(buf):
    n = len(buf)
    m = next_pow2(n)
    temp = np.zeros(m, dtype=buf.dtype)
    temp[:n] = buf.copy()
    d = 1
    while d < m:
        for i in range(0, m, 2*d):
            temp[i + 2*d - 1] += temp[i + d - 1]
        d *= 2

    temp[m-1] = 0
    d = m // 2
    while d >= 1:
        for i in range(0, m, 2*d):
            t = temp[i + d - 1]
            temp[i + d - 1] = temp[i + 2*d - 1]
            temp[i + 2*d - 1] += t
        d //= 2

    buf[:] = temp[:n]

@njit(parallel=True)
def block_blelloch_scan(a, block_size=1 << 16):
    n = len(a)
    nb = (n + block_size - 1) // block_size
    block_totals = np.zeros(nb, dtype=a.dtype)
    for b in prange(nb):
        start = b * block_size
        end = min(start + block_size, n)
        if end > start:
            buf = a[start:end].copy()
            blelloch_scan(buf)
            a[start:end] = buf
            block_totals[b] = buf[-1] + a[start]

    offsets = np.zeros_like(block_totals)
    acc = 0
    for b in range(nb):
        offsets[b] = acc
        acc += block_totals[b]

    for b in prange(nb):
        off = offsets[b]
        start = b * block_size
        end = min(start + block_size, n)
        if off != 0:
            for i in range(start, end):
                a[i] += off

    return a

def check_and_bench(dtype: np.dtype, count: int, block_size: int = 1 << 16, samples: int = 5):
    rng = np.random.default_rng(12345)
    if np.issubdtype(dtype, np.floating):
        base = rng.uniform(-100.0, 100.0, size=count).astype(dtype)
    else:
        base = rng.integers(-1000, 1000, size=count, dtype=dtype)
    elem_bytes = base.dtype.itemsize

    print(f"\n=== dtype={dtype} count={count:,} block_size={block_size} ===")

    seq_sum = base.sum()
    times = []
    for _ in range(samples):
        arr = base.copy()
        t0 = now()
        psum = parallel_sum(arr)
        dt = now() - t0
        times.append(dt)
    t = sum(times)/len(times)
    abs_err = float(psum - seq_sum)
    rel_err = abs_err / (abs(seq_sum)+1e-30)
    gb_per_s = (count * elem_bytes)/(t*1e9)
    print(f"SUM: parallel={psum} seq={seq_sum} abs_err={abs_err} rel_err={rel_err:.3e} time={t:.6f}s GB/s≈{gb_per_s:.3f}")

    seq_min = base.min()
    times = []
    for _ in range(samples):
        arr = base.copy()
        t0 = now()
        pmin = parallel_min(arr)
        dt = now() - t0
        times.append(dt)
    t = sum(times)/len(times)
    gb_per_s = (count * elem_bytes)/(t*1e9)
    print(f"MIN: parallel={pmin} seq={seq_min} diff={pmin - seq_min} time={t:.6f}s GB/s≈{gb_per_s:.3f}")

    seq_max = base.max()
    times = []
    for _ in range(samples):
        arr = base.copy()
        t0 = now()
        pmax = parallel_max(arr)
        dt = now() - t0
        times.append(dt)
    t = sum(times)/len(times)
    gb_per_s = (count * elem_bytes)/(t*1e9)
    print(f"MAX: parallel={pmax} seq={seq_max} diff={pmax - seq_max} time={t:.6f}s GB/s≈{gb_per_s:.3f}")

    times = []
    seq = np.cumsum(base)
    for _ in range(samples):
        arr = base.copy()
        t0 = now()
        block_blelloch_scan(arr, block_size=block_size)
        dt = now() - t0
        times.append(dt)
        if np.issubdtype(dtype, np.floating):
            abs_err = float(np.max(np.abs(arr - seq)))
            rel_err = float(np.max(np.abs((arr - seq)/(np.abs(seq)+1e-30))))
        else:
            max_err = int(np.max(np.abs(arr - seq)))
    t = sum(times)/len(times)
    gb_per_s = (2 * count * elem_bytes)/(t*1e9)
    if np.issubdtype(dtype, np.floating):
        print(f"SCAN: max_abs_err={abs_err:.3e} max_rel_err={rel_err:.3e} time={t:.6f}s GB/s≈{gb_per_s:.3f}")
    else:
        print(f"SCAN: max_err={max_err} time={t:.6f}s GB/s≈{gb_per_s:.3f}")

def main():
    tested_types = [np.int64, np.float64]
    tested_counts = [1 << 10, 1 << 20, 1 << 24]
    block_size = 1 << 16
    samples = 3

    print("NumPy version:", np.__version__)
    import sys
    print("Python:", sys.version.splitlines()[0])

    for dtype in tested_types:
        for count in tested_counts:
            check_and_bench(dtype, count, block_size=block_size, samples=samples)
            print("-"*70)

if __name__ == "__main__":
    main()
