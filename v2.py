import numpy as np
import time
from numba import njit, prange

def now():
    return time.perf_counter()

@njit(parallel=True)
def psum(a):
    n = len(a)
    nb = 64
    bs = (n + nb - 1) // nb
    block = np.zeros(nb, a.dtype)
    for b in prange(nb):
        s = b * bs
        e = min(s + bs, n)
        tmp = 0
        for i in range(s, e):
            tmp += a[i]
        block[b] = tmp
    total = 0
    for i in range(nb):
        total += block[i]
    return total

@njit(parallel=True)
def pmin(a):
    n = len(a)
    nb = 64
    bs = (n + nb - 1) // nb
    block = np.full(nb, np.inf, a.dtype)
    for b in prange(nb):
        s = b * bs
        e = min(s + bs, n)
        if s < e:
            m = a[s]
            for i in range(s + 1, e):
                if a[i] < m:
                    m = a[i]
            block[b] = m
    out = block[0]
    for i in range(1, nb):
        if block[i] < out:
            out = block[i]
    return out

@njit(parallel=True)
def pmax(a):
    n = len(a)
    nb = 64
    bs = (n + nb - 1) // nb
    block = np.full(nb, -np.inf, a.dtype)
    for b in prange(nb):
        s = b * bs
        e = min(s + bs, n)
        if s < e:
            m = a[s]
            for i in range(s + 1, e):
                if a[i] > m:
                    m = a[i]
            block[b] = m
    out = block[0]
    for i in range(1, nb):
        if block[i] > out:
            out = block[i]
    return out

@njit
def blelloch(x):
    n = len(x)
    m = 1
    while m < n:
        m <<= 1
    t = np.zeros(m, x.dtype)
    t[:n] = x
    d = 1
    while d < m:
        for i in range(0, m, 2 * d):
            t[i + 2 * d - 1] += t[i + d - 1]
        d <<= 1
    t[m - 1] = 0
    d = m >> 1
    while d:
        for i in range(0, m, 2 * d):
            v = t[i + d - 1]
            t[i + d - 1] = t[i + 2 * d - 1]
            t[i + 2 * d - 1] += v
        d >>= 1
    x[:] = t[:n]

@njit(parallel=True)
def scan(a, bs):
    n = len(a)
    orig = a.copy()
    nb = (n + bs - 1) // bs
    totals = np.zeros(nb, a.dtype)
    for b in prange(nb):
        s = b * bs
        e = min(s + bs, n)
        if s < e:
            buf = orig[s:e].copy()
            blelloch(buf)
            a[s:e] = buf
            totals[b] = buf[-1] + orig[e - 1]
    offs = np.zeros(nb, a.dtype)
    acc = 0
    for i in range(nb):
        offs[i] = acc
        acc += totals[i]
    for b in prange(nb):
        s = b * bs
        e = min(s + bs, n)
        o = offs[b]
        for i in range(s, e):
            a[i] += o
    for i in prange(n):
        a[i] += orig[i]

def bench(dtype, n, block_size):
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.floating):
        a = rng.uniform(-100, 100, n).astype(dtype)
    else:
        a = rng.integers(-1000, 1000, n, dtype=dtype)
    size = a.dtype.itemsize

    print(f"{'Op':<6} {'Parallel':>15} {'Seq':>15} {'AbsErr':>12} {'Time[s]':>10} {'GB/s':>10}")
    
    # SUM
    t0 = now()
    r = psum(a)
    dt = now() - t0
    err = abs(r - a.sum())
    print(f"{'SUM':<6} {r:15} {a.sum():15} {err:12.3e} {dt:10.6f} {(n*size)/(dt*1e9):10.2f}")
    
    # MIN
    t0 = now()
    r = pmin(a)
    dt = now() - t0
    err = abs(r - a.min())
    print(f"{'MIN':<6} {r:15} {a.min():15} {err:12.3e} {dt:10.6f} {(n*size)/(dt*1e9):10.2f}")
    
    # MAX
    t0 = now()
    r = pmax(a)
    dt = now() - t0
    err = abs(r - a.max())
    print(f"{'MAX':<6} {r:15} {a.max():15} {err:12.3e} {dt:10.6f} {(n*size)/(dt*1e9):10.2f}")
    
    # SCAN
    seq = np.cumsum(a)
    b = a.copy()
    t0 = now()
    scan(b, block_size)
    dt = now() - t0
    err = np.max(np.abs(b - seq))
    print(f"{'SCAN':<6} {0:15} {0:15} {err:12.3e} {dt:10.6f} {(2*n*size)/(dt*1e9):10.2f}")
    print()

def main():
    types = [np.int64, np.float64]
    sizes = [1 << 10, 1 << 20, 1 << 24]
    block_size = 1 << 16
    for t in types:
        for n in sizes:
            print(f"Type: {t}, Size: {n}")
            bench(t, n, block_size)

if __name__ == "__main__":
    main()
