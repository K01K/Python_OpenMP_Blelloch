import numpy as np
import time
from numba import njit, prange
import csv

openmp_results = "openmp_results.csv"

with open(openmp_results, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dtype", "size", "op", "parallel", "seq", "abs_err", "time", "GBs"])

def now():
    return time.perf_counter()

@njit(parallel=True)
def psum(a):
    n = len(a)
    nb = min(64, n)
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
    nb = min(64, n)
    bs = (n+nb-1)// nb
    block = np.empty(nb, a.dtype)
    for b in prange(nb):
        s = b*bs
        e = min(s+bs,n)
        m = a[s]
        for i in range(s+1,e):
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
    nb = min(64, n)
    bs = (n+nb-1) // nb
    block = np.empty(nb, a.dtype)
    for b in prange(nb):
        s = b*bs
        e = min(s+bs, n)
        m = a[s]
        for i in range(s+1, e):
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
        for i in range(0, m, 2*d):
            t[i+2*d-1]+= t[i+d-1]
        d <<= 1
    t[m-1] = 0
    d = m >> 1
    while d:
        for i in range(0, m, 2*d):
            v = t[i+d-1]
            t[i+d-1] = t[i+2*d-1]
            t[i + 2 * d - 1] += v
        d >>= 1
    x[:] = t[:n]

@njit(parallel=True)
def scan(a, bs):
    n = len(a)
    orig = a.copy()
    nb = (n+bs-1) // bs
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

    with open(openmp_results, "a", newline="") as f:
        w = csv.writer(f)

        #SUM
        print(f"{'Op'} {'Parallel'} {'Seq'} {'AbsErr'} {'Time[s]'} {'GB/s'}")
        t0 = now()
        r = psum(a)
        dt = now() - t0
        err = abs(r - a.sum())
        gbs = (n*size)/(dt*1e9)
        print(f"{'SUM'} {r} {a.sum()} {err} {dt} {gbs}")
        w.writerow([str(dtype), n, "SUM", r, a.sum(), err, dt, gbs])

        # MIN
        t0 = now()
        r = pmin(a)
        dt = now() - t0
        err = abs(r - a.min())
        gbs = (n*size)/(dt*1e9)
        print(f"{'MIN'} {r} {a.sum()} {err} {dt} {gbs}")
        w.writerow([str(dtype), n, "MIN", r, a.min(), err, dt, gbs])

        # MAX
        t0 = now()
        r = pmax(a)
        dt = now() - t0
        err = abs(r - a.max())
        gbs = (n*size)/(dt*1e9)
        print(f"{'MAX'} {r} {a.sum()} {err} {dt} {gbs}")
        w.writerow([str(dtype), n, "MAX", r, a.max(), err, dt, gbs])

        # SCAN
        seq = np.cumsum(a)
        b = a.copy()
        t0 = now()
        scan(b, block_size)
        dt = now() - t0
        err = np.max(np.abs(b - seq))
        gbs = (2*n*size)/(dt*1e9)
        print(f"{'SCAN'} {r} {a.sum()} {err} {dt} {gbs}")
        w.writerow([str(dtype), n, "SCAN", 0, 0, err, dt, gbs])


def main():
    types = [np.int64, np.int32, np.float64, np.float32]
    sizes = [1 << 10, 1 << 20, 1 << 24]
    block_size = 1 << 16
    for t in types:
        for n in sizes:
            print(f"Type: {t}, Size: {n}")
            bench(t, n, block_size)

if __name__ == "__main__":
    main()
