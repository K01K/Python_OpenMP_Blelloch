import numpy as np
import time
import csv
from mpi4py import MPI

mpi_results = "mpi_results.csv"

def now():
    return time.perf_counter()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_world = comm.Get_size()

def ensure_divisible(n):
    return n % size_world == 0

def bench_mpi(dtype, n):
    if rank == 0:
        rng = np.random.default_rng(0)
        if np.issubdtype(dtype, np.floating):
            full = rng.uniform(-100, 100, n).astype(dtype)
        else:
            full = rng.integers(-1000, 1000, n, dtype=dtype)
        
        seq_sum = full.sum()
        seq_min = full.min()
        seq_max = full.max()
        seq_scan = np.cumsum(full)
    else:
        full = None
        seq_sum = seq_min = seq_max = seq_scan = None

    itemsize = np.dtype(dtype).itemsize
    block_len = n // size_world
    local = np.empty(block_len, dtype=dtype)

    if rank == 0:
        sendcounts = [block_len] * size_world
        displs = [i * block_len for i in range(size_world)]
    else:
        sendcounts = None
        displs = None

    comm.Scatterv([full, sendcounts, displs, MPI._typedict[np.dtype(dtype).char]] if rank == 0 else None,
                  local, root=0)

    if rank == 0:
        f = open(mpi_results, "a", newline="")
        w = csv.writer(f)

    # SUM
    t0 = now()
    local_sum = local.sum()
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    dt = now() - t0
    if rank == 0:
        err = abs(global_sum - seq_sum)
        gbs = (n * itemsize) / (dt * 1e9) if dt > 0 else 0.0
        print(f"{'SUM'} {str(global_sum)} {str(seq_sum)} {err} {dt} {gbs}")
        w.writerow([str(dtype), n, "SUM", "MPI_Allreduce", global_sum, seq_sum, err, dt, gbs])

    # MIN
    t0 = now()
    local_min = local.min()
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    dt = now() - t0
    if rank == 0:
        err = abs(global_min - seq_min)
        gbs = (n * itemsize) / (dt * 1e9) if dt > 0 else 0.0
        print(f"{'MIN'} {str(global_sum)} {str(seq_sum)} {err} {dt} {gbs}")
        w.writerow([str(dtype), n, "MIN", "MPI_Allreduce", global_min, seq_min, err, dt, gbs])

    # MAX
    t0 = now()
    local_max = local.max()
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    dt = now() - t0
    if rank == 0:
        err = abs(global_max - seq_max)
        gbs = (n * itemsize) / (dt * 1e9) if dt > 0 else 0.0
        print(f"{'MAX'} {str(global_sum)} {str(seq_sum)} {err} {dt} {gbs}")
        w.writerow([str(dtype), n, "MAX", "MPI_Allreduce", global_max, seq_max, err, dt, gbs])

    # SCAN
    if ensure_divisible(n):
        t0 = now()
        local_scan = np.cumsum(local)
        local_total = local_scan[-1]

        if size_world > 1:
            offset = comm.exscan(local_total)
            if rank == 0 or offset is None:
                offset = 0
        else:
            offset = 0

        local_scan += offset
        local_scan = local_scan.astype(dtype)
        
        dt_compute = now() - t0

        recvbuf = np.empty(n, dtype=dtype) if rank == 0 else None
        recvcounts = [block_len] * size_world
        displs = [i * block_len for i in range(size_world)]

        comm.Gatherv([local_scan, MPI._typedict[np.dtype(dtype).char]],
                     [recvbuf, recvcounts, displs, MPI._typedict[np.dtype(dtype).char]],
                     root=0)

        dt_total = dt_compute
        if rank == 0:
            err = np.max(np.abs(recvbuf - seq_scan))
            gbs = (2 * n * itemsize) / (dt_total * 1e9) if dt_total > 0 else 0.0
            print(f"{'SCAN'} {str(global_sum)} {str(seq_sum)} {err} {dt} {gbs}")
            w.writerow([str(dtype), n, "SCAN", "MPI_Exscan+Gath", np.nan, np.nan, err, dt_total, gbs])

    if rank == 0:
        f.close()

def main():
    if rank == 0:
        with open(mpi_results, "w", newline="") as f:
            csv.writer(f).writerow(["dtype", "size", "op", "method", "parallel", "seq", "abs_err", "time", "GBs"])
    
    comm.Barrier()

    types = [np.int64, np.int32, np.float64, np.float32]
    sizes = [1 << 10, 1 << 20, 1 << 24]

    for t in types:
        for n in sizes:
            if rank == 0:
                print(f"\nType: {t}, Size: {n}")
                print(f"{'Op'} {'Parallel'} {'Seq'} {'AbsErr'} {'Time[s]'} {'GB/s'}")
            
            bench_mpi(t, n)
            comm.Barrier()

if __name__ == "__main__":
    main()