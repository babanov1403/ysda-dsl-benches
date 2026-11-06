import torch

from typing import List

import cutlass
import cutlass.cute as cute

from cutlass.memory.smem import SmemAllocator

@cute.kernel
def rms_norm_kernel(
    a: cute.Tensor,
    residual: cute.Tensor,
    weight: cute.Tensor,
    eps: cutlass.Float32,
    out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = tidx + bidx * bdim

    n = a.shape[0]

    shmemAlloc = SmemAllocator()

    acc = shmemAlloc.allocate_tensor(cutlass.Float32, 1)

    if idx < n:
        out[idx] = a[idx] + residual[idx]

    if tidx == 0:
        acc[0] = 0.

    cute.arch.sync_threads()

    for blstart in range(0, n, bdim):
        if blstart + tidx < n:
            tmp = out[tidx + blstart]
            tmp *= tmp
            cute.arch.atomic_add(acc.iterator, tmp)

    cute.arch.sync_threads()

    if tidx == 0:
        acc[0] /= n

    cute.arch.sync_threads()

    if idx < n:
        out[idx] = out[idx] / cute.math.sqrt(acc[0] + eps) * weight[idx]

@cute.jit
def rms_norm(
    a: cute.Tensor,
    residual: cute.Tensor,
    weight: cute.Tensor,
    eps: cutlass.Float32,
    out: cute.Tensor
):
    threads_per_block = 256
    n = a.shape[0]

    kernel_instance = rms_norm_kernel(a, residual, weight, eps, out)

    kernel_instance.launch(
        grid=((n + threads_per_block - 1) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )