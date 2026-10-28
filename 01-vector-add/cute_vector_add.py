import torch

from typing import List

import cutlass
import cutlass.cute as cute

@cute.kernel
def vector_add_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    index = tidx + bidx * bdim

    n = a.shape[0]
    if (index < n):
        out[index] = a[index] + b[index]

@cute.jit
def add(
    a: cute.Tensor,
    b: cute.Tensor,
    out: cute.Tensor,
):
    threads_per_block = 256
    n = a.shape[0]

    kernel_instance = vector_add_kernel(a, b, out)

    kernel_instance.launch(
        grid = (n // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )