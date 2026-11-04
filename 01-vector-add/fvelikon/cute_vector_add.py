import cutlass
import cutlass.cute as cute
from quack.autotuner import AutotuneConfig

from common.run import autotunable


@cute.kernel
def vector_add_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    index = tidx + bidx * bdim

    n = a.shape[0]
    if index < n:
        out[index] = a[index] + b[index]


@autotunable(
    configs=[AutotuneConfig(threads_per_block=t) for t in (64, 128, 256, 512, 1024)],
)
@cute.jit
def add(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, threads_per_block: cutlass.Constexpr = 256):
    n = a.shape[0]

    vector_add_kernel(a, b, out).launch(
        grid=((n + threads_per_block - 1) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )
