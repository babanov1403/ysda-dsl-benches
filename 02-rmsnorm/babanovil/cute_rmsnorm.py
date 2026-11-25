import cutlass
import cutlass.cute as cute
from quack.autotuner import AutotuneConfig

from common.run import autotunable

EPS = 1e-6
GROUP_SIZE = 256
REDUCE_STEPS = (GROUP_SIZE - 1).bit_length()

@cute.jit
def block_sum(partial: cutlass.Float32) -> cutlass.Float32:
    shmem = cutlass.Array(cutlass.Float32, (GROUP_SIZE,), space=cutlass.AddressSpace.smem)

    tid, _, _ = cute.arch.thread_idx()

    shmem[tid] = partial
    cute.arch.sync_threads()
    
    for k in cutlass.range_constexpr(REDUCE_STEPS):
        stride = 1 << (REDUCE_STEPS - k - 1)
        if tid < stride and tid + stride < GROUP_SIZE:
            shmem[tid] = shmem[tid] + shmem[tid + stride]
        cute.arch.sync_threads()

    return shmem[0]

@cute.kernel
def rms_norm_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    out: cute.Tensor,
    eps: cutlass.Float32,
    V: cutlass.Constexpr
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    xRow = cutlass.make_array_view(x[bid, None])

    tidx = tid * V
    
    xVec = xRow[tidx:V]
    w = cutlass.make_array_view(weight)[tidx:V]

    reduced = (xVec * xVec).reduce("add")
    sumRow = block_sum(reduced)
    invRms = cute.math.rsqrt(sumRow / x.shape[-1] + eps)

    outRow = cutlass.make_array_view(out[bid, None])
    outRow[tidx:V] = xVec * invRms * w

@cute.kernel
def rms_norm_kernel_scalar(
    x: cute.Tensor,
    weight: cute.Tensor,
    out: cute.Tensor,
    eps: cutlass.Float32,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    partial = cutlass.Float32(0)
    for j in cutlass.range(tid, x.shape[-1], bdim):
        value = x[bid, j]
        partial += value * value

    sumRow = block_sum(partial)
    invRms = cute.math.rsqrt(sumRow / x.shape[-1] + eps)

    for j in cutlass.range(tid, x.shape[-1], bdim):
        out[bid, j] = x[bid, j] * invRms * weight[j]



@autotunable(configs=[AutotuneConfig()])
@cute.jit
def rms_norm(x: cute.Tensor, weight: cute.Tensor, out: cute.Tensor):
    V = cute.ceil_div(x.shape[-1], GROUP_SIZE)
    if cutlass.const_expr(x.shape[-1] == V * GROUP_SIZE and V in (1, 2, 4, 8, 16)):
        rms_norm_kernel(x, weight, out, EPS, V).launch(
            grid=(x.shape[0], 1, 1),
            block=(GROUP_SIZE, 1, 1)
        )
    else:
        rms_norm_kernel_scalar(x, weight, out, EPS).launch(
            grid=(x.shape[0], 1, 1),
            block=(GROUP_SIZE, 1, 1)
        )
