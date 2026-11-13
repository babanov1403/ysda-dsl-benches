import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.base_dsl.typing import Vector
from cutlass.utils import SmemAllocator
from quack.autotuner import AutotuneConfig

from common.run import autotunable

EPS = 1e-6
WIDTH = 4  # floats per memory operation in the vectorized kernel
CONFIGS = [AutotuneConfig(threads_per_block=t) for t in (64, 128, 256, 512, 1024)]


@cute.jit
def block_sum(value: cutlass.Float32, num_warps: cutlass.Constexpr) -> cutlass.Float32:
    """Sum of value over the whole block, returned to every thread. Needs num_warps floats of dynamic smem."""
    partials = SmemAllocator().allocate_tensor(cutlass.Float32, cute.make_layout(num_warps))
    value = cute.arch.warp_reduction_sum(value)
    # warp_idx() is a full-warp shuffle, so every lane must call it, not only lane 0.
    warp = cute.arch.warp_idx()
    if cute.arch.lane_idx() == 0:
        partials[warp] = value
    cute.arch.sync_threads()
    total = cutlass.Float32(0.0)
    for w in cutlass.range_constexpr(num_warps):
        total += partials[w]
    return total


@cute.kernel
def rmsnorm_kernel(x: cute.Tensor, w: cute.Tensor, out: cute.Tensor, eps: cutlass.Constexpr, num_warps: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    n = x.shape[1]

    sum_sq = cutlass.Float32(0.0)
    for col in range(tidx, n, bdim):
        sum_sq += x[row, col] * x[row, col]
    inv_rms = cute.math.rsqrt(block_sum(sum_sq, num_warps) / n + eps)

    for col in range(tidx, n, bdim):
        out[row, col] = x[row, col] * inv_rms * w[col]


@cute.kernel
def rmsnorm_vec_kernel(x: cute.Tensor, w: cute.Tensor, out: cute.Tensor, eps: cutlass.Constexpr, num_warps: cutlass.Constexpr):
    # Same as rmsnorm_kernel, but every load and store moves WIDTH floats. Needs n % WIDTH == 0.
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    n = x.shape[1]
    vector = ir.VectorType.get([WIDTH], cutlass.Float32.mlir_type)  # == float4
    load = lambda tensor, offset: Vector(cute.arch.load((tensor.iterator + offset).llvm_ptr, vector))

    sum_sq = cutlass.Float32(0.0)
    for col in range(tidx * WIDTH, n, bdim * WIDTH):
        v = load(x, row * n + col)
        for i in cutlass.range_constexpr(WIDTH):
            sum_sq += v[i] * v[i]
    inv_rms = cute.math.rsqrt(block_sum(sum_sq, num_warps) / n + eps)

    for col in range(tidx * WIDTH, n, bdim * WIDTH):
        y = load(x, row * n + col) * load(w, col) * inv_rms
        cute.arch.store((out.iterator + row * n + col).llvm_ptr, y.ir_value())


@autotunable(configs=CONFIGS)
@cute.jit
def rmsnorm(x: cute.Tensor, w: cute.Tensor, out: cute.Tensor, threads_per_block: cutlass.Constexpr = 256):
    rmsnorm_kernel(x, w, out, EPS, threads_per_block // 32).launch(
        grid=(x.shape[0], 1, 1), block=(threads_per_block, 1, 1), smem=threads_per_block // 32 * 4
    )


@autotunable(configs=CONFIGS)
@cute.jit
def rmsnorm_vec(x: cute.Tensor, w: cute.Tensor, out: cute.Tensor, threads_per_block: cutlass.Constexpr = 256):
    rmsnorm_vec_kernel(x, w, out, EPS, threads_per_block // 32).launch(
        grid=(x.shape[0], 1, 1), block=(threads_per_block, 1, 1), 
        smem=threads_per_block // 32 * 4
    )
