import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator
from quack.autotuner import AutotuneConfig

from common.run import autotunable

EPS = 1e-6
DEFAULT_WIDTH = 2
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
def rmsnorm_vec_kernel(
    x: cute.Tensor,
    w: cute.Tensor,
    out: cute.Tensor,
    eps: cutlass.Constexpr,
    num_warps: cutlass.Constexpr,
    WIDTH: cutlass.Constexpr = DEFAULT_WIDTH,
):
    # Packed FP32 FMA requires SM100+.
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    n = x.shape[1]
    x_row = x[row, None]
    out_row = out[row, None]

    sum_sq_lo = cutlass.Float32(0.0)
    sum_sq_hi = cutlass.Float32(0.0)
    extra_sq_lo = cutlass.Float32(0.0)
    extra_sq_hi = cutlass.Float32(0.0)
    for tile in range(tidx, n // WIDTH, bdim):
        v = cute.local_tile(x_row, (WIDTH,), (tile,)).load()
        pair = (v[0], v[1])
        sum_sq_lo, sum_sq_hi = cute.arch.fma_packed_f32x2(pair, pair, (sum_sq_lo, sum_sq_hi))
        if cutlass.const_expr(WIDTH == 4):
            extra_pair = (v[2], v[3])
            extra_sq_lo, extra_sq_hi = cute.arch.fma_packed_f32x2(
                extra_pair, extra_pair, (extra_sq_lo, extra_sq_hi)
            )
    sum_sq = sum_sq_lo + sum_sq_hi
    if cutlass.const_expr(WIDTH == 4):
        sum_sq += extra_sq_lo + extra_sq_hi
    inv_rms = cute.math.rsqrt(block_sum(sum_sq, num_warps) / n + eps)

    for tile in range(tidx, n // WIDTH, bdim):
        x_tile = cute.local_tile(x_row, (WIDTH,), (tile,))
        w_tile = cute.local_tile(w, (WIDTH,), (tile,))
        out_tile = cute.local_tile(out_row, (WIDTH,), (tile,))
        out_tile.store(x_tile.load() * w_tile.load() * inv_rms)


@autotunable(configs=CONFIGS)
@cute.jit
def rmsnorm(x: cute.Tensor, w: cute.Tensor, out: cute.Tensor, threads_per_block: cutlass.Constexpr = 256):
    rmsnorm_kernel(x, w, out, EPS, threads_per_block // 32).launch(
        grid=(x.shape[0], 1, 1), block=(threads_per_block, 1, 1), smem=threads_per_block // 32 * 4
    )


@autotunable(configs=CONFIGS)
@cute.jit
def rmsnorm_vec(
    x: cute.Tensor,
    w: cute.Tensor,
    out: cute.Tensor,
    threads_per_block: cutlass.Constexpr = 256,
    WIDTH: cutlass.Constexpr = DEFAULT_WIDTH,
):
    """Normalize rows divisible by WIDTH, with WIDTH * 4-byte vector alignment."""
    assert WIDTH == 2 or WIDTH == 4, "WIDTH must be 2 or 4"
    if cutlass.const_expr(cute.is_static(x.shape[1])):
        assert x.shape[1] % WIDTH == 0, "row width must be divisible by WIDTH"
    rmsnorm_vec_kernel(x, w, out, EPS, threads_per_block // 32, WIDTH).launch(
        grid=(x.shape[0], 1, 1), block=(threads_per_block, 1, 1), smem=threads_per_block // 32 * 4
    )
