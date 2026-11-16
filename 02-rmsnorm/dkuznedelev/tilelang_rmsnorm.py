import torch
import tilelang
import tilelang.language as T

EPS = 1e-6


@tilelang.jit
def rmsnorm_(N: int, D: int, BLOCK_SIZE: int = 256, dtype: str = "float32"):
    """Compile a contiguous N-by-D RMSNorm, with one thread block per row."""
    if N < 0 or D <= 0:
        raise ValueError("N must be nonnegative and D must be positive")
    if BLOCK_SIZE < 32 or BLOCK_SIZE & (BLOCK_SIZE - 1):
        raise ValueError("BLOCK_SIZE must be a power of two and at least 32")
    if dtype not in ("float32", "float16", "bfloat16"):
        raise TypeError("RMSNorm supports float32, float16, and bfloat16")

    @T.prim_func()
    def rmsnorm_kernel(
        x: T.Tensor((N, D), dtype),
        w: T.Tensor((D,), dtype),
        out: T.Tensor((N, D), dtype),
    ):
        with T.Kernel(N, threads=128) as row:
            sum_sq = T.alloc_fragment((BLOCK_SIZE,), "float32")
            total = T.alloc_fragment((1,), "float32")
            T.clear(sum_sq)
            # Accumulate strided tiles, then reduce just once per row. The
            # explicit guard excludes the padded tail from the denominator.
            for tile in T.serial(T.ceildiv(D, BLOCK_SIZE)):
                for i in T.Parallel(BLOCK_SIZE):
                    col = tile * BLOCK_SIZE + i
                    if col < D:
                        value = T.cast(x[row, col], "float32")
                        sum_sq[i] += value * value
            T.reduce_sum(sum_sq, total, dim=0)
            total[0] = T.rsqrt(total[0] / D + EPS)
            for tile in T.serial(T.ceildiv(D, BLOCK_SIZE)):
                for i in T.Parallel(BLOCK_SIZE):
                    col = tile * BLOCK_SIZE + i
                    if col < D:
                        out[row, col] = (
                            T.cast(x[row, col], "float32")
                            * total[0]
                            * T.cast(w[col], "float32")
                        )

    return rmsnorm_kernel


def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """RMS-normalize rows of x and apply w; inputs must be contiguous CUDA tensors."""
    if x.ndim != 2 or w.ndim != 1 or w.shape[0] != x.shape[1]:
        raise ValueError("Expected x of shape (N, D) and w of shape (D,)")
    if x.shape[1] == 0:
        raise ValueError("The normalized dimension must be nonempty")
    if not x.is_cuda or not w.is_cuda or x.device != w.device:
        raise ValueError("x and w must be on the same CUDA device")
    if x.dtype != w.dtype or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("x and w must have the same float32, float16, or bfloat16 dtype")
    if not x.is_contiguous() or not w.is_contiguous():
        raise ValueError("x and w must be contiguous")
    out = torch.empty_like(x)
    if x.shape[0] == 0:
        return out
    with torch.cuda.device(x.device):
        kernel = rmsnorm_(*x.shape, dtype=str(x.dtype).removeprefix("torch."))
        kernel(x, w, out)
    return out
