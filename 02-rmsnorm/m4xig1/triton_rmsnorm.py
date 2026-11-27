import triton
import triton.language as tl

from common.run import autotunable


@autotunable(
    configs=[
        triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps)
        for block_size in (512, 1024, 2048, 4096, 8192)
        for num_warps in (2, 4, 8, 16)
    ],
    key=["n"],
)
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride, n, eps, BLOCK_SIZE: tl.constexpr = 1024):
    # One program per row: sum of squares over the row, then scale it in a second pass.
    row = tl.program_id(0)
    x_ptr += row * stride
    out_ptr += row * stride

    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(0, n, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets, mask=offsets < n, other=0.0)
        sum_sq += x * x
    inv_rms = tl.rsqrt(tl.sum(sum_sq) / n + eps)

    for start in range(0, n, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        w = tl.load(w_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x * inv_rms * w, mask=mask)

"""
A = [....] \in R^N

kernel 1: \sum_{i = j}^{j + BLOCK_K} --> S_j, j = 0, BLOCK_K, ..., N/BLOCK_K

kernel 2: \sum S_j \forall j


Triton with Thread block clusters and Distributed Shared Memory


"""
