import triton
import triton.language as tl

from common.run import autotunable


@autotunable(
    configs=[
        triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps)
        for block_size in (64, 128, 256, 512, 1024)
        for num_warps in (1, 2, 4, 8)
    ],
    key=["n"],
)
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr = 256):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    tl.store(out_ptr + offsets, a + b, mask=mask)
