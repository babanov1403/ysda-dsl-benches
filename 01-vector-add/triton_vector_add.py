import torch

import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    out = a + b

    tl.store(out_ptr + offsets, out, mask=mask)

def add(a: torch.Tensor, b: torch.Tensor):
    output = torch.empty_like(a)

    if not a.is_cuda or a.device != b.device:
        raise ValueError("Inputs must be on the same CUDA device")

    n = output.numel()

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )

    with torch.cuda.device(a.device):
        vector_add_kernel[grid](a, b, output, n, BLOCK_SIZE=256)

    return output