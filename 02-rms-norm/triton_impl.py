import torch

import triton
import triton.language as tl

@triton.jit
def residual_rms_norm_kernel(a_ptr, residual_ptr, w_ptr, eps, res_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask)
    residual = tl.load(residual_ptr + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)

    x = a + residual
    mean_sum_of_squares = tl.sum(x * x, axis = 0) / n + eps
    res = x * w / tl.sqrt(mean_sum_of_squares)

    tl.store(res_ptr + offsets, res, mask=mask)

def residual_rms_norm(a: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    output = torch.empty_like(a)
    n = output.numel()

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
    residual_rms_norm_kernel[grid](a, residual, weight, eps, output, n, BLOCK_SIZE=256)
    return output