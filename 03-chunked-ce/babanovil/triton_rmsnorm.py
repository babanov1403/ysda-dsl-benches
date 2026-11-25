import triton
import triton.language as tl


@triton.jit
def residual_rms_norm_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr, row_stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr = 256,
):
    tl.static_assert(False, "TODO: implement me!")
