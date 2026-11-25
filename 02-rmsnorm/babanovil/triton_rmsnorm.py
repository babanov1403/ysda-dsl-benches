import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr, weight_ptr, out_ptr, row_stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr = 256,
):
    # TODO: compute one output row per program. weight is shared across rows.
    pass
