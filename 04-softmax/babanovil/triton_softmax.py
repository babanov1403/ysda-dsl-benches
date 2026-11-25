import triton
import triton.language as tl


@triton.jit
def softmax_wide_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr = 256):
    tl.static_assert(False, "TODO: implement me!")


@triton.jit
def softmax_online_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr = 256):
    tl.static_assert(False, "TODO: implement me!")
