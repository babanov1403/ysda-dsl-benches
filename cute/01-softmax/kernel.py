import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def softmax(x: cute.Tensor, out: cute.Tensor):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    lane = tx % 32
    row = bx * 4 + tx // 32
    # Entire warps participate in the reductions, including lanes past the tail.
    if row < x.shape[0]:
        maximum = cutlass.Float32(float('-inf'))
        for col in range(lane, x.shape[1], 32):
            maximum = cute.arch.fmax(maximum, x[row, col])
        maximum = cute.arch.warp_reduction_max(maximum)
        total = cutlass.Float32(0.0)
        for col in range(lane, x.shape[1], 32):
            total += cute.exp(x[row, col] - maximum)
        total = cute.arch.warp_reduction_sum(total)
        for col in range(lane, x.shape[1], 32):
            out[row, col] = cute.exp(x[row, col] - maximum) / total


@cute.jit
def launch(x: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
    softmax(x, out).launch(
        grid=((x.shape[0] + 3) // 4, 1, 1), block=(128, 1, 1), stream=stream
    )


def prepare(inputs, out):
    (x,) = inputs
    args = from_dlpack(x), from_dlpack(out)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(launch, *args, stream)
    return lambda: compiled(*args, stream)
