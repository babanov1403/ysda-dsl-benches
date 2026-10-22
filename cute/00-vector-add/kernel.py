import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def vector_add(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    i = bx * 256 + tx
    if i < a.shape[0]:
        out[i] = a[i] + b[i]


@cute.jit
def launch(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
    vector_add(a, b, out).launch(
        grid=((a.shape[0] + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
    )


def prepare(inputs, out):
    a, b = inputs
    args = tuple(from_dlpack(x) for x in (a, b, out))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(launch, *args, stream)
    return lambda: compiled(*args, stream)
