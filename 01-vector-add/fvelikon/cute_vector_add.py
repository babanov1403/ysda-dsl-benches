import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.base_dsl.typing import Vector
from quack.autotuner import AutotuneConfig

from common.run import autotunable

WIDTH = 4  # floats per memory operation; wider is faster, see PTX ld.global.v4.b32


@cute.kernel
def vector_add_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    index = (tidx + bidx * bdim) * WIDTH

    n = a.shape[0]
    if index + WIDTH <= n:
        vector = ir.VectorType.get([WIDTH], cutlass.Float32.mlir_type)
        x = Vector(cute.arch.load((a.iterator + index).llvm_ptr, vector))
        y = Vector(cute.arch.load((b.iterator + index).llvm_ptr, vector))
        cute.arch.store((out.iterator + index).llvm_ptr, (x + y).ir_value())
    else:
        # A full vector would touch memory past n, so the tail stays scalar.
        for offset in range(WIDTH - 1):
            i = index + offset
            if i < n:
                out[i] = a[i] + b[i]


@autotunable(
    configs=[AutotuneConfig(threads_per_block=t) for t in (64, 128, 256, 512, 1024)],
)
@cute.jit
def add(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, threads_per_block: cutlass.Constexpr = 256):
    elements_per_block = threads_per_block * WIDTH

    vector_add_kernel(a, b, out).launch(
        grid=((a.shape[0] + elements_per_block - 1) // elements_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )
