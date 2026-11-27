import cutlass
import cutlass.cute as cute
from quack.autotuner import AutotuneConfig

import operator

from common.run import autotunable

GROUP_SIZE = 256

def block_reduce(binary_operand):
    @cute.jit
    def kern(partial: cutlass.Float32) -> cutlass.Float32:
        shmem = cutlass.Array(cutlass.Float32, (GROUP_SIZE,), space=cutlass.AddressSpace.smem)

        tid, _, _ = cute.arch.thread_idx()

        shmem[tid] = partial
        cute.arch.sync_threads()
        REDUCE_STEPS = (GROUP_SIZE - 1).bit_length()
        for k in cutlass.range_constexpr(REDUCE_STEPS):
            stride = 1 << (REDUCE_STEPS - k - 1)
            if tid < stride and tid + stride < GROUP_SIZE:
                shmem[tid] = binary_operand(shmem[tid], shmem[tid + stride])
            cute.arch.sync_threads()

        return shmem[0]

    return kern

block_max = block_reduce(cute.arch.fmax)
block_sum = block_reduce(operator.add)

@cute.kernel
def softmax_wide_kernel(x: cute.Tensor, out: cute.Tensor, V: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    if bid < x.shape[0]:
        xRow = cutlass.make_array_view(x[bid, None])
        tidx = tid * V

        xVec = xRow[tidx:V]

        # compute maximum
        max = block_max(xVec.reduce("max"))

        xVec -= max
        xExpVec = cute.math.exp(xVec)

        sum = block_sum(xExpVec.reduce("add"))

        xExpVec /= sum

        outRow = cutlass.make_array_view(out[bid, None])
        outRow[tidx:V] = xExpVec

@cute.kernel
def softmax_wide_kernel_scalar(x: cute.Tensor, out: cute.Tensor, V: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    if bid < x.shape[0]:
        xRow = cutlass.make_array_view(x[bid, None])
        tidx = tid * V

        partial_max = cute.Float32(float("-inf"))
        for idx in cutlass.range(V):
            if tidx + idx < x.shape[-1]:
                partial_max = cute.arch.fmax(partial_max, xRow[tidx + idx])

        # compute maximum
        max = block_max(partial_max)

        partial_exp_sum = cute.Float32(0)
        for idx in cutlass.range(V):
            if tidx + idx < x.shape[-1]:
                partial_exp_sum += cute.math.exp(xRow[tidx + idx] - max)

        sum = block_sum(partial_exp_sum)

        outRow = cutlass.make_array_view(out[bid, None])
        for idx in cutlass.range(V):
            if tidx + idx < x.shape[-1]:
                outRow[tidx + idx] = cute.math.exp(xRow[tidx + idx] - max) / sum

@autotunable(configs=[AutotuneConfig()])
@cute.jit
def softmax_wide(x: cute.Tensor, out: cute.Tensor):
    V = cute.ceil_div(x.shape[-1], GROUP_SIZE)
    print(V)
    if cutlass.const_expr(x.shape[-1] == V * GROUP_SIZE and V in (1, 2, 4, 8, 16)):
        softmax_wide_kernel(x, out, V).launch(
            grid=(x.shape[0], 1, 1),
            block=(GROUP_SIZE, 1, 1)
        )
    else:
        softmax_wide_kernel_scalar(x, out, V).launch(
            grid=(x.shape[0], 1, 1),
            block=(GROUP_SIZE, 1, 1)
        )

@autotunable(configs=[AutotuneConfig()])
@cute.jit
def softmax_online(x: cute.Tensor, out: cute.Tensor):
    raise NotImplementedError("TODO: implement me!")
