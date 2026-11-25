import cutlass
import cutlass.cute as cute
from quack.autotuner import AutotuneConfig

from common.run import autotunable

EPS = 1e-6
GROUP_SIZE = 256

@cute.jit
def block_sum(partial: cutlass.Float32) -> cutlass.Float32:
    raise NotImplementedError("TODO: implement me!")

@cute.kernel
def rms_norm_kernel(
    x: cute.Tensor,
    residual: cute.Tensor,
    weight: cute.Tensor,
    out: cute.Tensor,
    eps: cutlass.Float32,
    V: cutlass.Constexpr
):
    raise NotImplementedError("TODO: implement me!")

@cute.kernel
def rms_norm_kernel_scalar(
    x: cute.Tensor,
    residual: cute.Tensor,
    weight: cute.Tensor,
    out: cute.Tensor,
    eps: cutlass.Float32,
):
    raise NotImplementedError("TODO: implement me!")



@autotunable(configs=[AutotuneConfig()])
@cute.jit
def rms_norm(x: cute.Tensor, residual: cute.Tensor, weight: cute.Tensor, out: cute.Tensor):
    raise NotImplementedError("TODO: implement me!")
