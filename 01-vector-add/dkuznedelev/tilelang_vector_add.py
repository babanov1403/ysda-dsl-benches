import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def add_(N: int, BLOCK_SIZE: int = 256, dtype: str = "float32"):

    @T.prim_func()
    def add_kernel(
        a: T.Tensor((N,), dtype),
        b: T.Tensor((N,), dtype),
        c: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_SIZE)) as block_idx:
            for thread_idx in T.Parallel(BLOCK_SIZE):
                idx = BLOCK_SIZE * block_idx + thread_idx
                c[idx] = a[idx] + b[idx]

    return add_kernel


def add(a: torch.Tensor, b: torch.Tensor):
    c = torch.empty_like(a)
    N = c.numel()
    kernel = add_(N)
    kernel(a, b, c)
    return c
