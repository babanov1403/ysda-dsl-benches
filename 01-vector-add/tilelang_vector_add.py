import tilelang
import tilelang.language as T


@tilelang.jit
def add(a, b, BLOCK_SIZE: int = 256):
    N = T.const("N")
    a: T.Tensor((N,), T.float32)
    b: T.Tensor((N,), T.float32)
    c = T.empty((N,), T.float32)
    with T.Kernel(N // BLOCK_SIZE, threads=256) as bidx:
        for tidx in T.Parallel(BLOCK_SIZE):
            index = bidx * BLOCK_SIZE + tidx
            c[index] = a[index] + b[index]
    return c
