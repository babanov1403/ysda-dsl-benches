import triton
import triton.language as tl


@triton.jit
def vector_add(a, b, out, n: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + i, i < n, 0)
    y = tl.load(b + i, i < n, 0)
    tl.store(out + i, x + y, i < n)


def prepare(inputs, out):
    a, b = inputs
    n = a.numel()
    return lambda: vector_add[(triton.cdiv(n, 256),)](a, b, out, n, 256)
