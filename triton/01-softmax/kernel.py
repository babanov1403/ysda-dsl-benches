import triton
import triton.language as tl


@triton.jit
def softmax(x, out, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(x + row * COLS + col, col < COLS, float('-inf'))
    values = values - tl.max(values, axis=0)
    e = tl.exp(values)
    result = e / tl.sum(e, axis=0)
    tl.store(out + row * COLS + col, result, col < COLS)


def prepare(inputs, out):
    (x,) = inputs
    rows, cols = x.shape
    block = triton.next_power_of_2(cols)
    if block > 16384:
        raise ValueError('Initial softmax baseline supports at most 16384 columns')
    return lambda: softmax[(rows,)](x, out, cols, block, num_warps=4)
