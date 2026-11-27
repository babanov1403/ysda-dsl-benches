import triton
import triton.language as tl

from common.run import autotunable


def rmsnorm_grid(meta):
    return (triton.cdiv(meta["m"], meta.get("ROWS_PER_PROGRAM", 1)),)


@autotunable(
    configs=[
        triton.Config({"ROWS_PER_PROGRAM": rows}, num_warps=warps)
        for rows in (1, 2, 4)
        for warps in (1, 2, 4, 8)
    ],
    key=["m", "n"],
)
@triton.jit
def rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    x_stride,
    out_stride,
    m,
    n: tl.constexpr,
    eps,
    ROWS_PER_PROGRAM: tl.constexpr = 1,
):
    BLOCK_COLS: tl.constexpr = min(triton.next_power_of_2(n), 4096)
    rows = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    cols = tl.arange(0, BLOCK_COLS)
    row_mask = rows[:, None] < m
    x_rows = x_ptr + rows[:, None] * x_stride
    out_rows = out_ptr + rows[:, None] * out_stride

    if n <= BLOCK_COLS:
        col_mask = cols < n
        mask = row_mask & col_mask[None, :]
        x = tl.load(x_rows + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / n + eps)
        w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        tl.store(out_rows + cols[None, :], x * inv_rms[:, None] * w[None, :], mask=mask)
    else:
        # Bound the accumulator size for rows that span multiple tiles.
        sum_sq = tl.zeros((ROWS_PER_PROGRAM, BLOCK_COLS), dtype=tl.float32)
        for start in range(0, n, BLOCK_COLS):
            offsets = start + cols
            mask = row_mask & (offsets[None, :] < n)
            x = tl.load(x_rows + offsets[None, :], mask=mask, other=0.0).to(tl.float32)
            sum_sq += x * x
        inv_rms = tl.rsqrt(tl.sum(sum_sq, axis=1) / n + eps)

        for start in range(0, n, BLOCK_COLS):
            offsets = start + cols
            col_mask = offsets < n
            mask = row_mask & col_mask[None, :]
            x = tl.load(x_rows + offsets[None, :], mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
            tl.store(out_rows + offsets[None, :], x * inv_rms[:, None] * w[None, :], mask=mask)
