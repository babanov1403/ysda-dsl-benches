import torch
import tilelang
import tilelang.language as T

from common.run import _do_bench


@tilelang.jit
def softmax_kernel_(rows, cols, block_size, threads, dtype, vectorized=True):
    """Compile for contiguous, 32-byte-aligned input and output buffers."""
    if threads not in (32, 64, 128, 256) or block_size < threads or block_size & (block_size - 1):
        raise ValueError("Use a power-of-two block size >= threads, with 32, 64, 128 or 256 threads")
    # Full tiles keep both the row and tile offsets aligned for wide copies.
    vector_io = vectorized and cols % block_size == 0

    @T.prim_func
    def softmax_kernel(X: T.Tensor((rows, cols), dtype), Y: T.Tensor((rows, cols), dtype)):
        with T.Kernel(rows, threads=threads) as row:
            values = T.alloc_fragment((block_size,), "float32")
            weights = T.alloc_fragment((block_size,), "float32")
            row_max = T.alloc_fragment((1,), "float32")
            inv_sum = T.alloc_fragment((1,), "float32")
            for i in T.Parallel(block_size):
                values[i] = T.if_then_else(i < cols, T.cast(X[row, i], "float32"), -T.infinity("float32"))

            if cols <= block_size:
                # A whole row fits in registers; don't reread it.
                T.reduce_max(values, row_max, dim=0)
                for i in T.Parallel(block_size):
                    weights[i] = T.__exp(values[i] - row_max[0])
                T.reduce_sum(weights, inv_sum, dim=0)
                inv_sum[0] = 1.0 / inv_sum[0]
                for i in T.Parallel(block_size):
                    if i < cols:
                        Y[row, i] = weights[i] * inv_sum[0]
            else:
                # Keep an online (max, sum) per fragment element, so the scan
                # needs no block-wide reduction until the end of the row.
                for i in T.Parallel(block_size):
                    weights[i] = 1.0
                io_tile = T.alloc_fragment((block_size,), dtype)
                for tile in T.serial(1, T.ceildiv(cols, block_size)):
                    if vector_io:
                        start = tile * block_size
                        T.copy(X[row, start : start + block_size], io_tile)
                    for i in T.Parallel(block_size):
                        col = tile * block_size + i
                        if col < cols:
                            if vector_io:
                                x = T.cast(io_tile[i], "float32")
                            else:
                                x = T.cast(X[row, col], "float32")
                            # Rescale only the smaller term; equality also handles -inf.
                            scale = T.if_then_else(x == values[i], 1.0, T.__exp(-T.abs(x - values[i])))
                            weights[i] = T.if_then_else(x > values[i], weights[i] * scale + 1.0, weights[i] + scale)
                            values[i] = T.max(values[i], x)
                T.reduce_max(values, row_max, dim=0)
                for i in T.Parallel(block_size):
                    weights[i] *= T.__exp(values[i] - row_max[0])
                T.reduce_sum(weights, inv_sum, dim=0)
                inv_sum[0] = 1.0 / inv_sum[0]
                if vector_io:
                    for tile in T.serial(cols // block_size):
                        start = tile * block_size
                        T.copy(X[row, start : start + block_size], io_tile)
                        for i in T.Parallel(block_size):
                            io_tile[i] = T.__exp(T.cast(io_tile[i], "float32") - row_max[0]) * inv_sum[0]
                        T.copy(io_tile, Y[row, start : start + block_size])
                else:
                    for tile in T.serial(T.ceildiv(cols, block_size)):
                        for i in T.Parallel(block_size):
                            col = tile * block_size + i
                            if col < cols:
                                Y[row, col] = T.__exp(T.cast(X[row, col], "float32") - row_max[0]) * inv_sum[0]

    return softmax_kernel


_configs = {}


def _dtype(x):
    return str(x.dtype).removeprefix("torch.")


def _default_config(cols):
    return min(4096, max(32, 1 << (cols - 1).bit_length())), 128 if cols > 128 else 32


def _aligned(x):
    # Vectorized global loads require alignment even for contiguous views.
    return x if x.data_ptr() % 32 == 0 else x.clone()


def tune_softmax(x):
    """Tune on a contiguous CUDA input; cache by shape, dtype and device."""
    rows, cols = x.shape
    if not x.numel():
        raise ValueError("Cannot tune an empty input")
    key = (x.device, x.dtype, rows, cols)
    if key not in _configs:
        largest, _ = _default_config(cols)
        smallest = min(128, largest // 2)
        with torch.cuda.device(x.device):
            x = _aligned(x)
            out = torch.empty_like(x)
            kernels = {
                (block_size, threads): softmax_kernel_(rows, cols, block_size, threads, _dtype(x))
                for block_size in (32, 64, 128, 256, 512, 1024, 2048, 4096)
                for threads in (32, 64, 128, 256)
                if smallest <= block_size <= largest and threads <= block_size <= threads * 64
            }

            def latency(config):
                kernel = kernels[config]
                return _do_bench(lambda: kernel(x, out), warmup=25, rep=100, return_mode="median")

            _configs[key] = min(kernels, key=latency)
    return _configs[key]


def specialize(kernel, x, *, autotune=False):
    config = tune_softmax(x) if autotune else _default_config(x.shape[1])
    return kernel(*x.shape, *config, _dtype(x))


def softmax(x, *, autotune=False):
    """Row-wise softmax of a contiguous 2-D CUDA tensor with FP32 accumulation."""
    if x.ndim != 2 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("Expected a contiguous 2-D CUDA tensor")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("Expected float32, float16 or bfloat16")
    out = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            x = _aligned(x)
            specialize(softmax_kernel_, x, autotune=autotune)(x, out)
    return out
