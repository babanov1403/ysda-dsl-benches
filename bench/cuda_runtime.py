"""Thin C ABI bridge; CUDA C++ kernels do not depend on PyTorch headers."""
import ctypes
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


@lru_cache
def library(arch):
    path = ROOT / 'build' / f'sm_{arch}' / 'libysda_kernels.so'
    if not path.exists():
        raise RuntimeError(f'{path} missing; run bash scripts/build.sh on this GPU')
    lib = ctypes.CDLL(str(path))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.ysda_vector_add.argtypes = [ptr, ptr, ptr, integer, ptr]
    lib.ysda_vector_add.restype = integer
    lib.ysda_softmax.argtypes = [ptr, ptr, integer, integer, ptr]
    lib.ysda_softmax.restype = integer
    lib.ysda_cuda_error.argtypes = [integer]
    lib.ysda_cuda_error.restype = ctypes.c_char_p
    return lib


def prepare(task, inputs, out):
    a, b = torch.cuda.get_device_capability(out.device)
    lib = library(f'{a}{b}')
    stream = torch.cuda.current_stream(out.device).cuda_stream
    if task == '00-vector-add':
        x, y = inputs
        args = x.data_ptr(), y.data_ptr(), out.data_ptr(), x.numel(), stream
        func = lib.ysda_vector_add
    elif task == '01-softmax':
        (x,) = inputs
        args = x.data_ptr(), out.data_ptr(), *x.shape, stream
        func = lib.ysda_softmax
    else:
        raise ValueError(task)

    def launch():
        code = func(*args)
        if code:
            raise RuntimeError(lib.ysda_cuda_error(code).decode())
    return launch
