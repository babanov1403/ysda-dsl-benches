"""One contract for all languages. No allocations or references in timed code."""
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ('triton', 'cute', 'cuda')
TASKS = ('00-vector-add', '01-softmax')


@lru_cache
def load(language, task):
    if language not in LANGUAGES or task not in TASKS:
        raise ValueError(f'Unknown implementation: {language}/{task}')
    path = ROOT / language / task / 'kernel.py'
    name = f'ysda_{language}_{task.replace("-", "_")}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(language, task, inputs, out):
    if not inputs or any(t.device != out.device for t in inputs):
        raise ValueError('All tensors must be on the same CUDA device')
    for t in (*inputs, out):
        if not t.is_cuda or t.dtype != torch.float32 or not t.is_contiguous():
            raise ValueError('Initial kernels require contiguous CUDA float32 tensors')
        if t.numel() < 1 or t.numel() >= 2**31:
            raise ValueError('Expected 1 <= numel < 2**31')
    if out.shape != inputs[0].shape:
        raise ValueError('Output must have the same shape as the first input')
    out_begin = out.data_ptr()
    out_end = out_begin + out.numel() * out.element_size()
    if any(out_begin < t.data_ptr() + t.numel() * t.element_size()
           and t.data_ptr() < out_end for t in inputs):
        raise ValueError('Output must not alias inputs')
    if task == '00-vector-add':
        if len(inputs) != 2 or inputs[0].ndim != 1 or inputs[1].shape != out.shape:
            raise ValueError('Vector add requires two equally-sized 1D tensors')
    elif task == '01-softmax':
        if len(inputs) != 1 or out.ndim != 2 or out.shape[1] > 16384:
            raise ValueError('Softmax requires one [rows, cols] tensor; cols <= 16384')
    with torch.cuda.device(out.device):
        return load(language, task).prepare(inputs, out)


def make_inputs(task, *, size=1048579, rows=256, cols=1024, seed=0):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    if task == '00-vector-add':
        if size < 1:
            raise ValueError('size must be positive')
        return tuple(torch.randn(size, device='cuda', generator=generator) for _ in range(2))
    if task == '01-softmax':
        if rows < 1 or not 1 <= cols <= 16384:
            raise ValueError('rows must be positive; 1 <= cols <= 16384')
        return (torch.randn((rows, cols), device='cuda', generator=generator),)
    raise ValueError(task)


def reference(task, inputs):
    # FP64 reference computed outside timing; output contract is FP32.
    if task == '00-vector-add':
        return (inputs[0].double() + inputs[1].double()).float()
    if task == '01-softmax':
        return torch.softmax(inputs[0].double(), dim=-1).float()
    raise ValueError(task)


def check(task, out, expected):
    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-6, equal_nan=False)
    if task == '01-softmax':
        torch.testing.assert_close(out.sum(dim=-1), torch.ones_like(out[:, 0]), rtol=2e-5, atol=2e-6)
