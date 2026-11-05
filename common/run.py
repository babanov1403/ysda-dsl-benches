import gc
import tempfile
from functools import partial
from pathlib import Path

import torch
import triton
from triton import runtime
from triton.testing import _summarize_statistics

from common.asm import cubin_asm, so_asm

_compiled = {}  # hash -> CompiledKernel, in compile order
_triton_compile = triton.compile


def _do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    `triton.testing.do_bench` implementation BUT with warmup and repeat arguments passed as counts insterad of milliseconds

    Benchmark the runtime of the provided function. By default, returns the mean runtime of :code:`fn` as a single float.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in COUNTS). Controls how long the function is run before timing begins.
    :type warmup: int
    :param rep: Repetition time (in COUNTS). Controls the total duration of the timed runs.
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensors to None before each run,
        to avoid gradient accumulation affecting timing.
    :type grad_to_none: list[torch.Tensor], optional
    :param quantiles: If provided, return these quantiles of the runtime distribution instead
        of a summary statistic. For example, ``[0.2, 0.5, 0.8]`` returns the 20th, 50th, and
        80th percentile runtimes in ms. When set, ``return_mode`` is ignored.
    :type quantiles: list[float], optional
    :param return_mode: The summary statistic to return when ``quantiles`` is not set.
        ``"mean"`` and ``"median"`` return a single float. ``"min"`` and ``"max"`` return
        the fastest and slowest run respectively. ``"all"`` returns the raw list of
        per-run timings in ms. Default is ``"mean"``.
    :type return_mode: str
    :return: A single float by default, a list of floats if ``quantiles`` is provided,
        or a list of all per-run timings if ``return_mode="all"``.
    :rtype: float | list[float]
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # compute number of warmup and repeat
    n_warmup = warmup
    n_repeat = rep
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)




def _recording_compile(*args, **kwargs):
    # torch.compile's static launcher drops PTX/cubin, so keep them at compile time.
    kernel = _triton_compile(*args, **kwargs)
    _compiled[kernel.hash] = kernel
    return kernel


def _torch(kernel):
    return kernel, dict


def _torch_compile(kernel):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    torch._inductor.config.force_disable_caches = True
    # Worker processes compile kernels out of reach of the triton.compile hook below.
    torch._inductor.config.compile_threads = 1
    triton.compile = _recording_compile
    tuners = lambda: [o for o in gc.get_objects() if type(o) is CachingAutotuner]
    known = {id(t) for t in tuners()}

    def asm():
        winners = {t.compile_results[0].kernel.hash for t in tuners() if id(t) not in known}
        # Autotuning picks a new kernel per shape: the latest compile wins each name.
        return {k.name: cubin_asm(k.asm) for h, k in _compiled.items() if h in winners}

    return torch.compile(kernel), asm


def autotunable(configs, key=None):
    """Declares autotuning on a kernel: triton.autotune for triton.Config, Quack's autotune for AutotuneConfig.

    decorated(autotune=False) skips the autotuner entirely, so the kernel runs with its own defaults.
    """

    def decorate(jit_fn):
        def select(autotune=True):
            if isinstance(configs[0], triton.Config):
                return triton.autotune(configs, key)(jit_fn) if autotune else jit_fn
            return partial(_cute_runner, jit_fn, configs if autotune else None, key)

        return select

    return decorate


def _triton(kernel, grid, extra_args=lambda *tensors: (), **constants):
    last = None

    def call(*inputs):
        nonlocal last
        out = torch.empty_like(inputs[0])
        last = kernel[grid](*inputs, out, *extra_args(*inputs, out), **constants)
        return out

    return call, lambda: {last.name: cubin_asm(last.asm)}


def _cute_bench(fn, quantiles):
    # Quack's default CUDA-graph benchmark captures nothing from a CuTe launch and reports 0 ms.
    return _do_bench(fn, warmup=5, rep=50, quantiles=list(quantiles))


def _cute_runner(kernel, configs, key):
    import cutlass.cute as cute
    from cutlass.base_dsl.compiler import DumpDir
    from cutlass.cute.runtime import from_dlpack
    from quack.autotuner import autotune

    cache = {}
    last = last_config = held = held_wrapped = None

    def call(*inputs, **config):
        nonlocal last, last_config, held, held_wrapped
        # from_dlpack costs ~5 us per tensor, so reuse the wrappers while the inputs stay the same.
        if held is None or any(x is not y for x, y in zip(inputs, held)):
            held, held_wrapped = inputs, [from_dlpack(t, assumed_align=16) for t in inputs]
        tensors = (*inputs, torch.empty_like(inputs[0]))
        wrapped = [*held_wrapped, from_dlpack(tensors[-1], assumed_align=16)]
        spec = (tuple((t.device, t.dtype, t.shape, t.stride()) for t in tensors), *config.items())
        if spec not in cache:
            # CuTe reads the dumped PTX and cubin into memory, so the files don't need to outlive compile.
            # Line info adds .loc directives for asm() without changing the generated instructions.
            with tempfile.TemporaryDirectory() as dump_dir:
                options = cute.KeepPTX, cute.KeepCUBIN, cute.GenerateLineInfo(True), DumpDir(dump_dir)
                cache[spec] = cute.compile[options](kernel, *wrapped, **config)
        last, last_config = cache[spec], config
        last(*wrapped)
        return tensors[-1]

    def asm():
        listing = cubin_asm({"ptx": last.__ptx__, "cubin": last.__cubin__})
        return {kernel.__name__: {**listing, "config": last_config}}

    if not configs:
        return call, asm
    return autotune(configs, key, do_bench=_cute_bench, cache_results=False)(call), asm


def _cute(selected):
    return selected()


def _tilelang(kernel, specialize, name="tilelang"):
    # specialize maps runtime inputs to the JITKernel that will be launched.
    last = None

    def call(*inputs):
        nonlocal last
        out = torch.empty_like(inputs[0])
        compiled = specialize(kernel, *inputs)
        compiled(*inputs, out)
        last = compiled
        return out

    def asm():
        if last is None:
            raise RuntimeError("Call the TileLang runner before exporting assembly")
        from tilelang.contrib import nvcc

        # TileLang's export helpers recompile generated CUDA with the kernel's
        # target and flags; this is not necessarily the binary used at launch.
        with last.target:
            options = nvcc.default_compile_options(last.compile_flags)
            source = last.get_kernel_source()
            ptx = nvcc.compile_cuda(source, target_format="ptx", options=options).decode("utf-8")
            cubin = nvcc.compile_cuda(source, target_format="cubin", options=options)
        return {name: cubin_asm({"ptx": ptx, "cubin": cubin})}

    return call, asm


def _cuda(source, entry):
    from torch.utils.cpp_extension import load

    source = Path(source).resolve()
    module = None

    def call(*inputs):
        nonlocal module
        if module is None:
            module = load(name=source.stem, sources=[str(source)], extra_cuda_cflags=["-O3", "-lineinfo"])
        return getattr(module, entry)(*inputs)

    return call, lambda: {source.stem: so_asm(module.__file__)}


BACKENDS = {"torch": _torch, "torch_compile": _torch_compile, "triton": _triton, "cute": _cute, "cuda": _cuda,
            "tilelang": _tilelang}


def make_runner(backend, kernel, **options):
    """Returns fn(*inputs) -> output, with fn.asm() -> {name: {"ptx", "sass"}} after a call."""
    call, asm = BACKENDS[backend](kernel, **options)

    def runner(*inputs):
        return call(*inputs)

    runner.asm = asm
    return runner


@torch.no_grad()
def run_benchmark(impls, inputs, reference, flops, baseline="torch_compile", warmup=25, rep=100):
    expected = reference(*inputs)
    useful_bytes = sum(t.numel() * t.element_size() for t in (*inputs, expected))
    results = {}
    for name, fn in impls.items():
        torch.testing.assert_close(fn(*inputs), expected, rtol=1e-5, atol=1e-6, msg=name)
        p20, median, p80 = _do_bench(lambda: fn(*inputs), warmup=warmup, rep=rep, quantiles=[0.2, 0.5, 0.8])
        seconds = median * 1e-3
        results[name] = {
            "median_us": median * 1e3,
            "p20_us": p20 * 1e3,
            "p80_us": p80 * 1e3,
            "TFLOP/s": flops / seconds / 1e12,
            "effective_GB/s": useful_bytes / seconds / 1e9,
        }
    for name, r in results.items():
        speedup = results[baseline]["median_us"] / r["median_us"]
        print(f"{name:>14}: {r['median_us']:.3f} us (p20-p80: {r['p20_us']:.3f}-{r['p80_us']:.3f}) | "
              f"{r['TFLOP/s']:.6f} TFLOP/s | {r['effective_GB/s']:.2f} GB/s | {speedup:.2f}x")
    return results


