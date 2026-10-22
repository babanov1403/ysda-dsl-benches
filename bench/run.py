"""Correctness first, then steady-state CUDA-graph timings on one explicit stream."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch

from bench.doctor import report, gpu_snapshot
from bench.tasks import LANGUAGES, TASKS, check, make_inputs, prepare, reference


def measure(launch, stream, samples, inner):
    # JIT/allocations must have finished before capture. All implementations use
    # this stream (CuTe and C++ bind it in prepare).
    for _ in range(10):
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(inner):
            launch()
    for _ in range(5):
        graph.replay()
    stream.synchronize()
    elapsed = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record(stream)
        graph.replay()
        end.record(stream)
        end.synchronize()
        elapsed.append(start.elapsed_time(end) * 1000 / inner)
    return {
        'median_us': float(np.median(elapsed)),
        'p10_us': float(np.quantile(elapsed, 0.1)),
        'p90_us': float(np.quantile(elapsed, 0.9)),
        'samples_us': elapsed,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, default='00-vector-add')
    p.add_argument('--languages', nargs='+', choices=LANGUAGES, default=list(LANGUAGES))
    p.add_argument('--size', type=int, default=1048579)
    p.add_argument('--rows', type=int, default=256)
    p.add_argument('--cols', type=int, default=1024)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--inner', type=int, default=32)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.samples < 1 or args.inner < 1:
        p.error('samples and inner must be positive')
    environment = report()
    stream = torch.cuda.Stream()
    order = list(dict.fromkeys(args.languages))
    random.Random(args.seed).shuffle(order)
    results = []
    with torch.cuda.stream(stream):
        inputs = make_inputs(args.task, size=args.size, rows=args.rows, cols=args.cols, seed=args.seed)
        expected = reference(args.task, inputs)
        out = torch.empty_like(inputs[0])
        for language in order:
            out.fill_(float('nan'))
            launch = prepare(language, args.task, inputs, out)
            launch()
            stream.synchronize()
            check(args.task, out, expected)
            timing = measure(launch, stream, args.samples, args.inner)
            check(args.task, out, expected)
            result = {'language': language, 'correct': True, **timing}
            results.append(result)
            print(f'{language:8} PASS  {timing["median_us"]:.3f} us  '
                  f'p10={timing["p10_us"]:.3f}  p90={timing["p90_us"]:.3f}')
    payload = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': args.task, 'shape': list(inputs[0].shape), 'dtype': 'float32',
        'seed': args.seed, 'language_order': order,
        'method': {'timer': 'CUDA events around graph replay', 'inner': args.inner,
                   'samples': args.samples, 'cache': 'warm, reused input/output buffers',
                   'scope': 'steady-state kernels; excludes JIT, allocations, transfers, reference',
                   'qualification': 'baseline implementations, shared GPU allowed; not a DSL ranking'},
        'environment': environment, 'results': results,
        'telemetry_after': gpu_snapshot(environment['gpu']['uuid']),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
