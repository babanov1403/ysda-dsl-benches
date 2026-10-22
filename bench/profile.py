"""Launch one implementation within a CUDA profiler range (no timings)."""
import argparse

import torch

from bench.tasks import LANGUAGES, TASKS, check, make_inputs, prepare, reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--language', choices=LANGUAGES, required=True)
    p.add_argument('--task', choices=TASKS, default='01-softmax')
    p.add_argument('--rows', type=int, default=256)
    p.add_argument('--cols', type=int, default=1024)
    args = p.parse_args()
    with torch.cuda.stream(torch.cuda.Stream()):
        inputs = make_inputs(args.task, rows=args.rows, cols=args.cols)
        out = torch.empty_like(inputs[0])
        expected = reference(args.task, inputs)
        launch = prepare(args.language, args.task, inputs, out)
        launch()
        torch.cuda.synchronize()
        check(args.task, out, expected)
        torch.cuda.profiler.start()
        try:
            launch()
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()


if __name__ == '__main__':
    main()
