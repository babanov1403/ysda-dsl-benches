from bench.cuda_runtime import prepare as _prepare


def prepare(inputs, out):
    return _prepare('00-vector-add', inputs, out)
