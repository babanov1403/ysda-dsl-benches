from bench.cuda_runtime import prepare as _prepare


def prepare(inputs, out):
    return _prepare('01-softmax', inputs, out)
