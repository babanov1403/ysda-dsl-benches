import pytest
import torch

from bench.tasks import LANGUAGES, check, make_inputs, prepare, reference

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU required')]


def run_checked(language, task, inputs):
    # A non-default stream catches accidental implicit stream-0 launches.
    stream = torch.cuda.current_stream()
    expected = reference(task, inputs)
    out = torch.full_like(inputs[0], float('nan'))
    originals = [x.clone() for x in inputs]
    launch = prepare(language, task, inputs, out)
    launch()
    stream.synchronize()
    check(task, out, expected)
    for x, old in zip(inputs, originals):
        torch.testing.assert_close(x, old, rtol=0, atol=0)
    # Every backend must also support CUDA graph capture on the same stream.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    out.fill_(float('nan'))
    graph.replay()
    stream.synchronize()
    check(task, out, expected)


@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('size', [1, 31, 32, 255, 256, 257, 4099])
def test_vector_add(language, size):
    with torch.cuda.stream(torch.cuda.Stream()):
        inputs = make_inputs('00-vector-add', size=size, seed=17)
        run_checked(language, '00-vector-add', inputs)


@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('shape,pattern', [
    ((1, 1), 'random'), ((3, 31), 'random'), ((7, 33), 'random'),
    ((5, 129), 'random'), ((8, 1024), 'random'), ((2, 4097), 'random'),
    ((1, 16384), 'random'), ((5, 257), 'constant'), ((5, 257), 'large'),
])
def test_softmax(language, shape, pattern):
    with torch.cuda.stream(torch.cuda.Stream()):
        inputs = make_inputs('01-softmax', rows=shape[0], cols=shape[1], seed=21)
        if pattern == 'constant':
            inputs[0].fill_(-1234.0)
        if pattern == 'large':
            inputs[0].mul_(200).add_(10000)
        run_checked(language, '01-softmax', inputs)


@pytest.mark.parametrize('language', LANGUAGES)
def test_reject_noncontiguous(language):
    x = torch.ones((4, 8), device='cuda')[:, ::2]
    with pytest.raises(ValueError, match='contiguous'):
        prepare(language, '01-softmax', (x,), torch.empty_like(x))


def test_reject_alias():
    x = torch.ones((3, 32), device='cuda')
    with pytest.raises(ValueError, match='alias'):
        prepare('cuda', '01-softmax', (x,), x)


def test_reject_partial_overlap():
    storage = torch.ones(65, device='cuda')
    x, out = storage[:64], storage[1:]
    with pytest.raises(ValueError, match='alias'):
        prepare('cuda', '00-vector-add', (x, x), out)
