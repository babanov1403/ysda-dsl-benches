import torch


def softmax_reference(x):
    return torch.softmax(x, dim=-1)


def torch_softmax_wide(x):
    row_max = x.amax(dim=-1, keepdim=True)
    weights = (x - row_max).exp()
    return weights / weights.sum(dim=-1, keepdim=True)
