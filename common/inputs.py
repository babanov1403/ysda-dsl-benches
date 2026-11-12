import torch


def make_inputs(*shapes, dtype=torch.float32, seed=42, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn(shape, dtype=dtype, device=device, generator=generator)
        for shape in shapes
    )
