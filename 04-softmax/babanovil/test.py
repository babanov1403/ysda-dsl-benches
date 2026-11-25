import torch
from common.run import make_runner
from cute_softmax import softmax_wide

torch.manual_seed(1)

N = 512
run = make_runner("cute", softmax_wide(autotune=False))
x = torch.randn((3, N), device="cuda", dtype=torch.float32)
print(x)
out = run(x)
torch.cuda.synchronize()
torch.testing.assert_close(out, x.softmax(dim=-1), rtol=1e-5, atol=1e-6)
print(f"OK: {tuple(x.shape)}")