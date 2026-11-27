import unittest

import torch

from tilelang_softmax import softmax, softmax_kernel_, tune_softmax


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SoftmaxTest(unittest.TestCase):
    def test_single_tile_and_default_online(self):
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for cols in (1, 127, 4096, 8192, 8193):
                with self.subTest(dtype=dtype, cols=cols):
                    x = torch.randn(8, cols, device="cuda", dtype=dtype)
                    torch.testing.assert_close(softmax(x), x.softmax(-1))

    def test_online_tail_and_numerics(self):
        x = torch.zeros(7, 513, device="cuda")
        x[0, -1] = 90.0  # The maximum arrives in the last tile.
        x[1, 0] = 90.0
        x[2, :256] = -float("inf")
        x[3] = -float("inf")
        x[4, 0] = float("inf")
        x[5, 0] = float("nan")
        x[6, 256:512] = -float("inf")
        x[6, 256] = float("nan")
        out = torch.empty_like(x)
        softmax_kernel_(*x.shape, 128, 64, "float32")(x, out)
        torch.testing.assert_close(out, x.softmax(-1), rtol=1e-5, atol=1e-6, equal_nan=True)

    def test_tuned(self):
        x = torch.randn(8, 65, device="cuda")
        torch.testing.assert_close(softmax(x, autotune=True), x.softmax(-1))
        self.assertEqual(tune_softmax(x), tune_softmax(x.clone()))

    def test_contiguous_view_with_unaligned_offset(self):
        x = torch.randn(8 * 4096 + 1, device="cuda")[1:].view(8, 4096)
        torch.testing.assert_close(softmax(x), x.softmax(-1))

    def test_empty(self):
        for shape in ((0, 5), (5, 0)):
            x = torch.empty(shape, device="cuda")
            torch.testing.assert_close(softmax(x), x.softmax(-1))


if __name__ == "__main__":
    unittest.main()
