#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

constexpr int kThreads = 256;
constexpr float kEps = 1e-6f;

__global__ void rms_norm_kernel(
    const float* x, const float* residual, const float* weight, float* out,
    int64_t n_cols, float eps
) {
    // TODO: compute one output row per block. weight is shared across rows.
}

torch::Tensor cuda_rms_norm(torch::Tensor x, torch::Tensor residual, torch::Tensor weight) {
    // TODO
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &cuda_rms_norm);
}
