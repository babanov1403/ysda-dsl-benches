#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define GROUP_SIZE 256
#define EPS 1e-6f

// One block per row: sum of squares over the row, then scale it in a second pass.
__global__ void rmsnorm(const float* x, const float* w, float* out, unsigned int n)
{
    __shared__ float partials[GROUP_SIZE / 32];
    x += blockIdx.x * n;
    out += blockIdx.x * n;

    float sum_sq = 0.0f;
    for (unsigned int col = threadIdx.x; col < n; col += blockDim.x) {
        sum_sq += x[col] * x[col];    
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);    
    }

    if (threadIdx.x % 32 == 0) {
        partials[threadIdx.x / 32] = sum_sq;    
    }
    __syncthreads();

    float total = 0.0f;
    for (int i = 0; i < GROUP_SIZE / 32; ++i)
        total += partials[i];
    const float inv_rms = rsqrtf(total / n + EPS);

    for (unsigned int col = threadIdx.x; col < n; col += blockDim.x)
        out[col] = x[col] * inv_rms * w[col];
}

torch::Tensor cuda_rmsnorm(torch::Tensor x, torch::Tensor w) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto out = torch::empty_like(x);
    const auto rows = x.size(0);
    const auto n = x.size(1);

    rmsnorm<<<rows, GROUP_SIZE, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &cuda_rmsnorm);
}
