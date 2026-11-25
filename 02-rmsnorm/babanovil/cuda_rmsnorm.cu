#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda/std/bit>

constexpr int GROUP_SIZE = 256;
constexpr float EPS = 1e-6f;
constexpr auto REDUCE_STEPS = cuda::std::bit_width(GROUP_SIZE - 1u);

__device__ float block_sum(float partial) {
    __shared__ float shmem[GROUP_SIZE];

    const auto tid = threadIdx.x;
    shmem[tid] = partial;

    __syncthreads();

    for (int k = 0; k < REDUCE_STEPS; k++) {
        auto stride = 1 << (REDUCE_STEPS - k - 1);
        if (tid < stride && tid + stride < GROUP_SIZE) {
            shmem[tid] = shmem[tid] + shmem[tid + stride];
        }

        __syncthreads();
    }

    return shmem[0];
}

__global__ void rms_norm_kernel(
    const float* x, const float* weight, float* out,
    int64_t n_cols, float eps
) {
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;
    const auto bdim = blockDim.x;

    float partial = 0.;
    for (auto j = tid; j < n_cols; j += bdim) {
        auto value = x[bid * n_cols + j];
        partial += value * value;
    }

    auto sumRow = block_sum(partial);
    const float invRms = rsqrtf(sumRow / static_cast<float>(n_cols) + eps);
    const int64_t rowStart = static_cast<int64_t>(blockIdx.x) * n_cols;

    for (int64_t j = threadIdx.x; j < n_cols; j += blockDim.x) {
        out[rowStart + j] = x[rowStart + j] * invRms * weight[j];
    }
}

torch::Tensor cuda_rms_norm(torch::Tensor x, torch::Tensor weight) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto out = torch::empty_like(x);
    const auto rows = x.size(0);
    const auto n = x.size(1);

    rms_norm_kernel<<<rows, GROUP_SIZE, 0, stream>>>(x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), n, EPS);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &cuda_rms_norm);
}
