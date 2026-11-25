#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda/std/bit>

constexpr int GROUP_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr float EPS = 1e-6f;
constexpr auto REDUCE_STEPS = cuda::std::bit_width(GROUP_SIZE - 1u);

__device__ float block_sum(float partial) {
    __shared__ float shmem[GROUP_SIZE / WARP_SIZE];
    constexpr unsigned MASK = 0xffffffffu;

    const auto tid = threadIdx.x;
    const auto lane = tid % WARP_SIZE;
    const auto warpid = tid / WARP_SIZE;

    auto count = GROUP_SIZE;
    while (count > 1) {
        const auto groups = (count + WARP_SIZE - 1) / WARP_SIZE;

        if (warpid < groups) {
            // #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                partial += __shfl_down_sync(0xffffffffu, partial, offset);
            }
            if (lane == 0) {
                shmem[warpid] = partial;
            }
        }
        __syncthreads();

        count = groups;
        if (count > 1) {
            partial = tid < count ? shmem[tid] : 0.0f;
        }
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

__global__ void rms_norm_kernel_packed(
    const float* x, const float* weight, float* out,
    int64_t n_cols, float eps
) {
    constexpr auto N = 4096;
    assert(n_cols == N);

    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;
    const auto bdim = blockDim.x;

    const auto row = static_cast<int64_t>(blockIdx.x) * N;

    constexpr auto V = N / (GROUP_SIZE * 4);
    const auto tidx = tid * V;

    const auto* x4 = reinterpret_cast<const float4*>(x + row);
    const auto* w4 = reinterpret_cast<const float4*>(weight);
    auto* out4 = reinterpret_cast<float4*>(out + row);

    float4 vectorReg[V];
    float partial = 0.;

    // #pragma unroll
    for (int k = 0; k < V; ++k) {
        const float4 v = x4[tidx + k];
        vectorReg[k] = v;
        partial = fmaf(v.x, v.x, partial);
        partial = fmaf(v.y, v.y, partial);
        partial = fmaf(v.z, v.z, partial);
        partial = fmaf(v.w, v.w, partial);
    }

    auto sumRow = block_sum(partial);
    const float invRms = rsqrtf(sumRow / static_cast<float>(n_cols) + eps);

    for (int k = 0; k < V; ++k) {
        const float4 v = vectorReg[k];
        const float4 w = w4[tidx + k];
        out4[tidx + k] = make_float4(v.x * invRms * w.x, v.y * invRms * w.y,
                                     v.z * invRms * w.z, v.w * invRms * w.w);
    }
}

torch::Tensor cuda_rms_norm(torch::Tensor x, torch::Tensor weight) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto out = torch::empty_like(x);
    const auto rows = x.size(0);
    const auto n = x.size(1);

    if (n == 4096) {
        rms_norm_kernel_packed<<<rows, GROUP_SIZE, 0, stream>>>(x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), n, EPS);
    } else {
        rms_norm_kernel<<<rows, GROUP_SIZE, 0, stream>>>(x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), n, EPS);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &cuda_rms_norm);
}
