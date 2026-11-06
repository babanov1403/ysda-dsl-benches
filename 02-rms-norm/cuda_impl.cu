#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define GROUP_SIZE 256

__global__ void rms_norm_kernel(const float* a, const float* residual, const float* weight, float eps, int n, float* out) {
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        out[idx] = a[idx] + residual[idx];
    }

    __shared__ float sum;
    
    if (threadIdx.x == 0) {
        sum = 0.;
    }

    __syncthreads();

    for (int block_start = 0; block_start < n; block_start += blockDim.x) {
        if (block_start + threadIdx.x < n) {
            auto tmp = out[block_start + threadIdx.x];
            tmp *= tmp;

            atomicAdd(&sum, tmp);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        sum /= n;
    }

    __syncthreads();

    if (idx < n) {
        out[idx] = out[idx] / (sqrt(sum + eps)) * weight[idx];
    }
}

torch::Tensor cuda_rms_norm(torch::Tensor a, torch::Tensor residual, torch::Tensor weight, float eps = 1e-6) {
    c10::cuda::CUDAGuard guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto out = torch::empty_like(a);
    const auto n = a.numel();

    dim3 blocksCount((n + GROUP_SIZE - 1) / GROUP_SIZE, 1, 1);
    dim3 threadsCount(GROUP_SIZE, 1, 1);

    rms_norm_kernel<<<blocksCount, threadsCount, 0, stream>>>(a.data_ptr<float>(), residual.data_ptr<float>(), weight.data_ptr<float>(), eps, n, out.data_ptr<float>());
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &cuda_rms_norm);
}