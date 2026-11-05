#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

#define GROUP_SIZE 256

__global__ void add(const float* __restrict__ a,
                    const float* __restrict__ b,
                    float* __restrict__ c,
                    size_t n,
                    bool aligned)
{
    const size_t index = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;

    if (index >= n)
        return;

    if (aligned && n - index >= 4) {
        const float4 av = *reinterpret_cast<const float4*>(a + index);
        const float4 bv = *reinterpret_cast<const float4*>(b + index);
        *reinterpret_cast<float4*>(c + index) = make_float4(
            av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w);
    } else {
        for (size_t i = index; i < n && i - index < 4; ++i)
            c[i] = a[i] + b[i];
    }
}

torch::Tensor cuda_add(torch::Tensor a, torch::Tensor b) {
    c10::cuda::CUDAGuard guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto out = torch::empty_like(a);
    const auto n = a.numel();

    if (n == 0)
        return out;

    const auto* a_ptr = a.data_ptr<float>();
    const auto* b_ptr = b.data_ptr<float>();
    auto* out_ptr = out.data_ptr<float>();
    const bool aligned = (reinterpret_cast<std::uintptr_t>(a_ptr) % alignof(float4) == 0) &&
                         (reinterpret_cast<std::uintptr_t>(b_ptr) % alignof(float4) == 0) &&
                         (reinterpret_cast<std::uintptr_t>(out_ptr) % alignof(float4) == 0);
    dim3 blocksCount((n + 4 * GROUP_SIZE - 1) / (4 * GROUP_SIZE), 1, 1);
    dim3 threadsCount(GROUP_SIZE, 1, 1);

    add<<<blocksCount, threadsCount, 0, stream>>>(a_ptr, b_ptr, out_ptr, n, aligned);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &cuda_add);
}