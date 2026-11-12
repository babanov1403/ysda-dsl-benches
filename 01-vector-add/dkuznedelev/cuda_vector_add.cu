#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define BLOCK_SIZE 256
#define VEC_SIZE 4

__global__ void add(const float* __restrict__ a,
                    const float* __restrict__ b,
                    float* __restrict__ c,
                    size_t n)
{
    const size_t index = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;

    if (index + VEC_SIZE <= n) {
        const float4 av = *reinterpret_cast<const float4*>(a + index);
        const float4 bv = *reinterpret_cast<const float4*>(b + index);
        *reinterpret_cast<float4*>(c + index) = make_float4(
            av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w
        );
    } else {
        for (size_t i = index; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }
}

torch::Tensor cuda_add(torch::Tensor a, torch::Tensor b) {
    c10::cuda::CUDAGuard guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto out = torch::empty_like(a);
    const auto n = a.numel();

    dim3 blocksCount((n + VEC_SIZE * BLOCK_SIZE - 1) / (VEC_SIZE * BLOCK_SIZE), 1, 1);
    dim3 threadsCount(BLOCK_SIZE, 1, 1);

    add<<<blocksCount, threadsCount, 0, stream>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        out.data_ptr<float>(), 
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &cuda_add);
}