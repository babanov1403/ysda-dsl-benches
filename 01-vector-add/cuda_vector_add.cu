#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define GROUP_SIZE 256

__global__ void add(const float* a,
                    const float* b,
                          float* c,
                   unsigned int  n)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
        return;

    c[index] = a[index] + b[index];
}

torch::Tensor cuda_add(torch::Tensor a, torch::Tensor b) {
    c10::cuda::CUDAGuard guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto out = torch::empty_like(a);
    const auto n = a.numel();

    dim3 blocksCount((n + GROUP_SIZE - 1) / GROUP_SIZE, 1, 1);
    dim3 threadsCount(GROUP_SIZE, 1, 1);

    add<<<blocksCount, threadsCount, 0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &cuda_add);
}