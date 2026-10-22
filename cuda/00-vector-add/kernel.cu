#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" int ysda_vector_add(const float* a, const float* b, float* out,
                               int n, cudaStream_t stream) {
    vector_add<<<(n + 255) / 256, 256, 0, stream>>>(a, b, out, n);
    return static_cast<int>(cudaGetLastError());
}

extern "C" const char* ysda_cuda_error(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
