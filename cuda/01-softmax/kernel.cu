#include <cuda_runtime.h>
#include <math_constants.h>

// A deliberately simple baseline: one full warp per row, four rows per CTA.
// Re-reads input instead of keeping a large per-thread register tile.
__global__ void softmax(const float* x, float* out, int rows, int cols) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows) return;  // Entire warp takes the same branch.
    float maximum = -CUDART_INF_F;
    for (int col = lane; col < cols; col += 32)
        maximum = fmaxf(maximum, x[row * cols + col]);
    for (int offset = 16; offset > 0; offset /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    float total = 0.f;
    for (int col = lane; col < cols; col += 32)
        total += expf(x[row * cols + col] - maximum);
    for (int offset = 16; offset > 0; offset /= 2)
        total += __shfl_xor_sync(0xffffffff, total, offset);
    for (int col = lane; col < cols; col += 32)
        out[row * cols + col] = expf(x[row * cols + col] - maximum) / total;
}

extern "C" int ysda_softmax(const float* x, float* out, int rows, int cols,
                            cudaStream_t stream) {
    softmax<<<(rows + 3) / 4, 128, 0, stream>>>(x, out, rows, cols);
    return static_cast<int>(cudaGetLastError());
}
