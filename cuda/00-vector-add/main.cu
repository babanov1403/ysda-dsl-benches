#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" int ysda_vector_add(const float*, const float*, float*, int, cudaStream_t);
#define CHECK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)

int main() {
    constexpr int n = 4099;  // Intentionally not divisible by block size.
    std::vector<float> a(n, 1.25f), b(n, 2.5f), out(n);
    float *da, *db, *dc;
    CHECK(cudaMalloc(&da, n * sizeof(float)));
    CHECK(cudaMalloc(&db, n * sizeof(float)));
    CHECK(cudaMalloc(&dc, n * sizeof(float)));
    CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(static_cast<cudaError_t>(ysda_vector_add(da, db, dc, n, nullptr)));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (float v : out) if (v != 3.75f) return 2;
    CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc));
    std::puts("CUDA C++ runtime API: PASS (4099 elements)");
}
