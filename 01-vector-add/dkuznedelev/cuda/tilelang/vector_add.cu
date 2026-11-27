#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void add_kernel_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
extern "C" __global__ void __launch_bounds__(128, 1) add_kernel_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
  float4 __1;
    float4 v_ = *(float4*)(a + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4)));
    float4 v__1 = *(float4*)(b + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4)));
    *(float2*)(&(__1.x)) = tl::add2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
    *(float2*)(&(__1.z)) = tl::add2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
  *(float4*)(c + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4))) = __1;
}

