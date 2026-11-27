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

extern "C" __global__ void rmsnorm_kernel_kernel(float* __restrict__ out, const float* __restrict__ w, const float* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) rmsnorm_kernel_kernel(float* __restrict__ out, const float* __restrict__ w, const float* __restrict__ x) {
  float sum_sq[4];
  float2 total_pack[1];
  float total[1];
  extern __shared__ __align__(1024) float workspace[];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(sum_sq + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  for (int tile = 0; tile < 8; ++tile) {
    float4 value = *(float4*)(x + (((((int)blockIdx.x) * 4096) + (tile * 512)) + (((int)threadIdx.x) * 4)));
    float4 __1;
      float4 v_ = *(float4*)(sum_sq + 0);
      *(float2*)(&(__1.x)) = tl::fma2(*(float2*)(&(value.x)), *(float2*)(&(value.x)), *(float2*)(&(v_.x)));
      *(float2*)(&(__1.z)) = tl::fma2(*(float2*)(&(value.z)), *(float2*)(&(value.z)), *(float2*)(&(v_.z)));
    *(float4*)(sum_sq + 0) = __1;
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  total_pack[0] = make_float2(broadcast_var_1, broadcast_var_1);
  #pragma unroll
  for (int rv = 0; rv < 2; ++rv) {
    total_pack[0] = tl::add2(total_pack[0], *(float2*)(sum_sq + (rv * 2)));
  }
  total[0] = ((total_pack[0]).x + (total_pack[0]).y);
  total[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(total[0], (&(workspace[0])));
  total[0] = rsqrtf(((total[0] / 0x1p+12f/*4.096000e+03*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  for (int tile_1 = 0; tile_1 < 8; ++tile_1) {
    float4 __2;
      float4 __3;
        float4 v__1 = *(float4*)(x + (((((int)blockIdx.x) * 4096) + (tile_1 * 512)) + (((int)threadIdx.x) * 4)));
        float4 v__2 = make_float4(total[0], total[0], total[0], total[0]);
        *(float2*)(&(__3.x)) = tl::mul2(*(float2*)(&(v__1.x)), *(float2*)(&(v__2.x)));
        *(float2*)(&(__3.z)) = tl::mul2(*(float2*)(&(v__1.z)), *(float2*)(&(v__2.z)));
      float4 v__3 = *(float4*)(w + ((tile_1 * 512) + (((int)threadIdx.x) * 4)));
      *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(__3.x)), *(float2*)(&(v__3.x)));
      *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(__3.z)), *(float2*)(&(v__3.z)));
    *(float4*)(out + (((((int)blockIdx.x) * 4096) + (tile_1 * 512)) + (((int)threadIdx.x) * 4))) = __2;
  }
}

