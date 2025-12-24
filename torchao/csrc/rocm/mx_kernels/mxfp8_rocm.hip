// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// MXFP8 quantization kernel for AMD ROCm/HIP
// Optimized for AMD MI300X/MI350/MI355 (gfx942/gfx950)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mxfp8_hip {

using e8m0_t = uint8_t;

// Constants
constexpr int32_t FP32_MANTISSA_BITS = 23;
constexpr int32_t FP32_EXPONENT_BIAS = 127;
constexpr int32_t F8E4M3_MAX_POW2 = 8;
constexpr int32_t E8M0_EXPONENT_BIAS = 127;
constexpr float F8E4M3_FNUZ_MAX = 240.0f;  // Max value for E4M3 FNUZ format

// E8M0 scale calculation
__device__ __forceinline__ e8m0_t calculate_e8m0_scale(float amax) {
  // Handle special cases - never return 0xFF (NaN)
  if (isnan(amax) || isinf(amax) || amax == 0.0f) {
    return 0x7F;  // Return scale of 2^0 = 1.0 for edge cases
  }

  union { float f; uint32_t i; } u;
  u.f = amax;
  const int32_t extracted_pow2 = ((u.i >> FP32_MANTISSA_BITS) & 0xFF) - FP32_EXPONENT_BIAS;

  int32_t scale_unbiased = extracted_pow2 - F8E4M3_MAX_POW2;
  scale_unbiased = max(scale_unbiased, -E8M0_EXPONENT_BIAS);
  scale_unbiased = min(scale_unbiased, E8M0_EXPONENT_BIAS + 1);
  return static_cast<e8m0_t>(scale_unbiased + E8M0_EXPONENT_BIAS);
}

__device__ __forceinline__ float exp2f_rcp(e8m0_t biased_exp) {
  return (biased_exp == 0) ? 1.0f : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(biased_exp));
}

// AMD wave-level reduction (wave size 64)
__device__ __forceinline__ float wave_reduce_max(float val) {
  for (int offset = 32; offset > 0; offset /= 2) {
    float other = __shfl_down(val, offset, 64);
    val = fmaxf(val, other);
  }
  return __shfl(val, 0, 64);
}

// FP32 to FP8 E4M3 FNUZ conversion (AMD native format)
// Based on official HIP FP8 implementation
__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val, float scale_rcp) {
  float scaled = val * scale_rcp;

  #if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
    // Saturate to E4M3 FNUZ range (propagate NaN/Inf, no clipping otherwise)
    union { float f; uint32_t i; } u;
    u.f = scaled;
    if ((u.i & 0x7F800000) != 0x7F800000) {  // Not NaN/Inf
      scaled = __builtin_amdgcn_fmed3f(scaled, F8E4M3_FNUZ_MAX, -F8E4M3_FNUZ_MAX);
    }

    // Convert using AMD intrinsic (returns packed FP8 in lower byte)
    u.i = __builtin_amdgcn_cvt_pk_fp8_f32(scaled, scaled, 0, false);
    return u.i & 0xFF;  // Extract lower byte
  #else
    // Fallback for non-gfx940/950 architectures
    scaled = fmaxf(-F8E4M3_FNUZ_MAX, fminf(F8E4M3_FNUZ_MAX, scaled));
    union { float f; uint32_t i; } u;
    u.f = scaled;
    return static_cast<uint8_t>((u.i >> 16) & 0xFF);
  #endif
}

// Main quantization kernel
__global__ void mxfp8_quantize_colwise_f32_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    e8m0_t* __restrict__ scales,
    int64_t rows, int64_t cols,
    int64_t scale_dim_y) {

  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_block_id = blockIdx.y;

  if (col >= cols) return;

  const int64_t start_row = row_block_id * scale_dim_y;
  const int64_t end_row = min(start_row + scale_dim_y, rows);

  // Find max
  float thread_max = 0.0f;
  for (int64_t r = start_row + threadIdx.y; r < end_row; r += blockDim.y) {
    float val = input[r * cols + col];
    thread_max = fmaxf(thread_max, fabsf(val));
  }

  thread_max = wave_reduce_max(thread_max);

  // Calculate scale (lane 0 only)
  e8m0_t scale = 0;
  float scale_rcp = 1.0f;

  if ((threadIdx.x & 63) == 0) {  // First thread in wave
    scale = calculate_e8m0_scale(thread_max);
    scale_rcp = exp2f_rcp(scale);
  }

  // Broadcast scale_rcp from lane 0 to all lanes in wave
  scale_rcp = __shfl(scale_rcp, 0, 64);

  // Store scale
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const int64_t num_row_blocks = (rows + scale_dim_y - 1) / scale_dim_y;
    scales[col * num_row_blocks + row_block_id] = scale;
  }

  // Quantize and store (column-major)
  for (int64_t r = start_row + threadIdx.y; r < end_row; r += blockDim.y) {
    float val = input[r * cols + col];
    uint8_t quantized = float_to_fp8_e4m3(val, scale_rcp);
    output[col * rows + r] = quantized;
  }
}

// BF16 kernel
__global__ void mxfp8_quantize_colwise_bf16_kernel(
    const hip_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    e8m0_t* __restrict__ scales,
    int64_t rows, int64_t cols,
    int64_t scale_dim_y) {

  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_block_id = blockIdx.y;

  if (col >= cols) return;

  const int64_t start_row = row_block_id * scale_dim_y;
  const int64_t end_row = min(start_row + scale_dim_y, rows);

  float thread_max = 0.0f;
  for (int64_t r = start_row + threadIdx.y; r < end_row; r += blockDim.y) {
    hip_bfloat16 bf_val = input[r * cols + col]; float val = static_cast<float>(bf_val);
    thread_max = fmaxf(thread_max, fabsf(val));
  }

  thread_max = wave_reduce_max(thread_max);

  // Calculate scale (lane 0 only)
  e8m0_t scale = 0;
  float scale_rcp = 1.0f;

  if ((threadIdx.x & 63) == 0) {
    scale = calculate_e8m0_scale(thread_max);
    scale_rcp = exp2f_rcp(scale);
  }

  // Broadcast scale_rcp from lane 0 to all lanes in wave
  scale_rcp = __shfl(scale_rcp, 0, 64);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const int64_t num_row_blocks = (rows + scale_dim_y - 1) / scale_dim_y;
    scales[col * num_row_blocks + row_block_id] = scale;
  }

  for (int64_t r = start_row + threadIdx.y; r < end_row; r += blockDim.y) {
    hip_bfloat16 bf_val = input[r * cols + col]; float val = static_cast<float>(bf_val);
    uint8_t quantized = float_to_fp8_e4m3(val, scale_rcp);
    output[col * rows + r] = quantized;
  }
}

// Host launcher
extern "C" void mxfp8_quantize_hip_launcher(
    const void* input,
    void* output,
    void* scales,
    int64_t rows, int64_t cols,
    int64_t scale_dim_y,
    int input_dtype,  // 0=f32, 1=bf16
    hipStream_t stream) {

  dim3 block(32, 8);
  dim3 grid((cols + 31) / 32, (rows + scale_dim_y - 1) / scale_dim_y);

  if (input_dtype == 0) {  // FP32
    hipLaunchKernelGGL(
        mxfp8_quantize_colwise_f32_kernel,
        grid, block, 0, stream,
        (const float*)input,
        (uint8_t*)output,
        (e8m0_t*)scales,
        rows, cols, scale_dim_y
    );
  } else {  // BF16
    hipLaunchKernelGGL(
        mxfp8_quantize_colwise_bf16_kernel,
        grid, block, 0, stream,
        (const hip_bfloat16*)input,
        (uint8_t*)output,
        (e8m0_t*)scales,
        rows, cols, scale_dim_y
    );
  }
}

} // namespace mxfp8_hip
