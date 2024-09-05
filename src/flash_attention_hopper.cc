/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cuda.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include "cuda_runtime.h"

#include "flash.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

std::vector<int64_t> GetRowMajorStrides(ffi::Span<const int64_t> dimensions) {
  const size_t ndim = dimensions.size();
  std::vector<int64_t> strides(ndim);

  for (int i = ndim - 1, curr_stride = 1; i >= 0; i--) {
    strides[i] = curr_stride;
    curr_stride *= dimensions[i];
  }

  return strides;
}

/*
  Set the forward propagation params.

  Transcribed from the FlashAttention-3 torch api here:
  https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api.cpp
*/
void set_params_fprop(
    Flash_fwd_params& params,
    ffi::Buffer<ffi::DataType::F16> query,
    ffi::Buffer<ffi::DataType::F16> key,
    ffi::Buffer<ffi::DataType::F16> value,
    ffi::Result<ffi::Buffer<ffi::DataType::F16>> out,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    void* cu_seqlens_q_d,
    void* cu_seqlens_k_d,
    void* seqused_k,
    void* p_d,
    void* softmax_lse_d,
    float p_dropout,
    float softmax_scale,
    int window_size_left,
    int window_size_right,
    bool seqlenq_ngroups_swapped = false,
    bool unpadded_lse = false
) {
  const int64_t batch_size = query.dimensions()[0];
  const int64_t num_heads = query.dimensions()[2];
  const int64_t head_dim = query.dimensions()[3];
  const int64_t num_heads_kv = key.dimensions()[2];

  const int d_rounded = head_dim;

  // XLA forces a row-major layout on all arrays.
  const std::vector<int64_t> query_strides =
      GetRowMajorStrides(query.dimensions());
  const std::vector<int64_t> key_strides = GetRowMajorStrides(key.dimensions());
  const std::vector<int64_t> value_strides =
      GetRowMajorStrides(value.dimensions());
  const std::vector<int64_t> out_strides =
      GetRowMajorStrides(out->dimensions());

  // Reset the parameters
  params = {};

  params.is_bf16 = false;
  params.is_e4m3 = false;

  // Set the pointers and strides.
  params.q_ptr = query.untyped_data();
  params.k_ptr = key.untyped_data();
  params.v_ptr = value.untyped_data();
  params.o_ptr = out->untyped_data();

  // All stride are in elements, not bytes.
  params.q_row_stride = query_strides[1];
  params.k_row_stride = key_strides[1];
  params.v_row_stride = value_strides[1];
  params.q_head_stride = query_strides[2];
  params.k_head_stride = key_strides[2];
  params.v_head_stride = value_strides[2];
  params.o_row_stride = out_strides[1];
  params.o_head_stride = out_strides[2];

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = query_strides[0];
    params.k_batch_stride = key_strides[0];
    params.v_batch_stride = value_strides[0];
    params.o_batch_stride = out_strides[0];

    if (seqlenq_ngroups_swapped) {
      params.q_batch_stride *= seqlen_q;
      params.o_batch_stride *= seqlen_q;
    }
  }

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params.seqused_k = static_cast<int*>(seqused_k);

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_kv;
  params.h_h_k_ratio = num_heads / num_heads_kv;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = head_dim;
  params.d_rounded = d_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;
  __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
  __half2 scale_softmax_log2_half2 =
      __half2(scale_softmax_log2_half, scale_softmax_log2_half);
  params.scale_softmax_log2_half2 =
      reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;

  // Convert p from float to int so we don't have to conlovert the random uint
  // to float to compare. [Minor] We want to round down since when we do the
  // comparison we use <= instead of <
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

  // Causal is the special case where window_size_right == 0 and
  // window_size_left < 0. Local is the more general case where
  // window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;

  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_k;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

  params.is_seqlens_k_cumulative = true;
  params.unpadded_lse = unpadded_lse;
}


ffi::Error FlashAttentionHopperF16FwdImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F16> query,
    ffi::Buffer<ffi::DataType::F16> key,
    ffi::Buffer<ffi::DataType::F16> value,
    ffi::Buffer<ffi::DataType::S32> tile_count_semaphore,
    float softmax_scale,
    bool is_causal,
    ffi::Result<ffi::Buffer<ffi::DataType::F16>> res,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> softmax_lse) {
  if (query.dimensions().size() != 4 || key.dimensions().size() != 4 ||
      value.dimensions().size() != 4) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument, "query/key/value must be 4-dim.");
  }

  if (tile_count_semaphore.dimensions().size() != 1 &&
      tile_count_semaphore.dimensions()[0] != 1) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "tile_count_semaphore must be 1-dim and hold a single int32 value.");
  }

  const int seqlen_q = query.dimensions()[1];
  const int seqlen_k = query.dimensions()[1];

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  Flash_fwd_params params;
  set_params_fprop(
      params,
      query,
      key,
      value,
      res,
      seqlen_q, seqlen_k,
      seqlen_q_rounded, seqlen_k_rounded,
      /*cu_seqlens_q_d=*/nullptr,
      /*cu_seqlens_k_d=*/nullptr,
      /*seqused_k=*/nullptr,
      /*p_d*/nullptr,
      softmax_lse->untyped_data(),
      /*p_dropout=*/0.f,
      softmax_scale,
      /*window_size_left=*/-1,
      /*window_size_right=*/is_causal ? 0 : -1);
  params.tile_count_semaphore = tile_count_semaphore.typed_data();

  const int head_dim = query.dimensions()[3];
  if (head_dim == 64) {
    run_mha_fwd_<cutlass::half_t, 64>(params, stream);
  }
  else if (head_dim == 128) {
    run_mha_fwd_<cutlass::half_t, 128>(params, stream);
  }
  else if (head_dim == 256) {
    run_mha_fwd_<cutlass::half_t, 256>(params, stream);
  } else {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "Only head_dims of 64, 128 or 256 are supported right now.");
  }

  return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FlashAttentionHopperF16Fwd,
    FlashAttentionHopperF16FwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F16>>() // query
        .Arg<ffi::Buffer<ffi::DataType::F16>>() // key
        .Arg<ffi::Buffer<ffi::DataType::F16>>() // value
        .Arg<ffi::Buffer<ffi::DataType::S32>>() // tile_count_semaphore
        .Attr<float>("softmax_scale")
        .Attr<bool>("is_causal")
        .Ret<ffi::Buffer<ffi::DataType::F16>>() // res
        .Ret<ffi::Buffer<ffi::DataType::F32>>() // softmax_lse
);
