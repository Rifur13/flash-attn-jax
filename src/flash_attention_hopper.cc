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
#include <iostream>

#include "stdio.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iostream>


#include "flash.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#define CHECK_SHAPE(x, ...) { \
    const std::vector<int64_t> expected_shape = {__VA_ARGS__}; \
    if (!(x.dimensions() == ffi::Span<const int64_t>(expected_shape))) { \
        return ffi::Error( \
            ffi::ErrorCode::kInvalidArgument, \
            #x " must have shape (" #__VA_ARGS__ ")"); \
    } \
}

// TODO(pobudzey): replace with std::partial_sum
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
    const size_t batch_size,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t d,
    const size_t d_rounded,
    const std::vector<int64_t> query_strides,
    const std::vector<int64_t> key_strides,
    const std::vector<int64_t> value_strides,
    const std::vector<int64_t> out_strides,
    void* query_ptr,
    void* key_ptr,
    void* value_ptr,
    void* out_ptr,
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
  // Reset the parameters
  params = {};

  params.is_bf16 = false;
  params.is_e4m3 = false;

  // Set the pointers and strides.
  params.q_ptr = query_ptr,
  params.k_ptr = key_ptr,
  params.v_ptr = value_ptr;
  params.o_ptr = out_ptr;

  // All stride are in elements, not bytes.
  params.q_row_stride = query_strides[query_strides.size() - 3];
  params.k_row_stride = key_strides[key_strides.size() - 3];
  params.v_row_stride = value_strides[value_strides.size() - 3];
  params.o_row_stride = out_strides[out_strides.size() - 3];

  params.q_head_stride = query_strides[query_strides.size() - 2];
  params.k_head_stride = key_strides[key_strides.size() - 2];
  params.v_head_stride = value_strides[value_strides.size() - 2];
  params.o_head_stride = out_strides[out_strides.size() - 2];

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
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
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


ffi::Error run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
  if (params.d == 64) {
    run_mha_fwd_<cutlass::half_t, 64>(params, stream);
  }
  else if (params.d == 128) {
    run_mha_fwd_<cutlass::half_t, 128>(params, stream);
  }
  else if (params.d == 256) {
    run_mha_fwd_<cutlass::half_t, 256>(params, stream);
  } else {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "Only head_dims of 64, 128 or 256 are supported right now.");
  }

  return ffi::Error::Success();
}

ffi::Error FlashAttentionHopperF16FwdImpl(
    cudaStream_t stream,
    ffi::BufferR4<ffi::DataType::F16> query,
    ffi::BufferR4<ffi::DataType::F16> key,
    ffi::BufferR4<ffi::DataType::F16> value,
    ffi::BufferR1<ffi::DataType::S32> tile_count_semaphore,
    float softmax_scale,
    bool is_causal,
    ffi::Result<ffi::BufferR4<ffi::DataType::F16>> res,
    ffi::Result<ffi::BufferR3<ffi::DataType::F32>> softmax_lse) {

  CHECK_SHAPE(tile_count_semaphore, 1);

  const int64_t batch_size = query.dimensions()[0];
  const int64_t seqlen_q = query.dimensions()[1];
  const int64_t num_heads = query.dimensions()[2];
  const int64_t head_dim = query.dimensions()[3];

  const int64_t seqlen_k = key.dimensions()[1];
  const int64_t num_heads_k = key.dimensions()[2];

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  // XLA forces a row-major layout on all arrays.
  auto query_strides = GetRowMajorStrides(query.dimensions());
  auto key_strides = GetRowMajorStrides(key.dimensions());
  auto value_strides = GetRowMajorStrides(value.dimensions());
  auto out_strides = GetRowMajorStrides(res->dimensions());

  Flash_fwd_params params;
  set_params_fprop(
      params,
      batch_size,
      seqlen_q, seqlen_k,
      seqlen_q_rounded, seqlen_k_rounded,
      num_heads, num_heads_k,
      head_dim,
      /*head_size_rounded=*/head_dim,
      query_strides,
      key_strides,
      value_strides,
      out_strides,
      /*query_ptr=*/query.untyped_data(),
      /*key_ptr=*/key.untyped_data(),
      /*value_ptr=*/value.untyped_data(),
      /*out_ptr=*/res->untyped_data(),
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

  return run_mha_fwd(params, stream);
}

ffi::Error FlashAttentionHopperF16VarlenFwdImpl(
    cudaStream_t stream,
    ffi::BufferR3<ffi::DataType::F16> query,
    ffi::BufferR3<ffi::DataType::F16> key,
    ffi::BufferR3<ffi::DataType::F16> value,
    ffi::BufferR1<ffi::DataType::S32> cu_seqlens_q,
    ffi::BufferR1<ffi::DataType::S32> cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    int window_size_left,
    int window_size_right,        
    const float softmax_scale,
    bool causal,
    ffi::Result<ffi::BufferR3<ffi::DataType::F16>> res,
    ffi::Result<ffi::BufferR2<ffi::DataType::F32>> softmax_lse) {

  const int batch_size = cu_seqlens_q.element_count() - 1;

  const int total_q = query.dimensions()[0];  
  const int num_heads = query.dimensions()[1];
  const int head_size_og = query.dimensions()[2];

  const int total_k = key.dimensions()[0];
  const int num_heads_k = key.dimensions()[1];

  if (causal) { window_size_right = 0; }

  if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
  if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

  CHECK_SHAPE(query, total_q, num_heads, head_size_og);
  CHECK_SHAPE(key, total_k, num_heads_k, head_size_og);
  CHECK_SHAPE(value, total_k, num_heads_k, head_size_og);

  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

  // XLA forces a row-major layout on all arrays.
  auto query_strides = GetRowMajorStrides(query.dimensions());
  auto key_strides = GetRowMajorStrides(key.dimensions());
  auto value_strides = GetRowMajorStrides(value.dimensions());
  auto out_strides = GetRowMajorStrides(res->dimensions());

  Flash_fwd_params params;
  set_params_fprop(params,
                    batch_size,
                    max_seqlen_q, max_seqlen_k,
                    seqlen_q_rounded, seqlen_k_rounded,
                    num_heads, num_heads_k,
                    /*head_size=*/ head_size_og, 
                    /*head_size_rounded=*/ head_size_og,
                    query_strides,
                    key_strides,
                    value_strides,
                    out_strides,
                    /*query_ptr=*/query.untyped_data(),
                    /*key_ptr=*/key.untyped_data(),
                    /*value_ptr=*/value.untyped_data(),
                    /*out_ptr=*/res->untyped_data(),
                    /* cu_seqlens_q */ cu_seqlens_q.untyped_data(),
                    /* cu_seqlens_k */ cu_seqlens_k.untyped_data(),
                    /*seqused_k=*/nullptr,
                    /*p_d=*/nullptr,
                    softmax_lse->untyped_data(),
                    /*p_dropout=*/0.f,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
                    /*seqlenq_ngroups_swapped=*/false,
                    /*unpadded_lse=*/true);

  params.total_q = total_q;
  params.total_k = total_k;

  return run_mha_fwd(params, stream);
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FlashAttentionHopperF16Fwd,
    FlashAttentionHopperF16FwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR4<ffi::DataType::F16>>() // query
        .Arg<ffi::BufferR4<ffi::DataType::F16>>() // key
        .Arg<ffi::BufferR4<ffi::DataType::F16>>() // value
        .Arg<ffi::BufferR1<ffi::DataType::S32>>() // tile_count_semaphore
        .Attr<float>("softmax_scale")
        .Attr<bool>("is_causal")
        .Ret<ffi::BufferR4<ffi::DataType::F16>>() // res
        .Ret<ffi::BufferR3<ffi::DataType::F32>>() // softmax_lse
);


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FlashAttentionHopperF16VarlenFwd,
    FlashAttentionHopperF16VarlenFwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR3<ffi::DataType::F16>>() // query
        .Arg<ffi::BufferR3<ffi::DataType::F16>>() // key
        .Arg<ffi::BufferR3<ffi::DataType::F16>>() // value
        .Arg<ffi::BufferR1<ffi::DataType::S32>>() // cu_seqlens_q
        .Arg<ffi::BufferR1<ffi::DataType::S32>>() // cu_seqlens_k
        .Attr<int>("max_seqlen_q")
        .Attr<int>("max_seqlen_k")
        .Attr<int>("window_size_left")
        .Attr<int>("window_size_right")
        .Attr<float>("softmax_scale")
        .Attr<bool>("is_causal")
        .Ret<ffi::BufferR3<ffi::DataType::F16>>() // res
        .Ret<ffi::BufferR2<ffi::DataType::F32>>() // softmax_lse
);
