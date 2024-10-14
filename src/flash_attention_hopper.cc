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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iostream>
#include "stdio.h"

#include "flash.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#define CHECK_SHAPE(x, ...)                                              \
  {                                                                      \
    const std::vector<int64_t> expected_shape = {__VA_ARGS__};           \
    if (!(x.dimensions() == ffi::Span<const int64_t>(expected_shape))) { \
      return ffi::Error(                                                 \
          ffi::ErrorCode::kInvalidArgument,                              \
          #x " must have shape (" #__VA_ARGS__ ")");                     \
    }                                                                    \
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
    bool unpadded_lse = false) {
  // Reset the parameters
  params = {};

  params.is_bf16 = false;
  params.is_e4m3 = false;

  // Set the pointers and strides.
  params.q_ptr = query_ptr;
  params.k_ptr = key_ptr;
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

void set_params_dgrad(
    Flash_bwd_params& params,
    // sizes
    const size_t b,
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
    const std::vector<int64_t> dout_strides,
    // device pointers
    void* q_ptr,
    void* k_ptr,
    void* v_ptr,
    void* out_ptr,
    void* dout_ptr,
    void* dq_ptr,
    void* dk_ptr,
    void* dv_ptr,
    //
    void* cu_seqlens_q_d,
    void* cu_seqlens_k_d,
    void* seqused_q,
    void* seqused_k,
    void* dq_accum_d,
    void* dk_accum_d,
    void* dv_accum_d,
    void* softmax_lse_d,
    void* dsoftmax_sum_d,
    float p_dropout,
    float softmax_scale,
    int window_size_left,
    int window_size_right,
    bool deterministic) {
  set_params_fprop(
      params,
      b,
      seqlen_q,
      seqlen_k,
      seqlen_q_rounded,
      seqlen_k_rounded,
      h,
      h_k,
      d,
      d_rounded,
      query_strides,
      key_strides,
      value_strides,
      out_strides,
      q_ptr,
      k_ptr,
      v_ptr,
      out_ptr,
      cu_seqlens_q_d,
      cu_seqlens_k_d,
      seqused_k,
      /*p_d*/ nullptr,
      softmax_lse_d,
      p_dropout,
      softmax_scale,
      window_size_left,
      window_size_right);

  // Set the pointers and strides.
  params.do_ptr = dout_ptr;
  params.dq_ptr = dq_ptr;
  params.dk_ptr = dk_ptr;
  params.dv_ptr = dv_ptr;

  // q,k,v are forced to be row-major by XLA, so they will have the same strides
  // as dq, dk, dv.
  auto dq_strides = query_strides;
  auto dk_strides = key_strides;
  auto dv_strides = value_strides;

  params.do_row_stride = dout_strides[dout_strides.size() - 3];
  params.dq_row_stride = dq_strides[dq_strides.size() - 3];
  params.dk_row_stride = dk_strides[dk_strides.size() - 3];
  params.dv_row_stride = dv_strides[dv_strides.size() - 3];

  params.do_head_stride = dout_strides[dout_strides.size() - 2];
  params.dq_head_stride = dq_strides[dq_strides.size() - 2];
  params.dk_head_stride = dk_strides[dk_strides.size() - 2];
  params.dv_head_stride = dv_strides[dv_strides.size() - 2];

  if (cu_seqlens_q_d == nullptr) {
    params.do_batch_stride = dout_strides[0];
    params.dq_batch_stride = dq_strides[0];
    params.dk_batch_stride = dk_strides[0];
    params.dv_batch_stride = dv_strides[0];
  }

  params.dq_accum_ptr = dq_accum_d;
  params.dk_accum_ptr = dk_accum_d;
  params.dv_accum_ptr = dv_accum_d;

  // Softmax sum
  params.dsoftmax_sum = dsoftmax_sum_d;

  params.deterministic = deterministic;
}

ffi::Error run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  if (params.d == 64) {
    run_mha_fwd_<cutlass::half_t, 64>(params, stream);
  } else if (params.d == 128) {
    run_mha_fwd_<cutlass::half_t, 128>(params, stream);
  } else if (params.d == 256) {
    run_mha_fwd_<cutlass::half_t, 256>(params, stream);
  } else {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "Only head_dims of 64, 128 or 256 are supported right now.");
  }

  return ffi::Error::Success();
}

ffi::Error run_mha_bwd(Flash_bwd_params& params, cudaStream_t stream) {
  if (params.d == 64) {
    run_mha_bwd_<cutlass::half_t, 64>(params, stream);
  } else if (params.d == 128) {
    run_mha_bwd_<cutlass::half_t, 128>(params, stream);
  } else {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "Only head_dims of 64, 128 are supported right now.");
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
      seqlen_q,
      seqlen_k,
      seqlen_q_rounded,
      seqlen_k_rounded,
      num_heads,
      num_heads_k,
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
      /*p_d*/ nullptr,
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

  if (causal) {
    window_size_right = 0;
  }

  if (window_size_left >= max_seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= max_seqlen_k) {
    window_size_right = -1;
  }

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
  set_params_fprop(
      params,
      batch_size,
      max_seqlen_q,
      max_seqlen_k,
      seqlen_q_rounded,
      seqlen_k_rounded,
      num_heads,
      num_heads_k,
      /*head_size=*/head_size_og,
      /*head_size_rounded=*/head_size_og,
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

ffi::Error FlashAttentionHopperF16BwdImpl(
    cudaStream_t stream,
    ffi::BufferR4<ffi::DataType::F16> dout,
    ffi::BufferR4<ffi::DataType::F16> query,
    ffi::BufferR4<ffi::DataType::F16> key,
    ffi::BufferR4<ffi::DataType::F16> value,
    ffi::BufferR4<ffi::DataType::F16> out,
    ffi::BufferR3<ffi::DataType::F32> softmax_lse,
    ffi::BufferR3<ffi::DataType::F32> softmax_lse_log2,
    ffi::BufferR4<ffi::DataType::F32> dq_accum,
    ffi::BufferR3<ffi::DataType::S32> dq_semaphore,
    const float softmax_scale,
    const bool causal,
    int window_size_left,
    int window_size_right,
    const int deterministic,
    ffi::Result<ffi::BufferR4<ffi::DataType::F16>> dq,
    ffi::Result<ffi::BufferR4<ffi::DataType::F16>> dk,
    ffi::Result<ffi::BufferR4<ffi::DataType::F16>> dv,
    ffi::Result<ffi::BufferR3<ffi::DataType::F32>> softmax_d) {
  const int64_t batch_size = query.dimensions()[0];
  const int64_t seqlen_q = query.dimensions()[1];
  const int64_t num_heads = query.dimensions()[2];
  const int64_t head_dim = query.dimensions()[3];

  const int64_t seqlen_k = key.dimensions()[1];
  const int64_t num_heads_k = key.dimensions()[2];

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded =
      head_dim <= 64 ? 64 : round_multiple(head_dim, 32);

  const int kBlockM = head_dim <= 64 ? 128 : (head_dim < 256 ? 64 : 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, kBlockM);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  CHECK_SHAPE(query, batch_size, seqlen_q, num_heads, head_dim);
  CHECK_SHAPE(key, batch_size, seqlen_k, num_heads_k, head_dim);
  CHECK_SHAPE(value, batch_size, seqlen_k, num_heads_k, head_dim);
  CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_dim);
  CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_dim);

  // CHECK_SHAPE(softmax_d, batch_size, num_heads, seqlen_q_rounded);
  CHECK_SHAPE(softmax_lse_log2, batch_size, num_heads, seqlen_q_rounded);
  CHECK_SHAPE(dq_accum, batch_size, num_heads, seqlen_q_rounded, head_dim);

  // XLA forces a row-major layout on all arrays.
  auto query_strides = GetRowMajorStrides(query.dimensions());
  auto key_strides = GetRowMajorStrides(key.dimensions());
  auto value_strides = GetRowMajorStrides(value.dimensions());
  auto out_strides = GetRowMajorStrides(out.dimensions());
  auto dout_strides = GetRowMajorStrides(dout.dimensions());

  if (causal) {
    window_size_right = 0;
  }

  Flash_bwd_params params;
  set_params_dgrad(
      params,
      batch_size,
      seqlen_q,
      seqlen_k,
      seqlen_q_rounded,
      seqlen_k_rounded,
      num_heads,
      num_heads_k,
      /*head_size=*/head_dim,
      /*head_size_rounded=*/head_dim,
      query_strides,
      key_strides,
      value_strides,
      out_strides,
      dout_strides,
      /*q_ptr=*/query.untyped_data(),
      /*k_ptr=*/key.untyped_data(),
      /*v_ptr=*/value.untyped_data(),
      /*out_ptr=*/out.untyped_data(),
      /*dout_ptr=*/dout.untyped_data(),
      /*dq_ptr=*/dq->untyped_data(),
      /*dk_ptr=*/dk->untyped_data(),
      /*dv_ptr=*/dv->untyped_data(),
      /*cu_seqlens_q_d=*/nullptr,
      /*cu_seqlens_k_d=*/nullptr,
      /*seqused_q=*/nullptr,
      /*seqused_k=*/nullptr,
      /*dq_accum_d=*/dq_accum.untyped_data(),
      /*dk_accum_d=*/nullptr,
      /*dv_accum_d=*/nullptr,
      /*softmax_lse_d=*/softmax_lse.untyped_data(),
      /*dsoftmax_sum_d=*/softmax_d->untyped_data(),
      /*p_dropout=*/0.f,
      softmax_scale,
      window_size_left,
      window_size_right,
      deterministic);

  params.softmax_lse_log2_ptr = softmax_lse_log2.untyped_data();
  params.dq_semaphore = dq_semaphore.typed_data();

  return run_mha_bwd(params, stream);
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FlashAttentionHopperF16Bwd,
    FlashAttentionHopperF16BwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR4<ffi::DataType::F16>>() // dout
        .Arg<ffi::BufferR4<ffi::DataType::F16>>() // query
        .Arg<ffi::BufferR4<ffi::DataType::F16>>() // key
        .Arg<ffi::BufferR4<ffi::DataType::F16>>() // value
        .Arg<ffi::BufferR4<ffi::DataType::F16>>() // out
        .Arg<ffi::BufferR3<ffi::DataType::F32>>() // softmax_lse
        .Arg<ffi::BufferR3<ffi::DataType::F32>>() // softmax_lse_log2
        .Arg<ffi::BufferR4<ffi::DataType::F32>>() // dq_accum
        .Arg<ffi::BufferR3<ffi::DataType::S32>>() // dq_semaphore
        .Attr<float>("softmax_scale")
        .Attr<bool>("causal")
        .Attr<int>("window_size_left")
        .Attr<int>("window_size_right")
        .Attr<bool>("deterministic")
        .Ret<ffi::BufferR4<ffi::DataType::F16>>() // dq
        .Ret<ffi::BufferR4<ffi::DataType::F16>>() // dk
        .Ret<ffi::BufferR4<ffi::DataType::F16>>() // dv
        .Ret<ffi::BufferR3<ffi::DataType::F32>>() // softmax_d
);