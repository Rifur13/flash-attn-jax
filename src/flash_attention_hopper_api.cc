#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"

#include "flash_attention_hopper.h"

namespace nb = nanobind;

template <typename T>
nb::capsule EncapsulateFfiCall(T* fn) {
  // This check is optional, but it can be helpful for avoiding invalid
  // handlers.
  static_assert(
      std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
      "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void*>(fn));
}

NB_MODULE(flash_attn_jax_lib, m) {
  m.def("flash_attention_hopper_f16_fwd", []() {
    return EncapsulateFfiCall(FlashAttentionHopperF16Fwd);
  });

  m.def("flash_attention_hopper_f16_varlen_fwd", []() {
    return EncapsulateFfiCall(FlashAttentionHopperF16VarlenFwd);
  });

  m.def("flash_attention_hopper_f16_bwd", []() {
    return EncapsulateFfiCall(FlashAttentionHopperF16Bwd);
  });

  m.def("flash_attention_hopper_f16_varlen_bwd", []() {
    return EncapsulateFfiCall(FlashAttentionHopperF16VarlenBwd);
  });
}