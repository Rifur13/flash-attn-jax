cmake_minimum_required(VERSION 3.19)
project(flash-attn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90)

find_package(CUDA REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

add_library(flash-attn SHARED "src/flash_attention_hopper.cc")

####################
# Fetch dependencies
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax.extend import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

include(FetchContent)
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG         v3.5.1
)

FetchContent_Declare(
  dao-ai-lab
  GIT_REPOSITORY https://github.com/Dao-AILab/flash-attention.git
  GIT_TAG        v2.6.3
)

FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  # Copies the repo locally without building it.
  FetchContent_Populate(cutlass)
endif()

FetchContent_MakeAvailable(dao-ai-lab)
####################

# Find nanobind through pip
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)

list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

target_sources(
  flash-attn
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/flash_attention_hopper.cc
  ${dao-ai-lab_SOURCE_DIR}/hopper/flash_fwd_hdim64_fp16_sm90.cu
  ${dao-ai-lab_SOURCE_DIR}/hopper/flash_fwd_hdim128_fp16_sm90.cu
  ${dao-ai-lab_SOURCE_DIR}/hopper/flash_fwd_hdim256_fp16_sm90.cu
  ${dao-ai-lab_SOURCE_DIR}/hopper/flash_bwd_hdim128_fp16_sm90.cu
)

target_compile_features(flash-attn PUBLIC cxx_std_17)
target_compile_options(flash-attn PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       -gencode arch=compute_90a,code=sm_90a
                       --threads=4
                       -O3
                       -std=c++17
                       -U__CUDA_NO_HALF_OPERATORS__
                       -U__CUDA_NO_HALF_CONVERSIONS__
                       -U__CUDA_NO_BFLOAT16_OPERATORS__
                       -U__CUDA_NO_BFLOAT16_CONVERSIONS__
                       -U__CUDA_NO_BFLOAT162_OPERATORS__
                       -U__CUDA_NO_BFLOAT162_CONVERSIONS__
                       --expt-relaxed-constexpr
                       --expt-extended-lambda
                       --use_fast_math
                       --ptxas-options=-v  # printing out number of registers
                       --ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage  # printing out number of registers
                       -lineinfo
                       -DCUTLASS_DEBUG_TRACE_LEVEL=0  # Can toggle for debugging
                       -DNDEBUG  # Important, otherwise performance is severely impacted
                       >)

target_include_directories(
  flash-attn PUBLIC
  ${nanobind_INCLUDE_DIRS}
  ${Python_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${CUDA_INCLUDE_DIRS}
  ${XLA_DIR}
  ${cutlass_SOURCE_DIR}/include
  ${dao-ai-lab_SOURCE_DIR}/hopper)

# Create python bindings
nanobind_add_module(
  flash_attn_jax_lib
  NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN flash-attn
  ${CMAKE_CURRENT_SOURCE_DIR}/src/flash_attention_hopper_api.cc
)

target_link_libraries(flash_attn_jax_lib PUBLIC flash-attn)
target_include_directories(flash_attn_jax_lib PUBLIC 
    ${XLA_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

install(TARGETS flash_attn_jax_lib LIBRARY DESTINATION ${CMAKE_CURRENT_LIST_DIR})