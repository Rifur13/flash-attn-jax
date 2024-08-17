# FlashAttention-3 with JAX

FlashAttention-3 (https://arxiv.org/pdf/2407.08608) is an efficient attention implementation in cuda. It fullys utilize the new hardware capabilities present in the Hopper architecture. The code was released alongside the paper in the [Dao-AILab](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) repo.

We leverage XLA custom calls using the new jax FFI api to call the optimized cuda implementation. Please see the JAX FFI [docs](https://jax.readthedocs.io/en/latest/ffi.html) for an introduction.

## 1. Pre-Setup

Note that the cutlass version 3.5.1 fails to compile in some environments. See https://github.com/NVIDIA/cutlass/issues/1624.

One workaround is to preload the CUDA library **libcuda.so** before any other libraries are loaded. Try ```ldconfig -p | grep libcuda``` to find it.

For example:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so
```

## 2. Setup

```
git clone https://github.com/rifur13/flash-attn-jax.git
cd flash-attn-jax
```

Install the required dependencies:
```
pip install -r requirements.txt
```

Speed up compilation by using half of your available processing units.
```
NUM_WORKERS=$(( ( $(nproc) + 1 ) / 2 ))
```

Build with CMake
```
mkdir -p build && cd build
cmake .. && make -j $NUM_WORKERS
```

Make your new python bindings accessible to python.
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```


### 3. Usage
Simply import the python library
```
import flash_attn_jax_lib
```


### 4. Next steps
- Support more datatypes: bfloat16, and float8 when it's released.
- Support more head_dim sizes.
- Support variables sequence lengths.
- Relax conditions on the sequence length sizes.


