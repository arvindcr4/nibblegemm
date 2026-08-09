// Small sm_80 helpers shared by the kernels.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace nibble {

// Asynchronous global -> shared copy (Ampere `cp.async`).
//
// The pre-Ampere way to fill a shared tile is global -> register -> shared,
// which burns registers and forces the thread to stall on the load before it
// can issue the store. `cp.async` hands the copy to the memory pipeline and
// lets the thread keep issuing math, so the next tile lands while the current
// one is being consumed. That overlap is the whole point of double buffering.
//
// `.cg` bypasses L1 and caches at L2 only, which is what we want for streaming
// tiles that are read once and never revisited.
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(gmem_src));
}

// Same, but supplying a source size below 16 bytes makes the hardware zero-fill
// the remainder. Used to pad out-of-range rows without a branch.
__device__ __forceinline__ void cp_async_16_guarded(void* smem_dst, const void* gmem_src,
                                                    bool pred) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  const int bytes = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(addr), "l"(gmem_src),
               "r"(bytes));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::);
}

}  // namespace nibble
