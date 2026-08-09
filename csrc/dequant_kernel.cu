// Standalone dequantisation, exposed purely so the bit-trick decode can be
// tested in isolation.
//
// The lop3 path in dequant.cuh is the one place in this project where a wrong
// magic constant would produce plausible-looking output and a silently degraded
// model. Rather than trusting an end-to-end matmul tolerance to catch that, both
// the fast path and the naive shift/mask path are exposed here so tests can
// sweep every nibble value in every slot and demand bit-exact agreement.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "dequant.cuh"

namespace nibble {

__global__ void dequant_fast_kernel(const uint32_t* __restrict__ Wq, const half* __restrict__ S,
                                    half* __restrict__ out, int K, int N, int group_size) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int k8 = blockIdx.y;
  if (n >= N) return;

  const uint32_t q = Wq[k8 * N + n];
  const half sc = S[((k8 * 8) / group_size) * N + n];

  half2 w[4];
  dequant_8_scaled(q, sc, w);

  const half* vals = reinterpret_cast<const half*>(w);
#pragma unroll
  for (int j = 0; j < 8; ++j) out[(k8 * 8 + j) * N + n] = vals[j];
}

__global__ void dequant_scalar_kernel(const uint32_t* __restrict__ Wq, const half* __restrict__ S,
                                      half* __restrict__ out, int K, int N, int group_size) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int k8 = blockIdx.y;
  if (n >= N) return;

  const uint32_t q = Wq[k8 * N + n];
  const float sc = __half2float(S[((k8 * 8) / group_size) * N + n]);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    out[(k8 * 8 + j) * N + n] = __float2half(dequant_scalar(q, j, sc));
}

torch::Tensor dequant_w4(torch::Tensor Wq, torch::Tensor S, int64_t group_size, bool fast) {
  TORCH_CHECK(Wq.is_cuda() && S.is_cuda(), "tensors must live on CUDA");
  TORCH_CHECK(Wq.scalar_type() == torch::kInt32, "packed weights must be int32");
  TORCH_CHECK(S.scalar_type() == torch::kHalf, "scales must be fp16");
  TORCH_CHECK(Wq.is_contiguous() && S.is_contiguous(), "inputs must be contiguous");

  const at::cuda::CUDAGuard guard(Wq.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int rows = static_cast<int>(Wq.size(0));
  const int N = static_cast<int>(Wq.size(1));
  const int K = rows * 8;
  auto out = torch::empty({K, N}, S.options());

  const auto* w = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const auto* s = reinterpret_cast<const half*>(S.data_ptr<at::Half>());
  auto* o = reinterpret_cast<half*>(out.data_ptr<at::Half>());

  const int threads = 128;
  dim3 grid((N + threads - 1) / threads, rows);
  if (fast)
    dequant_fast_kernel<<<grid, threads, 0, stream>>>(w, s, o, K, N, (int)group_size);
  else
    dequant_scalar_kernel<<<grid, threads, 0, stream>>>(w, s, o, K, N, (int)group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace nibble
