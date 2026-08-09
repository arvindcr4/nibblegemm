// W4A16 kernels for the decode regime (small M).
//
// At batch sizes 1..16 this operation is not a GEMM in any useful sense -- there
// is almost no data reuse on the weights, so the kernel is pure streaming and
// its ceiling is HBM bandwidth. The entire reason 4-bit weights are worth the
// trouble is that they cut the bytes moved by ~4x versus fp16, and the job of
// the kernel is to convert that byte reduction into wall-clock without letting
// decode arithmetic or launch geometry get in the way.
//
// The file is deliberately a ladder: v0 through v4 are all correct, and each one
// changes exactly one thing relative to its predecessor so the benchmark can
// attribute the speedup. See docs/OPTIMIZATION_LOG.md.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "dequant.cuh"

namespace nibble {

// ---------------------------------------------------------------------------
// v0: the honest naive baseline.
//
// One thread per output element, one scalar dequantisation per weight, and a
// reload of the packed word for every one of the 8 nibbles it contains. Nothing
// here is a strawman -- it is the code you write first, and it is what the rest
// of the ladder is measured against.
// ---------------------------------------------------------------------------
__global__ void gemv_v0_naive(const half* __restrict__ X, const uint32_t* __restrict__ Wq,
                              const half* __restrict__ S, half* __restrict__ Y, int M, int K,
                              int N, int group_size) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  if (n >= N || m >= M) return;

  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    const uint32_t q = Wq[(k / 8) * N + n];  // reloaded 8x, once per nibble
    const float s = __half2float(S[(k / group_size) * N + n]);
    acc += __half2float(X[m * K + k]) * dequant_scalar(q, k % 8, s);
  }
  Y[m * N + n] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// v1: amortise the packed load.
//
// Same thread mapping, but each 32-bit load is now used for all 8 nibbles it
// carries instead of being re-fetched per k. Dequantisation is still the slow
// shift/mask/convert path, so this isolates the memory effect from the decode
// effect.
// ---------------------------------------------------------------------------
__global__ void gemv_v1_coalesced(const half* __restrict__ X, const uint32_t* __restrict__ Wq,
                                  const half* __restrict__ S, half* __restrict__ Y, int M, int K,
                                  int N, int group_size) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  if (n >= N || m >= M) return;

  float acc = 0.f;
  for (int k8 = 0; k8 < K / 8; ++k8) {
    const uint32_t q = Wq[k8 * N + n];
    const int k0 = k8 * 8;
    const float s = __half2float(S[(k0 / group_size) * N + n]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc += __half2float(X[m * K + k0 + j]) * dequant_scalar(q, j, s);
    }
  }
  Y[m * N + n] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// v2: fast lop3 decode + half2 math + activations staged in shared memory.
//
// Replaces 8 shift/mask/I2F/mul chains with 4 lop3 + 2 hsub2 + 2 hfma2, and
// switches the inner product to packed half2 FMAs. Activations are broadcast
// from shared memory so the weight stream owns L1.
// ---------------------------------------------------------------------------
template <int GROUP, int THREADS>
__global__ void gemv_v2_fastdq(const half* __restrict__ X, const uint32_t* __restrict__ Wq,
                               const half* __restrict__ S, half* __restrict__ Y, int M, int K,
                               int N) {
  const int n = blockIdx.x * THREADS + threadIdx.x;
  const int m = blockIdx.y;
  __shared__ half xs[GROUP];

  float acc = 0.f;
  for (int g = 0; g < K / GROUP; ++g) {
    for (int i = threadIdx.x; i < GROUP; i += THREADS) xs[i] = X[m * K + g * GROUP + i];
    __syncthreads();

    if (n < N) {
      const half sc = S[g * N + n];
      half2 psum = __float2half2_rn(0.f);
#pragma unroll
      for (int j = 0; j < GROUP / 8; ++j) {
        const uint32_t q = Wq[(g * GROUP / 8 + j) * N + n];
        half2 w[4];
        dequant_8(q, w);
        const half2* xv = reinterpret_cast<const half2*>(&xs[j * 8]);
#pragma unroll
        for (int t = 0; t < 4; ++t) psum = __hfma2(w[t], xv[t], psum);
      }
      acc += (__low2float(psum) + __high2float(psum)) * __half2float(sc);
    }
    __syncthreads();
  }
  if (n < N) Y[m * N + n] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// v3 / v4: vectorised loads, register blocking over N, and split-K.
//
// Two changes that matter more than anything else in this file:
//
// * **128-bit loads.** Each thread takes 4 adjacent output columns and pulls
//   them with one `uint4`, so a warp moves 512 contiguous bytes per instruction
//   and covers 32 weights per thread per load. Vectorising along N rather than
//   K is what keeps the access pattern contiguous.
//
// * **Split-K.** This is the difference between a kernel that works and one
//   that leaves the GPU idle. A 4096x4096 layer with 512 columns per block is
//   8 blocks; the A100 has 108 SMs, so without splitting the reduction axis
//   over 90% of the machine never gets work. Partials go to a `[splits, M, N]`
//   fp32 workspace and are summed by a second pass -- deterministic, unlike
//   atomics, and the workspace is a few hundred KB.
//
// `partial == nullptr` selects the single-split path and writes fp16 straight
// out, skipping the reduction launch.
//
// Accumulation is half2 within a short window then flushed to fp32. The window
// (`FLUSH_WORDS` x 8 values) trades a little accuracy for packed-FMA
// throughput; the group scale folds in at flush time, which removes a whole
// accumulator array from the register budget. tests/test_numerics.py measures
// what the window actually costs.
// ---------------------------------------------------------------------------
template <int MT, int GROUP, int THREADS, int FLUSH_WORDS>
__global__ void gemv_v4_splitk(const half* __restrict__ X, const uint32_t* __restrict__ Wq,
                               const half* __restrict__ S, half* __restrict__ Y,
                               float* __restrict__ partial, int M, int K, int N,
                               int groups_per_split) {
  constexpr int TN = 4;               // output columns per thread (one uint4)
  constexpr int WORDS = GROUP / 8;    // packed words per group
  const int n0 = (blockIdx.x * THREADS + threadIdx.x) * TN;
  const int split = blockIdx.y;

  const int total_groups = K / GROUP;
  const int g_begin = split * groups_per_split;
  const int g_end = min(g_begin + groups_per_split, total_groups);

  __shared__ half xs[MT][GROUP];

  float acc[MT][TN];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int c = 0; c < TN; ++c) acc[m][c] = 0.f;

  for (int g = g_begin; g < g_end; ++g) {
    for (int idx = threadIdx.x; idx < MT * GROUP; idx += THREADS) {
      const int m = idx / GROUP, kk = idx % GROUP;
      xs[m][kk] = (m < M) ? X[m * K + g * GROUP + kk] : __float2half(0.f);
    }
    __syncthreads();

    if (n0 < N) {
      half sc[TN];
#pragma unroll
      for (int c = 0; c < TN; ++c) sc[c] = S[g * N + n0 + c];

#pragma unroll
      for (int jw = 0; jw < WORDS / FLUSH_WORDS; ++jw) {
        half2 psum[MT][TN];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
          for (int c = 0; c < TN; ++c) psum[m][c] = __float2half2_rn(0.f);

#pragma unroll
        for (int jj = 0; jj < FLUSH_WORDS; ++jj) {
          const int j = jw * FLUSH_WORDS + jj;
          // One 128-bit load: 4 output columns x 8 reduction steps.
          const uint4 pk =
              *reinterpret_cast<const uint4*>(&Wq[(g * WORDS + j) * N + n0]);
          const uint32_t qs[TN] = {pk.x, pk.y, pk.z, pk.w};

          half2 xv[MT][4];
#pragma unroll
          for (int m = 0; m < MT; ++m) {
            const uint4 raw = *reinterpret_cast<const uint4*>(&xs[m][j * 8]);
            *reinterpret_cast<uint4*>(&xv[m][0]) = raw;
          }

#pragma unroll
          for (int c = 0; c < TN; ++c) {
            half2 w[4];
            dequant_8(qs[c], w);
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
              for (int t = 0; t < 4; ++t) psum[m][c] = __hfma2(w[t], xv[m][t], psum[m][c]);
          }
        }

#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
          for (int c = 0; c < TN; ++c)
            acc[m][c] +=
                (__low2float(psum[m][c]) + __high2float(psum[m][c])) * __half2float(sc[c]);
      }
    }
    __syncthreads();
  }

  if (n0 >= N) return;
  if (partial != nullptr) {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m >= M) break;
#pragma unroll
      for (int c = 0; c < TN; ++c)
        partial[(static_cast<size_t>(split) * M + m) * N + n0 + c] = acc[m][c];
    }
  } else {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m >= M) break;
#pragma unroll
      for (int c = 0; c < TN; ++c) Y[m * N + n0 + c] = __float2half(acc[m][c]);
    }
  }
}

__global__ void reduce_splits(const float* __restrict__ partial, half* __restrict__ Y, int splits,
                              int MN) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= MN) return;
  float s = 0.f;
  for (int t = 0; t < splits; ++t) s += partial[static_cast<size_t>(t) * MN + i];
  Y[i] = __float2half(s);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
namespace {

void check_inputs(const torch::Tensor& X, const torch::Tensor& Wq, const torch::Tensor& S,
                  int64_t group_size) {
  TORCH_CHECK(X.is_cuda() && Wq.is_cuda() && S.is_cuda(), "all tensors must live on CUDA");
  TORCH_CHECK(X.scalar_type() == torch::kHalf, "activations must be fp16");
  TORCH_CHECK(S.scalar_type() == torch::kHalf, "scales must be fp16");
  TORCH_CHECK(Wq.scalar_type() == torch::kInt32, "packed weights must be int32");
  TORCH_CHECK(X.is_contiguous() && Wq.is_contiguous() && S.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(X.dim() == 2 && Wq.dim() == 2 && S.dim() == 2, "expected 2-D inputs");
  const int64_t K = X.size(1), N = Wq.size(1);
  TORCH_CHECK(Wq.size(0) * 8 == K, "packed weight rows disagree with K");
  TORCH_CHECK(S.size(0) * group_size == K, "scale rows disagree with K/group_size");
  TORCH_CHECK(S.size(1) == N, "scale columns disagree with N");
  TORCH_CHECK(N % 4 == 0, "N must be a multiple of 4 for 128-bit weight loads");
  TORCH_CHECK(group_size == 64 || group_size == 128, "group_size must be 64 or 128");
}

// Choose how many ways to split the reduction axis.
//
// The subtlety is the rounding. Converting a split count into groups-per-split
// with ceil() and back again silently loses parallelism: 32 groups split 27
// ways gives 2 groups each, which is only 16 splits -- half the grid that was
// asked for. Flooring instead keeps the requested width and lets the last
// split be short.
int pick_splits(int64_t N, int64_t K, int64_t group_size, int cols_per_block, int waves) {
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const int col_blocks = static_cast<int>((N + cols_per_block - 1) / cols_per_block);
  const int target = props->multiProcessorCount * waves;
  const int max_useful = static_cast<int>(K / group_size);
  int splits = (target + col_blocks - 1) / col_blocks;
  splits = std::max(1, std::min(splits, max_useful));
  return splits;
}

// Choose the block width.
//
// This started as a heuristic that narrowed the block when the grid looked too
// small to fill the machine, on the theory that more blocks would mean more
// concurrent loads in flight. The sweep in bench/autotune.py disproved it: on a
// 4096x4096 layer, 32, 64 and 128 threads per block all land within 0.1% of one
// another.
//
// The reason is that block width cannot change the total amount of work in
// flight. Each thread owns TN=4 output columns, so the total thread count is
// (N / 4) * splits no matter how those threads are grouped -- narrowing the
// block just makes more, smaller blocks out of the same warps. The knobs that
// genuinely add memory-level parallelism are the split count (capped at
// K/group_size) and TN.
//
// The override is kept so the sweep can keep re-proving this on other hardware,
// but the default is simply the widest block, which minimises grid launch and
// scale-reload overhead.
int pick_block_threads(int64_t, int64_t, int64_t) { return 128; }

template <int MT, int GROUP, int THREADS>
void launch_v4(const half* x, const uint32_t* w, const half* s, half* y, float* partial, int M,
               int K, int N, int splits, int groups_per_split, cudaStream_t stream) {
  constexpr int cols_per_block = THREADS * 4;
  dim3 grid((N + cols_per_block - 1) / cols_per_block, splits);
  gemv_v4_splitk<MT, GROUP, THREADS, 4>
      <<<grid, THREADS, 0, stream>>>(x, w, s, y, partial, M, K, N, groups_per_split);
}

template <int GROUP, int THREADS>
void dispatch_batch(int M, const half* x, const uint32_t* w, const half* s, half* y,
                    float* partial, int K, int N, int splits, int groups_per_split,
                    cudaStream_t stream) {
#define NIBBLE_CALL(MT) \
  launch_v4<MT, GROUP, THREADS>(x, w, s, y, partial, M, K, N, splits, groups_per_split, stream)
  if (M <= 1) NIBBLE_CALL(1);
  else if (M <= 2) NIBBLE_CALL(2);
  else if (M <= 4) NIBBLE_CALL(4);
  else if (M <= 8) NIBBLE_CALL(8);
  else TORCH_CHECK(false, "gemv path supports M <= 8; use gemm_w4a16 for larger M");
#undef NIBBLE_CALL
}

template <int GROUP>
void dispatch_width(int threads, int M, const half* x, const uint32_t* w, const half* s, half* y,
                    float* partial, int K, int N, int splits, int groups_per_split,
                    cudaStream_t stream) {
  if (threads <= 32)
    dispatch_batch<GROUP, 32>(M, x, w, s, y, partial, K, N, splits, groups_per_split, stream);
  else if (threads <= 64)
    dispatch_batch<GROUP, 64>(M, x, w, s, y, partial, K, N, splits, groups_per_split, stream);
  else
    dispatch_batch<GROUP, 128>(M, x, w, s, y, partial, K, N, splits, groups_per_split, stream);
}

}  // namespace

torch::Tensor gemv_w4a16(torch::Tensor X, torch::Tensor Wq, torch::Tensor S, int64_t group_size,
                         int64_t version, int64_t splits_override, int64_t threads_override) {
  check_inputs(X, Wq, S, group_size);
  const at::cuda::CUDAGuard guard(X.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int M = static_cast<int>(X.size(0));
  const int K = static_cast<int>(X.size(1));
  const int N = static_cast<int>(Wq.size(1));
  auto Y = torch::empty({M, N}, X.options());

  const auto* x = reinterpret_cast<const half*>(X.data_ptr<at::Half>());
  const auto* w = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const auto* s = reinterpret_cast<const half*>(S.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(Y.data_ptr<at::Half>());

  if (version == 0 || version == 1) {
    dim3 block(256), grid((N + 255) / 256, M);
    if (version == 0)
      gemv_v0_naive<<<grid, block, 0, stream>>>(x, w, s, y, M, K, N, (int)group_size);
    else
      gemv_v1_coalesced<<<grid, block, 0, stream>>>(x, w, s, y, M, K, N, (int)group_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
  }

  if (version == 2) {
    dim3 grid((N + 128 - 1) / 128, M);
    if (group_size == 128)
      gemv_v2_fastdq<128, 128><<<grid, 128, 0, stream>>>(x, w, s, y, M, K, N);
    else
      gemv_v2_fastdq<64, 128><<<grid, 128, 0, stream>>>(x, w, s, y, M, K, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
  }

  // v3 is v4 pinned to a single split, which is exactly the "before" case for
  // the split-K change and costs no extra code. It also stays at the widest
  // block, so the v3 -> v4 delta isolates split-K instead of mixing in the
  // block-width heuristic.
  int threads;
  if (threads_override > 0)
    threads = static_cast<int>(threads_override);
  else if (version == 4)
    threads = pick_block_threads(N, K, group_size);
  else
    threads = 128;
  TORCH_CHECK(threads == 32 || threads == 64 || threads == 128,
              "block_threads must be 32, 64 or 128");
  const int cols_per_block = threads * 4;

  int splits = 1;
  if (version == 4) {
    splits = splits_override > 0 ? static_cast<int>(splits_override)
                                 : pick_splits(N, K, group_size, cols_per_block, 4);
  }

  const int total_groups = K / static_cast<int>(group_size);
  // Floor, not ceil: see pick_splits.
  const int groups_per_split = std::max(1, total_groups / splits);
  splits = (total_groups + groups_per_split - 1) / groups_per_split;  // trim empty splits

  torch::Tensor partial;
  float* partial_ptr = nullptr;
  if (splits > 1) {
    partial = torch::empty({splits, M, N}, X.options().dtype(torch::kFloat32));
    partial_ptr = partial.data_ptr<float>();
  }

  if (group_size == 128)
    dispatch_width<128>(threads, M, x, w, s, y, partial_ptr, K, N, splits, groups_per_split,
                        stream);
  else
    dispatch_width<64>(threads, M, x, w, s, y, partial_ptr, K, N, splits, groups_per_split,
                       stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (splits > 1) {
    const int MN = M * N;
    const int threads = 256;
    reduce_splits<<<(MN + threads - 1) / threads, threads, 0, stream>>>(partial_ptr, y, splits, MN);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return Y;
}

}  // namespace nibble
