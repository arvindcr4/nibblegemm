// W4A16 kernel for the prefill regime (large M).
//
// Once M is large the weights are reused across rows, HBM stops being the
// limit, and the binding constraint becomes arithmetic intensity: how many FLOP
// the kernel extracts per byte it pulls into shared memory.
//
// That number is set almost entirely by the block tile, and it is worth doing
// the arithmetic before writing any code. For a BM x BN tile stepping BK:
//
//     FLOP  per tile = 2 * BM * BN * BK
//     bytes per tile = BM * BK * 2   (fp16 activations)
//                    + BN * BK / 2   (INT4 weights)
//
// A 64x64 tile therefore runs at 51 FLOP/byte. The A100 needs roughly
// 312 TFLOP/s / 1.4 TB/s ~= 220 FLOP/byte to saturate its tensor cores, so a
// 64x64 tile is memory bound *by construction* and caps out near
// 51 * 1.4 TB/s ~= 71 TFLOP/s no matter how well the inner loop is written.
// The first version of this kernel measured 73.8 TFLOP/s, which is that ceiling
// rather than a defect -- and no amount of inner-loop tuning would have moved
// it.
//
// Doubling the tile to 128x128 doubles intensity to 102 FLOP/byte and moves the
// ceiling to ~143 TFLOP/s. BK drops to 32 to keep both double-buffered tiles
// inside the 48 KB shared-memory budget, which costs nothing: BK affects the
// pipeline depth, not the intensity, since it cancels out of the ratio above.
//
// Structure: 128x128 block tile, 256 threads (8 warps) in a 4x2 arrangement,
// each warp owning a 32x64 quadrant as 2x4 wmma 16x16x16 fragments, with
// double-buffered shared tiles. Activations stream in via `cp.async`; weights
// are read packed, decoded in register with the lop3 path, and written to
// shared already scaled -- only 4 bits per weight ever cross HBM.
//
// Shared-memory layout note: the B tile is stored transposed, `[BN][BK]`, so a
// thread that decodes one packed word writes its 8 consecutive k-values as a
// single 16-byte store instead of 8 strided 2-byte stores, and wmma reads it
// back as a `col_major` fragment at no cost.
//
// This uses the wmma API rather than raw `mma.sync` + `ldmatrix`. wmma is
// correct by construction but re-loads fragments through a fixed layout and
// gives no control over shared-memory swizzling; see docs/OPTIMIZATION_LOG.md
// for the measured gap that remains and what closing it would take.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

#include "common.cuh"
#include "dequant.cuh"

namespace nibble {

namespace {
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int PAD = 8;         // keeps rows 16-byte aligned for vector stores
constexpr int LDA = BK + PAD;  // As is [BM][LDA], row-major
constexpr int LDB = BK + PAD;  // Bs is [BN][LDB], i.e. B transposed
constexpr int A_ELEMS = BM * LDA;
constexpr int B_ELEMS = BN * LDB;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int WARP_M = 32, WARP_N = 64;  // 4x2 warp grid over the 128x128 tile
constexpr int FRAG_M = WARP_M / 16;      // 2
constexpr int FRAG_N = WARP_N / 16;      // 4
}  // namespace

template <bool USE_CP_ASYNC>
__global__ void gemm_v6_wmma(const half* __restrict__ X, const uint32_t* __restrict__ Wq,
                             const half* __restrict__ S, half* __restrict__ Y, int M, int K, int N,
                             int group_size) {
  extern __shared__ char smem_raw[];
  half* As = reinterpret_cast<half*>(smem_raw);
  half* Bs = As + 2 * A_ELEMS;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int warp_m = (warp / 2) * WARP_M;
  const int warp_n = (warp % 2) * WARP_N;

  using namespace nvcuda::wmma;
  fragment<accumulator, 16, 16, 16, float> c_frag[FRAG_M][FRAG_N];
#pragma unroll
  for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
    for (int j = 0; j < FRAG_N; ++j) fill_fragment(c_frag[i][j], 0.0f);

  // Stage the activation tile. Rows past M are zero-filled: with cp.async a
  // zero transfer size makes the hardware pad, so the edge case costs no branch
  // in the steady state.
  auto load_A = [&](int buf, int k0) {
    half* dst = As + buf * A_ELEMS;
#pragma unroll
    for (int c = tid; c < BM * (BK / 8); c += THREADS) {
      const int row = c / (BK / 8);
      const int col = (c % (BK / 8)) * 8;
      const bool ok = (m0 + row) < M;
      const int safe_row = ok ? (m0 + row) : (M - 1);
      const half* src = X + static_cast<size_t>(safe_row) * K + k0 + col;
      half* d = dst + row * LDA + col;
      if (USE_CP_ASYNC) {
        cp_async_16_guarded(d, src, ok);
      } else {
        if (ok)
          *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(src);
        else
          *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
      }
    }
  };

  // Stage the weight tile: one packed word per thread-iteration, decoded and
  // scaled in register, then written transposed as one 16-byte store.
  auto load_B = [&](int buf, int k0) {
    half* dst = Bs + buf * B_ELEMS;
#pragma unroll
    for (int e = tid; e < (BK / 8) * BN; e += THREADS) {
      const int r8 = e / BN;
      const int c = e % BN;
      const int n = n0 + c;
      uint4 payload = make_uint4(0, 0, 0, 0);
      if (n < N) {
        const uint32_t q = Wq[static_cast<size_t>(k0 / 8 + r8) * N + n];
        const half sc = S[static_cast<size_t>((k0 + r8 * 8) / group_size) * N + n];
        half2 w[4];
        dequant_8_scaled(q, sc, w);
        payload = *reinterpret_cast<uint4*>(w);
      }
      *reinterpret_cast<uint4*>(dst + c * LDB + r8 * 8) = payload;
    }
  };

  const int ntiles = K / BK;

  load_A(0, 0);
  load_B(0, 0);
  if (USE_CP_ASYNC) cp_async_commit();

  for (int t = 0; t < ntiles; ++t) {
    const int buf = t & 1;

    if (t + 1 < ntiles) {
      load_A(buf ^ 1, (t + 1) * BK);
      load_B(buf ^ 1, (t + 1) * BK);
      if (USE_CP_ASYNC) cp_async_commit();
    }
    if (USE_CP_ASYNC) {
      // Leave the prefetch in flight; only demand the tile about to be used.
      if (t + 1 < ntiles)
        cp_async_wait_group<1>();
      else
        cp_async_wait_group<0>();
    }
    __syncthreads();

    const half* Ab = As + buf * A_ELEMS;
    const half* Bb = Bs + buf * B_ELEMS;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      fragment<matrix_a, 16, 16, 16, half, row_major> a_frag[FRAG_M];
      fragment<matrix_b, 16, 16, 16, half, col_major> b_frag[FRAG_N];
#pragma unroll
      for (int i = 0; i < FRAG_M; ++i)
        load_matrix_sync(a_frag[i], Ab + (warp_m + i * 16) * LDA + kk, LDA);
#pragma unroll
      for (int j = 0; j < FRAG_N; ++j)
        load_matrix_sync(b_frag[j], Bb + (warp_n + j * 16) * LDB + kk, LDB);
#pragma unroll
      for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j) mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
    }
    __syncthreads();  // release this buffer before the next prefetch targets it
  }

  // Epilogue. A full fp32 [BM][BN] staging tile would be 64 KB, more shared
  // memory than the whole kernel has, so fragments are drained one 16x16 tile
  // at a time through a per-warp 1 KB scratch carved out of the (now dead) A
  // buffer. That keeps the fragment layout opaque -- wmma does not document it
  // -- while never needing more than 8 KB.
  __syncthreads();
  float* stage = reinterpret_cast<float*>(smem_raw) + warp * 256;

#pragma unroll
  for (int i = 0; i < FRAG_M; ++i) {
#pragma unroll
    for (int j = 0; j < FRAG_N; ++j) {
      store_matrix_sync(stage, c_frag[i][j], 16, mem_row_major);
      __syncwarp();
      const int gm = m0 + warp_m + i * 16;
      const int gn = n0 + warp_n + j * 16;
      for (int e = lane; e < 256; e += 32) {
        const int r = e >> 4, c = e & 15;
        if (gm + r < M && gn + c < N)
          Y[static_cast<size_t>(gm + r) * N + gn + c] = __float2half(stage[e]);
      }
      __syncwarp();
    }
  }
}

torch::Tensor gemm_w4a16(torch::Tensor X, torch::Tensor Wq, torch::Tensor S, int64_t group_size) {
  TORCH_CHECK(X.is_cuda() && Wq.is_cuda() && S.is_cuda(), "all tensors must live on CUDA");
  TORCH_CHECK(X.scalar_type() == torch::kHalf && S.scalar_type() == torch::kHalf,
              "activations and scales must be fp16");
  TORCH_CHECK(Wq.scalar_type() == torch::kInt32, "packed weights must be int32");
  TORCH_CHECK(X.is_contiguous() && Wq.is_contiguous() && S.is_contiguous(),
              "inputs must be contiguous");

  const int M = static_cast<int>(X.size(0));
  const int K = static_cast<int>(X.size(1));
  const int N = static_cast<int>(Wq.size(1));
  TORCH_CHECK(Wq.size(0) * 8 == K, "packed weight rows disagree with K");
  TORCH_CHECK(K % BK == 0, "K must be a multiple of ", BK);
  TORCH_CHECK(group_size == 64 || group_size == 128, "group_size must be 64 or 128");

  const at::cuda::CUDAGuard guard(X.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto Y = torch::empty({M, N}, X.options());

  const auto* x = reinterpret_cast<const half*>(X.data_ptr<at::Half>());
  const auto* w = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const auto* s = reinterpret_cast<const half*>(S.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(Y.data_ptr<at::Half>());

  const size_t smem = static_cast<size_t>(2 * A_ELEMS + 2 * B_ELEMS) * sizeof(half);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_v6_wmma<true><<<grid, THREADS, smem, stream>>>(x, w, s, y, M, K, N, (int)group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Y;
}

}  // namespace nibble
