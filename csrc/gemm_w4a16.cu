// W4A16 kernels for the prefill regime (large M).
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
// BK cancels, so the tile shape alone sets the ceiling. A 64x64 tile runs at
// 51 FLOP/byte; the A100 needs roughly 312 TFLOP/s / 1.4 TB/s ~= 220 FLOP/byte
// to saturate its tensor cores, so a 64x64 tile is memory bound *by
// construction* and caps near 71 TFLOP/s. The first version of this kernel
// measured 73.8 -- its roofline, not a defect. The 128x128 tile used here
// doubles intensity to 102 FLOP/byte.
//
// Both kernels below share that tile and differ only in how the inner product
// is issued:
//
//   v6  nvcuda::wmma        -- correct by construction, but `load_matrix_sync`
//                              recomputes fragment addressing on every call
//   v7  mma.sync + ldmatrix -- fragment addresses computed once, outside the
//                              k-loop, and the epilogue writes straight to
//                              global with no shared-memory staging
//
// Keeping both makes the cost of the wmma abstraction measurable rather than
// asserted; see docs/OPTIMIZATION_LOG.md.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

#include "common.cuh"
#include "dequant.cuh"
#include "mma.cuh"

namespace nibble {

namespace {
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int PAD = 8;         // see csrc/mma.cuh: makes ldmatrix conflict-free
constexpr int LDA = BK + PAD;  // As is [BM][LDA], row-major, k contiguous
constexpr int LDB = BK + PAD;  // Bs is [BN][LDB], i.e. B transposed, k contiguous
constexpr int A_ELEMS = BM * LDA;
constexpr int B_ELEMS = BN * LDB;
constexpr int THREADS = 256;
constexpr int WARP_M = 32, WARP_N = 64;  // 4x2 warp grid over the 128x128 tile
}  // namespace

// ---------------------------------------------------------------------------
// Tile staging, shared by both kernels
// ---------------------------------------------------------------------------

// Rows past M are zero-filled: with cp.async a zero transfer size makes the
// hardware pad, so the edge case costs no branch in the steady state.
//
// Index arithmetic is done in 32-bit and widened only at the final offset.
// Widening first makes every address a 64-bit IMAD, and IMAD issues on the FMA
// pipe -- which the ncu breakdown showed was carrying more of this kernel than
// the tensor cores were. `gemm_w4a16` checks that the shapes fit.
template <bool USE_CP_ASYNC>
__device__ __forceinline__ void load_A_tile(half* dst, const half* __restrict__ X, int m0, int k0,
                                            int M, int K, int tid) {
#pragma unroll
  for (int c = tid; c < BM * (BK / 8); c += THREADS) {
    const int row = c / (BK / 8);
    const int col = (c % (BK / 8)) * 8;
    const bool ok = (m0 + row) < M;
    const int safe_row = ok ? (m0 + row) : (M - 1);
    const half* src = X + static_cast<size_t>(safe_row * K + k0 + col);
    half* d = dst + row * LDA + col;
    if (USE_CP_ASYNC) {
      cp_async_16_guarded(d, src, ok);
    } else {
      *reinterpret_cast<uint4*>(d) =
          ok ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
    }
  }
}

// One packed word per thread-iteration, decoded and scaled in register, then
// written transposed as a single 16-byte store. Storing [BN][BK] rather than
// [BK][BN] is what turns 8 strided 2-byte stores into one vector store, and it
// is also the layout ldmatrix wants.
//
// GROUP is a template parameter rather than an argument for one reason: the
// scale lookup needs `(k0 + r8*8) / GROUP`, and integer division by a *runtime*
// value compiles to a multi-instruction sequence on the FMA pipe, executed once
// per thread per iteration. As a compile-time power of two it is a shift. An
// ncu pipe breakdown showed 64% of this kernel's instructions landing on the
// ALU and FMA pipes -- integer address arithmetic, not math -- which is what
// pointed here.
template <int GROUP>
__device__ __forceinline__ void load_B_tile(half* dst, const uint32_t* __restrict__ Wq,
                                            const half* __restrict__ S, int n0, int k0, int N,
                                            int tid) {
#pragma unroll
  for (int e = tid; e < (BK / 8) * BN; e += THREADS) {
    const int r8 = e / BN;
    const int c = e % BN;
    const int n = n0 + c;
    uint4 payload = make_uint4(0, 0, 0, 0);
    if (n < N) {
      const uint32_t q = Wq[static_cast<size_t>((k0 / 8 + r8) * N + n)];
      const half sc = S[static_cast<size_t>(((k0 + r8 * 8) / GROUP) * N + n)];
      half2 w[4];
      dequant_8_scaled(q, sc, w);
      payload = *reinterpret_cast<uint4*>(w);
    }
    *reinterpret_cast<uint4*>(dst + c * LDB + r8 * 8) = payload;
  }
}

// ---------------------------------------------------------------------------
// v6: wmma
// ---------------------------------------------------------------------------
template <int GROUP>
__global__ void gemm_v6_wmma(const half* __restrict__ X, const uint32_t* __restrict__ Wq,
                             const half* __restrict__ S, half* __restrict__ Y, int M, int K,
                             int N) {
  constexpr int FRAG_M = WARP_M / 16;  // 2
  constexpr int FRAG_N = WARP_N / 16;  // 4
  extern __shared__ char smem_raw[];
  half* As = reinterpret_cast<half*>(smem_raw);
  half* Bs = As + 2 * A_ELEMS;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int warp_m = (warp / 2) * WARP_M, warp_n = (warp % 2) * WARP_N;

  using namespace nvcuda::wmma;
  fragment<accumulator, 16, 16, 16, float> c_frag[FRAG_M][FRAG_N];
#pragma unroll
  for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
    for (int j = 0; j < FRAG_N; ++j) fill_fragment(c_frag[i][j], 0.0f);

  const int ntiles = K / BK;
  load_A_tile<true>(As, X, m0, 0, M, K, tid);
  load_B_tile<GROUP>(Bs, Wq, S, n0, 0, N, tid);
  cp_async_commit();

  for (int t = 0; t < ntiles; ++t) {
    const int buf = t & 1;
    if (t + 1 < ntiles) {
      load_A_tile<true>(As + (buf ^ 1) * A_ELEMS, X, m0, (t + 1) * BK, M, K, tid);
      load_B_tile<GROUP>(Bs + (buf ^ 1) * B_ELEMS, Wq, S, n0, (t + 1) * BK, N, tid);
      cp_async_commit();
      cp_async_wait_group<1>();
    } else {
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

  // A full fp32 [BM][BN] staging tile would need 64 KB, more shared memory than
  // the kernel has, so fragments drain one 16x16 tile at a time through a
  // per-warp 1 KB scratch carved out of the now-dead A buffer.
  __syncthreads();
  float* stage = reinterpret_cast<float*>(smem_raw) + warp * 256;
#pragma unroll
  for (int i = 0; i < FRAG_M; ++i) {
#pragma unroll
    for (int j = 0; j < FRAG_N; ++j) {
      store_matrix_sync(stage, c_frag[i][j], 16, mem_row_major);
      __syncwarp();
      const int gm = m0 + warp_m + i * 16, gn = n0 + warp_n + j * 16;
      for (int e = lane; e < 256; e += 32) {
        const int r = e >> 4, c = e & 15;
        if (gm + r < M && gn + c < N)
          Y[static_cast<size_t>((gm + r) * N + gn + c)] = __float2half(stage[e]);
      }
      __syncwarp();
    }
  }
}

// ---------------------------------------------------------------------------
// v7: mma.sync + ldmatrix
//
// Three differences from v6, all aimed at the instruction count rather than the
// memory system (which the tile size already fixed):
//
// 1. Fragment addresses are computed **once**, before the k-loop. Each
//    ldmatrix then costs one add off a precomputed base instead of a fresh
//    address calculation per call.
// 2. One `ldmatrix.x4` feeds two n-tiles, so a 16-wide B load serves two
//    `mma.sync` issues -- the {r0,r2} / {r1,r3} split documented in mma.cuh.
// 3. The epilogue writes **straight to global memory**. Because the accumulator
//    layout puts d0/d1 in adjacent columns, each pair is one 4-byte `half2`
//    store; no shared staging, no extra barrier, and the whole shared buffer
//    dies with the loop.
// ---------------------------------------------------------------------------
template <int GROUP>
__global__ void gemm_v7_mma(const half* __restrict__ X, const uint32_t* __restrict__ Wq,
                            const half* __restrict__ S, half* __restrict__ Y, int M, int K,
                            int N) {
  constexpr int FRAG_M = WARP_M / 16;     // 2 m-tiles of 16
  constexpr int FRAG_N = WARP_N / 8;      // 8 n-tiles of 8
  constexpr int B_GROUPS = WARP_N / 16;   // 4 ldmatrix loads, 16 n each

  extern __shared__ char smem_raw[];
  half* As = reinterpret_cast<half*>(smem_raw);
  half* Bs = As + 2 * A_ELEMS;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int warp_m = (warp / 2) * WARP_M, warp_n = (warp % 2) * WARP_N;

  float acc[FRAG_M][FRAG_N][4];
#pragma unroll
  for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
    for (int j = 0; j < FRAG_N; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

  // Hoisted addressing: everything lane- and warp-dependent is folded in here,
  // so the inner loop only adds the buffer and k offsets.
  const int lrow = ldmatrix_row_off(lane), lcol = ldmatrix_col_off(lane);
  uint32_t a_base[FRAG_M], b_base[B_GROUPS];
#pragma unroll
  for (int i = 0; i < FRAG_M; ++i)
    a_base[i] = smem_u32(As + (warp_m + i * 16 + lrow) * LDA + lcol);
#pragma unroll
  for (int g = 0; g < B_GROUPS; ++g)
    b_base[g] = smem_u32(Bs + (warp_n + g * 16 + lrow) * LDB + lcol);

  const int ntiles = K / BK;
  load_A_tile<true>(As, X, m0, 0, M, K, tid);
  load_B_tile<GROUP>(Bs, Wq, S, n0, 0, N, tid);
  cp_async_commit();

  for (int t = 0; t < ntiles; ++t) {
    const int buf = t & 1;
    if (t + 1 < ntiles) {
      load_A_tile<true>(As + (buf ^ 1) * A_ELEMS, X, m0, (t + 1) * BK, M, K, tid);
      load_B_tile<GROUP>(Bs + (buf ^ 1) * B_ELEMS, Wq, S, n0, (t + 1) * BK, N, tid);
      cp_async_commit();
      cp_async_wait_group<1>();
    } else {
      cp_async_wait_group<0>();
    }
    __syncthreads();

    const uint32_t a_off = buf * A_ELEMS * sizeof(half);
    const uint32_t b_off = buf * B_ELEMS * sizeof(half);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      const uint32_t k_off = kk * sizeof(half);

      uint32_t a_frag[FRAG_M][4];
#pragma unroll
      for (int i = 0; i < FRAG_M; ++i) ldmatrix_x4(a_frag[i], a_base[i] + a_off + k_off);

      uint32_t b_raw[B_GROUPS][4];
#pragma unroll
      for (int g = 0; g < B_GROUPS; ++g) ldmatrix_x4(b_raw[g], b_base[g] + b_off + k_off);

#pragma unroll
      for (int i = 0; i < FRAG_M; ++i) {
#pragma unroll
        for (int g = 0; g < B_GROUPS; ++g) {
          // Tiles {0,2} are the low/high k halves of the first n-tile in this
          // group; {1,3} are the same for the second.
          const uint32_t b0[2] = {b_raw[g][0], b_raw[g][2]};
          const uint32_t b1[2] = {b_raw[g][1], b_raw[g][3]};
          mma_m16n8k16(acc[i][2 * g], a_frag[i], b0);
          mma_m16n8k16(acc[i][2 * g + 1], a_frag[i], b1);
        }
      }
    }
    __syncthreads();
  }

  // Direct-to-global epilogue. d0/d1 are adjacent columns and d2/d3 are the
  // same pair eight rows down, so each accumulator quad is two half2 stores.
  const int group = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int i = 0; i < FRAG_M; ++i) {
#pragma unroll
    for (int j = 0; j < FRAG_N; ++j) {
      const int col = n0 + warp_n + j * 8 + tig * 2;
      if (col >= N) continue;
      const bool pair = (col + 1) < N;
      const int r0 = m0 + warp_m + i * 16 + group;
      const int r1 = r0 + 8;

      if (r0 < M) {
        half* p = Y + static_cast<size_t>(r0 * N + col);
        if (pair)
          *reinterpret_cast<half2*>(p) = __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
        else
          *p = __float2half(acc[i][j][0]);
      }
      if (r1 < M) {
        half* p = Y + static_cast<size_t>(r1 * N + col);
        if (pair)
          *reinterpret_cast<half2*>(p) = __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
        else
          *p = __float2half(acc[i][j][2]);
      }
    }
  }
}

torch::Tensor gemm_w4a16(torch::Tensor X, torch::Tensor Wq, torch::Tensor S, int64_t group_size,
                         int64_t version) {
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
  TORCH_CHECK(N % 2 == 0, "N must be even");
  TORCH_CHECK(group_size == 64 || group_size == 128, "group_size must be 64 or 128");
  // The kernels index in 32-bit to keep address IMADs off the 64-bit path.
  constexpr int64_t kIndexLimit = 1LL << 31;
  TORCH_CHECK(static_cast<int64_t>(M) * N < kIndexLimit &&
                  static_cast<int64_t>(M) * K < kIndexLimit &&
                  static_cast<int64_t>(K) * N < kIndexLimit,
              "shape too large for 32-bit indexing in the prefill kernel");

  const at::cuda::CUDAGuard guard(X.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto Y = torch::empty({M, N}, X.options());

  const auto* x = reinterpret_cast<const half*>(X.data_ptr<at::Half>());
  const auto* w = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const auto* s = reinterpret_cast<const half*>(S.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(Y.data_ptr<at::Half>());

  const size_t smem = static_cast<size_t>(2 * A_ELEMS + 2 * B_ELEMS) * sizeof(half);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

#define NIBBLE_GEMM_LAUNCH(G)                                                        \
  do {                                                                               \
    if (version == 6)                                                                \
      gemm_v6_wmma<G><<<grid, THREADS, smem, stream>>>(x, w, s, y, M, K, N);          \
    else                                                                             \
      gemm_v7_mma<G><<<grid, THREADS, smem, stream>>>(x, w, s, y, M, K, N);           \
  } while (0)

  if (group_size == 128)
    NIBBLE_GEMM_LAUNCH(128);
  else
    NIBBLE_GEMM_LAUNCH(64);
#undef NIBBLE_GEMM_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Y;
}

}  // namespace nibble
