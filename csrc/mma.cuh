// Raw `mma.sync` / `ldmatrix` wrappers for sm_80, plus the fragment layouts.
//
// The wmma API is correct by construction but recomputes fragment addressing
// inside every `load_matrix_sync` call and gives no control over the shared
// layout. Profiling the wmma prefill kernel showed 9.5 non-tensor instructions
// issued per tensor instruction -- the kernel was issue-bound on bookkeeping,
// not math. Dropping to the underlying instructions removes that bookkeeping,
// at the cost of having to get the register layouts exactly right by hand.
//
// The layouts below are the part that is easy to get wrong and hard to debug,
// so they are written out rather than left to the reader's memory of the PTX
// ISA document. Throughout: `lane` is 0..31, `group = lane / 4` and
// `tig = lane % 4` ("thread in group").
//
// ---------------------------------------------------------------------------
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// ---------------------------------------------------------------------------
// A is 16x16 (M x K) row-major, 8 halves per thread in 4 registers:
//     a0 -> (row = group,     col = 2*tig    )  and col+1
//     a1 -> (row = group + 8, col = 2*tig    )  and col+1
//     a2 -> (row = group,     col = 2*tig + 8)  and col+1
//     a3 -> (row = group + 8, col = 2*tig + 8)  and col+1
//
// B is 16x8 (K x N) col-major, 4 halves per thread in 2 registers:
//     b0 -> (col = group, row = 2*tig    )  and row+1
//     b1 -> (col = group, row = 2*tig + 8)  and row+1
//
// C/D is 16x8 fp32, 4 floats per thread:
//     d0,d1 -> (row = group,     col = 2*tig) and col+1
//     d2,d3 -> (row = group + 8, col = 2*tig) and col+1
//
// The d0/d1 pair being adjacent *columns* is what lets the epilogue store
// results straight to global memory as one 4-byte `half2` write, with no
// shared-memory staging pass at all.
//
// ---------------------------------------------------------------------------
// ldmatrix.sync.aligned.m8n8.x4.shared.b16
// ---------------------------------------------------------------------------
// Loads four 8x8 tiles of 16-bit elements. Each lane supplies the address of
// one row: lanes 0-7 give the rows of tile 0, lanes 8-15 tile 1, 16-23 tile 2,
// 24-31 tile 3. Afterwards each lane holds two elements of each tile, at
// row = group, cols = 2*tig and 2*tig + 1.
//
// Feeding it addresses laid out as
//
//     row_offset = (lane % 8) + 8 * ((lane / 8) % 2)
//     col_offset = 8 * (lane / 16)
//
// makes the four tiles come out as (rows 0-7, cols 0-7), (rows 8-15, cols 0-7),
// (rows 0-7, cols 8-15), (rows 8-15, cols 8-15) -- which is exactly a0..a3 for
// a row-major A fragment, in order, with no shuffling. The same address pattern
// applied to a K-contiguous (transposed) B tile yields b0/b1 for two adjacent
// n-tiles: {r0, r2} for the first, {r1, r3} for the second.
//
// ---------------------------------------------------------------------------
// Bank conflicts
// ---------------------------------------------------------------------------
// ldmatrix issues in phases of 8 lanes, each reading 16 bytes. With a row
// stride of 40 halves (80 bytes = 20 banks), the 8 rows of a phase land on bank
// offsets 20*r mod 32 = {0, 20, 8, 28, 16, 4, 24, 12}. Each lane covers 4
// consecutive banks from there, and those 8 spans tile the 32 banks exactly
// once -- conflict-free without needing an XOR swizzle. This is why the tiles
// are padded to BK + 8 rather than any other value.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace nibble {

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// Row/column offsets a lane must apply so that ldmatrix.x4 yields the four
// tiles in (0-7, 0-7), (8-15, 0-7), (0-7, 8-15), (8-15, 8-15) order.
__device__ __forceinline__ int ldmatrix_row_off(int lane) {
  return (lane & 7) + 8 * ((lane >> 3) & 1);
}
__device__ __forceinline__ int ldmatrix_col_off(int lane) { return 8 * (lane >> 4); }

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

}  // namespace nibble
