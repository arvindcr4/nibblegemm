// Fast INT4 -> fp16 decode.
//
// The naive way to turn a 4-bit integer into fp16 is shift, mask, I2F convert,
// multiply. The I2F is the expensive part: on Ampere it runs on the same
// pipe as other conversions and, at one per weight, it becomes the bottleneck
// long before HBM does. A W4A16 kernel that decodes this way is arithmetic
// bound while the whole point of 4-bit weights is to be bandwidth bound.
//
// The standard trick (FasterTransformer, AWQ, Marlin) removes the conversion
// entirely by exploiting the fp16 bit layout. An fp16 whose exponent field
// encodes 2^10 is 1024.0 with a zero mantissa:
//
//     0x6400 = 0 11001 0000000000 = 1024.0
//
// The low mantissa bits are integer-valued at that exponent, so OR-ing a
// nibble n into 0x6400 yields exactly the float 1024 + n. Recovering n is then
// one subtract, and because two fp16 live in a 32-bit register the subtract is
// a packed `hsub2` handling two weights at once.
//
// Extraction is a single `lop3.b32` -- one instruction for `(a & b) | c` --
// applied with two masks that each pick up a pair of nibbles 16 bits apart, so
// the results land pre-packed as half2. Four lop3 ops decode all eight nibbles.
//
// Masking at bits 4..7 instead of 0..3 leaves the value 16x too large; rather
// than shifting, that lane is corrected by an `hfma2` that folds the 1/16 and
// the bias into one instruction. The quantiser's implicit zero-point of 8 is
// folded into the same bias, so subtracting it is free.
//
// Cost: 4 lop3 + 2 hsub2 + 2 hfma2 = 8 instructions for 8 weights, versus
// roughly 8 shifts + 8 masks + 8 I2F + 8 multiplies. All results are exact --
// every intermediate (<= 1264) is an integer inside fp16's exactly-representable
// range, so this is a pure speed win with no numerical cost.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace nibble {

// lop3 immLut for (a & b) | c, derived the standard way by evaluating the
// expression on the canonical operands ta=0xF0, tb=0xCC, tc=0xAA.
static constexpr uint32_t kLutAndOr = (0xF0 & 0xCC) | 0xAA;  // 0xEA

static constexpr uint32_t kMaskLo = 0x000f000f;  // nibble slots {0, 4}
static constexpr uint32_t kMaskHi = 0x00f000f0;  // nibble slots {1, 5}
static constexpr uint32_t kMagic = 0x64006400;   // two copies of fp16 1024.0

__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(d)
               : "r"(a), "r"(b), "r"(c), "n"(kLutAndOr));
  return d;
}

// Decode one packed uint32 (eight nibbles) into four half2 holding logical
// offsets k = 0..7 in ascending order, already zero-point corrected but not yet
// scaled. Ordering falls out of the nibble interleave applied at pack time --
// see python/nibblegemm/quant.py. No shuffling is needed here.
__device__ __forceinline__ void dequant_8(uint32_t q, half2 (&out)[4]) {
  const uint32_t q_hi = q >> 8;

  // Each lop3 yields two fp16 of the form 1024 + n (lo masks) or 1024 + 16n
  // (hi masks), pre-packed into a half2.
  uint32_t r0 = lop3_and_or(q, kMaskLo, kMagic);     // slots 0, 4 -> k 0, 1
  uint32_t r1 = lop3_and_or(q, kMaskHi, kMagic);     // slots 1, 5 -> k 2, 3
  uint32_t r2 = lop3_and_or(q_hi, kMaskLo, kMagic);  // slots 2, 6 -> k 4, 5
  uint32_t r3 = lop3_and_or(q_hi, kMaskHi, kMagic);  // slots 3, 7 -> k 6, 7

  // 1024 + n            -> n - 8
  const half2 bias_lo = __float2half2_rn(1024.0f + 8.0f);
  // (1024 + 16n)/16 - 72 -> n - 8
  const half2 mul_hi = __float2half2_rn(1.0f / 16.0f);
  const half2 bias_hi = __float2half2_rn(-(64.0f + 8.0f));

  out[0] = __hsub2(*reinterpret_cast<half2*>(&r0), bias_lo);
  out[1] = __hfma2(*reinterpret_cast<half2*>(&r1), mul_hi, bias_hi);
  out[2] = __hsub2(*reinterpret_cast<half2*>(&r2), bias_lo);
  out[3] = __hfma2(*reinterpret_cast<half2*>(&r3), mul_hi, bias_hi);
}

// Same decode, then apply a broadcast group scale. Kept separate so callers
// that hoist the scale multiply into an accumulator can skip it.
__device__ __forceinline__ void dequant_8_scaled(uint32_t q, half scale, half2 (&out)[4]) {
  dequant_8(q, out);
  const half2 s = __half2half2(scale);
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = __hmul2(out[i], s);
}

// Straightforward shift/mask/convert decode. This is what the fast path is
// replacing; it exists so the v0 kernel is honest about its baseline and so
// tests can compare the two paths bit for bit.
__device__ __forceinline__ float dequant_scalar(uint32_t q, int k_off, float scale) {
  // Undo the pack-time interleave: even k in slots 0..3, odd k in slots 4..7.
  const int slot = (k_off & 1) ? (4 + (k_off >> 1)) : (k_off >> 1);
  const int nib = static_cast<int>((q >> (4 * slot)) & 0xF);
  return static_cast<float>(nib - 8) * scale;
}

}  // namespace nibble
