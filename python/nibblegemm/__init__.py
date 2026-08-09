"""nibblegemm -- INT4 weight-only GEMM kernels for NVIDIA Ampere."""

from .gptq import (
    HessianAccumulator,
    find_decoder_blocks,
    gptq_quantize,
    quantize_model_gptq,
)
from .ops import DECODE_MAX_M, QuantLinear, dequant, extension, gemm, gemv, matmul
from .quant import (
    QuantizedWeight,
    dequantize,
    pack_nibbles,
    quantize,
    reference_matmul,
    unpack_nibbles,
)

__all__ = [
    "DECODE_MAX_M",
    "QuantLinear",
    "QuantizedWeight",
    "dequant",
    "dequantize",
    "HessianAccumulator",
    "extension",
    "find_decoder_blocks",
    "gptq_quantize",
    "quantize_model_gptq",
    "gemm",
    "gemv",
    "matmul",
    "pack_nibbles",
    "quantize",
    "reference_matmul",
    "unpack_nibbles",
]
__version__ = "0.1.0"
