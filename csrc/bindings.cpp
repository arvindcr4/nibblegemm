#include <torch/extension.h>

namespace nibble {
torch::Tensor gemv_w4a16(torch::Tensor X, torch::Tensor Wq, torch::Tensor S, int64_t group_size,
                         int64_t version, int64_t splits_override, int64_t threads_override);
torch::Tensor gemm_w4a16(torch::Tensor X, torch::Tensor Wq, torch::Tensor S, int64_t group_size,
                         int64_t version);
torch::Tensor dequant_w4(torch::Tensor Wq, torch::Tensor S, int64_t group_size, bool fast);
}  // namespace nibble

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_w4a16", &nibble::gemv_w4a16,
        "W4A16 matmul, decode regime (M <= 8). version selects the ladder stage.",
        py::arg("X"), py::arg("Wq"), py::arg("S"), py::arg("group_size"), py::arg("version") = 4,
        py::arg("splits_override") = 0, py::arg("threads_override") = 0);
  m.def("gemm_w4a16", &nibble::gemm_w4a16,
        "W4A16 matmul, prefill regime. version 6 = wmma (default, marginally faster), "
        "7 = mma.sync + ldmatrix.",
        py::arg("X"), py::arg("Wq"), py::arg("S"), py::arg("group_size"), py::arg("version") = 6);
  m.def("dequant_w4", &nibble::dequant_w4, "Dequantise packed INT4 to fp16.", py::arg("Wq"),
        py::arg("S"), py::arg("group_size"), py::arg("fast") = true);
}
