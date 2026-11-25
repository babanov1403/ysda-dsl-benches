#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

torch::Tensor softmax_wide(torch::Tensor x) {
    TORCH_CHECK(false, "TODO: implement me!");
}

torch::Tensor softmax_online(torch::Tensor x) {
    TORCH_CHECK(false, "TODO: implement me!");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_wide", &softmax_wide);
    m.def("softmax_online", &softmax_online);
}
