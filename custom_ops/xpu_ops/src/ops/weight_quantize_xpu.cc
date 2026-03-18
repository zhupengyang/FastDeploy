// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <infer_ops.h>
#include <infer_ops_eb.h>
#include <paddle/extension.h>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename T>
std::vector<paddle::Tensor> WeightQuantizeKernel(
    const paddle::Tensor &x,
    const std::string &algo,
    const int32_t arch,
    const int32_t group_size,
    const std::string &input_layout,
    const std::string &output_layout) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  int64_t k = x.shape()[0];
  int64_t n = x.shape()[1];
  if (input_layout == "nk") {
    std::swap(n, k);
  }
  if (algo == "weight_only_int4") {
    PD_CHECK(k % 2 == 0, "k must be even when algo is weight_only_int4");
  }
  int64_t k_out = algo == "weight_only_int4" ? k / 2 : k;

  int ret = -1;
  paddle::Tensor x_nk = x;
  if (input_layout == "kn") {
    x_nk = paddle::empty({n, k}, x.dtype(), x.place());
    ret = baidu::xpu::api::transpose<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(x.data<T>()),
        reinterpret_cast<XPUType *>(x_nk.data<T>()),
        {k, n},
        {1, 0});
    PD_CHECK(ret == 0);
  }
  paddle::Tensor out_nk =
      paddle::empty({n, k_out}, paddle::DataType::INT8, x.place());
  paddle::Tensor out = out_nk;
  if (output_layout == "kn") {
    out = paddle::empty({k_out, n}, paddle::DataType::INT8, x.place());
  }
  paddle::Tensor scale =
      paddle::empty({n}, paddle::DataType::FLOAT32, x.place());

  if (algo == "weight_only_int8") {
    ret = infer_ops::quant2d_per_token<XPUType, float, int8_t>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(x_nk.data<T>()),
        nullptr,
        out_nk.data<int8_t>(),
        scale.data<float>(),
        n,
        k);
    PD_CHECK(ret == 0);
  } else if (algo == "weight_only_int4") {
    ret = infer_ops::quant2d_per_token<XPUType, float, int4_t>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(x_nk.data<T>()),
        nullptr,
        reinterpret_cast<int4_t *>(out_nk.data<int8_t>()),
        scale.data<float>(),
        n,
        k);
    PD_CHECK(ret == 0);
  } else {
    PD_THROW(
        "Weight quantize only supports weight_only_int8 or weight_only_int4 on "
        "XPU now.");
  }

  if (output_layout == "kn") {
    ret = baidu::xpu::api::transpose<int8_t>(xpu_ctx->x_context(),
                                             out_nk.data<int8_t>(),
                                             out.data<int8_t>(),
                                             {n, k_out},
                                             {1, 0});
    PD_CHECK(ret == 0);
  }
  return {out, scale};
}

std::vector<paddle::Tensor> WeightQuantize(const paddle::Tensor &x,
                                           const std::string &algo,
                                           const int32_t arch,
                                           const int32_t group_size,
                                           const std::string &input_layout,
                                           const std::string &output_layout) {
  PD_CHECK(algo == "weight_only_int8" || algo == "weight_only_int4",
           "algo must be weight_only_int8 or weight_only_int4, but get ",
           algo);
  PD_CHECK(group_size == -1, "group_size must be -1, but get ", group_size);
  PD_CHECK(input_layout == "nk" || input_layout == "kn",
           "input_layout must be kn or nk, but get ",
           input_layout);
  PD_CHECK(output_layout == "kn" || output_layout == "nk",
           "output_layout must be kn or nk, but get ",
           output_layout);
  const auto x_type = x.dtype();
#define APPLY_WEIGHT_QUANTIZE_KERNEL(TX) \
  return WeightQuantizeKernel<TX>(       \
      x, algo, arch, group_size, input_layout, output_layout);
  if (x_type == paddle::DataType::BFLOAT16) {
    APPLY_WEIGHT_QUANTIZE_KERNEL(paddle::bfloat16);
  } else if (x_type == paddle::DataType::FLOAT32) {
    APPLY_WEIGHT_QUANTIZE_KERNEL(float);
  } else {
    PD_THROW("WeightQuantize not support x_type==%d", static_cast<int>(x_type));
    return {};
  }
}

std::vector<std::vector<int64_t>> WeightQuantizeInferShape(
    const std::vector<int64_t> &x_shape,
    const std::string &algo,
    const int32_t arch,
    const int32_t group_size,
    const std::string &input_layout,
    const std::string &output_layout) {
  PD_CHECK(algo == "weight_only_int8" || algo == "weight_only_int4",
           "algo must be weight_only_int8 or weight_only_int4");
  PD_CHECK(input_layout == "nk" || input_layout == "kn",
           "input_layout must be kn or nk");
  PD_CHECK(output_layout == "kn" || output_layout == "nk",
           "output_layout must be kn or nk");
  int64_t k = x_shape[0];
  int64_t n = x_shape[1];
  if (input_layout == "nk") {
    std::swap(n, k);
  }
  if (algo == "weight_only_int4") {
    PD_CHECK(k % 2 == 0, "k must be even when algo is weight_only_int4");
    k = k / 2;
  }
  if (output_layout == "kn") {
    return {{k, n}, {n}};
  } else {
    return {{n, k}, {n}};
  }
}

std::vector<paddle::DataType> WeightQuantizeInferDtype(
    const paddle::DataType &x_dtype,
    const std::string &algo,
    const int32_t arch,
    const int32_t group_size,
    const std::string &input_layout,
    const std::string &output_layout) {
  if (algo == "weight_only_int8") {
    return {paddle::DataType::INT8, paddle::DataType::FLOAT32};
  } else if (algo == "weight_only_int4") {
    return {paddle::DataType::INT8, paddle::DataType::FLOAT32};
  } else {
    PD_THROW("weight_quantize not support algo=%s", algo);
  }
}

PD_BUILD_STATIC_OP(weight_quantize_xpu)
    .Inputs({"x"})
    .Outputs({"out", "scale"})
    .Attrs({"algo: std::string",
            "arch: int",
            "group_size: int",
            "input_layout: std::string",
            "output_layout: std::string"})
    .SetKernelFn(PD_KERNEL(WeightQuantize))
    .SetInferShapeFn(PD_INFER_SHAPE(WeightQuantizeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WeightQuantizeInferDtype));
