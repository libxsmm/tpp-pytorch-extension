#include <ATen/record_function.h>
#include <torch/torch.h>
// #include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
using namespace torch::autograd;

#include "tensor_helper.h"

#define PRINT_T_SIZE(x) std::cout << #x << ": " << x.sizes() << std::endl
#define PRINT_T(x) std::cout << #x << ": " << x << std::endl

REGISTER_LOCAL_SCOPE(fused_gemm_act, "fused_gemm_act");
REGISTER_LOCAL_SCOPE(d_bias, "d_bias");
REGISTER_LOCAL_SCOPE(d_act, "d_act");
REGISTER_LOCAL_SCOPE(di_fused_gemm, "di_gemm");
REGISTER_LOCAL_SCOPE(dw_gemm, "dw_gemm");
REGISTER_LOCAL_SCOPE(di_0_gemm, "dii_gemm");

template <typename Tout>
inline void omp_reduce_buf(
    int num_threads,
    int N,
    float** ptrs,
    Tout* buf,
    bool accumulate = false) {
  ScopedTimer _t(EW_RED);
#pragma omp for
  for (int i = 0; i < N; i++) {
    float sum = 0.0;
    for (int j = 0; j < num_threads; j++) {
      sum += ptrs[j][i];
    }
    if (accumulate) {
      buf[i] += sum;
    } else {
      buf[i] = sum;
    }
  }
}

// inputs: x1, w1, b1, w2, b2
// return: save_bwd - x1, w1, y1, x2, w2, y2
static std::vector<at::Tensor> tpp_forward(
    bool bias,
    int nLayers,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  std::vector<at::Tensor> save_bwd;
  at::Tensor t_in = inputs[0];

  if (t_in.dtype() == at::kFloat) {
    typedef float Tin;
    typedef float Tout;
#include "mlp_fwd_tmpl.h"
  } else if (t_in.dtype() == at::kBFloat16) {
    typedef bfloat16 Tin;
    typedef bfloat16 Tout;
#include "mlp_fwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
  return save_bwd;
}

// saved_tensors: x1, w1, y1, x2, w2, y2 from forward pass
// grad_act: from Autograd
// return: ret_list - grad_in0, grad_wt0, grad_bias0, grad_wt1, grad_bias1
static std::vector<at::Tensor> tpp_backward(
    int nLayers,
    std::vector<at::Tensor> saved_tensors,
    at::Tensor t_grad_act) {
  std::vector<at::Tensor>
      ret_list; // grad_in0, grad_wt1, grad_bias1, grad_wt0, grad_bias0

  if (t_grad_act.dtype() == at::kFloat) {
    typedef float Tin;
    typedef float Tout;
#include "mlp_bwd_tmpl.h"
  } else if (t_grad_act.dtype() == at::kBFloat16) {
    typedef bfloat16 Tin;
    typedef bfloat16 Tout;
#include "mlp_bwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }

  return ret_list;
}

// inputs: x1, w1, b1, w2, b2
// return: save_bwd - x1, w1, y1, x2, w2, y2
static std::vector<at::Tensor> cpp_forward(
    bool bias,
    int nLayers,
    std::vector<at::Tensor> inputs) {
  std::vector<at::Tensor> save_bwd;
  at::Tensor t_in = inputs[0];
  int wb_i = 1; // weight bias index
  for (int i = 0; i < nLayers; i++) {
    auto t_wt = inputs[wb_i++]; // w_i
    auto t_bias = inputs[wb_i++]; // b_i
    auto mm = torch::addmm(t_bias, t_in, t_wt.t());
    auto act = torch::sigmoid(mm);

    save_bwd.push_back(t_in); // in, wt, act
    save_bwd.push_back(t_wt);
    save_bwd.push_back(act);
    t_in = act;
  }
  return save_bwd;
}

// saved_tensors: x1, w1, y1, x2, w2, y2 from forward pass
// grad_act: from Autograd
// return: ret_list - grad_in0, grad_wt1, grad_bias1, grad_wt0, grad_bias0
static std::vector<at::Tensor> cpp_backward(
    int nLayers,
    std::vector<at::Tensor> saved_tensors,
    at::Tensor grad_act) {
  std::vector<at::Tensor>
      ret_list; // grad_in0, grad_wt1, grad_bias1, grad_wt0, grad_bias0
  // dummy push
  for (int i = 0; i < 2 * nLayers + 1; i++) {
    ret_list.push_back(grad_act);
  }

  int wb_i = 3 * nLayers -
      3; // wb_i initialized to index of in2; -- in1, wt1, act1, in2, wt2, act2
  int r_i = 2 * nLayers;
  for (int i = nLayers - 1; i >= 0; i--) {
    auto dy = at::sigmoid_backward(grad_act, saved_tensors[wb_i + 2]); // act[i]
    auto grad_in = torch::mm(dy, saved_tensors[wb_i + 1]); // wt[i]
    auto grad_wt = torch::mm(dy.t(), saved_tensors[wb_i]); // in[i]);
    auto grad_bias = dy.sum(0, true);

    if (i == 0) {
      ret_list[0] = grad_in;
    }
    ret_list[r_i--] = grad_bias;
    ret_list[r_i--] = grad_wt;

    grad_act = grad_in;
    wb_i -= 3;
  }
  return ret_list;
}

REGISTER_SUBMODULE(_mlp_cpp, m) {
  m.def("forward", &tpp_forward, "MLP Forward");
  m.def("backward", &tpp_backward, "MLP Backward");
  m.def("cpp_forward", &cpp_forward, "MLP Forward");
  m.def("cpp_backward", &cpp_backward, "MLP Backward");
}
