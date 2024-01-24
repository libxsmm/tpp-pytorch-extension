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

bool isActSigmoid = false;

REGISTER_LOCAL_SCOPE(gemm, "gemm");
REGISTER_LOCAL_SCOPE(d_bias, "d_bias");
REGISTER_LOCAL_SCOPE(d_act, "d_act");
REGISTER_LOCAL_SCOPE(di_gemm, "di_gemm");
REGISTER_LOCAL_SCOPE(dw_gemm, "dw_gemm");

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

// s'(z) = (1 - s(z)) * s(z)
at::Tensor d_sigmoid(at::Tensor grad_act, at::Tensor act) {
  //    auto s = torch::sigmoid(z);
  return grad_act * (1 - act) * act;
}

/*
at::Tensor forward(at::Tensor t_in, at::Tensor t_wt, at::Tensor t_bias) {
    auto y = torch::addmm(t_bias, t_in, t_wt.t());
    auto act = torch::sigmoid(y);
    return act;
}
*/
static std::vector<at::Tensor> tpp_forward(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias) {
  GlobalPass _gp(FWD);
  if (t_in.dtype() == at::kFloat) {
    typedef float Tin;
    typedef float Tout;
#include "perceptron_fwd_tmpl.h"
  } else if (t_in.dtype() == at::kBFloat16) {
    typedef bfloat16 Tin;
    typedef bfloat16 Tout;
#include "perceptron_fwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}
/*
std::vector<at::Tensor> backward(at::Tensor grad_act, at::Tensor in, at::Tensor
wt, at::Tensor act) {
//    auto dy = at::sigmoid_backward(grad_act, act);
    auto dy = d_sigmoid(grad_act, act);
    auto d_in = torch::mm(dy, wt); //dy*wt.t()
    auto d_wt = torch::mm(dy.t(), in); //in.t * dy - vnni
    auto d_bias = dy.sum(0, true);//torch::mm(dy, input.t()); dy.sum(dim, keep
dim) return {d_in, d_wt, d_bias};
}
*/
static std::vector<at::Tensor> tpp_backward(
    at::Tensor _t_grad_act,
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_act,
    at::Tensor t_relu_mask) {
  auto t_grad_act = _t_grad_act.contiguous();

  GlobalPass _gp(BWD);
  if (t_grad_act.dtype() == at::kFloat) {
    typedef float Tin;
    typedef float Tout;
#include "perceptron_bwd_tmpl.h"
  } else if (t_grad_act.dtype() == at::kBFloat16) {
    typedef bfloat16 Tin;
    typedef bfloat16 Tout;
#include "perceptron_bwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

class PerceptronFunction : public Function<PerceptronFunction> {
 public:
  static at::Tensor forward(
      AutogradContext* ctx,
      at::Tensor input,
      at::Tensor weight,
      at::Tensor bias) {
    auto output = tpp_forward(input, weight, bias);
    auto t_out = output[0];
    auto t_relu_mask = output[1];
    ctx->save_for_backward({input, weight, bias, t_out, t_relu_mask});
    return t_out; // output;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];
    auto act = saved[3];
    auto relu_mask = saved[4];

    auto outTensorList =
        tpp_backward(grad_output[0], input, weight, act, relu_mask);
    return outTensorList;
  }
};

static at::Tensor perceptron_global(
    bool _isActSigmoid,
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias) {
  isActSigmoid = _isActSigmoid;

  auto out = PerceptronFunction::apply(input, weight, bias);
  return out;
}

REGISTER_SUBMODULE(_perceptron_cpp, m) {
  m.def("perceptron_global", &perceptron_global, "Perceptron interface");
  //  m.def("forward", &forward, "Perceptron Forward");
  //  m.def("backward", &backward, "Perceptron Backward");
}
