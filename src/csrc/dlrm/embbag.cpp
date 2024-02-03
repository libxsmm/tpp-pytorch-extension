#include <ATen/record_function.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <omp.h>
#include <time.h>
#include <iostream>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

// #define NO_TPP_FWD
// #define NO_TPP_BWD
// #define NO_TPP_UPD
// #define NO_TPP_DOT

using namespace tpp;
using namespace torch::autograd;

#define MYASSERT(x)                     \
  do {                                  \
    if (!(x)) {                         \
      printf("Assert failed %s\n", #x); \
      exit(1);                          \
    }                                   \
  } while (0)

using namespace torch::autograd::profiler;

REGISTER_LOCAL_SCOPE(ebag, "embbag");
REGISTER_LOCAL_SCOPE(d_ebag, "d_embbag");
REGISTER_LOCAL_SCOPE(bdot, "bdot");
REGISTER_LOCAL_SCOPE(d_bdot, "d_bdot");

#ifdef FP32_OUTPUT
#define out_scalar_t float
#else
#define out_scalar_t scalar_t
#endif

template <typename scalar_t>
void tpp_embedding_bag_forward_tmpl(
    torch::Tensor t_weight,
    torch::Tensor t_input,
    torch::Tensor t_offsets,
    torch::Tensor& t_output) {
  auto N = t_offsets.size(0);
  auto NS = t_input.size(0);
  // auto M = t_weight.size(0);
  auto E = t_weight.size(1);
  t_input = t_input.contiguous();
  t_offsets = t_offsets.contiguous();

  DECL_VLA_PTR_PT(scalar_t, weight, [E], t_weight);
  DECL_VLA_PTR_PT(out_scalar_t, output, [E], t_output);
  int64_t* input = t_input.data_ptr<int64_t>();
  int64_t* offsets = t_offsets.data_ptr<int64_t>();

  auto embbag =
      SCOPEIT((EmbBagFwdTPP<scalar_t, out_scalar_t, int64_t>(E)), ROW_GT);

  {
    RECORD_SCOPE(ebag, {t_weight, t_input, t_offsets});
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      auto start = offsets[n];
      auto end = (n < N - 1 ? offsets[n + 1] : NS);

      embbag(output[n], weight[0], &input[start], end - start);
    }
  }
}

// kHalf, kBFloat16, kFloat
at::Tensor tpp_embedding_bag_forward(
    torch::Tensor weight,
    torch::Tensor input,
    torch::Tensor offsets) {
  GlobalPass _gp(FWD);
  auto N = offsets.size(0);
  // auto NS = input.size(0);
  auto E = weight.size(1);
#ifdef FP32_OUTPUT
  auto opts = weight.options().dtype(at::kFloat);
#else
  auto opts = weight.options();
#endif
  at::Tensor output = at::empty({N, E}, opts);
  if (weight.dtype() == at::kFloat) {
    tpp_embedding_bag_forward_tmpl<float>(weight, input, offsets, output);
  } else if (weight.dtype() == at::kBFloat16) {
    tpp_embedding_bag_forward_tmpl<bfloat16>(weight, input, offsets, output);
  } else {
    TPP_ASSERT(0, "This datatype is not supported\n");
  }
  return output;
}

template <typename scalar_t>
inline void tpp_embedding_bag_backward_tmpl(
    torch::Tensor t_gradout,
    torch::Tensor t_weight,
    torch::Tensor t_input,
    torch::Tensor t_offsets,
    torch::Tensor t_values) {
  auto N = t_offsets.size(0);
  auto NS = t_input.size(0);
  auto E = t_gradout.size(1);

  // DECL_VLA_PTR_PT(scalar_t, weight, [E], t_weight);
  DECL_VLA_PTR_PT(scalar_t, values, [E], t_values);
  DECL_VLA_PTR_PT(out_scalar_t, gradout, [E], t_gradout);
  // int64_t *input = t_input.data_ptr<int64_t>();
  int64_t* offsets = t_offsets.data_ptr<int64_t>();
  auto embbag_bwd = SCOPEIT((EmbBagBwdTPP<out_scalar_t, scalar_t>(E)), EW_COPY);
  {
    RECORD_SCOPE(d_ebag, {t_gradout, t_input, t_offsets});
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      auto start = offsets[n];
      auto end = (n < N - 1 ? offsets[n + 1] : NS);
      embbag_bwd(gradout[n], values[start], end - start);
    }
  }
}

at::Tensor tpp_embedding_bag_backward(
    torch::Tensor gradout,
    torch::Tensor weight,
    torch::Tensor input,
    torch::Tensor offsets) {
  GlobalPass _gp(BWD);
  auto NS = input.size(0);
  auto E = gradout.size(1);
  auto values = at::empty({NS, E}, weight.options());
  auto indices = input.reshape({1, -1});
  if (weight.dtype() == at::kFloat) {
    tpp_embedding_bag_backward_tmpl<float>(
        gradout, weight, input, offsets, values);
  } else if (weight.dtype() == at::kBFloat16) {
    tpp_embedding_bag_backward_tmpl<bfloat16>(
        gradout, weight, input, offsets, values);
  } else {
    TPP_ASSERT(0, "This datatype is not supported\n");
  }

  auto grad_weight =
      at::_sparse_coo_tensor_unsafe(indices, values, weight.sizes());

  return grad_weight;
}

#if 0
libxsmm_smmfunction get_smm_kernel(int M, int N, int K) {
  libxsmm_smmfunction mm_kernel;
  float alpha = 1.0;
  float beta = 0.0;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  mm_kernel = libxsmm_smmdispatch( N, M, K, NULL, NULL, NULL, &alpha, &beta, &flags, NULL );
  MYASSERT(mm_kernel);
  return mm_kernel;
}

libxsmm_bmmfunction get_bmm_kernel(int M, int N, int K) {
  libxsmm_bmmfunction mm_kernel;
  float alpha = 1.0;
  float beta = 0.0;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
  mm_kernel = libxsmm_bmmdispatch( N, M, K, NULL, NULL, NULL, &alpha, &beta, &flags, NULL );
  MYASSERT(mm_kernel);
  return mm_kernel;
}
#endif

at::Tensor tpp_bdot_forward(at::Tensor& in) {
  GlobalPass _gp(FWD);
  auto sizes = in.sizes();
  int MB = sizes[0];
  unsigned int M = sizes[1];
  unsigned int K = sizes[2];
  at::Tensor out = at::empty({MB, M, M}, in.options());

  RECORD_SCOPE(bdot, {in});
  if (in.scalar_type() == at::kFloat) {
    // std::cout << "tpp_dot: float " << in.scalar_type() << std::endl;
    // libxsmm_smmfunction mm_kernel_f32 = get_smm_kernel( M, M, K );
    auto mm_kernel_f32_tpp =
        SCOPEIT((BrgemmTPP<float, float>(M, M, K, M * K, M * K, 0.0)));
    auto tr_kernel_tpp =
        SCOPEIT(XformExtTPP<float>(M, K, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    float* input = in.data_ptr<float>();
    float* output = out.data_ptr<float>();

#pragma omp parallel for
    for (int i = 0; i < MB; ++i) {
      float tmpa[M * K];
      tr_kernel_tpp(&input[i * M * K], tmpa);
      // mm_kernel_f32( tmpa, &input[i*M*K], &output[i*M*M] );
      mm_kernel_f32_tpp(&input[i * M * K], tmpa, &output[i * M * M], 1);
    }
  } else if (in.scalar_type() == at::kBFloat16) {
    // std::cout << "tpp_dot: bfloat " << in.scalar_type() << std::endl;
    MYASSERT(K % 2 == 0);
    // unsigned int K2 = K / 2;
    // libxsmm_bmmfunction mm_kernel_bf16 = get_bmm_kernel( M, M, K );
    auto mm_kernel_bf16_tpp =
        SCOPEIT((BrgemmTPP<bfloat16, bfloat16>(M, M, K, M * K, M * K, 0.0)));
    auto tr_kernel_tpp = SCOPEIT(
        XformExtTPP<at::BFloat16>(M, K, XformTPP::XFORM_XPOSE_N2V_TPP), XPOSE);
    auto* input = in.data_ptr<at::BFloat16>();
    auto* output = out.data_ptr<at::BFloat16>();

#pragma omp parallel for
    for (int i = 0; i < MB; ++i) {
      at::BFloat16 tmpa[M * K];
      // at::BFloat16 tmpa[4096];
      tr_kernel_tpp(&input[i * M * K], tmpa);
      // mm_kernel_bf16( (libxsmm_bfloat16*)tmpa,
      // (libxsmm_bfloat16*)&input[i*M*K], (libxsmm_bfloat16*)&output[i*M*M] );
      mm_kernel_bf16_tpp(&input[i * M * K], tmpa, &output[i * M * M], 1);
    }
  } else {
    MYASSERT(0);
  }
  return out;
}

at::Tensor tpp_bdot_backward(at::Tensor& in1, at::Tensor& in2) {
  GlobalPass _gp(BWD);
  auto sizes1 = in1.sizes();
  auto sizes2 = in2.sizes();
  int MB = sizes1[0];
  unsigned int M = sizes1[1];
  unsigned int K = sizes1[2];
  unsigned int N = sizes2[2];
  MYASSERT(M == K);
  at::Tensor out = at::empty({MB, M, N}, in1.options());
  // at::Tensor out = at::zeros({MB, M, N}, in1.options());
  // libxsmm_smmfunction mm_kernel_f32 = get_smm_kernel( M, N, K );
  auto mm_kernel_f32_tpp =
      SCOPEIT((BrgemmTPP<float, float>(M, N, K, M * K, N * K, 0.0)));

  RECORD_SCOPE(d_bdot, {in1, in2});
  // std::cout << "in1: " << sizes1 << ", in2: " << sizes2 << ", out: " <<
  // out.sizes() << ", dt: " << in1.scalar_type() << std::endl;
  if (in1.scalar_type() == at::kFloat) {
    auto input1 = in1.data_ptr<float>();
    auto input2 = in2.data_ptr<float>();
    auto output = out.data_ptr<float>();
#pragma omp parallel for
    for (int i = 0; i < MB; ++i) {
      float tmpa[M * K];
      // float tmpa[4096];
      for (unsigned int j = 0; j < M; j++) {
        for (unsigned int k = 0; k < K; k++) {
          tmpa[j * K + k] =
              input1[i * M * K + j * K + k] + input1[i * M * K + k * M + j];
        }
      }
      // mm_kernel_f32( &input2[i*K*N], tmpa, &output[i*M*N] );
      mm_kernel_f32_tpp(tmpa, &input2[i * K * N], &output[i * M * N], 1);
    }
  } else if (in1.scalar_type() == at::kBFloat16) {
    auto input1 = in1.data_ptr<at::BFloat16>();
    auto input2 = in2.data_ptr<at::BFloat16>();
    auto output = out.data_ptr<at::BFloat16>();
    auto cvt_in1 = SCOPEIT((ConvertTPP<at::BFloat16, float>(M * K)), EW_COPY);
    auto cvt_in2 = SCOPEIT((ConvertTPP<at::BFloat16, float>(K * N)), EW_COPY);
    auto cvt_out = SCOPEIT((ConvertTPP<float, at::BFloat16>(M * N)), EW_COPY);
#pragma omp parallel for
    for (int i = 0; i < MB; ++i) {
      float tmpa[M * K], tmpb[K * N], tmpc[M * N], tmpaa[M * K];
      // float tmpa[4096], tmpb[4096], tmpc[4096];
      cvt_in1(&input1[i * M * K], tmpa);
      cvt_in2(&input2[i * K * N], tmpb);
      for (unsigned int j = 0; j < M; j++) {
        for (unsigned int k = 0; k < K; k++) {
          tmpaa[j * K + k] = tmpa[j * K + k] + tmpa[k * M + j];
        }
      }
      // mm_kernel_f32( tmpb, tmpa, tmpc );
      mm_kernel_f32_tpp(tmpaa, tmpb, tmpc, 1);
      cvt_out(tmpc, &output[i * M * N]);
    }
  } else {
    MYASSERT(0);
  }
  return out;
}

REGISTER_SUBMODULE(_embbag_cpp, m) {
  m.def(
      "embbag_forward",
      &tpp_embedding_bag_forward,
      "Tpp Embedding Bag forward");
  m.def(
      "embbag_backward",
      &tpp_embedding_bag_backward,
      "Tpp Embedding Bag backward");
  m.def("bdot_forward", &tpp_bdot_forward, "Tpp batch dot forward");
  m.def("bdot_backward", &tpp_bdot_backward, "Tpp batch dot backward");
}
