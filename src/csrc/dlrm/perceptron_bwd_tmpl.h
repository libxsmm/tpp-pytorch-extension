RECORD_FUNCTION("perceptron_bwd", std::vector<c10::IValue>());
auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto N1 = in_sizes[0];
auto C1 = in_sizes[1];
auto N2 = in_sizes[2];
auto C2 = in_sizes[3];

auto K1 = wt_sizes[0];
auto K2 = wt_sizes[3];
auto padded_K2 = K2;
const int VBS = get_vnni_block_size<Tin>();
const auto grad_wt_flag =
    (t_wt.dim() == 5 ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_NONE_TPP);
const auto input_trans_flag =
    (t_in.dtype() == at::kFloat ? XformTPP::XFORM_XPOSE_TPP
                                : XformTPP::XFORM_NONE_TPP);
auto t_wt_TV = wt_tensor_for_bwd(K1, K2, C1, C2, t_wt);

if (t_wt_TV.numel() != t_wt.numel()) {
  padded_K2 = t_wt_TV.size(2) * t_wt_TV.size(4);
}

auto t_in_T = t_in;

if (input_trans_flag == XformTPP::XFORM_NONE_TPP) {
  t_in_T = act_tensor_trans(N1, C1, N2, C2, t_in);
}
auto t_grad_out = t_in.new_empty({N1, K1, N2, padded_K2});
auto t_grad_in = at::empty_like(t_in);
auto t_grad_wt = at::empty_like(t_wt);
auto t_grad_bias = t_wt.new_empty({K1 * K2});
auto t_grad_out_V = t_grad_out;

if (t_grad_out.dtype() != at::kFloat) {
  t_grad_out_V = t_grad_out.new_empty({N1, K1, N2 / VBS, K2, VBS});
}
auto relu_rd = (N2 * K2 + 15) / 16;
auto relu_mask = GetVLAPtr<short>(t_relu_mask, {K1, relu_rd});
auto in_T = GetVLAPtr<Tin>(t_in_T, {C1, C2* N2});
auto wt_TV = GetVLAPtr<Tin>(t_wt_TV, {C1, C2* K2});
auto grad_in = GetVLAPtr<Tout>(t_grad_in, {C1, N2* C2});
auto grad_wt = GetVLAPtr<Tout>(t_grad_wt, {C1, C2* K2});
auto grad_bias = GetVLAPtr<Tout>(t_grad_bias, {K2});
auto grad_act = GetVLAPtr<Tin>(t_grad_act, {K1, N2* K2});
auto act = GetVLAPtr<Tin>(t_act, {K1, N2* K2});
auto grad_out = GetVLAPtr<Tin>(t_grad_out, {K1, N2* padded_K2});
auto grad_out_V = GetVLAPtr<Tin>(t_grad_out_V, {K1, N2* K2});

auto K1b = K1;
// if (K1 > C1 && K1 % C1 == 0) {
//  K1b = C1;
//}
auto grad_bias_tpp = SCOPEIT(GradBiasTPP<Tin>(N2, K2), BIAS);
auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(K1 * K2), EW_ZERO);
auto n2v_tpp =
    SCOPEIT(XformExtTPP<Tin>(N2, K2, XformTPP::XFORM_N2V_TPP, true), VNNI);

auto grad_sigmoid_tpp =
    SCOPEIT((SigmoidBwdTPP<Tin, Tout>(N2, K2, K2, K2)), ACT);

auto grad_relu_bwd_tpp = SCOPEIT((ReLUBwdTPP<Tin, Tout>(N2, K2, true)), ACT);

auto pad_tpp = SCOPEIT(PadTPP<Tin>(N2, K2, N2, padded_K2), EW_COPY);

auto di_gemm_b1_tpp = SCOPEITGEMM((BrgemmExtTPP<Tin, Tout>(
    N2,
    C2,
    padded_K2,
    N2* padded_K2,
    C1* padded_K2* C2,
    1.0,
    XformTPP::XFORM_NONE_TPP,
    0,
    K1b)));
auto dw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<Tin, Tout>(
    C2,
    K2,
    N2,
    C1* N2* C2,
    K1* N2* K2,
    1.0,
    (XformTPP::XFORM_TYPE)grad_wt_flag,
    input_trans_flag,
    N1)));
if (isActSigmoid) {
  RECORD_SCOPE(d_bias, {t_grad_out});
  tensor_set_zero(N1 * K1, N2 * padded_K2, t_grad_out);
  tensor_set_zero(K1, K2, t_grad_bias);
  int num_threads = omp_get_max_threads();
  float* bias_ptrs[num_threads];
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      float prv_grad_bias[K1][K2];
      bias_ptrs[tid] = prv_grad_bias[0];
      set_zero_tpp(prv_grad_bias[0]);
#pragma omp for collapse(2) // reduction(+:grad_bias[:K1][:K2])
      for (int n1 = 0; n1 < N1; n1++) {
        for (int k1 = 0; k1 < K1; k1++) {
          Tin tmp[N2 * K2];
          Tin* tmp_p = padded_K2 != K2 ? tmp : grad_out[n1][k1];
          grad_sigmoid_tpp(grad_act[n1][k1], act[n1][k1], tmp_p);
          grad_bias_tpp(tmp_p, prv_grad_bias[k1]);
          n2v_tpp(tmp_p, grad_out_V[n1][k1]);
          if (padded_K2 != K2)
            pad_tpp(tmp_p, grad_out[n1][k1]);
        }
      }
#pragma omp barrier
      omp_reduce_buf(num_threads, K1 * K2, bias_ptrs, grad_bias[0]);
    }
  }
} else {
  RECORD_SCOPE(d_bias, {t_grad_out});
  tensor_set_zero(N1 * K1, N2 * padded_K2, t_grad_out);
  tensor_set_zero(K1, K2, t_grad_bias);
  int num_threads = omp_get_max_threads();
  float* bias_ptrs[num_threads];
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      float prv_grad_bias[K1][K2];
      bias_ptrs[tid] = prv_grad_bias[0];
      set_zero_tpp(prv_grad_bias[0]);
#pragma omp for collapse(2) // reduction(+:grad_bias[:K1][:K2])
      for (int n1 = 0; n1 < N1; n1++) {
        for (int k1 = 0; k1 < K1; k1++) {
          Tin tmp[N2 * K2];
          Tin* tmp_p = padded_K2 != K2 ? tmp : grad_out[n1][k1];
          grad_relu_bwd_tpp(
              grad_act[n1][k1], tmp_p, nullptr, relu_mask[n1][k1]);
          grad_bias_tpp(tmp_p, prv_grad_bias[k1]);
          n2v_tpp(tmp_p, grad_out_V[n1][k1]);
          if (padded_K2 != K2)
            pad_tpp(tmp_p, grad_out[n1][k1]);
        }
      }
#pragma omp barrier
      omp_reduce_buf(num_threads, K1 * K2, bias_ptrs, grad_bias[0]);
    }
  }
}
{
  RECORD_SCOPE(di_gemm, {t_grad_out, t_wt_TV});
  // if(K1 != K1b) t_grad_in.zero_();
  tensor_set_zero(N1 * C1, N2 * C2, t_grad_in);
  for (int k1 = 0; k1 < K1; k1 += K1b) {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int n1 = 0; n1 < N1; n1++) {
      for (int c1 = 0; c1 < C1; c1++) {
        di_gemm_b1_tpp(grad_out[n1][k1], wt_TV[k1][c1], grad_in[n1][c1], K1b);
      }
    }
  }
}
{
  RECORD_SCOPE(dw_gemm, {t_in_T, t_grad_out_V});
  tensor_set_zero(K1 * C1, K2 * C2, t_grad_wt);
  constexpr int BS = 16;
  for (int n1 = 0; n1 < N1; n1 += BS) {
    int count = (n1 + BS <= N1 ? BS : N1 - n1);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int k1 = 0; k1 < K1; k1++) {
      for (int c1 = 0; c1 < C1; c1++) {
        dw_gemm_tpp(in_T[n1][c1], grad_out_V[n1][k1], grad_wt[k1][c1], count);
      }
    }
  }
}

return std::vector<at::Tensor>({t_grad_in, t_grad_wt, t_grad_bias});
