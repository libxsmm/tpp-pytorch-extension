RECORD_FUNCTION("mlp_bwd", std::vector<c10::IValue>());

// dummy push
// ret_list: grad_in0, grad_wt0, grad_bias0, grad_wt1, grad_bias1
// 2*nLayers for grad_wt and grad_bias for all layers;
// +1 for grad_in0
for (int i = 0; i < 2 * nLayers + 1; i++) {
  ret_list.push_back(t_grad_act);
}

// index for accessing vector of tensors saved in fwd pass
// -- in0, wt0, act0, in1, wt1, act1
// in: x (input); wt: weight; act: y (activation output)
// fwd_i initialized to index of in1 (input of last layer)
int fwd_i = 3 * nLayers - 3;
// index for accessing ret_list: return vector of tensors
// rw_i index for grad_weight tensor in ret_list
// rb_i index for grad_bias tensor in ret_list
// ret_list: grad_in0, grad_wt0, grad_bias0, grad_wt1, grad_bias1
// rw_i initialized to index of grad_wt1
// rb_i initialized to index of grad_bias1
int rw_i = 2 * nLayers - 1;
int rb_i = 2 * nLayers;

auto t_grad_out = at::empty_like(t_grad_act);
auto t_grad_out_V = t_grad_out; // VNNI equivalent of t_grad_out

for (int i = nLayers - 1; i >= 0; i--) {
  auto t_in = saved_tensors[fwd_i];
  auto t_wt = saved_tensors[fwd_i + 1];
  auto t_act = saved_tensors[fwd_i + 2];

  auto in_sizes = t_in.sizes();
  auto wt_sizes = t_wt.sizes();
  auto N1 = in_sizes[0];
  auto C1 = in_sizes[1];
  auto N2 = in_sizes[2];
  auto C2 = in_sizes[3];
  auto K1 = wt_sizes[0];
  auto K2 = wt_sizes[3];
  auto padded_K2 = K2; // pad K2 if it is odd dim
  auto t_wt_TV = wt_tensor_for_bwd(K1, K2, C1, C2, t_wt);

  if (t_wt_TV.numel() != t_wt.numel()) {
    padded_K2 = t_wt_TV.size(1) * t_wt_TV.size(3);
  }
  t_grad_out = t_grad_act.new_empty({N1, K1, N2, padded_K2});
  auto t_in_T = t_in;

  const int VBS = get_vnni_block_size<Tin>(); // TODO Tin or Tout?
  const auto grad_wt_flag =
      (t_wt.dim() == 5 ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_NONE_TPP);
  const auto input_trans_flag =
      (t_in.dtype() == at::kFloat ? XformTPP::XFORM_XPOSE_TPP
                                  : XformTPP::XFORM_NONE_TPP);

  if (input_trans_flag == XformTPP::XFORM_NONE_TPP) {
    t_in_T = act_tensor_trans(N1, C1, N2, C2, t_in);
  }
  if (t_grad_out.dtype() != at::kFloat) {
    TPP_ASSERT(N2 % VBS == 0, "mini batch, N2, need to be VNNI aligned");
    t_grad_out_V = t_grad_out.new_empty(
        {N1, K1, N2 / VBS, K2, VBS}); // N2 is nBatch and VNNI aligned
  }

  auto t_grad_in = at::empty_like(t_in);
  auto t_grad_wt = at::empty_like(t_wt);
  auto t_grad_bias = t_wt.new_empty({K1 * K2});

  auto in_T = GetVLAPtr<Tin>(t_in_T, {C1, C2 * N2});
  auto wt_TV = GetVLAPtr<Tin>(t_wt_TV, {C1, C2 * padded_K2});
  auto act = GetVLAPtr<Tin>(t_act, {K1, N2 * K2});
  auto grad_in = GetVLAPtr<Tout>(t_grad_in, {C1, N2 * C2});
  auto grad_wt = GetVLAPtr<Tout>(t_grad_wt, {C1, C2 * K2});
  auto grad_bias = GetVLAPtr<Tout>(t_grad_bias, {K2});
  auto grad_act = GetVLAPtr<Tin>(t_grad_act, {K1, N2 * K2});
  auto grad_out = GetVLAPtr<Tin>(t_grad_out, {K1, N2 * padded_K2});
  auto grad_out_V = GetVLAPtr<Tin>(t_grad_out_V, {K1, N2 * K2});

  auto K1b = K1;
  // if (K1 > C1 && K1 % C1 == 0) {
  //  K1b = C1;
  //}

  auto grad_bias_tpp = SCOPEIT(GradBiasTPP<Tin>(N2, K2), BIAS);
  // to compute grad_bias of next layer
  auto grad_bias_n_tpp = SCOPEIT(GradBiasTPP<Tin>(N2, C2), BIAS);
  auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(K1 * K2), EW_ZERO);
  auto n2v_tpp =
      SCOPEIT(XformExtTPP<Tin>(N2, K2, XformTPP::XFORM_N2V_TPP, true), VNNI);
  auto pad_tpp = SCOPEIT(PadTPP<Tin>(N2, K2, N2, padded_K2), ACT);

  auto grad_sigmoid_tpp =
      SCOPEIT((SigmoidBwdTPP<Tin, Tout>(N2, K2, K2, K2)), ACT);
  // to compute grad_sigmoid of next layer
  auto grad_sigmoid_n_tpp =
      SCOPEIT((SigmoidBwdTPP<Tin, Tout>(N2, C2, C2, C2)), ACT);
  // to compute grad_input
  auto di_gemm_b1_tpp = SCOPEITGEMM((BrgemmExtTPP<Tin, Tout>(
      N2,
      C2,
      K2,
      N2 * K2,
      C1 * K2 * C2,
      1.0,
      XformTPP::XFORM_NONE_TPP,
      0,
      K1b)));
  // to compute grad_weight
  auto dw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<Tin, Tout>(
      C2,
      K2,
      N2,
      C1 * N2 * C2,
      K1 * N2 * K2,
      1.0,
      (XformTPP::XFORM_TYPE)grad_wt_flag,
      input_trans_flag,
      N1)));

  // for last layer compute grad_bias and grad_sigmoid (grad_out)
  if (i == nLayers - 1) // last layer
  {
    RECORD_SCOPE(d_bias, {t_grad_out});
    tensor_set_zero(N1 * K1, N2 * K2, t_grad_out);
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

#pragma omp for collapse(2)
        for (int n1 = 0; n1 < N1; n1++) {
          for (int k1 = 0; k1 < K1; k1++) {
            Tin tmp[N2 * K2];
            grad_sigmoid_tpp(
                grad_act[n1][k1], act[n1][k1], tmp /*grad_out[n1][k1]*/);
            grad_bias_tpp(tmp /*grad_out[n1][k1]*/, prv_grad_bias[k1]);
            n2v_tpp(tmp /*grad_out[n1][k1]*/, grad_out_V[n1][k1]);
            pad_tpp(tmp, grad_out[n1][k1]);
          }
        }
#pragma omp barrier
        omp_reduce_buf(num_threads, K1 * K2, bias_ptrs, grad_bias[0]);
      }
    }
    ret_list[rb_i] = t_grad_bias;
    rb_i -= 2;
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
    ret_list[rw_i] = t_grad_wt;
    rw_i -= 2;
  }

  // compute grad_input for current layer and fuse grad_sigmoid and grad_bias
  // computation for the next layer
  if (i > 0) // for all layers except layer=0
  {
    RECORD_SCOPE(di_fused_gemm, {t_grad_out, t_wt_TV});
    // temp tensors for computing grad_out and grad_bias for next layer
    auto t_grad_bias_n = t_wt.new_empty({C1 * C2});
    auto t_act_n = saved_tensors[fwd_i + 2 - 3]; // t_act of prev layer
    auto t_grad_out_n = at::empty_like(t_grad_in);
    auto t_grad_out_V_n = t_grad_out_n;

    if (t_grad_out_n.dtype() != at::kFloat) {
      t_grad_out_V_n = t_grad_out_n.new_empty({N1, C1, N2 / VBS, C2, VBS});
    }

    auto act_n = GetVLAPtr<Tin>(t_act_n, {C1, N2 * C2});
    auto grad_bias_n = GetVLAPtr<Tout>(t_grad_bias_n, {C2});
    auto grad_out_n = GetVLAPtr<Tout>(t_grad_out_n, {C1, N2 * C2});
    auto grad_out_V_n = GetVLAPtr<Tin>(t_grad_out_V_n, {C1, N2 * C2});

    tensor_set_zero(N1 * C1, N2 * C2, t_grad_out_n);
    tensor_set_zero(N1 * C1, N2 * C2, t_grad_out_V_n);
    tensor_set_zero(C1, C2, t_grad_bias_n);
    tensor_set_zero(N1 * C1, N2 * C2, t_grad_in);

    int num_threads = omp_get_max_threads();
    float* bias_ptrs[num_threads];

#pragma omp parallel num_threads(num_threads)
    {
      int tid = omp_get_thread_num();
      float prv_grad_bias[C1][C2];
      bias_ptrs[tid] = prv_grad_bias[0];
      set_zero_tpp(prv_grad_bias[0]);

      for (int k1 = 0; k1 < K1; k1 += K1b) {
#pragma omp for collapse(2)
        for (int c1 = 0; c1 < C1; c1++) {
          for (int n1 = 0; n1 < N1; n1++) {
            // for (int c1 = 0; c1 < C1; c1++) {
            di_gemm_b1_tpp(
                grad_out[n1][k1], wt_TV[k1][c1], grad_in[n1][c1], K1b);
            if (k1 + K1b >= K1) {
              grad_sigmoid_n_tpp(
                  grad_in[n1][c1], act_n[n1][c1], grad_out_n[n1][c1]);
              grad_bias_n_tpp(grad_out_n[n1][c1], prv_grad_bias[c1]);
              n2v_tpp(grad_out_n[n1][c1], grad_out_V_n[n1][c1]);
            }
          }
        }
      }
#pragma omp barrier
      omp_reduce_buf(num_threads, C1 * C2, bias_ptrs, grad_bias_n[0]);
    }
    ret_list[rb_i] = t_grad_bias_n;
    rb_i -= 2;
    t_grad_out = t_grad_out_n;
    t_grad_out_V = t_grad_out_V_n;
  } else // layer 0
  {
    RECORD_SCOPE(di_0_gemm, {t_grad_out, t_wt_TV});
    tensor_set_zero(N1 * C1, N2 * C2, t_grad_in);
    for (int k1 = 0; k1 < K1; k1 += K1b) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int c1 = 0; c1 < C1; c1++) {
        for (int n1 = 0; n1 < N1; n1++) {
          // for (int c1 = 0; c1 < C1; c1++) {
          di_gemm_b1_tpp(grad_out[n1][k1], wt_TV[k1][c1], grad_in[n1][c1], K1b);
        }
      }
    }
    ret_list[0] = t_grad_in;
  }
  t_grad_act = t_grad_in;
  fwd_i -= 3;
} // for loop over layers
