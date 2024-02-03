RECORD_FUNCTION("mlp_fwd", std::vector<c10::IValue>());

// index for accessing vector of tensors from python
// vector: in0, wt0, bias0, wt1, bias1
// wb_i initialized to 1, index of wt0
int wb_i = 1;

for (int i = 0; i < nLayers; i++) {
  auto t_wt = inputs[wb_i++];
  auto t_bias = inputs[wb_i++];

  auto in_sizes = t_in.sizes();
  auto wt_sizes = t_wt.sizes();
  auto N1 = in_sizes[0];
  auto C1 = in_sizes[1];
  auto N2 = in_sizes[2];
  auto C2 = in_sizes[3];
  auto K1 = wt_sizes[0];
  auto K2 = wt_sizes[3];
  auto padded_C2 = C2;
  // auto relu_rd = (N2 * K2 + 15) / 16;

  auto t_wt_V = wt_tensor_for_fwd(K1, K2, C1, C2, t_wt);
  if (t_wt_V.numel() != t_wt.numel()) { // if padded
    padded_C2 = t_wt_V.size(2) * t_wt_V.size(4);
    t_in = get_padded_activation_for_vnni(t_in);
  }
  auto t_out = t_in.new_empty({N1, K1, N2, K2});

  auto in = GetVLAPtr<Tin>(t_in, {C1, N2 * padded_C2});
  auto wt_V = GetVLAPtr<Tin>(t_wt_V, {C1, C2 * K2});
  auto bias = GetVLAPtr<Tin>(t_bias, {K2});
  auto out = GetVLAPtr<Tout>(t_out, {K1, N2 * K2});

  auto C1b = C1;
  if (C1 > K1 && C1 % K1 == 0) {
    C1b = K1;
  }

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<Tin>(N2, K2), BIAS);
  auto brgemm_tpp = SCOPEITGEMM((BrgemmExtTPP<Tin, Tin>(
      N2, K2, C2, N2 * C2, K2 * C2, 1.0, XformTPP::XFORM_NONE_TPP, 0, C1b)));

  auto sigmoid_tpp = SCOPEIT((SigmoidFwdTPP<Tin, Tout>(N2, K2, K2, K2)), ACT);

  {
    RECORD_SCOPE(fused_gemm_act, {t_in, t_wt_V});
    for (int c1 = 0; c1 < C1; c1 += C1b) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int n1 = 0; n1 < N1; n1++) {
        for (int k1 = 0; k1 < K1; k1++) {
          if (c1 == 0) {
            copy_bias_tpp(bias[k1], out[n1][k1]);
          }
          // if k1==0
          // Tin tmp[N2*C2padded];
          // pad_tpp(in[n1][c1], tmp);
          // brgemm_tpp(tmp, wt_V.....);
          brgemm_tpp(in[n1][c1], wt_V[k1][c1], out[n1][k1], C1b);
          if (c1 + C1b >= C1)
            sigmoid_tpp(out[n1][k1], out[n1][k1]);
        }
      }
    }
  }

  save_bwd.push_back(t_in);
  save_bwd.push_back(t_wt);
  save_bwd.push_back(t_out);

  t_in = t_out;
}
