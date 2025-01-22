
// T = int8_t
{
  RECORD_SCOPE(pln_gemm, {t_in, t_idx, t_wt});
  auto in_sizes = t_in.sizes();
  auto wt_sizes = t_wt.sizes();
  auto BS = t_idx.size(0);
  auto C = in_sizes[1];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto Ncb = Nc;
  auto BSb = 32L;
  auto rem = BS % 32;

  static_assert(std::is_same<T, int8_t>::value, "T must be int8_t\n");
  typedef int32_t BrTout;
  TPP_ASSERT(
      t_in.is_quantized() && t_in.qscheme() == at::kPerBlockAffine,
      "Input not quantized\n");
  TPP_ASSERT(
      t_wt.is_quantized() && t_wt.qscheme() == at::kPerBlockAffine,
      "Weight not quantized\n");
  auto constexpr pack_size = 1;
  long block_size = q_per_block_block_size(t_wt);
  TPP_ASSERT(block_size % Hc == 0, "Block size mismatch\n");
  long n_Hc_blocks = block_size / Hc;
  long ScNc = Nc / n_Hc_blocks;
  auto wt_V = GetVLAPtr<T>(t_wt, {Nc, (Hc * Hk) / pack_size});
  auto t_w_scf = q_per_block_scales(t_wt);
  auto t_i_scf = q_per_block_scales(t_in);
  auto w_scf = GetVLAPtr<float>(t_w_scf, {ScNc, Hk});
  auto i_scf = GetVLAPtr<float>(t_i_scf, {ScNc});
  auto b_vnni = 1;

  auto in = GetVLAPtr<T>(t_in, {C});
  auto idx = GetVLAPtr<Tidx>(t_idx);
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  auto gather_tpp = SCOPEIT((EmbeddingFwdTPP<T, Tidx, T>(BSb, C, C, C)), ROW_GT);
  auto gather_tpp_rem = SCOPEIT((EmbeddingFwdTPP<T, Tidx, T>(rem, C, C, C)), ROW_GT);
  auto gather_scf_tpp = SCOPEIT((EmbeddingFwdTPP<float, Tidx, float>(BSb, ScNc, ScNc, ScNc)), ROW_GT);
  auto gather_scf_tpp_rem = SCOPEIT((EmbeddingFwdTPP<float, Tidx, float>(rem, ScNc, ScNc, ScNc)), ROW_GT);

  auto brgemm_tpp = SCOPEIT((BrgemmTPP<T, BrTout, T>(
        BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 0.0, 0, 1, b_vnni)));
  auto brgemm_tpp_rem = SCOPEIT((BrgemmTPP<T, BrTout, T>(
        rem, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 0.0, 0, 1, b_vnni)));
  auto dequant_acc = SCOPEIT(
        (DequantTPP<BrTout, Tout, float>(BSb, Hk, Hk, K, ScNc)), EW_RCP);
  auto dequant_acc_rem = SCOPEIT(
        (DequantTPP<BrTout, Tout, float>(rem, Hk, Hk, K, ScNc)), EW_RCP);
  {
    RECORD_OMP_TIME();
    auto loop_scheme = "A";
    auto ogemm_loop = ThreadedLoop<1>(
        {LoopSpecs{0L, BS, BSb}},
        loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int s1 = ind[0];
          auto count = Nc;
          T tmp[BSb][Nc][Hc];
          float tmp_scf[BSb][ScNc];
          BrTout tmp_out[BSb * Hk];
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            gather_tpp(in[0], &idx[s1], tmp[0][0]);
            gather_scf_tpp(i_scf[0], &idx[s1], tmp_scf[0]);
            for (int nk = 0; nk < Nk; nk++) {
              zero_tpp(out[s1][nk]);
              for (int c = 0; c < Nc; c += n_Hc_blocks) {
              brgemm_tpp(tmp[0][c], wt_V[nk][c], tmp_out, n_Hc_blocks, true);
              dequant_acc(
                  tmp_out,
                  out[s1][nk],
                  &tmp_scf[0][c / n_Hc_blocks],
                  w_scf[nk][c / n_Hc_blocks]);
              }
            }
          } else {
            gather_tpp_rem(in[0], &idx[s1], tmp[0][0]);
            gather_scf_tpp_rem(i_scf[0], &idx[s1], tmp_scf[0]);
            for (int nk = 0; nk < Nk; nk++) {
              zero_tpp_rem(out[s1][nk]);
              for (int c = 0; c < Nc; c += n_Hc_blocks) {
                brgemm_tpp_rem(tmp[0][c], wt_V[nk][c], tmp_out, 1, false);
                dequant_acc_rem(
                    tmp_out,
                    out[s1][nk],
                    &tmp_scf[0][c / n_Hc_blocks],
                    w_scf[nk][c / n_Hc_blocks]);
              }
            }
          }
        },
      [&]() {
        TimerStart();
          brgemm_tpp.config();
        },
        [&]() {
          brgemm_tpp.release();
          TimerEnd();
        });
  }

  auto out_attn = GetVLAPtr<Tout>(t_out_attn, {H}); // N, H

  auto t_attn = t_attn_3d.view({H * F});
  auto attn = GetVLAPtr<Tout>(t_attn, {F}); // nk, bk

  auto in_attn = GetVLAPtr<Tout>(t_out, {H, F});

  auto mul_reduce_tpp = SCOPEIT((MulReduceTPP<Tout, Tout, Tout>(H, F)), EW_MUL);
  {
    RECORD_OMP_TIME();
#pragma omp parallel for
    for (int n = 0; n < BS; n++) {
      mul_reduce_tpp(attn[0], in_attn[n][0], out_attn[n]);
    }
  }
}
