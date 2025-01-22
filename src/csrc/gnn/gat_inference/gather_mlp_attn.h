
{
  RECORD_SCOPE(pln_gemm, {t_in, t_idx, t_wt});
  auto in_sizes = t_in.sizes();
  auto C = in_sizes[1];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto Ncb = Nc;
  auto BSb = 32L;
  auto rem = BS % 32;

  auto in = GetVLAPtr<T>(t_in, {C});
  auto idx = GetVLAPtr<Tidx>(t_idx);
  auto wt_V = GetVLAPtr<T>(t_wt, {Nc, Hc * Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto gather_tpp = SCOPEIT((EmbeddingFwdTPP<T, Tidx, T>(BSb, C, C, C)), ROW_GT);
  auto gather_tpp_rem = SCOPEIT((EmbeddingFwdTPP<T, Tidx, T>(rem, C, C, C)), ROW_GT);
  auto brgemm_tpp = SCOPEIT(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEIT(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));

  {
    RECORD_OMP_TIME();
    auto loop_scheme = "aB";
    auto ogemm_loop = ThreadedLoop<2>(
        {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{0L, BS, BSb}},
        loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          T tmp[BSb][Nc][Hc];
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            gather_tpp(in[0], &idx[s1], tmp[0][0]);
            for(int nk=0; nk < Nk; nk++) {
              if (nc == 0)
                zero_tpp(out[s1][nk]);
              brgemm_tpp(tmp[0][0], wt_V[nk][nc], out[s1][nk], count, true);
            }
          } else {
            gather_tpp_rem(in[0], &idx[s1], tmp[0][0]);
            for(int nk=0; nk < Nk; nk++)
               if (nc == 0) {
                 zero_tpp_rem(out[s1][nk]);
               brgemm_tpp_rem(tmp[0][0], wt_V[nk][nc], out[s1][nk], count, false);
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
  auto out_attn = GetVLAPtr<T>(t_out_attn, {H}); // N, H

  auto t_attn = t_attn_3d.view({H * F});
  auto attn = GetVLAPtr<T>(t_attn, {F}); // nk, bk

  auto in_attn = GetVLAPtr<T>(t_out, {H, F});

  auto mul_reduce_tpp = SCOPEIT((MulReduceTPP<T, T, T>(H, F)), EW_MUL);
  {
    RECORD_OMP_TIME();
#pragma omp parallel for
    for (int n = 0; n < BS; n++) {
      mul_reduce_tpp(attn[0], in_attn[n][0], out_attn[n]);
    }
  }
}
