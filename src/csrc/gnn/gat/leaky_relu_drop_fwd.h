/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("leaky_relu_drop_fwd", std::vector<c10::IValue>());

auto t_in = inp;
at::Tensor t_lrelu_mask;
at::Tensor t_dp_mask;

auto N = t_in.numel();
int dN = (N + 15) / 16;
auto t_out = at::empty_like(t_in);
auto out = t_out.data_ptr<T>();

t_lrelu_mask = at::empty({dN}, at::kShort);
t_dp_mask = at::empty({dN}, at::kShort);
auto in = t_in.data_ptr<T>();
auto lrelu_mask = t_lrelu_mask.data_ptr<short>();
auto dp_mask = t_dp_mask.data_ptr<short>();
const int BS = 256; // Define the block size

if (training) {
  auto leaky_relu_fwd_tpp = SCOPEIT(LeakyReLUFwdTPP<T>(BS, alpha), ACT);
  auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(BS, p), DROPOUT);
  {
    RECORD_SCOPE(go_lrelu_drop, {t_in});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      long n;
      /* The omp parallel loop will run for all the blocks of N except the last
         block by using lastprivate() the ALIGNDOWN() it takes care of the
         blocks for the 1D tensor*/
#pragma omp parallel for lastprivate(n)
      for (n = 0; n < ALIGNDOWN(N, BS); n += BS) {
        leaky_relu_fwd_tpp(&in[n], &out[n], &lrelu_mask[n / 16]);
        dropout_fwd_tpp(
            &out[n], (void*)get_rng_state(), &out[n], &dp_mask[n / 16]);
      }
      // The reminder part is handled here
      if (n < N) {
        auto leaky_relu_fwd_tpp =
            SCOPEIT(LeakyReLUFwdTPP<T>(N - n, alpha), ACT);
        auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(N - n, p), DROPOUT);
        leaky_relu_fwd_tpp(&in[n], &out[n], &lrelu_mask[n / 16]);
        dropout_fwd_tpp(
            &out[n], (void*)get_rng_state(), &out[n], &dp_mask[n / 16]);
      }
    }
  }
} else {
  auto leaky_relu_fwd_tpp = SCOPEIT(LeakyReLUFwdTPP<T>(BS, alpha), ACT);
  long n;
#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(N, BS); n += BS) {
    leaky_relu_fwd_tpp(&in[n], &out[n], &lrelu_mask[n / 16]);
  }
  if (n < N) {
    auto leaky_relu_fwd_tpp = SCOPEIT(LeakyReLUFwdTPP<T>(N - n, alpha), ACT);
    leaky_relu_fwd_tpp(&in[n], &out[n], &lrelu_mask[n / 16]);
  }
}

return {t_out, t_lrelu_mask, t_dp_mask};
