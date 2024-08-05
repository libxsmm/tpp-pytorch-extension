/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("mlp_attn_scf", std::vector<c10::IValue>());

at::Tensor t_in, t_in_scf, t_attn_3d, t_wt, t_bias;
int i = 0;

t_in = inputs[i++];
t_in_scf = inputs[i++];
t_wt = inputs[i++];
t_attn_3d = inputs[i++];
if (add_bias)
  t_bias = inputs[i++];
else
  t_bias = at::empty(0, at::kBFloat16);

auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto scf_sizes = t_in_scf.sizes();
auto N = in_sizes[0];
auto blocks = scf_sizes[1];

auto nk = wt_sizes[0];
auto nc = wt_sizes[1];
auto bc = wt_sizes[2];
if (t_wt.dim() == 5)
  bc = bc * wt_sizes[4];
auto bk = wt_sizes[3];
auto bcp = bc;
auto K = nk * bk;
auto C = in_sizes[1];
auto Cb = C / blocks;

if (t_wt.dim() == 5) {
  auto lp = get_vnni_block_size(t_wt.dtype());
  auto d = bc % lp;
  bcp = d > 0 ? bc + (lp - d) : bc;
}

auto t_wt_V = wt_tensor_for_fwd(nk, bk, nc, bc, t_wt);

auto cvt_f32_bf16_tpp = SCOPEIT((ConvertTPP<float, bfloat16>(1, C)), EW_COPY);

libxsmm_meltw_unary_shape deq_shape;
deq_shape.m = Cb;
deq_shape.n = 1;
deq_shape.ldi = Cb;
deq_shape.ldo = Cb;
deq_shape.in0_type = LIBXSMM_DATATYPE_I8;
deq_shape.out_type = LIBXSMM_DATATYPE_F32;
deq_shape.comp_type = LIBXSMM_DATATYPE_F32;

libxsmm_meltwfunction_unary dequant_kernel;
dequant_kernel = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_DEQUANT, deq_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

at::Tensor t_in_tmp = at::empty({N,C}, at::kBFloat16);
auto scf = GetVLAPtr<float>(t_in_scf, {blocks});
auto in_dq = GetVLAPtr<Tact>(t_in, {blocks, Cb});
auto in_tmp = GetVLAPtr<bfloat16>(t_in_tmp, {C});

#pragma omp parallel for
for (int n=0; n < N; n++) {
  float tmp[blocks][Cb];
  libxsmm_meltw_unary_param unary_param;
  for(auto j = 0; j<blocks; j++) {
    unary_param.in.primary    = (void*)in_dq[n][j];
    unary_param.in.secondary  = (void*)&scf[n][j];
    unary_param.out.primary   = (void*)tmp[j];
    dequant_kernel( &unary_param );
  }
  cvt_f32_bf16_tpp(tmp[0], in_tmp[n]);
}

at::Tensor t_out_mlp = fc_plain<bfloat16>(t_in_tmp, t_wt_V, t_bias);

auto attn_sizes = t_attn_3d.sizes(); // 3D shape [1, H, F] = [1, 4, 128] let

auto H = attn_sizes[1]; // 4
auto F = attn_sizes[2]; // 128

auto t_out_attn = t_out_mlp.new_empty({N, H});
auto out_attn = GetVLAPtr<bfloat16>(t_out_attn, {H}); // N, H

auto t_attn = t_attn_3d.view({H * F});
auto attn = GetVLAPtr<bfloat16>(t_attn, {F}); // nk, bk

auto in_attn = GetVLAPtr<bfloat16>(t_out_mlp, {H, F});

auto mul_reduce_tpp = SCOPEIT((MulReduceTPP<bfloat16, bfloat16, bfloat16>(H, F)), EW_MUL);
{
  RECORD_SCOPE(o_attn, {t_out_attn});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      mul_reduce_tpp(attn[0], in_attn[n][0], out_attn[n]);
    }
  }
}

return {t_out_mlp, t_out_attn.view({N, H, 1})};
