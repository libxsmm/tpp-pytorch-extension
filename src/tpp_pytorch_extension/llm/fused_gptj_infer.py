###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

import math
import torch
from torch import nn
from tpp_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
    get_blocking_signature,
)
from tpp_pytorch_extension.utils.xsmm import get_vnni_blocking
from tpp_pytorch_extension._C import _fused_gptj_infer as fused_gptj_cpp
import time
from contextlib import contextmanager
from typing import Optional, Tuple, Union
import transformers
from transformers.utils import is_torch_fx_proxy
import numpy as np

USE_LOW_PREC_PARAMS = True
LAYER_NORM_USE_FP32_PARAMS = True
global_layer_dtype = torch.float32
unpad = True
print_cou = 0


def compare(ref, opt, name=""):
    ref = ref.detach()
    opt = opt.detach()
    allclose = ref.allclose(opt, atol=1e-6, rtol=1e-6)
    # print(f"{name}: ref: {ref.abs().mean():14g} allclose: {allclose}  shape: {ref.shape}")
    if not allclose:
        print(f"ref = {ref.view([-1])[:8]}, xsmm = {opt.view([-1])[:8]}")
        avg = ref.abs().mean()
        adiff = (ref - opt).abs()
        rdiff = adiff / avg
        err = 1e-6
        for ind, rd in np.ndenumerate(rdiff):
            if rd > err:
                print(
                    f"{ind}: ref: {ref[ind].item():.7g} opt: {opt[ind].item():.7g} diff: {adiff[ind].item():.7g}  rdiff: {rd:.7g}"
                )
                err = rd


def print_grad_hook(var, name):
    if not hasattr(var, "grad_fn"):
        return

    def register_grad(grad_input, grad_output):
        global print_cou
        print(f"TESTGRADU {name}: {var.grad_fn.name()} - {grad_input[0].abs().sum()}")
        torch.save(grad_input, "tmp_u_%d.pt" % print_cou)
        print_cou += 1

    var.grad_fn.register_hook(register_grad)


def generate_mask(attention_mask, use_unpad):
    assert not attention_mask is None, "attention_mask is None"
    B, _, _, S = attention_mask.shape
    S1, S2 = BlockedModule.default_blocking_factors(S)
    attention_mask = attention_mask.view([B, S]).clone()
    S2a = S2 if use_unpad else S
    nnz = (((attention_mask + 10000).count_nonzero(dim=-1) + (S2a - 1)) // S2a) * S2a
    # nnz = (((attention_mask+10000).count_nonzero(dim=-1) + (S - 1))//S)*S
    nnz1 = nnz.unsqueeze(dim=1).expand([-1, S])
    a = torch.arange(S).expand([B, -1])
    a1 = torch.arange(B).unsqueeze(dim=1).expand([B, S])
    msk = a < nnz1
    bmap = a1[msk].view([-1, S2])[:, 0].squeeze().contiguous()
    attention_mask = attention_mask[msk].clone()
    seq_offsets = torch.cat([torch.zeros([1]), nnz // S2]).to(torch.long)
    # seq_sqr_offsets = seq_offsets * seq_offsets
    seq_offsets = seq_offsets.cumsum(dim=0)
    # seq_sqr_offsets = seq_sqr_offsets.cumsum(dim=0)
    return msk, attention_mask, seq_offsets, bmap, S2


class PadInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, msk, padded_shape):
        ctx.save_for_backward(msk)
        output = input.new_zeros(padded_shape)

        output[msk, :] = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (msk,) = ctx.saved_tensors

        grad_input = grad_output[msk, :]
        return grad_input, None, None


class UnpadInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, msk):
        ctx.save_for_backward(msk)
        ctx.shape = input.shape

        output = input[msk, :]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (msk,) = ctx.saved_tensors

        grad_input = grad_output.new_zeros(ctx.shape)
        grad_input[msk, :] = grad_output

        return grad_input, None


class BlockedLinear(BlockedModule, torch.nn.Linear):
    def maybe_block_params(self):
        self.weight.block()
        if self.bias is not None:
            self.bias.block()

    def parallelize(self, dim, rank, size):
        if size <= 1: return
        ShardLinear(self, dim, rank, size)
        self.model_parallel = True
        self.parallel_dim = dim
        self.parallel_rank = rank
        self.parallel_size = size

    def forward(self, input):
        #if self.model_parallel == True and self.parallel_dim == 1:
        #    input = input.chunk(self.parallel_size, dim=-1)[self.parallel_rank].contiguous()
        #self.maybe_block_params()
        bias = (
            self.bias if self.bias is not None else torch.Tensor().to(self.weight.dtype)
        )
        # print("BIas:", bias.shape, bias.dtype)
        input = input.to(self.weight.dtype)
        parallel_dim = self.parallel_dim if self.model_parallel == True else -1
        ret = torch.ops.tpp_gptj.fc_plain(input, self.weight, bias, parallel_dim)
        # if self.model_parallel == True:
        #     with torch.inference_mode(False):
        #         if self.parallel_dim == 0:
        #             agret = [t.view_as(ret) for t in ret.new_empty([self.parallel_size]+list(ret.shape)).chunk(self.parallel_size)]
        #             torch.distributed.all_gather(agret, ret)
        #             ret = torch.cat(agret, dim = -1)
        #         else:
        #             torch.distributed.all_reduce(ret)
        return ret


class BlockedLayerNorm(BlockedModule, torch.nn.LayerNorm):
    def maybe_block_params(self):
        if self.elementwise_affine:
            self.weight.block()
            if self.bias is not None:
                self.bias.block()


def FixLinear(self, bk=None, bc=None, layer_dtype=global_layer_dtype, parallel_dim=None):
    if not isinstance(self, torch.nn.Linear):
        return
    if isinstance(self, BlockedLinear): return
    self.__class__ = BlockedLinear
    self.model_parallel = False
    if parallel_dim is not None:
        self.parallelize(parallel_dim, get_rank(), get_size())
    self.weight = BlockedParameter(self.weight.data)
    self.weight.set_blocking_param(
        (
            [bk, bc],
            [0, 2, 3, 1],
        )
    )
    layer_use_low_prec = layer_dtype != torch.float32
    if layer_use_low_prec == True and USE_LOW_PREC_PARAMS:
        low_prec_vnni_blocking = get_vnni_blocking(layer_dtype)
        self.weight.set_blocking_param(
            (
                [
                    bk,
                    [
                        bc // low_prec_vnni_blocking,
                        low_prec_vnni_blocking,
                    ],
                ],
                [0, 2, 3, 1, 4],
                layer_dtype,
            )
        )

    if self.bias is not None:
        self.bias = BlockedParameter(self.bias.data)
        self.bias.set_blocking_param((None, None, layer_dtype))
    


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq
    ).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


# @torch.fx.wrap
# def get_embed_positions(embed_positions, position_ids):
#     return embed_positions.to(position_ids.device).repeat(position_ids.shape[0], 1, 1)
# 
# 
# def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
#     x1 = x[:, :, :, ::2]
#     x2 = x[:, :, :, 1::2]
#     x = torch.stack((-x2, x1), dim=-1)
#     return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
# 
# 
# def apply_rotary_pos_emb(
#     tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
# ) -> torch.Tensor:
#     sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
#     cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
#     return (tensor * cos) + (rotate_every_two(tensor) * sin)
# 
# 
# class GPTJAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
# 
#         max_positions = config.max_position_embeddings
#         self.register_buffer(
#             "bias",
#             torch.tril(
#                 torch.ones((max_positions, max_positions), dtype=torch.bool)
#             ).view(1, 1, max_positions, max_positions),
#         )
#         self.register_buffer("masked_bias", torch.tensor(-1e9))
# 
#         self.attn_dropout = nn.Dropout(config.attn_pdrop)
#         self.resid_dropout = nn.Dropout(config.resid_pdrop)
# 
#         self.embed_dim = config.hidden_size
#         self.num_attention_heads = config.num_attention_heads
#         self.head_dim = self.embed_dim // self.num_attention_heads
#         if self.head_dim * self.num_attention_heads != self.embed_dim:
#             raise ValueError(
#                 f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
#                 f" `num_attention_heads`: {self.num_attention_heads})."
#             )
#         self.scale_attn = torch.sqrt(
#             torch.tensor(self.head_dim, dtype=torch.float32)
#         ).to(torch.get_default_dtype())
# 
#         self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
#         self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
#         self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
#         self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
#         self.rotary_dim = config.rotary_dim
#         pos_embd_dim = self.rotary_dim or self.embed_dim
#         self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)
# 
#     def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
#         """
#         Splits hidden dim into attn_head_size and num_attention_heads
#         """
#         new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
#         tensor = tensor.view(new_shape)
#         if rotary:
#             return tensor
#         if len(tensor.shape) == 5:
#             return tensor.permute(
#                 0, 1, 3, 2, 4
#             )  # (batch, blocks, head, block_length, head_features)
#         elif len(tensor.shape) == 4:
#             return tensor.permute(
#                 0, 2, 1, 3
#             )  # (batch, head, seq_length, head_features)
#         else:
#             raise ValueError(
#                 f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
#             )
# 
#     def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
#         """
#         Merges attn_head_size dim and num_attn_heads dim into hidden dim
#         """
#         if len(tensor.shape) == 5:
#             tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
#         elif len(tensor.shape) == 4:
#             tensor = tensor.permute(0, 2, 1, 3).contiguous()
#         else:
#             raise ValueError(
#                 f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
#             )
#         new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
#         return tensor.view(new_shape)
# 
#     def _attn(
#         self,
#         query,
#         key,
#         value,
#         attention_mask=None,
#         head_mask=None,
#     ):
#         # compute causal mask from causal mask buffer
#         query_length, key_length = query.size(-2), key.size(-2)
#         causal_mask = self.bias[
#             :, :, key_length - query_length : key_length, :key_length
#         ]
# 
#         # Keep the attention weights computation in fp32 to avoid overflow issues
#         query = query.to(torch.float32)
#         key = key.to(torch.float32)
# 
#         attn_weights = torch.matmul(query, key.transpose(-1, -2))
# 
#         mask_value = torch.finfo(attn_weights.dtype).min
#         # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
#         # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
#         mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
#             attn_weights.device
#         )
#         attn_weights = torch.where(causal_mask, attn_weights, mask_value)
# 
#         attn_weights = attn_weights / self.scale_attn
# 
#         if attention_mask is not None:
#             # Apply the attention mask
#             attn_weights = attn_weights + attention_mask
# 
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)
#         attn_weights = attn_weights.to(value.dtype)
#         attn_weights = self.attn_dropout(attn_weights)
# 
#         # Mask heads if we want to
#         if head_mask is not None:
#             attn_weights = attn_weights * head_mask
# 
#         attn_output = torch.matmul(attn_weights, value)
# 
#         return attn_output, attn_weights
# 
#     def _get_embed_positions(self, position_ids):
#         embed_positions = self.embed_positions
#         if embed_positions.device != position_ids.device:
#             embed_positions = embed_positions.to(position_ids.device)
#             self.embed_positions = embed_positions
#         return embed_positions.repeat(position_ids.shape[0], 1, 1)
# 
#     def forward(
#         self,
#         hidden_states: torch.FloatTensor,
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = False,
#         output_attentions: Optional[bool] = False,
#     ) -> Union[
#         Tuple[torch.Tensor, Tuple[torch.Tensor]],
#         Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
#     ]:
#         query = self.q_proj(hidden_states)
#         key = self.k_proj(hidden_states)
#         value = self.v_proj(hidden_states)
# 
#         query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
#         key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
#         value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)
#         # print("query:", query.shape)
#         # print("1query:", query.dtype, query[1, :2, 0, :8])
#         query1 = query.to(torch.float32).clone()
#         key1 = key.to(torch.float32).clone()
#         fused_gptj_cpp.apply_rotary_pos_emb(
#             query1,
#             self.embed_positions,
#             position_ids.contiguous(),
#             self.num_attention_heads,
#             self.head_dim,
#         )
#         fused_gptj_cpp.apply_rotary_pos_emb(
#             key1,
#             self.embed_positions,
#             position_ids.contiguous(),
#             self.num_attention_heads,
#             self.head_dim,
#         )
# 
#         #'''
#         if is_torch_fx_proxy(position_ids):
#             # The logic to conditionally copy to GPU could not be traced, so we do this
#             # every time in the torch.fx case
#             embed_positions = get_embed_positions(self.embed_positions, position_ids)
#         else:
#             embed_positions = self._get_embed_positions(position_ids)
# 
#         repeated_position_ids = position_ids.unsqueeze(-1).repeat(
#             1, 1, embed_positions.shape[-1]
#         )
#         sincos = torch.gather(embed_positions, 1, repeated_position_ids)
#         sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
# 
#         if self.rotary_dim is not None:
#             k_rot = key[:, :, :, : self.rotary_dim]
#             k_pass = key[:, :, :, self.rotary_dim :]
# 
#             q_rot = query[:, :, :, : self.rotary_dim]
#             q_pass = query[:, :, :, self.rotary_dim :]
# 
#             k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
#             q_rot = apply_rotary_pos_emb(q_rot, sin, cos)
# 
#             key = torch.cat([k_rot, k_pass], dim=-1)
#             query = torch.cat([q_rot, q_pass], dim=-1)
#         else:
#             key = apply_rotary_pos_emb(key, sin, cos)
#             query = apply_rotary_pos_emb(query, sin, cos)
# 
#         #'''
#         # print("query:", query.dtype, query[1, :2, 0, :8])
#         # print("query1:", query1.dtype, query1[1, :2, 0, :8])
#         # compare(query, query1, "Query")
#         # compare(key, key1, "Key")
#         key = key.permute(0, 2, 1, 3)
#         query = query.permute(0, 2, 1, 3)
# 
#         if layer_past is not None:
#             past_key = layer_past[0]
#             past_value = layer_past[1]
#             key = torch.cat((past_key, key), dim=-2)
#             value = torch.cat((past_value, value), dim=-2)
# 
#         if use_cache is True:
#             present = (key, value)
#         else:
#             present = None
# 
#         # compute self-attention: V x Softmax(QK^T)
#         attn_output, attn_weights = self._attn(
#             query, key, value, attention_mask, head_mask
#         )
# 
#         attn_output = self._merge_heads(
#             attn_output, self.num_attention_heads, self.head_dim
#         )
#         attn_output = self.out_proj(attn_output)
#         attn_output = self.resid_dropout(attn_output)
# 
#         outputs = (attn_output, present)
#         if output_attentions:
#             outputs += (attn_weights,)
# 
#         return outputs  # a, present, (attentions)
# 
# 
# class GPTJMLP(nn.Module):
#     def __init__(
#         self, intermediate_size, config
#     ):  # in MLP: intermediate_size= 4 * embed_dim
#         super().__init__()
#         embed_dim = config.n_embd
# 
#         self.fc_in = nn.Linear(embed_dim, intermediate_size)
#         self.fc_out = nn.Linear(intermediate_size, embed_dim)
# 
#         self.act = ACT2FN[config.activation_function]
#         self.dropout = nn.Dropout(config.resid_pdrop)
# 
#     def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
#         hidden_states = self.fc_in(hidden_states)
#         hidden_states = self.act(hidden_states)
#         hidden_states = self.fc_out(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         return hidden_states


class GPTJBlock(BlockedModule):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTJAttention(config)
        self.mlp = GPTJMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        #print("HS:", hidden_states.shape, hidden_states.device, hidden_states.dtype)
        # print("layer_past:", layer_past[0].shape if layer_past is not None else layer_past)
        # print("attention_mask:", attention_mask.shape if attention_mask is not None else attention_mask)
        # print("position_ids:", position_ids.shape if position_ids is not None else position_ids)
        #'''
        if not hasattr(self, "cpp_block"):
            raise
            params = [self.ln_1.weight, self.ln_1.bias]
            params += [
                self.attn.q_proj.weight,
                self.attn.k_proj.weight,
                self.attn.v_proj.weight,
                self.attn.out_proj.weight,
            ]
            params += [self.mlp.fc_in.weight, self.mlp.fc_in.bias]
            params += [self.mlp.fc_out.weight, self.mlp.fc_out.bias]
            if hasattr(self.attn, "embed_positions"):
                embed_positions = self.attn.embed_positions
            else:
                max_positions = self.attn.bias.size(-1)
                pos_embd_dim = self.attn.rotary_dim or self.attn.embed_dim
                embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)
            params += [embed_positions]
            #self.cpp_block = fused_gptj_cpp.GPTJBlock(
            self.cpp_block = torch.classes.tpp_gptj.GPTJBlock(
                params,
                self.ln_1.eps,
                self.attn.num_attention_heads,
                self.attn.head_dim,
                self.attn.bias.size(-1),
                self.attn.rotary_dim,
            )
            self.blocked_input_signature = get_blocking_signature("BSF", "BSF")
        orig_hidden_states = hidden_states
        S = hidden_states.size(-2)
        hidden_states = self.get_blocked_tensor(hidden_states, self.blocked_input_signature, [None, None, self.features_block_size])
        inputs = [hidden_states]
        dummy_tensor = torch.Tensor().to(self.layer_dtype)

        def add_tensor_or_empty(t):
            inputs.append(t.contiguous() if t is not None else dummy_tensor)

        if layer_past is not None:
            #print("KP: ", layer_past[0].shape)
            #print("VP: ", layer_past[1].shape)
            add_tensor_or_empty(layer_past[0])
            add_tensor_or_empty(layer_past[1])
            # add_tensor_or_empty(self.get_blocked_tensor(layer_past[0], self.blocked_input_signature, [None, None, self.features_block_size]))
            # add_tensor_or_empty(self.get_blocked_tensor(layer_past[1], self.blocked_input_signature, [None, None, self.features_block_size]))
        else:
            inputs += [dummy_tensor, dummy_tensor]
        add_tensor_or_empty(attention_mask)
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            offset = 0
            if layer_past is not None:
                offset = layer_past[0].shape[1]
            position_ids = torch.arange(offset, offset+seq_len).repeat(hidden_states.shape[0], 1)

        add_tensor_or_empty(position_ids)
        inputs = [
            i.to(self.layer_dtype) if i.is_floating_point() else i for i in inputs
        ]
        #print("AM: ", inputs[-2].shape, inputs[-2].dtype)

        # print("PHS:", hidden_states.shape)
        hs, k, v = self.cpp_block.forward(inputs, use_cache)
        #print("K: ", k.shape)
        #print("V: ", v.shape)

        hs = BlockedTensor(hs, self.blocked_input_signature, orig_hidden_states.dtype)
        # k = BlockedTensor(k, self.blocked_input_signature).unblocked_tensor()
        # v = BlockedTensor(v, self.blocked_input_signature).unblocked_tensor()

        if use_cache:
            outputs = (hs, (k, v))
        else:
            outputs = (hs,)
        #'''

        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        #feed_forward_hidden_states = self.mlp(hidden_states)
        #hidden_states = attn_output + feed_forward_hidden_states + residual
        x = fused_gptj_cpp.fc_in(hidden_states, self.mlp.fc_in.weight, self.mlp.fc_in.bias)
        hidden_states = fused_gptj_cpp.fc_out(x, attn_output, residual, self.mlp.fc_out.weight, self.mlp.fc_out.bias)


        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        """

        return outputs  # hidden_states, present, (attentions)

def ShardLinear(m, dim, rank, size):
    # dim = 0 - shard output features
    # dim = 1 - shard input features
    m.weight.data = torch.chunk(m.weight.data, size, dim)[rank].contiguous()
    if m.weight.is_meta:
        m.weight = torch.nn.Parameter(torch.empty_like(m.weight.data, device='cpu'))
    if m.bias is not None:
        if dim == 0:
            m.bias.data = torch.chunk(m.bias.data, size, dim)[rank].contiguous()
        else:
            m.bias.data = m.bias.data / size
        if m.bias.is_meta:
            m.bias = torch.nn.Parameter(torch.empty_like(m.bias.data, device='cpu'))

def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank

def get_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        size = torch.distributed.get_world_size()
    else:
        size = 1
    return size

def all_reduce(t):
    with torch.autograd.profiler.record_function("allreduce"):
        torch.distributed.all_reduce(t)

def set_pg():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        fused_gptj_cpp.set_pg(torch.distributed.distributed_c10d._get_default_group())

def FixGPTJBlock(self, bk=None, bc=None, layer_dtype=global_layer_dtype):
    if not isinstance(self, transformers.models.gptj.modeling_gptj.GPTJBlock):
        return
    self.__class__ = GPTJBlock
    self.features_block_size = bc
    self.layer_dtype = layer_dtype
    rank = get_rank()
    wsize = get_size()
    if wsize > 1:
        ShardLinear(self.attn.q_proj, 0, rank, wsize)
        ShardLinear(self.attn.k_proj, 0, rank, wsize)
        ShardLinear(self.attn.v_proj, 0, rank, wsize)
        ShardLinear(self.attn.out_proj, 1, rank, wsize)
        ShardLinear(self.mlp.fc_in, 0, rank, wsize)
        ShardLinear(self.mlp.fc_out, 1, rank, wsize)
        self.model_parallel = True
    else:
        self.model_parallel = False
    for m in self.modules():
        for name in m._parameters.keys():
            if m._parameters[name] is None or not m._parameters[name].is_meta: continue
            param_cls = type(m._parameters[name])
            kwargs = m._parameters[name].__dict__
            m._parameters[name] = param_cls(torch.empty_like(m._parameters[name], device='cpu'), **kwargs)

        #if isinstance(m, transformers.models.gptj.modeling_gptj.GPTJAttention):
        #    m.__class__ = GPTJAttention
        if isinstance(m, torch.nn.Linear):
            FixLinear(m, bk, bc, layer_dtype)
    block(self)
    if not hasattr(self, "cpp_block"):
        params = [self.ln_1.weight, self.ln_1.bias]
        params += [
            self.attn.q_proj.weight,
            self.attn.k_proj.weight,
            self.attn.v_proj.weight,
            self.attn.out_proj.weight,
        ]
        params += [self.mlp.fc_in.weight, self.mlp.fc_in.bias]
        params += [self.mlp.fc_out.weight, self.mlp.fc_out.bias]
        if hasattr(self.attn, "embed_positions"):
            embed_positions = self.attn.embed_positions
        else:
            max_positions = self.attn.bias.size(-1)
            pos_embd_dim = self.attn.rotary_dim or self.attn.embed_dim
            embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)
        params += [embed_positions]
        self.cpp_block = torch.classes.tpp_gptj.GPTJBlock(
            params,
            self.ln_1.eps,
            self.attn.num_attention_heads // wsize,
            self.attn.head_dim,
            self.attn.bias.size(-1),
            self.attn.rotary_dim,
        )
        self.blocked_input_signature = get_blocking_signature("BSF", "BSF")

def OptimizeModelForGPTJ(model, dtype, device='cpu'):
    set_pg()

    for m in model.modules():
        if isinstance(m, transformers.models.gptj.modeling_gptj.GPTJBlock):
            FixGPTJBlock(m, 16, 64, dtype)
        elif isinstance(m, torch.nn.Linear):
            FixLinear(m, 100, 64, dtype, parallel_dim=0)
            block(m)
    for m in model.modules():
        for name in m._parameters.keys():
            if m._parameters[name] is None or not m._parameters[name].is_meta: continue
            param_cls = type(m._parameters[name])
            kwargs = m._parameters[name].__dict__
            m._parameters[name] = param_cls(torch.empty_like(m._parameters[name], device=device), **kwargs)


def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
    """
    This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
    [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
    beam_idx at every generation step.
    """
    ret = fused_gptj_cpp.reorder_cache(past, beam_idx)
    return tuple(
        tuple(p for p in layer_past) for layer_past in ret
    )
    # return tuple(
    #     tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
    #     for layer_past in past
    # )

transformers.models.gptj.modeling_gptj.GPTJForCausalLM._reorder_cache = staticmethod(_reorder_cache)

# bm_default_blocking_factors = BlockedModule.default_blocking_factors
# @staticmethod
# def custom_blocking_factors(S):
#     print(f"S = {S}")
#     if S % 32 == 0: return [S//32, 32]
#     return bm_default_blocking_factors
# BlockedModule.default_blocking_factors = custom_blocking_factors

try:
    import transformers

    transformers_orig_is_tensor = transformers.file_utils.is_tensor

    def is_tensor(x):
        """Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`."""
        if transformers_orig_is_tensor(x):
            return True
        if isinstance(x, BlockedTensor):
            return True
        return False

    transformers.file_utils.is_tensor = is_tensor
except:
    pass


@contextmanager
def tpp_impl(enable=True, use_low_prec=False, use_bf8=False):
    global global_layer_dtype
    try:
        import transformers

        orig_GPTJBlock = transformers.models.gptj.modeling_gptj.GPTJBlock
        orig_global_layer_dtype = global_layer_dtype
        try:
            if enable:
                transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock
                if use_low_prec:
                    global_layer_dtype = (
                        torch.bfloat8 if use_bf8 == True else torch.bfloat16
                    )
            yield
        finally:
            transformers.models.gptj.modeling_gptj.GPTJBlock = orig_GPTJBlock
            global_layer_dtype = orig_global_layer_dtype
    except ImportError as e:
        pass


def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()
