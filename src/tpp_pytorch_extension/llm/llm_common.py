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
from tpp_pytorch_extension._C import _fused_llm_infer as fused_llm_cpp
import time
from contextlib import contextmanager
from typing import Optional, Tuple, Union
import numpy as np
import os
import time
import inspect
from tpp_pytorch_extension._C import _fused_llm_infer as fused_llm_cpp

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (
    GenerationMixin,
)
import tpp_pytorch_extension as tpx

USE_LOW_PREC_PARAMS = True
LAYER_NORM_USE_FP32_PARAMS = True
global_layer_dtype = torch.float32

BATCH_DIM_IN_KV_CACHE = fused_llm_cpp.get_batch_dim_in_kv_cache()

tensor_parallel_enabled = True

use_discrete_kv = int(os.environ.get("USE_DISCRETE_KV_CACHE", 1))


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


def shm_allreduce(t):
    fused_llm_cpp.allreduce(t)


def sparse_model_config(model_config):
    embedding_size = None
    if hasattr(model_config, "hidden_size"):
        embedding_size = model_config.hidden_size
    elif hasattr(model_config, "n_embed"):
        embedding_size = model_config.n_embed
    elif hasattr(model_config, "n_embd"):
        embedding_size = model_config.n_embd

    num_head = None
    if hasattr(model_config, "num_attention_heads"):
        num_head = model_config.num_attention_heads
    elif hasattr(model_config, "n_head"):
        num_head = model_config.n_head

    if embedding_size is None or num_head is None or num_head == 0:
        raise ValueError("Check the model config")

    num_embedding_size_per_head = int(embedding_size / num_head)
    if hasattr(model_config, "n_layer"):
        num_layer = model_config.n_layer
    elif hasattr(model_config, "num_hidden_layers"):
        num_layer = model_config.num_hidden_layers
    else:
        raise ValueError(
            "Number of hidden layers couldn't be determined from the model config"
        )

    return num_layer, num_head, num_embedding_size_per_head


def generate_past_key_values(model, batch_size, seq_len, num_beams=1, indirect_kv=True):
    (
        num_block_layers,
        num_attention_heads,
        num_embedding_size_per_head,
    ) = sparse_model_config(model.config)
    if model.config.model_type == "bloom":
        past_key_values = tuple(
            (
                torch.empty(
                    int(num_attention_heads * batch_size),
                    num_embedding_size_per_head,
                    seq_len,
                )
                .to(model.dtype)
                .to(model.device),
                torch.empty(
                    int(num_attention_heads * batch_size),
                    seq_len,
                    num_embedding_size_per_head,
                )
                .to(model.dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    else:
        if indirect_kv == True:
            num_tensors = 7 if num_beams > 1 else 6
            past_key_values = tuple(
                (
                    torch.empty(
                        batch_size, num_attention_heads, 0, num_embedding_size_per_head
                    )
                    .to(model.dtype)
                    .to(model.device),
                    torch.empty(
                        batch_size, num_attention_heads, 0, num_embedding_size_per_head
                    )
                    .to(model.dtype)
                    .to(model.device),
                    torch.empty([seq_len, batch_size * num_beams], dtype=torch.long).to(
                        model.device
                    ),
                    torch.tensor(0),
                    torch.empty(
                        batch_size,
                        num_attention_heads,
                        seq_len,
                        num_embedding_size_per_head,
                    )
                    .to(model.dtype)
                    .to(model.device),
                    torch.empty(
                        batch_size,
                        num_attention_heads,
                        seq_len,
                        num_embedding_size_per_head,
                    )
                    .to(model.dtype)
                    .to(model.device),
                    torch.empty(batch_size, 1, dtype=torch.int64).to(model.device),
                )[:num_tensors]
                for _ in range(num_block_layers)
            )
        else:
            past_key_values = tuple(
                (
                    torch.empty(
                        batch_size,
                        num_attention_heads,
                        seq_len,
                        num_embedding_size_per_head,
                    )
                    .to(model.dtype)
                    .to(model.device),
                    torch.empty(
                        batch_size,
                        num_attention_heads,
                        seq_len,
                        num_embedding_size_per_head,
                    )
                    .to(model.dtype)
                    .to(model.device),
                    torch.zeros([batch_size * num_beams], dtype=torch.long).to(
                        model.device
                    ),
                )
                for _ in range(num_block_layers)
            )
    return past_key_values


def prepare_jit_inputs(inputs, model, tokenizer, num_beams):
    batch_size = len(inputs)
    dummy_input = tokenizer.batch_encode_plus(inputs, return_tensors="pt")
    dummy_input = dummy_input.to(model.device)
    if model.config.use_cache:
        assert (
            use_discrete_kv == 1
        ), "Using TorchScript JIT requires use_discrete_kv = 1"
        dummy_input["past_key_values"] = generate_past_key_values(
            model, batch_size, 1, num_beams=num_beams, indirect_kv=use_discrete_kv
        )
    if len(dummy_input["past_key_values"][0]) < 4:
        dummy_input["attention_mask"] = torch.cat(
            [
                torch.zeros(dummy_input["attention_mask"].shape[0], 1)
                .to(dummy_input["attention_mask"].dtype)
                .to(model.device),
                dummy_input["attention_mask"],
            ],
            -1,
        )
    return dummy_input


class _ModelFallbackWrapper(GenerationMixin):
    __slots__ = ("_optimized", "_default")

    def __init__(
        self, optimized, default, num_beams, enable_profile=False, default_kv=False
    ):
        self._optimized = optimized
        self._default = default
        self.num_beams = num_beams
        self.enable_profile = enable_profile
        self.default_kv = default_kv
        self.token_latency = None
        self.output_past_key_values = None
        self.saved_input_ids = None
        self.saved_past_key_values = None

    def __call__(self, *args, **kwargs):
        if not "past_key_values" in kwargs:
            kwargs["past_key_values"] = None
        if len(args) > 0:
            kwargs["input_ids"] = args[0]
            assert len(args) == 1, "More than 1 positional arguments not supported"
        first_token = True if kwargs["past_key_values"] is None else False
        if (
            kwargs["past_key_values"] is None
            and self._default.config.use_cache
            and not self.default_kv
        ):
            kwargs["past_key_values"] = generate_past_key_values(
                self._default,
                kwargs["input_ids"].shape[0],
                0,
                num_beams=self.num_beams,
                indirect_kv=use_discrete_kv,
            )
        # kwargs.pop("position_ids", None)
        if first_token == True and self.num_beams > 1:
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and k in [
                    "input_ids",
                    "position_ids",
                    "attention_mask",
                    "token_type_ids",
                ]:
                    kwargs[k] = kwargs[k][:: self.num_beams].contiguous()
        for k in list(kwargs.keys()):
            if kwargs[k] is None or isinstance(kwargs[k], bool):
                kwargs.pop(k)
        outputs = self._optimized(**kwargs)
        lm_logits = outputs[0]
        if first_token == True and self.num_beams > 1:
            _, ret = self._default._expand_inputs_for_generation(
                expand_size=self.num_beams, output=lm_logits
            )
            lm_logits = ret["output"]

        past_key_values = outputs[1]
        if self.output_past_key_values == True:
            self.saved_past_key_values = past_key_values
        fixed_output = CausalLMOutputWithPast(
            loss=None,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

        if self.enable_profile and first_token == True:
            tpx.print_debug_timers(detailed=False)
            tpx.reset_debug_timers()
        return fixed_output

    # @torch.no_grad()
    def generate(self, *args, **kwargs):
        self.output_past_key_values = kwargs.pop("output_past_key_values", None)
        self.token_latency = kwargs.pop("token_latency", None)
        if self.token_latency == True:
            self.token_latencies = []

        # print("inp=",kwargs["input_ids"].shape)
        output = super().generate(*args, **kwargs)
        if self.token_latency == True:
            self.token_latencies.append(time.time())
            latencies = []
            for i in range(len(self.token_latencies) - 1):
                latencies.append(self.token_latencies[i + 1] - self.token_latencies[i])
            self.token_latencies = []
            output = [
                output,
                latencies,
            ]
        if self.enable_profile:
            tpx.print_debug_timers(detailed=False)
            tpx.reset_debug_timers()

        if self.output_past_key_values == True:
            saved_input_ids = self.saved_input_ids
            saved_past_key_values = self.saved_past_key_values
            self.saved_input_ids = None
            self.saved_past_key_values = None
            output = [output, [saved_input_ids, saved_past_key_values]]
        return output

    def __getattr__(self, item):
        return getattr(self._default, item)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if self.output_past_key_values == True:
            self.saved_input_ids = input_ids
        if self.token_latency == True:
            self.token_latencies.append(time.time())

        return self._default.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return self._default._reorder_cache(past_key_values, beam_idx)


def jit_trace_model(
    model,
    tokenizer,
    num_beams,
    indirect_kv=True,
    enable_profile=False,
    only_last_logit=False,
):
    torch._C._jit_set_texpr_fuser_enabled(False)
    jit_input_texts = ["enable jit"]
    jit_inputs = prepare_jit_inputs(jit_input_texts, model, tokenizer, num_beams)
    past_key_values = jit_inputs.pop("past_key_values")
    jit_inputs = model.prepare_inputs_for_generation(**jit_inputs)
    jit_inputs["past_key_values"] = past_key_values
    model.config.return_dict = False
    model.config.only_last_logit = only_last_logit
    if hasattr(model, "forward"):
        sig = inspect.signature(model.forward)
    else:
        sig = inspect.signature(model.__call__)
    jit_inputs = tuple(
        jit_inputs[key]
        for key in sig.parameters
        if jit_inputs.get(key, None) is not None
    )
    traced_model = torch.jit.trace(model, jit_inputs, strict=False)
    traced_model = torch.jit.freeze(traced_model.eval())
    traced_model(*jit_inputs)
    traced_model(*jit_inputs)

    model = _ModelFallbackWrapper(
        traced_model, model, num_beams, enable_profile=enable_profile
    )
    return model


def optimize_for_first_token(
    model, num_beams, enable_profile=False, only_last_logit=False, default_kv=False
):
    model.config.only_last_logit = only_last_logit
    model = _ModelFallbackWrapper(
        model, model, num_beams, enable_profile=enable_profile, default_kv=default_kv
    )
    return model


def fc_plain(input, weight, bias=torch.Tensor(), parallel_dim=-1, split_sizes=[]):
    return torch.ops.tpp_llm.fc_plain(input, weight, bias, parallel_dim, split_sizes)


class BlockedLinear(BlockedModule, torch.nn.Linear):
    def maybe_block_params(self):
        self.weight.block()
        if self.bias is not None:
            self.bias.block()

    def parallelize(self, dim, rank, size, block_size=1):
        if size <= 1:
            return
        ShardLinear(self, dim, rank, size, block_size)
        self.model_parallel = True
        self.parallel_dim = dim
        self.parallel_rank = rank
        self.parallel_size = size

    def forward(self, input):
        # if self.model_parallel == True and self.parallel_dim == 1:
        #    input = input.chunk(self.parallel_size, dim=-1)[self.parallel_rank].contiguous()
        self.maybe_block_params()
        bias = (
            self.bias if self.bias is not None else torch.Tensor().to(self.weight.dtype)
        )
        # print("BIas:", bias.shape, bias.dtype)
        input = input.to(self.weight.dtype)
        parallel_dim = self.parallel_dim if self.model_parallel == True else -1
        split_sizes = self.split_sizes if hasattr(self, "split_sizes") else []
        ret = fc_plain(input, self.weight, bias, parallel_dim, split_sizes)
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


def FixLinear(
    self,
    bk=None,
    bc=None,
    layer_dtype=global_layer_dtype,
    parallel_dim=None,
    block_size=1,
    weight_dtype=None,
):
    if not isinstance(self, torch.nn.Linear):
        return
    if isinstance(self, BlockedLinear):
        return
    self.__class__ = BlockedLinear
    self.model_parallel = False
    if parallel_dim is not None:
        self.parallelize(parallel_dim, get_rank(), get_size(), block_size=block_size)
    if weight_dtype is None:
        weight_dtype = layer_dtype
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
                weight_dtype,
            )
        )

    if self.bias is not None:
        self.bias = BlockedParameter(self.bias.data)
        self.bias.set_blocking_param((None, None, layer_dtype))


def ShardLinear(m, dim, rank, size, block_size=1):
    # dim = 0 - shard output features
    # dim = 1 - shard input features
    dim_size = m.weight.shape[dim]
    assert (
        dim_size % block_size == 0
    ), f"dim_size ({dim_size}) is not multiple of block_size ({block_size})"
    num_blocks = dim_size // block_size
    split_offsets = [0] + [
        ((num_blocks * (i + 1)) // size) * block_size for i in range(size)
    ]
    start = split_offsets[rank]
    end = split_offsets[rank + 1]
    m.split_sizes = [split_offsets[i + 1] - split_offsets[i] for i in range(size)]
    # if rank == 0:
    #     print(f"ShardLinear: offsets: {split_offsets}")
    #     print(f"ShardLinear: split_sizes: {m.split_sizes}")
    if dim == 0:
        m.weight.data = m.weight.data[start:end, :].contiguous()
    else:
        m.weight.data = m.weight.data[:, start:end].contiguous()
    if m.weight.is_meta:
        m.weight = torch.nn.Parameter(torch.empty_like(m.weight.data, device="cpu"))
    if m.bias is not None:
        if dim == 0:
            m.bias.data = m.bias.data[start:end].contiguous()
        else:
            m.bias.data = m.bias.data / size
        if m.bias.is_meta:
            m.bias = torch.nn.Parameter(torch.empty_like(m.bias.data, device="cpu"))


def get_rank():
    if not tensor_parallel_enabled:
        return 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank


def get_size():
    if not tensor_parallel_enabled:
        return 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        size = torch.distributed.get_world_size()
    else:
        size = 1
    return size


def all_reduce(t):
    with torch.autograd.profiler.record_function("allreduce"):
        torch.distributed.all_reduce(t)


def set_pg():
    if not tensor_parallel_enabled:
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        fused_llm_cpp.set_pg(torch.distributed.distributed_c10d._get_default_group())


def get_layer_past_and_offset(
    layer_past: Optional[Tuple[torch.Tensor]], discrete_kv: bool
):
    if layer_past is None:
        return ([], 0)
    n = len(layer_past)
    B_DIM = BATCH_DIM_IN_KV_CACHE
    if n < 4:  # cache with beam_idx
        if discrete_kv == True and use_discrete_kv == 1:
            B1, N, S, H = layer_past[0].shape
            if n == 3:
                B2 = layer_past[2].shape[0]
            else:
                B2 = B1
            inc_size = int(os.environ.get("KV_CACHE_INC_SIZE", "128"))
            capacity = S + inc_size
            if B_DIM == 1:
                new_key = layer_past[0].new_zeros([capacity, B2, N, H])
                new_value = layer_past[1].new_zeros([capacity, B2, N, H])
            else:
                new_key = layer_past[0].new_zeros([B2, N, capacity, H])
                new_value = layer_past[1].new_zeros([B2, N, capacity, H])
            if B1 == B2:
                if B_DIM == 1:
                    new_key[:S, :, :, :].copy_(layer_past[0].permute([2, 0, 1, 3]))
                    new_value[:S, :, :, :].copy_(layer_past[1].permute([2, 0, 1, 3]))
                else:
                    new_key[:, :, :S, :].copy_(layer_past[0])
                    new_value[:, :, :S, :].copy_(layer_past[1])
                new_beam_idx = (
                    torch.arange(B2).unsqueeze(0).expand([capacity, B2]).contiguous()
                )
                if n == 3:
                    new_beam_idx[S - 1] = layer_past[2]
            else:
                assert B2 % B1 == 0, f"B1 = {B1}, B2 = {B2}"
                assert n == 3, f"Must use lazy kv cache reorder but n = {n}"
                num_beams = B2 // B1
                if B_DIM == 1:
                    new_key[:S, ::num_beams, :, :] = layer_past[0].permute([2, 0, 1, 3])
                    new_value[:S, ::num_beams, :, :] = layer_past[1].permute(
                        [2, 0, 1, 3]
                    )
                else:
                    new_key[::num_beams, :, :S, :] = layer_past[0]
                    new_value[::num_beams, :, :S, :] = layer_past[1]
                new_beam_idx = layer_past[2].new_zeros([capacity, B2]).contiguous()
                # beam_idx was adjusted in reorder_cache by num_beams, so fit it back
                new_beam_idx[:S] = layer_past[2] * num_beams
            offset = torch.tensor(S)
            if B_DIM == 1:
                key = new_key[:S, :, :, :].permute([1, 2, 0, 3])
                value = new_value[:S, :, :, :].permute([1, 2, 0, 3])
            else:
                key = new_key[:, :, :S, :]
                value = new_value[:, :, :S, :]
            layer_past = (
                key,
                value,
                new_beam_idx,
                offset,
                new_key,
                new_value,
            )
            return (layer_past, offset)
        else:
            return (layer_past, layer_past[0].shape[2])

    else:
        # B1, N, S, H = layer_past[0].shape
        # B2 = layer_past[2].shape[0]
        # print(f"pkv{n} B1: {B1}  B2: {B2} t_offset: {layer_past[3]}")
        # print(f"pkv{n} : layer_past[3]{layer_past[3]}")
        return (layer_past, layer_past[3])


def _reorder_cache(
    past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
) -> Tuple[Tuple[torch.Tensor]]:
    """
    This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
    [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
    beam_idx at every generation step.
    """
    # print(f"_reorder_cache: len(pkv) = {len(past[0])}, {past[0][0].shape}  beam_idx = {beam_idx}")
    if len(past[0]) >= 4:  # discrete kv_cache
        assert (
            len(past[0]) >= 6
        ), f"Invalid past key_value tuple length ({len(past[0])})"
        B_DIM = BATCH_DIM_IN_KV_CACHE
        B1 = past[0][4].shape[B_DIM]
        B2 = beam_idx.shape[0]
        # print(f"_reorder_cache: B1: {past[0][0].shape}, beam_idx: {beam_idx}")
        # print(f"B1 = {B1}, B2 = {B2}")
        if B1 != B2:
            assert B2 % B1 == 0, f"B1 = {B1}, B2 = {B2}"
            num_beams = B2 // B1
            tmp = (
                past[0][2]
                .repeat_interleave(num_beams, dim=1)
                .mul(num_beams)
                .contiguous()
            )
            tmp[past[0][3] - 1] = beam_idx
            remapped_ind = fused_llm_cpp.remap_indices(tmp, past[0][3])
            new_past = []
            S = past[0][0].shape[2]
            for layer_past in past:
                layer_past_4 = (
                    layer_past[4].repeat_interleave(num_beams, dim=B_DIM).contiguous()
                )
                layer_past_5 = (
                    layer_past[5].repeat_interleave(num_beams, dim=B_DIM).contiguous()
                )
                layer_past_2 = (
                    layer_past[2]
                    .repeat_interleave(num_beams, dim=1)
                    .mul(num_beams)
                    .contiguous()
                )
                layer_past_2[layer_past[3] - 1] = beam_idx
                if B_DIM == 1:
                    layer_past_0 = layer_past_4[:S].permute([1, 2, 0, 3])
                    layer_past_1 = layer_past_5[:S].permute([1, 2, 0, 3])
                else:
                    layer_past_0 = layer_past_4[:, :, :S, :]
                    layer_past_1 = layer_past_5[:, :, :S, :]
                new_past.append(
                    (
                        layer_past_0,
                        layer_past_1,
                        layer_past_2,
                        layer_past[3],
                        layer_past_4,
                        layer_past_5,
                        remapped_ind,
                    )
                )

            return tuple(new_past)
        else:
            for layer_past in past:
                layer_past[2][layer_past[3] - 1] = beam_idx
            remapped_ind = fused_llm_cpp.remap_indices(past[0][2], past[0][3])
            new_past = []
            for layer_past in past:
                l_layer_past = list(layer_past)
                if len(layer_past) == 6:
                    l_layer_past.append(remapped_ind)
                else:
                    l_layer_past[6] = remapped_ind
                new_past.append(tuple(l_layer_past))
            past = tuple(new_past)
            return past
    else:
        B1 = past[0][0].shape[0]
        B2 = beam_idx.shape[0]
        if B1 != B2:
            assert B2 % B1 == 0, f"B1 = {B1}, B2 = {B2}"
            num_beams = B2 // B1
            beam_idx = beam_idx // num_beams
            # print(f"B1 = {B1}, B2 = {B2}, beam_idx: {beam_idx}")
        return tuple(tuple(layer_past) + (beam_idx,) for layer_past in past)

    # ret = fused_llm_cpp.reorder_cache(past, beam_idx)
    # return tuple(
    #     tuple(p for p in layer_past) for layer_past in ret
    # )

    # return tuple(
    #     tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
    #     for layer_past in past
    # )


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


def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()
