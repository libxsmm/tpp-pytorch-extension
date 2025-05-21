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
from typing import List, Tuple, Dict, Any
from tpp_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
    get_blocking_signature,
)
from tpp_pytorch_extension.utils.xsmm import get_vnni_blocking
from tpp_pytorch_extension._C import _fused_llm_infer as fused_llm_cpp
from tpp_pytorch_extension._C import _qtype as qtype
import tpp_pytorch_extension
import time
from contextlib import contextmanager
from typing import Optional, Tuple, Union
import numpy as np
import os
import time
import inspect

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (
    GenerationMixin,
)
from transformers.cache_utils import Cache, DynamicCache
import tpp_pytorch_extension as tpx

USE_LOW_PREC_PARAMS = True
LAYER_NORM_USE_FP32_PARAMS = True
global_layer_dtype = torch.float32

tensor_parallel_enabled = True


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


def print_line_info():
    cf = inspect.currentframe().f_back
    ln = cf.f_lineno
    fn = cf.f_code.co_name
    pfn = cf.f_back.f_code.co_name
    pf = cf.f_back.f_code.co_filename
    pfl = cf.f_back.f_lineno
    print(f"Running At {fn}():{ln}  from {pf}:{pfl}:{pfn}()")


class TppCache(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    """

    def __init__(self, **kwargs) -> None:
        print_line_info()
        self.tpp_cache = torch.classes.tpp_llm.TppCache()

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        # print_line_info()
        return self.tpp_cache.size()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        print_line_info()
        # Update the cache
        ret = self.tpp_cache.update(key_states, value_states, layer_idx)
        return ret

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # print_line_info()
        return self.tpp_cache.get_seq_length(layer_idx)

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        print_line_info()
        return None

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        seq_len = self.get_seq_length()
        if max_length < 0:
            max_length = seq_len - abs(max_length)

        if seq_len <= max_length:
            return

        print(f"seq_len = {seq_len}, max_length = {max_length} size = {len(self)}")
        self.tpp_cache.crop(max_length)

    def align_and_invert_mask(
        self, mask: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
        else:
            mask = mask.to(inputs_embeds.dtype)

        min_val = torch.finfo(inputs_embeds.dtype).min
        return self.tpp_cache.align_and_invert_mask(mask, min_val)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        # print_line_info()
        self.tpp_cache.reorder_cache(beam_idx.to(torch.long))

    def get_cpp(self):
        return self.tpp_cache


def generate_past_key_values(model, batch_size, seq_len, num_beams=1):
    return TppCache()


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
        if isinstance(kwargs["past_key_values"], Cache):
            first_token = (
                True if kwargs["past_key_values"].get_seq_length() == 0 else False
            )
        if (
            kwargs["past_key_values"] is None
            and self._default.config.use_cache
            and not self.default_kv
        ):
            kwargs["past_key_values"] = TppCache()
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


def prepare_jit_inputs(inputs, model, tokenizer, num_beams):
    batch_size = len(inputs)
    dummy_input = tokenizer.batch_encode_plus(inputs, return_tensors="pt")
    dummy_input = dummy_input.to(model.device)
    if model.config.use_cache:
        dummy_input["past_key_values"] = generate_past_key_values(
            model,
            batch_size,
            1,
            num_beams=num_beams,
        )
    return dummy_input


def jit_trace_model(
    model,
    tokenizer,
    num_beams,
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
        if hasattr(self.weight, "block"):
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
    elif isinstance(weight_dtype, str):
        if weight_dtype not in ["mxfp4", "qint8"]:
            try:
                weight_dtype = getattr(torch, weight_dtype)
            except:
                raise ValueError(f"Unknown weight_dtype {weight_dtype}")
    else:
        if not isinstance(weight_dtype, torch.dtype):
            raise ValueError(
                f"weight_dtype must be either str or torch.dtype but is {type(weight_dtype)}"
            )

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
    block(self)
    if isinstance(weight_dtype, str):
        if weight_dtype == "mxfp4":
            self.weight = torch.nn.Parameter(
                qtype.remap_and_quantize_mxfp4(self.weight), requires_grad=False
            )
        elif weight_dtype == "qint8":
            self.weight = torch.nn.Parameter(
                qtype.remap_and_quantize_qint8(self.weight), requires_grad=False
            )
    else:
        self.weight = torch.nn.Parameter(
            self.weight.data.to(weight_dtype), requires_grad=False
        )


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
