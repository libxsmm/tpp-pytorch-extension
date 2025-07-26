import torch
from typing import Dict, List, Optional, Tuple
from tpp_pytorch_extension.utils.xsmm import get_vnni_blocking
import os
import time
from functools import wraps

try:
    from vllm.model_executor.layers.linear import (
        UnquantizedLinearMethod,
        LinearBase,
        LinearMethodBase,
    )
except:
    UnquantizedLinearMethod = LinearBase = None

TPP_XGEMM = int(os.environ.get("TPP_XGEMM", "1"))
TPP_XGEMM_PREPACK_LEVEL = int(os.environ.get("TPP_XGEMM_PREPACK_LEVEL", "1"))
TPP_XGEMM_USE_FP8 = int(os.environ.get("TPP_XGEMM_USE_FP8", "0"))
try:
    if TPP_XGEMM == 1:
        import XGEMM as xgemm

        # xgemm.set_log_level("INFO")
        xgemm.set_log_level("WARNING")
        # xgemm.set_num_devices(1)
    else:
        xgemm = None
except:
    xgemm = None


def gemm_timer(func):
    @wraps(func)
    def wrapper(input, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(input, *args, **kwargs)
        end_time = time.perf_counter()
        exec_time = end_time - start_time
        M, K, N = (input.shape[0], input.shape[1], result.shape[-1])
        flops = 2 * M * N * K / exec_time / 1e12
        in_dt = input.dtype
        if M > 100:
            print(
                f"Func '{func.__name__}' took {exec_time*1e3:.3f} ms ({flops:.2f} TF/s) [{M} {N} {K} {in_dt}]."
            )
        return result

    return wrapper


@gemm_timer
def fc_xgemm(input, weight, w_trans, bias, K):
    sizes = list(input.shape)
    sizes[1] = K
    output = torch.empty(sizes, dtype=input.dtype, device=input.device)
    if TPP_XGEMM_USE_FP8 > 0:
        input = input.to(torch.float8_e4m3fn)
    if bias is None:
        if isinstance(weight, torch.Tensor):
            xgemm.xgemm(input, weight, output, 1.0, 0.0, aOp=0, bOp=w_trans)
        else:
            xgemm.xgemm_packed(input, weight, output, 1.0, 0.0, aOp=0, bOp=w_trans)
    else:
        if isinstance(weight, torch.Tensor):
            xgemm.xgemm(input, weight, output, 1.0, 0.0, aOp=0, bOp=w_trans, bias=bias)
        else:
            xgemm.xgemm_packed(
                input, weight, output, 1.0, 0.0, aOp=0, bOp=w_trans, bias=bias
            )
    # if bias:
    #     output += bias
    return output


@gemm_timer
def fc_plain(input, weight, bias=torch.Tensor(), parallel_dim=-1, split_sizes=[]):
    return torch.ops.tpp_llm.fc_plain(input, weight, bias, parallel_dim, split_sizes)


class TppLinear(torch.nn.Linear):
    def forward(self, input):
        N = input.numel() // input.shape[-1]
        if hasattr(self, "weight_2"):
            weight = self.weight_2
            if hasattr(self, "weight_1") and N > 192:
                weight = self.weight_1
        elif hasattr(self, "weight_1"):
            weight = self.weight_1
        else:
            weight = self.weight.t()
        input = input.to(weight.dtype)
        bias = (
            self.bias.to(weight.dtype)
            if self.bias is not None
            else torch.Tensor().to(weight.dtype)
        )

        ret = fc_plain(input, weight, bias)
        return ret


def BlockedWeight(weight, bk, bc, dtype):
    K, C = weight.shape
    if dtype == torch.float32:
        new_weight = (
            weight.view(K // bk, bk, C // bc, bc)
            .permute([0, 2, 3, 1])
            .to(dtype)
            .contiguous()
        )
    else:
        VBS = get_vnni_blocking(dtype)
        new_weight = (
            weight.view(K // bk, bk, C // bc, bc // VBS, VBS)
            .permute([0, 2, 3, 1, 4])
            .to(dtype)
            .contiguous()
        )
    return new_weight


def FixLinear(
    self,
    bkbc_2,
    bkbc_1=None,
    layer_dtype=None,
    weight_dtype=None,
):
    if not isinstance(self, torch.nn.Linear):
        return
    if isinstance(self, TppLinear):
        return
    self.__class__ = TppLinear
    if layer_dtype is None:
        layer_dtype = self.weight.dtype
    if weight_dtype is None:
        weight_dtype = layer_dtype
    bk, bc = bkbc_2
    self.weight_2 = BlockedWeight(self.weight.data, bk, bc, layer_dtype)
    if bkbc_1 is not None:
        bk, bc = bkbc_1
        self.weight_1 = BlockedWeight(self.weight.data, bk, bc, layer_dtype)

    if self.bias is not None:
        self.bias.data = self.bias.data.to(layer_dtype)


if LinearMethodBase:

    class TppLinearMethod(LinearMethodBase):
        """Linear method without quantization."""

        def create_weights(
            self,
            layer: torch.nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: List[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
        ):
            raise "NotImplemented"

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:

            N = x.numel() // x.shape[-1]
            if xgemm and N > 60 and hasattr(layer, "weight_t"):
                bias_t = layer.bias_t if bias else None

                ret = fc_xgemm(x, layer.weight_t, 1, bias_t, layer.weight.shape[0])
                return ret

            if layer.weight.dtype == torch.float16:
                ret = self.orig_method.apply(layer, x, bias)
                return ret

            if hasattr(layer, "weight_2"):
                weight = layer.weight_2
                if hasattr(layer, "weight_1") and N > 192:
                    weight = layer.weight_1
            elif hasattr(layer, "weight_1"):
                weight = layer.weight_1
            else:
                ret = self.orig_method.apply(layer, x, bias)
                return ret

            if weight.dtype != torch.float8_e4m3fn:
                x = x.to(weight.dtype)
                bias = (
                    bias.to(weight.dtype)
                    if bias is not None
                    else torch.Tensor().to(weight.dtype)
                )
            else:
                bias = (
                    bias.to(x.dtype) if bias is not None else torch.Tensor().to(x.dtype)
                )
            # if N > 250:
            #     print(f"TPP: {weight.dtype} {x.shape[0]}, {x.shape[1]}, {weight.shape}, {bias.shape}")
            ret = fc_plain(x, weight, bias)
            return ret


def FixLinearBase(
    self,
    bkbc_2,
    bkbc_1=None,
    layer_dtype=None,
    weight_dtype=None,
):
    if not isinstance(self, LinearBase):
        return
    if not isinstance(self.quant_method, UnquantizedLinearMethod):
        return
    use_tpp = False
    if layer_dtype is None:
        layer_dtype = self.weight.dtype
    # if layer_dtype == torch.float16:
    #     layer_dtype = torch.bfloat16
    if weight_dtype is None:
        weight_dtype = layer_dtype
    ofm, ifm = self.weight.shape
    bk, bc = bkbc_2
    if ofm % bk == 0 and ifm % bc == 0:
        self.weight_2 = BlockedWeight(self.weight.data, bk, bc, layer_dtype)
        if TPP_XGEMM_USE_FP8 > 0:
            self.weight_2 = self.weight_2.to(torch.float8_e4m3fn)
        use_tpp = True
    if bkbc_1 is not None:
        bk, bc = bkbc_1
        if ofm % bk == 0 and ifm % bc == 0:
            self.weight_1 = BlockedWeight(self.weight.data, bk, bc, layer_dtype)
            use_tpp = True
    if xgemm is not None:
        weight = self.weight
        if TPP_XGEMM_USE_FP8 > 0:
            weight = weight.to(torch.float8_e4m3fn)
        self.weight_t = xgemm.Matrix(
            weight, "b", TPP_XGEMM_PREPACK_LEVEL, split=1, Op=1
        )
        if self.bias:
            self.bias_t = xgemm.Matrix(
                self.bias, "d", TPP_XGEMM_PREPACK_LEVEL, split=0, Op=0
            )
        use_tpp = True
        # self.weight_t.shape = self.weight.shape
    if use_tpp:
        orig_method = self.quant_method
        self.quant_method = TppLinearMethod()
        self.quant_method.orig_method = orig_method


def OptimizeForLinear(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            ofm, ifm = m.weight.shape
            dtype = m.weight.dtype
            if ofm % 64 == 0 and ifm % 64 == 0:
                FixLinear(m, (16, 64), (64, 64))
        elif isinstance(m, LinearBase):
            FixLinearBase(m, (16, 64), (64, 64))
