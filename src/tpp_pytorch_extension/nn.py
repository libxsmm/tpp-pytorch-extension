import torch
from tpp_pytorch_extension.utils.xsmm import get_vnni_blocking


def fc_plain(input, weight, bias=torch.Tensor(), parallel_dim=-1, split_sizes=[]):
    return torch.ops.tpp_llm.fc_plain(input, weight, bias, parallel_dim, split_sizes)


class TppLinear(torch.nn.Linear):
    def forward(self, input):
        bias = (
            self.bias if self.bias is not None else torch.Tensor().to(self.weight.dtype)
        )
        N = input.numel() // input.shape[-1]
        if hasattr(self, "weight_2"):
            weight = self.weight_2
            if hasattr(self, "weight_1") and N > 192:
                weight = self.weight_1
        else:
            weight = self.weight
        input = input.to(weight.dtype)

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


def OptimizeForLinear(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            ofm, ifm = m.weight.shape
            dtype = m.weight.dtype
            if ofm % 64 == 0 and ifm % 64 == 0:
                FixLinear(m, (16, 64), (64, 64))
