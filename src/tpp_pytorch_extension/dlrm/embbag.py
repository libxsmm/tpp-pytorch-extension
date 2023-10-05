import torch
from torch.autograd import Function

from tpp_pytorch_extension._C import _embbag_cpp as embbag_cpp


torch_embedding_bag = torch.embedding_bag


def tpp_embedding_bag(
    weight,
    input,
    offsets,
    scale_grad_by_freq,
    mode_enum,
    sparse,
    per_sample_weights,
    include_last_offset=False,
    padding_index=None,
):
    if (
        sparse
        and mode_enum == 0
        and per_sample_weights is None
        and scale_grad_by_freq == False
        and weight.device == torch.device("cpu")
    ):
        ret = TppEmbeddingBagFunction.apply(
            weight, input.contiguous(), offsets.contiguous()
        )
        ret = (ret, None, None, None)
    else:
        ret = torch_embedding_bag(
            weight,
            input,
            offsets,
            scale_grad_by_freq,
            mode_enum,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_index,
        )

    return ret


def bdot(input):
    return BDotFunc.apply(input)


def override_embedding_bag():
    torch.embedding_bag = tpp_embedding_bag
    print("Using TPP EmbeddingBag Implementation")


class TppEmbeddingBagFunction(Function):
    @staticmethod
    def forward(ctx, weight, input, offsets):
        ctx.save_for_backward(weight, input, offsets)
        output = embbag_cpp.embbag_forward(weight, input, offsets)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        weight, input, offsets = ctx.saved_tensors
        grad_weight = grad_input = grad_offsets = None
        grad_weight = embbag_cpp.embbag_backward(grad_out, weight, input, offsets)
        return (grad_weight, grad_input, grad_offsets)


class BDotFunc(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = embbag_cpp.bdot_forward(input)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        (input,) = ctx.saved_tensors
        grad_inp = embbag_cpp.bdot_backward(grad_out, input)
        return grad_inp
