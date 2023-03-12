import torch
from torch import nn
import torch.nn.functional as F
import time

from tpp_pytorch_extension.dlrm import mlp as pclMLP
from torchrec.modules import mlp as trecMLP

from torchrec.modules.utils import extract_module_or_tensor_callable
import numpy as np


def compare(ref, opt, name=""):
    ref = ref.detach()
    opt = opt.detach()
    allclose = ref.allclose(opt, atol=1e-3, rtol=1e-4)
    print(
        f"{name}: ref: {ref.abs().mean():14g} allclose: {allclose}  shape: {ref.shape}"
    )
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


if __name__ == "__main__":
    torch.manual_seed(42)
    bias = True
    #    activation = extract_module_or_tensor_callable(torch.sigmoid)
    activation = torch.sigmoid
    device = None

    LOOP = 100
    bs = 2048
    in_features = 2048
    layer_sizes = [2048, 2048, 2048, 2048, 2048]
    BlockS = 32

    tMLP = trecMLP.MLP(in_features, layer_sizes, bias, activation, device)

    use_tpp = True
    use_bf16 = False
    with pclMLP.tpp_impl(use_tpp, use_bf16):
        #    with tpp_impl(use_tpp, use_bf16):
        pMLP = trecMLP.MLP(in_features, layer_sizes, bias, activation, device)

    for ii, (i, o) in enumerate(zip(tMLP.parameters(), pMLP.parameters())):
        o.data = i.data.clone().detach()

    for m in pMLP.modules():
        if hasattr(m, "set_blocking"):
            m.set_blocking(BlockS, BlockS, torch.float32)
            m.maybe_block_params()

    inp = torch.empty([bs, in_features]).uniform_(-1.0, 1.0)
    ##    inp = inp.to(torch.bfloat16).to(torch.float32)
    inp1 = inp.clone().detach().requires_grad_()
    inp2 = inp.clone().detach().requires_grad_()

    print([(n, p.shape) for n, p in tMLP.named_parameters()])
    print([(n, p.shape) for n, p in pMLP.named_parameters()])

    out = tMLP(inp1)
    tmp_t = torch.rand(out.shape)

    pout = None
    start_t = time.time()
    for i in range(LOOP):
        pout = pMLP(inp2)
        pout = pout * tmp_t
        poutLoss = pout.mean() * 100000
        poutLoss.backward()

    elapsed_pcl = time.time() - start_t
    print("pMLP, elapsed time: ", elapsed_pcl / LOOP)

    start_t = time.time()
    for i in range(LOOP):
        out = tMLP(inp1)
        out = out * tmp_t
        outLoss = out.mean() * 100000
        outLoss.backward()

    elapsed_tr = time.time() - start_t
    print("tMLP, elapsed time: ", elapsed_tr / LOOP)

    print("speedup: ", elapsed_tr / elapsed_pcl)

    with torch.autograd.profiler.profile(True, record_shapes=True) as prof:
        with torch.autograd.profiler.record_function("trec_mlp"):
            for i in range(LOOP):
                out = tMLP(inp1)
                out = out * tmp_t
                outLoss = out.mean() * 100000
                outLoss.backward()
        with torch.autograd.profiler.record_function("pcl_mlp"):
            for i in range(LOOP):
                pout = pMLP(inp2)
                pout = pout * tmp_t
                poutLoss = pout.mean() * 100000
                poutLoss.backward()

    file_prefix = "mlp_prof"
    with open("%s.prof" % file_prefix, "w") as prof_f:
        prof_f.write(
            prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total")
        )
    try:
        with open("%s.nested.prof" % file_prefix, "w") as prof_f:
            prof_f.write(prof.nested_key_averages().table(sort_by="cpu_time_total"))
        with open("%s.top_level.prof" % file_prefix, "w") as prof_f:
            prof_f.write(
                prof.nested_key_averages(only_top_level=True).table(
                    sort_by="cpu_time_total"
                )
            )
        prof.print_op_timings(False, file_prefix)
    except:
        pass

    if hasattr(pout, "unblocked_tensor"):
        pout = pout.unblocked_tensor()

    compare(out, pout, "Output")
    print(out.allclose(pout, atol=1e-4, rtol=1e-4))
    compare(inp1.grad, inp2.grad, "InpGrad")
    print(inp1.grad.allclose(inp2.grad, atol=1e-4, rtol=1e-4))
    diff = inp1.grad - inp2.grad
    print(f"INP: ref = {inp1.grad.view([-1])[:8]}, xsmm = {inp2.grad.view([-1])[:8]}")

    for ii, (i, o) in enumerate(zip(tMLP.parameters(), pMLP.parameters())):
        o.unblock()
        compare(i.grad, o.grad, ii)
