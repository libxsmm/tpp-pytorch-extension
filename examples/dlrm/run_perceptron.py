import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time

# import repro
# import pcl_bert
from tpp_pytorch_extension.dlrm import perceptron as pclPerceptron
from torchrec.modules.mlp import Perceptron as trecPerceptron

# from torchrec.modules.utils import extract_module_or_tensor_callable


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
    torch.manual_seed(1)

    LOOP = 10
    in_size = 512  # 2048
    out_size = 512  # 2048
    bs = 512
    bias = True
    #    activation = extract_module_or_tensor_callable(torch.sigmoid)
    #    activation = torch.sigmoid
    activation = torch.relu
    device = None

    trecPercep = trecPerceptron(in_size, out_size, bias, activation, device)
    pclPercep = pclPerceptron.Perceptron(in_size, out_size, bias, activation, device)

    # print([(n, p.shape) for n,p in trecPercep.named_parameters()])
    # print([(n, p.shape) for n,p in pclPercep.named_parameters()])
    for ii, (i, o) in enumerate(zip(trecPercep.parameters(), pclPercep.parameters())):
        # i.data.uniform_(0.0, 1.0)
        o.data = i.data.clone().detach()

    for m in pclPercep.modules():
        if hasattr(m, "set_blocking"):
            #            m.set_blocking(16, 16, torch.float32)
            m.set_blocking(16, 16, torch.bfloat16)
            m.maybe_block_params()

    inp = torch.empty([bs, in_size]).uniform_(-1.0, 1.0)
    inp.to(torch.bfloat16).to(torch.float32)

    inp1 = inp.clone().detach().requires_grad_()
    inp2 = inp.clone().detach().requires_grad_()

    out = trecPercep(inp1)
    tmp_t = torch.rand(out.shape)

    start_t = time.time()
    for i in range(LOOP):
        out = trecPercep(inp1)
        out = out * tmp_t
        outLoss = out.mean() * 100000
        outLoss.backward()
    elapsed_tr = time.time() - start_t
    print("torchrecPercep, elapsed time: ", elapsed_tr / LOOP)

    start_t = time.time()
    for i in range(LOOP):
        pout = pclPercep(inp2)
        pout = pout * tmp_t
        poutLoss = pout.mean() * 100000
        poutLoss.backward()
    elapsed_pcl = time.time() - start_t
    print("pclPercep, elapsed time: ", elapsed_pcl / LOOP)
    print("speedup: ", elapsed_tr / elapsed_pcl)

    #    with torch.autograd.profiler.profile(True, record_shapes=True) as prof:
    #      with torch.autograd.profiler.record_function("ref_bert"):
    #        for i in range(LOOP):
    #            out = trecPercep(inp1)
    #            out = out * tmp_t
    #            outLoss = out.mean()*100000
    #            outLoss.backward()
    #      with torch.autograd.profiler.record_function("pcl_bert"):
    #        for i in range(LOOP):
    #            pout = pclPercep(inp2)
    #            pout = pout * tmp_t
    #            poutLoss = pout.mean()*100000
    #            poutLoss.backward()
    #
    if hasattr(pout, "unblocked_tensor"):
        pout = pout.unblocked_tensor()

    compare(out, pout, "Output")
    print(out.allclose(pout, atol=1e-4, rtol=1e-4))
    compare(inp1.grad, inp2.grad, "InpGrad")
    print(inp1.grad.allclose(inp2.grad, atol=1e-4, rtol=1e-4))
    diff = inp1.grad - inp2.grad
    ##    print(f"INP: Max = {torch.max(diff.abs()):.7g}, avg = {diff.abs().mean():.7g}, ref mean = {inp1.grad.abs().mean():.7g}  rel = {diff.abs().max()/inp1.grad.abs().mean():.7g}")
    ##    #print(f"ref = {out[0].view([-1])[:8]}, xsmm = {pout[0].view([-1])[:8]}")
    print(f"INP: ref = {inp1.grad.view([-1])[:8]}, xsmm = {inp2.grad.view([-1])[:8]}")

    for ii, (i, o) in enumerate(zip(trecPercep.parameters(), pclPercep.parameters())):
        o.unblock()
        compare(i.grad, o.grad, ii)

#    file_prefix='repro_prof'
#    with open("%s.prof" % file_prefix, "w") as prof_f:
#        prof_f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))
#    # print(prof.key_averages().table(sort_by="cpu_time_total"))
#    try:
#        with open("%s.nested.prof" % file_prefix, "w") as prof_f:
#            prof_f.write(prof.nested_key_averages().table(sort_by="cpu_time_total"))
#        with open("%s.top_level.prof" % file_prefix, "w") as prof_f:
#            prof_f.write(prof.nested_key_averages(only_top_level=True).table(sort_by="cpu_time_total"))
#        prof.print_op_timings(False, file_prefix)
#    except:
#        pass
