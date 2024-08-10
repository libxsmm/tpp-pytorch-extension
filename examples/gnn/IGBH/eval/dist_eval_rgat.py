import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as ddp
import time, tqdm, numpy as np
import math
import tpp_pytorch_extension as ppx
from tpp_pytorch_extension._C._qtype import remap_and_quantize_qint8

import os, psutil
import os.path as osp
import collections
from contextlib import contextmanager

from tpp_pytorch_extension.gnn.gat_inference import fused_gat as tpp_gat
from tpp_pytorch_extension.gnn.common import gnn_utils

from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch.conv import GATConv
from dgl.distgnn.mpi import init_mpi
from dgl.distgnn.partitions import create_partition_book
from dgl.distgnn.communicate import alltoall_v_sync

from aux_func_eval import *
from dgl import apply_each
from dgl.distgnn.queue import gqueue

import warnings
warnings.filterwarnings("ignore")

capture_graph_stats = False
linear = None

global_layer_dtype = th.float32
global_use_qint8_gemm = False

@contextmanager
def opt_impl(enable=True, use_qint8_gemm=False, use_bf16=False):
    try:
        global GATConv
        global linear
        global global_layer_dtype
        global global_use_qint8_gemm
        orig_GATConv = GATConv
        orig_linear = nn.Linear

        try:
            if enable:
                linear = tpp_gat.LinearOut
                GATConv = tpp_gat.GATConvOpt
                if use_bf16:
                    global_layer_dtype = th.bfloat16
                if use_qint8_gemm:
                    global_use_qint8_gemm = use_qint8_gemm
            yield
        finally:
            GATConv = orig_GATConv
            linear = nn.Linear
    except ImportError as e:
        pass

class RGAT(nn.Module):
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers, n_heads, activation=None, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        n_hidden = h_feats // n_heads
        self.layers.append(HeteroGraphConv({
            etype: GATConv(in_feats, n_hidden, n_heads, 
                activation=activation, 
                feat_drop=dropout, 
                layer_dtype=global_layer_dtype,
                use_qint8_gemm=global_use_qint8_gemm,
            )
            for etype in etypes}))
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: GATConv(h_feats, n_hidden, n_heads, 
                     activation=activation, 
                     feat_drop=dropout, 
                     layer_dtype=global_layer_dtype,
                     use_qint8_gemm=global_use_qint8_gemm,
                )
                for etype in etypes}))
        self.layers.append(HeteroGraphConv({
            etype: GATConv(h_feats, n_hidden, n_heads,
                activation=None, layer_dtype=global_layer_dtype,
                use_qint8_gemm=global_use_qint8_gemm,
            )
            for etype in etypes}))
        
        self.linear = linear(h_feats, num_classes, global_layer_dtype)
        self.queue = gqueue()

    def forward(self, blocks, x, scfs):
        h = x
        scf = {}
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            for stype, etype, dtype in block.canonical_etypes:
                scf[etype] = (scfs[stype], scfs[dtype])
            h = layer(block, h, scf)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
        return self.linear(h['paper'])

def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def distgnn_eval(distgnn_inf, pb):

    tic = time.time()
    g_orig = pb.g_orig
    features   = distgnn_inf.get_feats
    in_feats   = features.shape[1]

    with opt_impl(args.use_tpp, args.use_qint8_gemm, args.use_bf16):
        model = RGAT(g_orig.etypes, in_feats, args.hidden_channels,
            args.n_classes, args.num_layers, args.num_heads, F.leaky_relu)
        model.requires_grad_(False)
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    if args.use_tpp:
        block(model)

    if args.use_qint8_gemm:
        for l in range(len(model.layers)):
            mkeys = model.layers[l].mods.keys()
            for k in mkeys:
                model.layers[l].mods[k].fc.weight = \
                    torch.nn.Parameter(remap_and_quantize_qint8(model.layers[l].mods[k].fc.weight), requires_grad=False)
    m = model

    if args.rank == 0:
        #print("###############################")
        #print("Model layers: ", m.layers)
        #print("###############################", flush=True)
        param_size = 0
        for param in m.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in m.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(size_all_mb))

    global_acc = 0
    s_batch_inf, e_batch_inf = distgnn_inf.init_setup(args.fan_out)

    model.eval()

    toc = time.time() - tic

    distgnn_inf.bootstrap()

    istep = 0
    acc_tensor = []
    with torch.autograd.profiler.profile(
        enabled=args.profile if args.rank==0 else False, record_shapes=False
    ) as prof:
        if args.use_tpp:
            ppx.reset_debug_timers()
        for step in range(s_batch_inf, e_batch_inf):
            t0 = time.time()
            predictions = []
            labels = []

            with torch.no_grad():
                _, blocks, batch_inputs, batch_scf, batch_labels, _ = distgnn_inf.batch_setup(step)
                   
                predictions.append(model(blocks, batch_inputs, batch_scf).argmax(1).clone().numpy())
                labels.append(batch_labels.clone().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                acc = sklearn.metrics.accuracy_score(labels, predictions)
                if math.isnan(acc):
                    acc = -1

                acc_tensor.append(acc)

            t1 = time.time()

            if args.rank == 0:
                if step > 0:
                    stime = distgnn_inf.ticd / step
                    ctime = distgnn_inf.ticlst / step
                else:
                    stime = distgnn_inf.ticd / (step+1)
                    ctime = distgnn_inf.ticlst / (step+1)

                print('Batch {:07d} | Sampling time {:.4f} (s) | Comms Time {:.4f} | Inference time {:.4f} (s) |'.format(istep, stime, ctime, t1-t0))
            istep += 1
    
    if args.use_tpp and args.profile and args.rank == 0:
        ppx.print_debug_timers(0)

    if prof and args.rank == 0:
        fname = "gat_"+args.dataset_size+".prof"
        with open(fname, "w") as prof_f:
            prof_f.write(
                prof.key_averages(group_by_input_shape=False).table(
                    sort_by="cpu_time_total"
                )
            )
        if ppx.extend_profiler:
            fname = "gat_"+args.dataset_size+"_nested.prof"
            with open(fname, "w") as prof_f:
                prof_f.write(
                    prof.nested_key_averages().table(sort_by=None, row_limit=1000)
                )
            fname = "gat_"+args.dataset_size+"_top_level.prof"
            with open(fname, "w") as prof_f:
                prof_f.write(
                    prof.nested_key_averages(only_top_level=True).table(
                        sort_by="cpu_time_total"
                    )
                )
            prefix = "gat_"+args.dataset_size+"_time"
            prof.print_op_timings(prefix=prefix)

    dist.barrier()

    acc_tensor = torch.tensor(acc_tensor[:distgnn_inf.my_steps])
    global_acc = acc_tensor.sum()
    dist.all_reduce(global_acc, op=dist.ReduceOp.SUM)

    if args.rank == 0:
        global_acc = global_acc / ((e_batch_inf-s_batch_inf)*args.world_size)
        print(f'global acc = {global_acc}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='',
        help='path containing the dataset')
    parser.add_argument('--dataset', type=str, default='IGBH',
            help='name of the dataset')
    parser.add_argument('--dataset_size', type=str, default='full',
        choices=['tiny', 'small', 'medium', 'large', 'full'],
        help='size of the datasets')
    parser.add_argument("--token", type=str, default="p",
            help='string to identify partition root-dir')
    parser.add_argument('--n_classes', type=int, default=19,
        choices=[19, 2983], help='number of classes')

    # Model
    parser.add_argument('--checkpoint', type=str, default='.')

    # Model parameters
    parser.add_argument('--fan_out', type=str, default='-1,-1,-1')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument( "--use_tpp", action="store_true",
        help="Whether to use optimized MLP impl when available",
    )
    parser.add_argument( "--use_int8", action="store_true",
        help="Whether to use int8 datatype when available",
    )
    parser.add_argument( "--use_bf16", action="store_true",
        help="Whether to use BF16 datatype when available",
    )
    parser.add_argument( "--use_qint8_gemm", action="store_true",
        help="Whether to use qint8 GEMM",
    )
    parser.add_argument('--val_fraction', type=float, default=0.025)
    parser.add_argument("--mode", type=str, default="iels")
    parser.add_argument("--part_method", type=str, default="random")
    parser.add_argument("--ielsqsize", type=int, default=1,
       help="#delay in delayed updates")
    parser.add_argument('--world-size', default=-1, type=int,
                         help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                         help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                         help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='mpi', type=str,
                            help='distributed backend')
    parser.add_argument('--use_ddp', action='store_true',
                            help='use or not torch distributed data parallel')
    parser.add_argument("--enable_iec", action='store_true', 
            help="enables original embedding cache")
    parser.add_argument( "--profile", action="store_true",
        help="Whether to profile",
    )
    args = parser.parse_args()

    args.rank, args.world_size = init_mpi(args.dist_backend, args.dist_url)

    seed = int(datetime.datetime.now().timestamp())
    seed = seed >> (seed.bit_length() - 31)
    th.manual_seed(seed)
    dgl.random.seed(seed)
    np.random.seed(seed)

    part_config = args.path

    pb = create_partition_book(args, part_config, 'paper')

    dist.barrier()
    if args.rank == 0:
        print('Data read time from disk {:.4f}'.format(pb.dle))

    if pb.g_orig is None:
        print('Unable to load original graph object! exiting...')
        os.sys.exit(1)

    t = time.time()
    distgnn_inf = distgnn_mb_inf(pb, args, 'paper') 
    dist.barrier()
    
    distgnn_eval(distgnn_inf, pb)

    distgnn_inf.finalize()

    if args.rank == 0:
        print("Run details:")
        print('exec file name:', os.path.basename(__file__))
        print("| world size: ", args.world_size, end="")
        print("| dataset:", args.dataset, end="")
        print("| fan_out: ", args.fan_out, end="")
        print("| batch_size: ", args.batch_size, end="")
        print("| dist_backend: ", args.dist_backend, "|")
        print("| comms delay: ", args.ielsqsize, end="")

        distgnn_inf.printname()
        print()

    dist.barrier()

