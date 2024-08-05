import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
import torch.nn.functional as F
from dgl import apply_each
from dgl.nn.pytorch.conv import GATConv, GraphConv, SAGEConv
from dgl.nn.pytorch import HeteroGraphConv
from contextlib import contextmanager
from tpp_pytorch_extension.gnn.gat_inference import fused_gat as tpp_gat
from tpp_pytorch_extension.gnn.common_inference import gnn_utils
from lp_dataset import IGBHeteroDGLDataset
import tpp_pytorch_extension as ppx
from tpp_pytorch_extension._C._qtype import remap_and_quantize_qint8

from dgl.dataloading import NeighborSampler
import os, psutil
import collections
import datetime

import warnings
warnings.filterwarnings("ignore")

use_tpp = False
linear = None

global_layer_dtype = torch.float32
global_use_qint8_gemm = False

@contextmanager
def opt_impl(enable=True, use_qint8_gemm=False, use_bf16=False):
    try:
        global GATConv
        global linear
        global use_tpp
        global global_layer_dtype
        global global_use_qint8_gemm
        orig_GATConv = GATConv
        linear = nn.Linear
        try:
            if enable:
                use_tpp = enable
                if use_bf16:
                    global_layer_dtype = torch.bfloat16
                if use_qint8_gemm:
                    global_use_qint8_gemm = use_qint8_gemm
                GATConv = tpp_gat.GATConvOpt
                linear = tpp_gat.LinearOut
            yield
        finally:
            GATConv = orig_GATConv
            linear = nn.Linear
    except ImportError as e:
        pass

class RGAT(nn.Module):
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers, n_heads, activation, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()

        if not use_tpp: activation = None
        self.layers.append(HeteroGraphConv({
            etype: GATConv(in_feats, h_feats // n_heads, n_heads, 
                activation=activation, 
                feat_drop=dropout, 
                layer_dtype=global_layer_dtype,
                use_qint8_gemm=global_use_qint8_gemm,
            )
            for etype in etypes}))
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: GATConv(h_feats, h_feats // n_heads, n_heads, 
                    activation=activation, 
                    feat_drop=dropout, 
                    layer_dtype=global_layer_dtype,
                    use_qint8_gemm=global_use_qint8_gemm,
                )
                for etype in etypes}))
        self.layers.append(HeteroGraphConv({
            etype: GATConv(h_feats, h_feats // n_heads, 
                n_heads, activation=None, 
                layer_dtype=global_layer_dtype, 
                use_qint8_gemm=global_use_qint8_gemm,
            )
            for etype in etypes}))
        if not use_tpp:
            self.dropout = nn.Dropout(dropout)
        self.linear = linear(h_feats, num_classes, global_layer_dtype)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.layers) - 1 and not use_tpp:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        return self.linear(h['paper'])   

def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

def load_subtensor_dict(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs={}
    ntypes = nfeat.keys()
    for ntype in ntypes:
        batch_inputs[ntype] = gnn_utils.gather_features(nfeat[ntype], input_nodes[ntype])

    batch_labels = labels[seeds]

    return batch_inputs, batch_labels

def track_acc(g, category, args, device):

    seed = int(datetime.datetime.now().timestamp())
    seed = seed >> (seed.bit_length() - 31)
    dgl.random.seed(seed)

    fanouts = [int(fanout) for fanout in args.fan_out.split(",")]
    sampler = NeighborSampler(fanouts, fused=True)

    val_nid = torch.nonzero(g.nodes[category].data['val_mask'], as_tuple=True)[0]
    if args.profile:
        val_nids = 10000
    else:
        val_nids = int(args.validation_frac * val_nid.shape[0])
    val_nid = val_nid[:val_nids]

    nfeat = g.ndata['feat']
    labels = g.ndata['label'][category]
    in_feats = g.ndata['feat'][category].shape[1]

    with opt_impl(args.use_tpp, args.use_qint8_gemm, args.use_bf16):
        model = RGAT(g.etypes, in_feats, args.hidden_channels,
                args.num_classes, args.num_layers, args.num_heads, 
                F.leaky_relu).to(device)
        model.requires_grad_(False)
    #print(model)

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

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    predictions = []
    lables = []
    global_acc = []
    block_acc = 0
    e_start = time.time()
    end = time.time()
    with torch.autograd.profiler.profile(
        enabled=args.profile, record_shapes=False
    ) as prof:
        for step_ in range(0, val_nid.shape[0], args.batch_size):
            if args.use_tpp and step_ == 0:
                cores = int(os.environ["OMP_NUM_THREADS"])
                gnn_utils.affinitize_cores(cores, 0)
                ppx.reset_debug_timers()

            step = int(step_ / args.batch_size)
            if step_ + args.batch_size < val_nid.shape[0]:
                seeds = val_nid[step_: step_ + args.batch_size]
            else:
                seeds = val_nid[step_:]
            seeds_dict = {category:seeds}

            seeds_dict, output_nodes, blocks = sampler.sample_blocks(g, seeds_dict)

            input_nodes = blocks[0].srcdata[dgl.NID]

            t0 = time.time()

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor_dict(
                nfeat, labels, seeds, input_nodes
            )
            #if args.data_type == 'hf8_mm':
            #    for k in batch_inputs.keys():
            #        batch_inputs[k] = batch_inputs[k].to(torch.float8_e4m3fn)
            t1 = time.time()

            if args.batch_size == 1:
                with torch.no_grad():
                    preds = [model(blocks, batch_inputs).argmax(1).numpy()]
                    labs = [blocks[-1].dstdata['label'][category].numpy()]

                    acc = sklearn.metrics.accuracy_score(labs, preds)
                    global_acc.append(acc)
                    block_acc = block_acc + acc
                t2 = time.time()
                if step > 0 and step % 70000 == 0:
                    print('Block acc {:.4f}'.format(block_acc/70000))
                    block_acc = 0
                elif step > 0 and step == 788378:
                    print('Block acc {:.4f}'.format(block_acc/18379))

                print('Batch {:07d} | Acc {:.4f} | Total time {:.4f} (s) |'
                      'Sample {:.4f} (s) | Gather {:.4f} (s) |'
                      'Infer {:.4f} (s)'.format(
                      step, acc, t2-end, t0-end, t1-t0, t2-t1)
                )
            elif args.batch_size > 1:
                with torch.no_grad():
                    predictions.append(model(blocks, batch_inputs).argmax(1).numpy())
                    lables.append(blocks[-1].dstdata['label'][category].numpy())

                t2 = time.time()
                print('Batch {:07d} | Total time {:.4f} (s) |'
                      'Sample {:.4f} (s) | Gather {:.4f} (s) |'
                      'Infer {:.4f} (s)'.format(
                      step, t2-end, t0-end, t1-t0, t2-t1)
                )

            end = time.time()

    if args.use_tpp and args.profile:
        ppx.print_debug_timers(0)

    if prof:
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

    if args.batch_size > 1:
        predictions = np.concatenate(predictions)
        lables = np.concatenate(lables)
        acc = sklearn.metrics.accuracy_score(lables, predictions)
    e_end = time.time()
    if args.batch_size == 1:
        print('Avg. batch inference time: {:.4f}'.format((e_end-e_start)/val_nid.shape[0]))
        print('Overall accuracy: {:.4f}'.format(np.mean(global_acc)))
    elif args.batch_size > 1:
        print('Avg. batch inference time: {:.4f}'.format((e_end-e_start)/(val_nid.shape[0]//args.batch_size)))
        print('Overall accuracy: {:.4f}'.format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/beegfs/savancha/IGBH',
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='full',
        choices=['tiny', 'small', 'medium', 'large', 'full'],
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=2983,
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=1,
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--data_type', type=str, default='int8',
        help='input data type')

    # Model
    parser.add_argument('--model_type', type=str, default='rgat',
                        choices=['rgat', 'rsage', 'rgcn'])
    parser.add_argument('--checkpoint', type=str, default='.')

    # Model parameters
    parser.add_argument('--fan_out', type=str, default='-1,-1,-1')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--validation_frac', type=float, default=0.025)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument( "--use_tpp", action="store_true",
        help="Whether to use optimized MLP impl when available",
    )
    parser.add_argument( "--use_bf16", action="store_true",
        help="Whether to use BF16 datatype when available",
    )
    parser.add_argument( "--use_qint8_gemm", action="store_true",
        help="Whether to use qint8 GEMM",
    )
    parser.add_argument( "--profile", action="store_true",
        help="Whether to profile",
    )

    args = parser.parse_args()

    use_label_2K = int(args.num_classes) == 2983
    dataset = IGBHeteroDGLDataset(args.path, args.dataset_size, args.in_memory, use_label_2K, args.data_type)
    g = dataset[0]
    category = g.predict

    track_acc(g, category, args, 'cpu')
