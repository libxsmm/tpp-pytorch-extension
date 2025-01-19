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
from rgat import RGAT, block, opt_impl
import tpp_pytorch_extension as ppx
from tpp_pytorch_extension._C._qtype import (
        remap_and_quantize_qint8,
        create_qtensor_int8sym
     )
import os.path as osp
from dgl.dataloading import NeighborSampler
import os, psutil
import collections
import datetime

import warnings
warnings.filterwarnings("ignore")

def load_data(args):
    use_label_2K = (args.num_classes == 2983)
    label_file = 'node_label_19.npy' if not use_label_2K else 'node_label_2K.npy'
    paper_lbl_path = osp.join(args.path, args.dataset_size, 'processed', 'paper', label_file)
    if args.dataset_size in ['large', 'full']:
        paper_node_labels = torch.from_numpy(np.fromfile(paper_lbl_path, dtype=np.float32)).to(torch.long)
    else:
        paper_node_labels = torch.from_numpy(np.load(paper_lbl_path)).long()

    root_name = 'node_feat'
    if args.data_type == 'bf16':
        root_name += '.pt'
    elif args.data_type == 'int8':
        root_name += '_int8.pt'
    elif args.data_type == 'hf8':
        root_name += '_hf8.pt'
    elif args.data_type == 'bf8':
        root_name += '_bf8.pt'

    nfeat = {}
    author_feat_path = osp.join(args.path, args.dataset_size, 'processed', 'author', root_name)
    author_node_features = torch.load(author_feat_path)

    if args.dataset_size in ['large', 'full']:
        conference_feat_path = osp.join(args.path, args.dataset_size, 'processed', 'conference', root_name)
        conference_node_features = torch.load(conference_feat_path)
        journal_feat_path = osp.join(args.path, args.dataset_size, 'processed', 'journal', root_name)
        journal_node_features = torch.load(journal_feat_path)

    fos_feat_path = osp.join(args.path, args.dataset_size, 'processed', 'fos', root_name)
    fos_node_features = torch.load(fos_feat_path)

    institute_feat_path = osp.join(args.path, args.dataset_size, 'processed','institute', root_name)
    institute_node_features = torch.load(institute_feat_path)

    paper_feat_path = osp.join(args.path, args.dataset_size, 'processed', 'paper', root_name)
    paper_node_features = torch.load(paper_feat_path)

    if args.data_type == 'int8':
        author_scf_path = osp.join(args.path, args.dataset_size, 'processed', 'author', 'node_feat_scf.pt')
        author_feat_scf = torch.load(author_scf_path)
        block_size = author_node_features.shape[1] // author_feat_scf.shape[1]
        nfeat.update({'author': create_qtensor_int8sym(author_node_features, author_feat_scf, block_size, 1, False)})

        if args.dataset_size in ['large', 'full']:
            conference_scf_path = osp.join(args.path, args.dataset_size, 'processed', 'conference', 'node_feat_scf.pt')
            conference_feat_scf = torch.load(conference_scf_path)
            nfeat.update({'conference': create_qtensor_int8sym(conference_node_features, conference_feat_scf, block_size, 1, False)})

        fos_scf_path = osp.join(args.path, args.dataset_size, 'processed', 'fos', 'node_feat_scf.pt')
        fos_feat_scf = torch.load(fos_scf_path)
        nfeat.update({'fos': create_qtensor_int8sym(fos_node_features, fos_feat_scf, block_size, 1, False)})

        institute_scf_path = osp.join(args.path, args.dataset_size, 'processed', 'institute', 'node_feat_scf.pt')
        institute_feat_scf = torch.load(institute_scf_path)
        nfeat.update({'institute': create_qtensor_int8sym(institute_node_features, institute_feat_scf, block_size, 1, False)})

        if args.dataset_size in ['large', 'full']:
            journal_scf_path = osp.join(args.path, args.dataset_size, 'processed', 'journal', 'node_feat_scf.pt')
            journal_feat_scf = torch.load(journal_scf_path)
            nfeat.update({'journal': create_qtensor_int8sym(journal_node_features, journal_feat_scf, block_size, 1, False)})

        paper_scf_path = osp.join(args.path, args.dataset_size, 'processed', 'paper', 'node_feat_scf.pt')
        paper_feat_scf = torch.load(paper_scf_path)
        nfeat.update({'paper': create_qtensor_int8sym(paper_node_features, paper_feat_scf, block_size, 1, False)})
    else:
        nfeat.update({'author': author_node_features})
        if args.dataset_size in ['large', 'full']:
            nfeat.update({'conference': conference_node_features})
        nfeat.update({'fos': fos_node_features})
        nfeat.update({'institute': institute_node_features})
        if args.dataset_size in ['large', 'full']:
            nfeat.update({'journal': journal_node_features})
        nfeat.update({'paper': paper_node_features})

    return nfeat, paper_node_labels, author_node_features.shape[1]

def track_acc(g, category, args, device):

    seed = int(datetime.datetime.now().timestamp())
    seed = seed >> (seed.bit_length() - 31)
    dgl.random.seed(seed)

    fanouts = [int(fanout) for fanout in args.fan_out.split(",")]
    sampler = NeighborSampler(fanouts, fused=True)

    val_nid = torch.nonzero(g.nodes[category].data['val_mask'], as_tuple=True)[0]
    if args.profile or args.tpp_profile:
        val_nids = 10000
    else:
        val_nids = int(args.validation_frac * val_nid.shape[0])
    val_nid = val_nid[:val_nids]

    nfeat, labels, in_feats = load_data(args)

    model = RGAT(g.etypes, args.use_tpp, args.use_qint8_gemm, args.use_bf16).to(device)
    model.eval()
    
    def load_state_dict(model_path):
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        for key in list(ckpt.keys()):
            newkey = key.replace('module.', '')
            ckpt[newkey] = ckpt.pop(key)
        return ckpt

    state_dict = load_state_dict(args.checkpoint)
    model.load_state_dict(state_dict)

    '''
    mkeys = model.model.layers[0].mods.keys()    
    for k in mkeys:
        model.model.layers[0].mods[k].copy_weights(int(os.environ["OMP_NUM_THREADS"]))
    '''
    
    if args.use_tpp:
        block(model.model)

    if args.use_qint8_gemm:
        for l in range(len(model.model.layers)):
            mkeys = model.model.layers[l].mods.keys()
            for k in mkeys:
                model.model.layers[l].mods[k].fc.weight = \
                    torch.nn.Parameter(remap_and_quantize_qint8(model.model.layers[l].mods[k].fc.weight), requires_grad=False)

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
                #cores = int(os.environ["OMP_NUM_THREADS"])
                #gnn_utils.affinitize_cores(cores, 0)
                if args.tpp_profile: ppx.reset_debug_timers()

            step = int(step_ / args.batch_size)
            if step_ + args.batch_size < val_nid.shape[0]:
                seeds = val_nid[step_: step_ + args.batch_size]
            else:
                seeds = val_nid[step_:]
            seeds_dict = {category:seeds}

            _, _, blocks = sampler.sample_blocks(g, seeds_dict)

            input_nodes = blocks[0].srcdata[dgl.NID]
            
            t0 = time.time()

            # Load the input features as well as output labels
            '''
            batch_inputs, batch_labels = load_subtensor_dict(
                nfeat, labels, seeds, input_nodes
            )
            '''
            t1 = time.time()

            if args.batch_size == 1:
                with torch.no_grad():
                    preds = [model(blocks, nfeat, input_nodes).argmax(1).numpy()] # batch_inputs).argmax(1).numpy()]
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
                    predictions.append(model.forward_gather(blocks, nfeat, input_nodes).argmax(1).numpy()) # batch_inputs).argmax(1).numpy())
                    #predictions.append(model(blocks, batch_inputs).argmax(1).numpy())
                    #predictions.append(model.forward_graph(blocks).argmax(1).numpy())
                    lables.append(blocks[-1].dstdata['label'][category].numpy())

                t2 = time.time()
                print('Batch {:07d} | Total time {:.4f} (s) |'
                      'Sample {:.4f} (s) | Gather {:.4f} (s) |'
                      'Infer {:.4f} (s)'.format(
                      step, t2-end, t0-end, t1-t0, t2-t1)
                )

            end = time.time()

    if args.use_tpp and args.tpp_profile:
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
    parser.add_argument('--path', type=str, default='/root/savancha/IGBH',
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
    parser.add_argument( "--tpp_profile", action="store_true",
        help="Whether to profile TPP",
    )

    args = parser.parse_args()

    use_label_2K = int(args.num_classes) == 2983
    dataset = IGBHeteroDGLDataset(args.path, args.dataset_size, args.in_memory, use_label_2K, args.data_type)
    g = dataset[0]
    category = g.predict

    track_acc(g, category, args, 'cpu')
