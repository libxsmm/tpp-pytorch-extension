import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as ddp
import time, tqdm, numpy as np
import tpp_pytorch_extension as ppx

import os, psutil
import os.path as osp
import collections
from contextlib import contextmanager

from tpp_pytorch_extension.gnn.gat import fused_gat as tpp_gat
from tpp_pytorch_extension.gnn.common import gnn_utils
from tpp_pytorch_extension.utils.blocked_layout import BlockedParameter

from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch.conv import GATConv
from dgl.distgnn.mpi import init_mpi
from dgl.distgnn.partitions import partition_book_random, create_partition_book
from dgl.distgnn.communicate import alltoall_v_sync

from aux_func import *
from dgl import apply_each
from dgl.distgnn.queue import gqueue

import mlperf_logging.mllog.constants as mllog_constants
from mlperf_logging_utils import get_mlperf_logger, submission_info

import warnings
warnings.filterwarnings("ignore")

mllogger = get_mlperf_logger(path=osp.dirname(osp.abspath(__file__)))

capture_graph_stats = False
linear = None

global_layer_dtype = th.float32

@contextmanager
def opt_impl(enable=True, use_bf16=False):
    try:
        global GATConv
        global linear
        global global_layer_dtype
        orig_GATConv = GATConv
        orig_linear = nn.Linear

        try:
            if enable:
                linear = tpp_gat.LinearOut
                GATConv = tpp_gat.GATConvOpt
                if use_bf16:
                    global_layer_dtype = th.bfloat16
            yield
        finally:
            GATConv = orig_GATConv
            linear = nn.Linear
    except ImportError as e:
        pass

class RGAT(nn.Module):
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers, n_heads, activation=None, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        n_hidden = h_feats // n_heads
        self.layers.append(HeteroGraphConv({
            etype: GATConv(in_feats, n_hidden, n_heads, activation=activation, feat_drop=dropout, input_needs_grad=False, layer_dtype=global_layer_dtype)
            for etype in etypes}))
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: GATConv(h_feats, n_hidden, n_heads, activation=activation, feat_drop=dropout, layer_dtype=global_layer_dtype)
                for etype in etypes}))
        self.layers.append(HeteroGraphConv({
            etype: GATConv(h_feats, n_hidden, n_heads, activation=None, layer_dtype=global_layer_dtype)
            for etype in etypes}))
        
        self.linear = linear(h_feats, num_classes, global_layer_dtype)
        self.queue = gqueue()

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h, mod_kwargs=None)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
        return self.linear(h['paper'])

    def param_sync(self):
        for param in self.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    ## async params comms start
    def param_sync_start(self):
        for param in self.parameters():
            if param.requires_grad and param.grad is not None:
                req = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
                self.queue.push(req)

    ## async params comms- wait
    def param_sync_end(self):
        while not self.queue.empty():
            req = self.queue.pop()
        req.wait()

def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

def unblock(model):
    for m in model.modules():
        if hasattr(m, "maybe_unblock_params"):
            m.maybe_unblock_params()

def evaluate(distgnn_inf, model, s_batch, e_batch, epoch_num):

    if args.rank == 0:
        mllogger.start(
            key=mllog_constants.EVAL_START,
            metadata={mllog_constants.EPOCH_NUM: epoch_num},
        )
    predictions = []
    labels = []

    distgnn_inf.epoch_setup()
    with torch.no_grad():
        for i in range(s_batch, e_batch):
            _, blocks, batch_inputs, batch_labels, _ = distgnn_inf.batch_setup(i)

            predictions.append(model(blocks, batch_inputs).argmax(1).clone().numpy())
            labels.append(batch_labels.clone().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        acc = sklearn.metrics.accuracy_score(labels, predictions)

        dist.barrier()
        acc_tensor = torch.tensor(acc)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        global_acc = acc_tensor.item() / args.world_size

    if args.rank == 0:
        mllogger.event(
            key=mllog_constants.EVAL_ACCURACY,
            value=global_acc,
            metadata={mllog_constants.EPOCH_NUM: epoch_num},
        )
        mllogger.end(
            key=mllog_constants.EVAL_STOP,
            metadata={mllog_constants.EPOCH_NUM: epoch_num},
        )

    return global_acc

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

def distgnn_train(distgnn, distgnn_inf, pb):

    tic = time.time()
    g_orig = pb.g_orig
    features   = distgnn.get_feats
    in_feats   = features.shape[1]

    with opt_impl(args.tpp_impl, args.use_bf16):
        model = RGAT(g_orig.etypes, in_feats, args.hidden_channels,
            args.n_classes, args.num_layers, args.num_heads, F.leaky_relu, args.dropout)
    
    block(model)

    if args.use_ddp:
        model = ddp(model, find_unused_parameters=True, gradient_as_bucket_view=True)

    if args.use_ddp:
        m = model.module
    else:
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

    loss_fcn = nn.CrossEntropyLoss()
    if args.tpp_impl:
        no_decay = ["bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in m.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.decay,
            },
            {
                "params": [
                    p
                    for n, p in m.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = ppx.optim.Adam(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
    else:
        optimizer = optim.Adam(m.parameters(),
            lr=args.learning_rate)

    s_batch, e_batch = distgnn.init_setup(args.fan_out, model, optimizer)
    s_batch_inf, e_batch_inf = distgnn_inf.init_setup(args.val_fan_out)

    batch_num = e_batch - distgnn.delay
    validation_freq = int(batch_num * args.validation_frac_in_epoch)
    target_reached = False
    toc = time.time() - tic

    for epoch in range(args.n_epochs):
        if args.rank==0: 
            mllogger.start(key=mllog_constants.EPOCH_START, metadata={"epoch_num": epoch+1})
        batch_fwd_time = AverageMeter()
        batch_bwd_time = AverageMeter()
        mbc_time = AverageMeter()
        ar_time = AverageMeter()

        ticd, ticlst, ticbs, ticf, ticb, ticar, ticos = 0, 0, 0, 0, 0, 0, 0
        ticd0, tic_gg = 0, time.time()

        distgnn.epoch_setup()

        es = time.time() - tic_gg

        model.train()
        total_loss = 0
        train_acc = 0
        idx = 0

        step = 0
        t0 = tic_gg
        for i in range(s_batch, e_batch):

            dist.barrier()
            input_nodes, blocks, batch_inputs, batch_labels, seeds = distgnn.batch_setup(i)

            if i > 1:
                t2 = time.time() 
                mbc = t2 - t0
                ticd += mbc
                mbc_time.update(mbc)
                
                t3 = time.time()
            batch_pred = model(blocks, batch_inputs).to(torch.float32)
            
            loss = loss_fcn(batch_pred, batch_labels)
            if args.use_bf16: loss = loss.to(torch.bfloat16)

            if i > 1:
                t4 = time.time()
                cf = t4 - t3
                ticf += cf
                batch_fwd_time.update(cf)

                t5 = time.time()

            optimizer.zero_grad()
            loss.backward()

            if i > 1:
                t6 = time.time()
                cb = t6 - t5
                ticb += cb
                batch_bwd_time.update(cb)

                t6 = time.time()

            distgnn.optimize(i)

            if i > 1:
                t7 = time.time()
                ticos += t7 - t6

            if args.rank == 0:
                if step > 0 and step % args.log_every == 0:
                    if not args.use_ddp and not distgnn.ps_overlap:
                        print("Epoch {:05d} | Step {:05d} | Loss {:.2f} | Time(s) {:.2f} |"
                                " MBC {:.2f} ({:.2f}) | FWD {:.2f} ({:.2f}) | BWD {:.2f} ({:.2f}) |"
                                " AR {:.2f} ({:.2f}) | OPT {:.2f} ({:.2f})"
                              .format( epoch, step, loss.item(), time.time() - t0, 
                                       mbc_time.val, mbc_time.avg, 
                                       batch_fwd_time.val, batch_fwd_time.avg,
                                       batch_bwd_time.val, batch_bwd_time.avg,
                                       ar_time.val, ar_time.avg, 
                                       ops_time.val, ops_time.avg,
                                       flush=True)
                        )
                    elif not args.use_ddp and distgnn.ps_overlap:
                        print("Epoch {:05d} | Step {:05d} | Loss {:.2f} | Time(s) {:.2f} |"
                                " MBC {:.2f} ({:.2f}) | FWD {:.2f} ({:.2f}) | BWD {:.2f} ({:.2f}) |"
                              .format( epoch, step, loss.item(), time.time() - t0, 
                                       mbc_time.val, mbc_time.avg, 
                                       batch_fwd_time.val, batch_fwd_time.avg,
                                       batch_bwd_time.val, batch_bwd_time.avg,
                                       flush=True)
                        )
                    elif args.use_ddp:
                        print("Epoch {:05d} | Step {:05d} | Loss {:.2f} | Time(s) {:.2f} |"
                                " MBC {:.2f} ({:.2f}) | FWD {:.2f} ({:.2f}) | BWD {:.2f} ({:.2f}) |"
                                " OPT {:.2f} ({:.2f})"
                              .format( epoch, step, loss.item(), time.time() - t0, 
                                       mbc_time.val, mbc_time.avg, 
                                       batch_fwd_time.val, batch_fwd_time.avg,
                                       batch_bwd_time.val, batch_bwd_time.avg,
                                       ops_time.val, ops_time.avg,
                                       flush=True)
                        )
            dist.barrier()
            if distgnn.ps_overlap: svar=1
            else: svar=0
            if step > svar and step % validation_freq == svar:
                model.eval()
                epoch_num = epoch + step / batch_num
                global_acc = evaluate(distgnn_inf, model, s_batch_inf, e_batch_inf, epoch_num)

                if global_acc >= args.target_acc:
                    if args.rank == 0:
                        if args.model_save:
                            mp = osp.join(args.modelpath, 'model_'+args.dataset_size+'.pt')
                            unblock(model)
                            torch.save(model.state_dict(), mp)
                    target_reached = True
                    break
                model.train()
            idx += 1
            step += 1
            t0 = time.time()

        if args.rank==0:
            mllogger.end(key=mllog_constants.EPOCH_STOP, metadata={"epoch_num": epoch+1})
        toc_gg = time.time()
        if capture_graph_stats:
            dist.all_reduce(halo_nodes)
            halo_nodes = halo_nodes / (args.world_size * max_steps)
            dist.all_reduce(total_nodes)
            total_nodes = total_nodes / (args.world_size * max_steps)

        #dist.barrier()

        #distgnn.profile()
        if args.rank == 0:
            print("Epoch: {} time: {:0.4f} sec".format(epoch, toc_gg - tic_gg), end="", flush=True)
            print(' dltime: {:.4f}, dlt: {:.4f}, ftime: {:.4f}, btime: {:.4f}, ' \
                    'all-red time: {:.4f}, opt step: {:.4f} sec' \
                  .format(ticd, ticlst, ticf, ticb, ticar, ticos), flush=True)
        #    process = psutil.Process(os.getpid())
        #    print('\t Mem usage in GB: ', process.memory_info().rss/1e9, flush=True)  # in bytes
        #    if capture_graph_stats:
        #        print('avg halo nodes: ', halo_nodes, flush=True)
        #        print('avg total nodes: ', total_nodes, flush=True)
        #    print()

        dist.barrier()

        if target_reached: break

        #train_acc /= idx
        #sched.step()

    if args.rank == 0:
        status = mllog_constants.SUCCESS if target_reached else mllog_constants.ABORTED
        mllogger.end(key=mllog_constants.RUN_STOP,
                     metadata={mllog_constants.STATUS: status,
                               mllog_constants.EPOCH_NUM: epoch_num,
                     }
        )
        print("Total time taken " + str(datetime.timedelta(seconds = int(time.time() - train_start))))

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
    parser.add_argument('--modelpath', type=str, default='.')
    parser.add_argument('--model_save', action="store_true",
            help="save model checkpoint?"
    )

    # Model parameters
    parser.add_argument('--fan_out', type=str, default='15,10,5')
    parser.add_argument('--val_fan_out', type=str, default='15,10,5')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument( "--adam_epsilon", default=1e-8, type=float)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--validation_frac_in_epoch', type=float, default=0.05)
    parser.add_argument('--log_every', type=int, default=5)
    parser.add_argument('--target_acc', type=float, default=1.0)

    parser.add_argument( "--tpp_impl", action="store_true",
        help="Whether to use optimized MLP impl when available",
    )
    parser.add_argument( "--use_int8", action="store_true",
        help="Whether to use int8 datatype when available",
    )
    parser.add_argument( "--use_bf16", action="store_true",
        help="Whether to use BF16 datatype when available",
    )

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
    args = parser.parse_args()

    args.rank, args.world_size = init_mpi(args.dist_backend, args.dist_url)

    seed = int(datetime.datetime.now().timestamp())
    seed = seed >> (seed.bit_length() - 31)
    th.manual_seed(seed)
    dgl.random.seed(seed)
    np.random.seed(seed)
    ppx.manual_seed(seed)


    if args.rank == 0:
        mllogger.event(key=mllog_constants.CACHE_CLEAR, value=True)
        mllogger.start(key=mllog_constants.INIT_START)
        submission_info(mllogger, mllog_constants.GNN, 'Intel', 'Intel Xeon(R) 8592, codenamed Emerald Rapids')
        mllogger.event(key=mllog_constants.GLOBAL_BATCH_SIZE, value=args.world_size*args.batch_size)
        mllogger.event(key=mllog_constants.GRADIENT_ACCUMULATION_STEPS, value=1)
        mllogger.event(key=mllog_constants.OPT_NAME, value='adam')
        mllogger.event(key=mllog_constants.OPT_BASE_LR, value=args.learning_rate)
        mllogger.event(key=mllog_constants.SEED,value=seed)
        mllogger.end(key=mllog_constants.INIT_STOP)
        mllogger.start(key=mllog_constants.RUN_START)

    part_config = os.path.join(args.path, args.dataset, args.dataset_size)
    category = 'paper'

    train_start = time.time()
    pb = create_partition_book(args, part_config, category)

    dist.barrier()
    if args.rank == 0:
        print('Data read time from disk {:.4f}'.format(pb.dle))

    if pb.g_orig is None:
        print('Unable to load original graph object! exiting...')
        os.sys.exit(1)

    if args.rank == 0:
        mllogger.event(key=mllog_constants.TRAIN_SAMPLES, value=pb.node_feats['train_samples'].item())
        mllogger.event(key=mllog_constants.EVAL_SAMPLES, value=pb.node_feats['eval_samples'].item())

    t = time.time()
    distgnn = distgnn_mb(pb, args, 'paper')
    distgnn_inf = distgnn_mb_inf(pb, distgnn.gobj, args, distgnn.acc_onodes, distgnn.acc_pnodes, distgnn.tr_ntype, 'paper') 
    dist.barrier()
    rpt = time.time() - t 
    
    distgnn_train(distgnn, distgnn_inf, pb)

    distgnn.finalize()
    distgnn_inf.finalize()

    if args.rank == 0:
        print("Run details:")
        print('exec file name:', os.path.basename(__file__))
        print("| world size: ", args.world_size, end="")
        print("| dataset:", args.dataset, end="")
        print("| lr: ", args.learning_rate, end="")
        print("| fan_out: ", args.fan_out, end="")
        print("| batch_size: ", args.batch_size, end="")
        print("| dist_backend: ", args.dist_backend, "|")
        print("| enable_iec: ", args.enable_iec, end="")
        if args.enable_iec:
            print('| Cache size: ', args.cache_size, end="")
        print("| comms delay: ", args.ielsqsize, end="")

        distgnn.printname()
        print()

    dist.barrier()

