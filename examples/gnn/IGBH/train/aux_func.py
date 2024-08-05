from math import ceil
import sys, time
import torch as th
from dgl.distgnn.communicate import mpi_allreduce
import torch.distributed as dist
from dgl.distgnn.queue import gqueue
from dgl.distgnn.iels import iels_master_func as iels_master
from dgl.distgnn.hels import hels_master_func as hels_master
from dgl.distgnn.iels import ielspp_batch
from tpp_pytorch_extension.gnn.common import gnn_utils
from dgl.dataloading import NeighborSampler
import dgl
from itertools import accumulate
import numpy as np
import psutil, os

class distgnn_mb:
    def __init__(self, pb, args, category=''):
        self.mode = args.mode
        if args.mode != 'iels' and args.mode != 'hels':
            print('wrong mode provided....exiting')
            sys.exit(0)

        if category != '':
            self.category = category
        self.g_orig = pb.g_orig
        self.ntypes = pb.g_orig.ntypes
        self.ntypes_id = []
        for keys, nt in enumerate(pb.g_orig.ntypes):
            self.ntypes_id.append(keys)
        
        self.num_onodes, self.num_pnodes = [], [] #num ntypes nodes per partition and in orig graph
        nsn = (pb.node_feats['inner_node'] == 1).sum()
        for keys, nt in enumerate(self.ntypes):
            self.num_onodes.append(pb.g_orig.num_nodes(nt))
            index = (pb.node_feats['_TYPE'][:nsn] == keys).nonzero(as_tuple=True)[0]
            self.num_pnodes.append((pb.node_feats['inner_node'][index]).sum())
            if nt == category: 
                self.tr_ntype = keys
  
        del index
        self.acc_onodes = np.insert(np.cumsum(self.num_onodes), 0, 0)
        self.acc_pnodes = np.insert(np.cumsum(self.num_pnodes), 0, 0)
        #print('acc_onodes: ', self.acc_onodes, flush=True)
        #print('acc_pnodes: ', self.acc_pnodes, flush=True)
         
        self.pb = pb
        if self.mode == 'iels':
            self.gobj = iels_master(pb, args)
        elif self.mode == 'hels':
            self.gobj = hels_master(pb, args)
        self.node_feats = pb.node_feats

        self.batch_size = args.batch_size  ## train batch size
        self.delay = args.ielsqsize
        self.rank = args.rank
        self.tpp_impl = args.tpp_impl

        self.use_ddp = args.use_ddp
        self.ps_overlap = not args.use_ddp

        self.adjust_batch('train')
        self.nn_mode = 'train'
        self.reset()

    def init_setup(self, fan_out, model, optimizer):
        self.po_map = self.gobj.ndata['orig']
        self.train_step_count = self.max_steps
        #print("train steps ",self.train_step_count, flush=True)
        #print("train nids ",self.tr_nid_part.shape[0], flush=True)
        #print("batch size ",self.batch_size, flush=True)

        self.labels = self.get_labels
        self.btchq = gqueue()

        self.model = model
        self.optimizer = optimizer

        if self.mode == 'iels':
            fanouts = [int(fanout) for fanout in fan_out.split(",")]
            self.sampler = NeighborSampler(fanouts, fused=False)
            self.iter_count = self.train_step_count + self.delay  # Added 'self.'
            return self.delay, self.iter_count
        elif self.mode == 'hels':
            self.sampler = NeighborSampler(self.gobj, fan_out)
            return 0, self.train_step_count
        else:
            print('Error: wrong mode')


    def adjust_batch(self, mask):
        self.tr_nid_part = th.nonzero(self.node_feats['train_mask'], as_tuple=True)[0]

        tsize = self.tr_nid_part.shape[0]

        my_steps = th.tensor(ceil(tsize/self.batch_size))

        my_steps_t = th.tensor(ceil(tsize/self.batch_size))
        dist.barrier()
        dist.all_reduce(my_steps_t, op=dist.ReduceOp.MAX)
        self.max_steps = int(my_steps_t)

        if self.rank == 0:
            print("| max_steps: {} | my_steps: {} |".
                  format(self.max_steps, int(my_steps)), flush=True)

        self.batch_size = tsize//self.max_steps
        self.bs_all = th.full([self.max_steps], self.batch_size,  dtype=th.int32)
        rem = tsize % self.max_steps
        self.bs_all[0:rem] += 1

        self.bs_all = th.cumsum(self.bs_all, 0)
        self.bs_all = th.cat((th.zeros(1, dtype=self.bs_all.dtype), self.bs_all), 0)

    def finalize(self):
        self.gobj.finalize()

    def printname(self):
        self.gobj.printname()

    def profile(self):
        self.gobj.profile()
        if self.rank == 0:
            print(f'ticd: {self.ticd: .3f}, ticlst: {self.ticlst: .3f}')

    def reset(self):
        self.ticd, self.ticlst = 0, 0

    def train(self):
        self.nn_mode = 'train'

    def val(self):
        self.nn_mode = 'val'

    def test(self):
        self.nn_mode = 'test'

    def optimize(self, step):
        if self.ps_overlap and (self.delay == 0  or step >= self.train_step_count - 1):
            self.model.param_sync()
            self.optimizer.step()
        elif not self.ps_overlap:
            if not self.use_ddp:
                self.model.param_sync()
            self.optimizer.step()

    def epoch_setup(self):
        self.reset()
        if self.mode == 'iels':
            self.ielspp_epoch_setup()
        else:
            self.hels_epoch_setup()

    def batch_setup(self, i):
        if self.mode == 'iels':
            return self.ielspp_batch_setup(i)
        else:
            return self.hels_batch_setup(i)

    def batch_apply(self, block, h):
        if self.mode=='hels':
            block.apply_hs_func(self.train_step_count, h)


    def ielspp_epoch_setup(self):
        self.gobj.reset()

        self.tr_nid_p = shuffle(self.tr_nid_part)
        self.tr_nid_o = self.po_map[self.tr_nid_p]

        self.btchq = gqueue()
        self.ielspp_batch_setup_startup()
        dist.barrier()

    def ielspp_batch_setup_startup(self):  # Added 'i' as an argument
        for i in range(self.delay):
            st, ed = self.bs_all[i], self.bs_all[i+1]
            seeds_o = self.tr_nid_o[st : ed ]  
            seeds_p = self.tr_nid_p[st : ed] 

            tic = time.time()
            if len(self.ntypes_id) > 1:
                seeds_od = {self.category: seeds_o - self.acc_onodes[self.tr_ntype]}
            else:
                seeds_od = seeds_o

            _, output_nodes, blocks = self.sampler.sample_blocks(self.g_orig, seeds_od)

            self.ticd += time.time() - tic
            input_nodes = blocks[0].srcdata[dgl.NID]
            batch = ielspp_batch(self.gobj, input_nodes)
            batch.blocks = blocks
            batch.labels = self.labels[seeds_p]  
            batch.seeds = output_nodes 
            batch.input_nodes = input_nodes
            batch.start_comm()
            self.btchq.push(batch)

    def ielspp_batch_setup(self, i):  
        if i < self.train_step_count:
            st, ed = self.bs_all[i], self.bs_all[i+1]
            seeds_o = self.tr_nid_o[st : ed ]  
            seeds_p = self.tr_nid_p[st : ed]  

            tic = time.time()
            if len(self.ntypes_id) > 1:
                seeds_od = {self.category: seeds_o - self.acc_onodes[self.tr_ntype]}
            else:
                seeds_od = seeds_o

            wait = 0
            if self.ps_overlap and (self.delay > 0 and i > self.delay):
                wait = 1
                self.model.param_sync_start() 

            _, output_nodes, blocks = self.sampler.sample_blocks(self.g_orig, seeds_od)

            self.ticd += time.time() - tic
            input_nodes = blocks[0].srcdata[dgl.NID]
            batch = ielspp_batch(self.gobj, input_nodes)
            batch.blocks = blocks
            batch.labels = self.labels[seeds_p]  # 'labels' needs to be defined
            batch.seeds = output_nodes #seeds_o
            batch.input_nodes = input_nodes

            if wait == 1:
                self.model.param_sync_end()
                self.optimizer.step()
            batch.start_comm()
            self.btchq.push(batch)

        if i >= self.delay:
            batch = self.btchq.pop()
            tic = time.time()
            batch.end_comm()
            self.ticlst += time.time() - tic

            batch_labels = batch.labels
            batch_seeds = batch.seeds
            input_nodes = batch.input_nodes
            if i < self.train_step_count:
                if self.btchq.size() != self.delay:
                    print('>> ', self.btchq.size(), ' ', self.delay)
                assert self.btchq.size() == self.delay

            return input_nodes, batch.blocks, batch.batch_feats, batch_labels, batch_seeds


    def hels_epoch_setup(self):
        self.train_nid_shuffled = shuffle(self.tr_nid_part)
        self.gobj.reset()
        dist.barrier()

    def hels_batch_setup(self, step):
        tic0 = time.time()
        sn = step * self.batch_size
        if step < self.train_step_count-1:
            seeds = self.train_nid_shuffled[sn: sn + self.batch_size]
        else:
            seeds = self.train_nid_shuffled[sn:]

        blocks = self.sampler.sample_blocks_syncord(seeds)  ##ndl w/ ord
        input_nodes = blocks[0].srcdata[dgl.NID]
        tic1 = time.time()
        self.ticd += tic1 - tic0
        batch_inputs, batch_labels = self.hels_load_subtensor(seeds, input_nodes)
        self.ticlst += time.time() - tic1

        for level in range(len(blocks)):
            blk = blocks[level]
            blk.init(self.gobj, input_nodes, level)

        return input_nodes, blocks, batch_inputs, batch_labels, seeds


    def hels_load_subtensor(self, seeds, input_nodes): ## assuming fp32 or bhf16, no conversion here
        feats = self.node_feats['feat']
        if self.tpp_impl:
            batch_inputs = gnn_utils.gather_features(feats, input_nodes.long())
        else:
            batch_inputs = feats[input_nodes]

        batch_labels = self.labels[seeds]
        return batch_inputs, batch_labels


    def hels_apply(self, block):
        block.apply_hs_func(self.nbatch)


    @property
    def get_labels(self):
        try:
            labels = self.node_feats['label'].long()
        except:
            labels = self.node_feats['labels'].long()

        return labels

    @property
    def get_feats(self):
        features  = self.node_feats['feat']
        return features

    @property
    def get_train(self):
        train_mask  = self.node_feats['train_mask']
        return train_mask

    @property
    def get_val(self):
        val_mask  = self.node_feats['val_mask']
        return val_mask

    @property
    def get_test(self):
        test_mask  = self.node_feats['test_mask']
        return test_mask


#####################################################################################
class distgnn_mb_inf:
    def __init__(self, pb, gobj, args, accon, accpn, ntid, category=''):
        self.mode = 'iels'

        self.g_orig = pb.g_orig
        self.pb = pb
        self.gobj = gobj

        self.ntypes = pb.g_orig.ntypes
        self.ntypes_id = []
        for keys, nt in enumerate(pb.g_orig.ntypes):
            self.ntypes_id.append(keys)
        
        self.acc_onodes = accon
        self.acc_pnodes = accpn
        self.tr_ntype = ntid
        self.node_feats = pb.node_feats

        self.val_batch_size = args.val_batch_size
        self.val_fraction = args.val_fraction
        self.delay = args.ielsqsize
        self.rank = args.rank
        self.category = category

        self.adjust_batch('val')


    def init_setup(self, fan_out):
        self.po_map = self.gobj.ndata['orig']
        self.ts_nid_o = self.po_map[self.ts_nid_part]
        self.val_step_count = self.max_steps
        self.btchq = gqueue()
        fanouts = [int(fanout) for fanout in fan_out.split(",")]
        self.sampler = NeighborSampler(fanouts, fused=False)

        self.labels = self.get_labels
        self.iter_count = self.val_step_count + self.delay
        return self.delay, self.iter_count


    def adjust_batch(self, mask):
        self.ts_nid_part = th.nonzero(self.node_feats['val_mask'], as_tuple=True)[0]
        val_nids = int(self.ts_nid_part.shape[0] * self.val_fraction)
        self.ts_nid_part = self.ts_nid_part[:val_nids]
        batch_size = self.val_batch_size

        my_steps = ceil(val_nids/batch_size)
        my_steps_t = th.tensor(my_steps)
        dist.barrier()
        dist.all_reduce(my_steps_t, op=dist.ReduceOp.MAX)
        self.max_steps = int(my_steps_t)

        if self.rank == 0:
            print("| max_val_steps: {} |  my_val_steps: {} |".
                  format(self.max_steps, int(my_steps)), flush=True)

        self.val_batch_size = val_nids//self.max_steps
        self.bs_all = th.full([self.max_steps], self.val_batch_size,  dtype=th.int32)
        rem = val_nids % self.max_steps
        self.bs_all[0:rem] += 1

        self.bs_all = th.cumsum(self.bs_all, 0)
        self.bs_all = th.cat((th.zeros(1, dtype=self.bs_all.dtype), self.bs_all), 0)


    def finalize(self):
        self.gobj.finalize()

    def profile(self):
        self.gobj.profile()

    def reset(self):
        pass

    def train(self):
        pass

    def epoch_setup(self):
        self.ielspp_epoch_setup()

    def batch_setup(self, i):
        return self.ielspp_batch_setup(i)

    def batch_apply(self, block, h):
        return

    def ielspp_epoch_setup(self):
        self.gobj.reset()

        self.ts_nid_p = self.ts_nid_part
        self.ts_nid_o = self.po_map[self.ts_nid_p]

        self.btchq = gqueue()
        self.ielspp_batch_setup_startup()
        dist.barrier()

    def ielspp_batch_setup_startup(self):
        for i in range(self.delay):
            st, ed = self.bs_all[i], self.bs_all[i+1]
            seeds_o = self.ts_nid_o[st : ed ]
            seeds_p = self.ts_nid_p[st : ed]

            if len(self.ntypes_id) > 1:
                seeds_od = {self.category: seeds_o - self.acc_onodes[self.tr_ntype]}
            else:
                seeds_od = seeds_o
            _, output_nodes, blocks = self.sampler.sample_blocks(self.g_orig, seeds_od)
            input_nodes = blocks[0].srcdata[dgl.NID]
            batch = ielspp_batch(self.gobj, input_nodes)
            batch.blocks = blocks
            batch.labels = self.labels[seeds_p]  # 'labels' needs to be defined
            batch.seeds = output_nodes #seeds_o
            batch.input_nodes = input_nodes
            batch.start_comm()
            self.btchq.push(batch)

        return None

    def ielspp_batch_setup(self, i):
        if i < self.val_step_count:
            st, ed = self.bs_all[i], self.bs_all[i+1]
            seeds_o = self.ts_nid_o[st : ed ]
            seeds_p = self.ts_nid_p[st : ed]

            if len(self.ntypes_id) > 1:
                seeds_od = {self.category: seeds_o - self.acc_onodes[self.tr_ntype]}
            else:
                seeds_od = seeds_o
            _, output_nodes, blocks = self.sampler.sample_blocks(self.g_orig, seeds_od)
            input_nodes = blocks[0].srcdata[dgl.NID]
            batch = ielspp_batch(self.gobj, input_nodes)
            batch.blocks = blocks
            batch.labels = self.labels[seeds_p]  # 'labels' needs to be defined
            batch.seeds = output_nodes #seeds_o
            batch.input_nodes = input_nodes
            batch.start_comm()
            self.btchq.push(batch)
            #return None

        if i >= self.delay:
            batch = self.btchq.pop()
            batch.end_comm()
            batch_labels = batch.labels
            batch_seeds = batch.seeds
            input_nodes = batch.input_nodes
            if i < self.val_step_count:
                if self.btchq.size() != self.delay:
                    print('>> ', self.btchq.size(), ' ', self.delay)
                assert self.btchq.size() == self.delay

            return input_nodes, batch.blocks, batch.batch_feats, batch_labels, batch_seeds

    @property
    def get_labels(self):
        try:
            labels = self.node_feats['label'].long()
        except:
            labels = self.node_feats['labels'].long()

        return labels

    @property
    def get_feats(self):
        features  = self.node_feats['feat']
        return features

    @property
    def get_train(self):
        train_mask  = self.node_feats['train_mask']
        return train_mask

    @property
    def get_val(self):
        val_mask  = self.node_feats['val_mask']
        return val_mask

    @property
    def get_test(self):
        test_mask  = self.node_feats['test_mask']
        return test_mask


def compute_nclasses(pb, num_ranks):
    tensor = th.unique(pb.node_feats['labels'])
    send_tensor = tensor
    scount = [tensor.shape[0]]
    for i in range(1, num_ranks):
        send_tensor = th.cat((send_tensor, tensor),0)
        scount.append(tensor.shape[0])

    send_sr, recv_sr = alltoall_s(scount)
    recv_labels = th.empty(recv_sr.sum(), tensor.dtype)
    dist.all_to_all_single(recv_labels, send_tensor, rcount.tolist(), scount.tolist())
    uniq = th.unique(th.sort(recv_labels))
    n_classes = torch.count_nonzero(~torch.isnan(tensor))
    return n_classes

def find_tot_nodes(node_feats):
    N = th.tensor((node_feats['inner_node'] == 1).sum(), dtype=th.int64)
    dist.all_reduce(N)
    return int(N)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print('[aux_func] Checkpointing..', flush=True)
    th.save(state, filename, _use_new_zipfile_serialization=True)

def shuffle(ten):
    idx = th.randperm(ten.shape[0])
    ten = ten[idx]
    return ten

class AverageMeter(object):
    """Computes and stores the average and current value"""

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
