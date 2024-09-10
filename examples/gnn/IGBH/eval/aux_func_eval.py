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
import os

class distgnn_mb_inf:
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
                self.val_ntype = keys
  
        del index
        self.acc_onodes = np.insert(np.cumsum(self.num_onodes), 0, 0)
        self.acc_pnodes = np.insert(np.cumsum(self.num_pnodes), 0, 0)
         
        self.pb = pb
        if self.mode == 'iels':
            self.gobj = iels_master(pb, args)
        elif self.mode == 'hels':
            self.gobj = hels_master(pb, args)

        self.node_feats = pb.node_feats

        self.batch_size = args.batch_size 
        self.delay = args.ielsqsize
        self.rank = args.rank
        self.use_tpp = args.use_tpp
        self.val_fraction = args.val_fraction
        self.profile = args.profile or args.tpp_profile

        self.adjust_batch('val')
        self.reset()

    def init_setup(self, fan_out):
        self.po_map = self.gobj.ndata['orig']
        self.val_step_count = self.max_steps

        self.labels = self.get_labels
        self.btchq = gqueue()

        fanouts = [int(fanout) for fanout in fan_out.split(",")]
        self.sampler = NeighborSampler(fanouts, fused=True)
        self.iter_count = self.val_step_count + self.delay  # Added 'self.'
        return self.delay, self.iter_count

    def adjust_batch(self, mask):
        self.val_nid_part = th.nonzero(self.node_feats['val_mask'], as_tuple=True)[0]
        if self.profile:
            val_nids = 10000
        else:
            val_nids = int(self.val_nid_part.shape[0] * self.val_fraction)
        self.val_nid_part = self.val_nid_part[:val_nids]

        my_steps = th.tensor(ceil(val_nids/self.batch_size))
        self.my_steps = my_steps

        dist.barrier()
        dist.all_reduce(my_steps, op=dist.ReduceOp.MAX)
        self.max_steps = int(my_steps)

        if self.rank == 0:
            print("| max_steps: {} | my_steps: {} |".
                  format(self.max_steps, int(my_steps)), flush=True)

        self.batch_size = val_nids//self.max_steps
        self.bs_all = th.full([self.max_steps], self.batch_size,  dtype=th.int32)
        rem = val_nids % self.max_steps
        self.bs_all[0:rem] += 1

        self.bs_all = th.cumsum(self.bs_all, 0)
        self.bs_all = th.cat((th.zeros(1, dtype=self.bs_all.dtype), self.bs_all), 0)

    def finalize(self):
        self.gobj.finalize()

    def printname(self):
        self.gobj.printname()

    def reset(self):
        self.ticd, self.ticlst = 0, 0

    def val(self):
        self.nn_mode = 'val'

    def bootstrap(self):
        self.ielspp_bootstrap()

    def batch_setup(self, i):
        return self.ielspp_batch_setup(i)

    def batch_apply(self, block, h):
        if self.mode=='hels':
            block.apply_hs_func(self.val_step_count, h)

    def ielspp_bootstrap(self):
        self.gobj.reset()
        self.val_nid_p = self.val_nid_part
        self.val_nid_o = self.po_map[self.val_nid_p]

        self.btchq = gqueue()
        self.ielspp_batch_setup_startup()
        dist.barrier()

    def ielspp_batch_setup_startup(self):  # Added 'i' as an argument
        for i in range(self.delay):
            st, ed = self.bs_all[i], self.bs_all[i+1]
            seeds_o = self.val_nid_o[st : ed ]  
            seeds_p = self.val_nid_p[st : ed] 

            tic = time.time()
            if len(self.ntypes_id) > 1:
                seeds_od = {self.category: seeds_o - self.acc_onodes[self.val_ntype]}
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
        if i < self.val_step_count:
            st, ed = self.bs_all[i], self.bs_all[i+1]
            seeds_o = self.val_nid_o[st : ed ]  
            seeds_p = self.val_nid_p[st : ed]  

            tic = time.time()
            if len(self.ntypes_id) > 1:
                seeds_od = {self.category: seeds_o - self.acc_onodes[self.val_ntype]}
            else:
                seeds_od = seeds_o

            _, output_nodes, blocks = self.sampler.sample_blocks(self.g_orig, seeds_od)

            self.ticd += time.time() - tic
            input_nodes = blocks[0].srcdata[dgl.NID]
            batch = ielspp_batch(self.gobj, input_nodes)
            batch.blocks = blocks
            batch.labels = self.labels[seeds_p]  # 'labels' needs to be defined
            batch.seeds = output_nodes #seeds_o
            batch.input_nodes = input_nodes

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
            batch_feats = batch.batch_feats
            batch_scf = batch.batch_scf
            if i < self.val_step_count:
                if self.btchq.size() != self.delay:
                    print('>> ', self.btchq.size(), ' ', self.delay)
                assert self.btchq.size() == self.delay

            return input_nodes, batch.blocks, batch_feats, batch_scf, batch_labels, batch_seeds

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
    def get_val(self):
        val_mask  = self.node_feats['val_mask']
        return val_mask

    @property
    def get_test(self):
        test_mask  = self.node_feats['test_mask']
        return test_mask

