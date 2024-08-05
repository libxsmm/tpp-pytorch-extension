import os, psutil, sys, gc, time, random
from math import ceil, floor
import dgl
import torch as th
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from .. import function as fn
from ..utils import expand_as_pair, check_eq_shape
from ..heterograph import DGLGraph as DGLHeteroGraph
from .communicate import alltoall_s
from tpp_pytorch_extension.gnn.common import gnn_utils


## Cache feats
class hels_cache:
    def __init__(self, dgmb, level):
        self.rank = dgmb.rank
        #self.num_parts = dgmb.num_parts
        self.min_life = dgmb.min_life
        self.life = dgmb.max_life   #5      ## expiry
        self.cache_size = dgmb.cache_size ## 10
        self.max_feat_size = 512    ## make a parameter from partition book (pb)
        self.N = dgmb.N
        self.feat_size = dgmb.feats.shape[1]
        self.feat_dtype = dgmb.feats.dtype

        self.piggy_bag = 1
        self.cache_ptr = 0
        self.level = level
        if dgmb.part_method == "metis":
            self.piggy_bag = 0

        self.batch_id = 0
        self.cachemap = th.tensor([-200 for i in range(dgmb.N)], dtype=th.int64)

        max_age = self.life + 1
        self.storage_aux = th.tensor([-99 for i in range(self.cache_size)], dtype = self.feat_dtype)
        self.storage_deg  =self.storage_aux
        #self.storage_nodes = th.empty(self.cache_size, dtype = th.int32)
        self.buf_feats = th.empty(self.cache_size * self.max_feat_size, dtype = self.feat_dtype)

        self.wt = th.tensor([max_age for i in range(self.cache_size)], dtype = th.int32)
        ## reverse pointer to cachemap which contains oid pointer to cache buffer
        self.rptr = th.tensor([-1 for i in range(self.cache_size)], dtype = th.int32)

        self.reset_timers()


    def reset_timers(self):
        self.batch_id = 0
        self.store_tim = 0
        self.load_tim = 0

    def display_timers(self):
        print('\tCache, load time: {:.4f} store_time: {:.4f}'.format(self.load_tim, self.store_tim))

    def inc_age(self):
        self.batch_id += 1
        self.wt = self.wt + 1

    def sort(self):
        self.age_index = th.empty([i for i in range(self.cache_size)], dtype = th.int32)

    def set_fs(self, feat_size):
        self.feat_size = feat_size
        assert self.feat_size <= self.max_feat_size
        newl = self.cache_size * self.feat_size
        self.buf_feats = self.buf_feats[:newl].view(self.cache_size, self.feat_size)

    def check_eq(self, a, b):
        assert a == b

    def cache_reset(self):
        if self.rank == 0:
            self.display_timers()

        self.feat_size = 0
        max_age = self.life + 1
        self.wt = th.tensor([max_age for i in range(self.cache_size)], dtype = th.int32)
        self.rptr = th.tensor([-1 for i in range(self.cache_size)], dtype = th.int32)
        self.storage_deg = th.tensor([-99 for i in range(self.cache_size)],
                                     dtype = self.feat_dtype)

        ind = th.nonzero(self.cachemap != -200, as_tuple=True)[0]
        if self.rank == 0:
            print('\t> hs cache fill, avg occ: {:.4f}, %occ: {:.2f}'.
                  format(ind.shape[0], ind.shape[0]*1.0/self.cache_size))

        self.cachemap[ind] = -200

        ## validation
        ind = th.nonzero(self.cachemap != -200, as_tuple=True)[0]
        assert ind.shape[0] == 0
        self.reset_timers()

    def cache_store(self, data):
        feats, nodes, degs = data
        tic = time.time()
        if self.feat_size == 0:
            self.set_fs(feats.shape[1])
        else:
            assert feats.shape[1] == self.feat_size

        cache_ptr_t = th.tensor([self.cache_ptr], dtype=th.int32)
        hval = -200
        rval = -1
        size = self.cache_size - self.cache_ptr
        if size < 0:
            self.cache_ptr = 0
            size = feats.size(0)

        if nodes.shape[0] > 0:
            cache_data = self.cachemap, self.rptr, self.wt, nodes, self.buf_feats,\
                     feats, feats[:size][:], feats[size:][:],\
                     cache_ptr_t, int(self.cache_size), int(hval), int(rval)
            gnn_utils.cache_store(cache_data)
            self.cache_ptr = int(cache_ptr_t)

        toc = time.time()
        self.store_tim += toc - tic

    def cache_lookup(self, oid):
        if self.feat_size == 0:
            print('Cache empty....')
            return
        tic = time.time()
        bitval_oid = self.cachemap[oid.long()]
        inda = th.nonzero(bitval_oid != -200, as_tuple=True)[0]
        bitval = bitval_oid[inda]

        if self.level == 0:
            lookup_loc = bitval
            oid_index = inda
        else:
            age = self.wt[bitval]
            indb = th.nonzero(((age >= self.min_life) & (age <= self.life)), as_tuple=True)[0]
            lookup_loc = bitval[indb]
            oid_index = inda[indb]   ## oid with non -200 and life < self.life

        toc = time.time()
        self.load_tim += toc - tic
        return oid_index, lookup_loc


    def cache_load(self, lookup_loc):
        if self.feat_size == 0:
            print('Cache empty....')
            return

        if lookup_loc.shape[0] > 0:
            assert self.feat_size != 0
            feats = self.buf_feats.view(self.cache_size, self.feat_size)
            gat_feats =  gnn_utils.gather_features(feats, lookup_loc)

            return gat_feats
        return None

    def fused_lookup_load(self, oid):
        if self.feat_size == 0:
            print('Cache empty....')
            return
        tic = time.time()

        assert self.feat_size != 0
        feats = self.buf_feats.view(self.cache_size, self.feat_size)
        if self.level == 0:
            oid_index, gat_data = gnn_utils.cache_load(self.cachemap, oid, feats)
        else:
            oid_index, gat_data = gnn_utils.cache_load(self.cachemap, oid, feats,\
                self.wt, self.level, self.min_life, self.life)

        #oid_index2, loc = cache_lookup(oid)
        #gat_data2 =  gnn_utils.gather_features(feats, lookup_loc)
        #assert th.allclose(g_orig.ndata['feat'][input_nodes], batch_inputs, atol=0.01) == Tr

        toc = time.time()
        self.load_tim += toc - tic

        return oid_index, gat_data, self.feat_size
