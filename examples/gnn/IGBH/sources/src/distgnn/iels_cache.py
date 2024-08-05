import os, psutil, sys, gc, time, random
from math import ceil, floor
import dgl
import torch as th
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from tpp_pytorch_extension.gnn.common import gnn_utils

stats=False
chk = True

class feat_cache:
    def __init__(self, pb, sn_db, args):
        self.rank = args.rank
        self.num_parts = pb.num_parts
        self.cache_size = args.cache_size ## 10
        self.N = pb.g_orig.number_of_nodes()      ## total nodes in original graph
        self.feat_size = pb.node_feats['feat'].shape[1]
        self.cur_itr = 0

        self.cache_offset, self.base = 0, 0
        self.n = (pb.graph.ndata['inner_node'] == 1).sum() ##pb.node_feats['feat'].shape[0]
        self.feat_dtype = pb.node_feats['feat'].dtype

        if sn_db == None:
            #self.cachemap = th.tensor([-1 for i in range(self.N)], dtype=th.int64)
            self.cachemap = th.tensor([-1 for i in range(self.N)], dtype=th.int32)
        else:
            self.cachemap = sn_db  ##resuse cachemap created by dms
            if chk:
                self.immutable = sn_db.clone()

        self.buf_feats = th.empty([self.cache_size, self.feat_size], dtype = self.feat_dtype)
        self.wt = th.zeros(self.cache_size, dtype=th.int32)
        ## reverse pointer to cachemap which contains oid pointer to cache buffer
        self.rptr = th.tensor([-1 for i in range(self.cache_size)], dtype = th.int64)

        self.reset()
        self.cache_occ = -1


    ## fused cache buffer with partition local features
    def fuse_buf(self, feats):
        self.buf_feats = th.cat((feats, self.buf_feats), 0)
        self.base = self.n
        self.fused = True
        return self.buf_feats

    def reset(self):
        self.load_tim, self.store_tim = 0, 0


    def display(self):
        print("load time: {:.4f} ".format(self.load_tim))
        print("save time: {:.4f}".format( self.store_tim))
        print('cache size: {:.4f}, occ: {:.4f}'.format(self.cache_size, self.cache_occ))
        print("############################################")

    def get_cache_lines(self, num_nodes_to_cache):
        if self.cache_size - self.cache_offset > num_nodes_to_cache:
            poffset = self.cache_offset
            self.cache_offset += num_nodes_to_cache
            index = th.tensor([ i for i in range(poffset, self.cache_offset)], dtype=th.int32)
        else:
            index1 = th.tensor([ i for i in range(self.cache_offset, self.cache_size)], dtype=th.int32)
            self.cache_offset = num_nodes_to_cache - (self.cache_size - self.cache_offset)
            index2 = th.tensor([ i for i in range(self.cache_offset)], dtype=th.int32)
            index = th.cat((index1, index2), 0)

        return index


    def sample_nodes(self, nodes, n, sample='random'):
        assert nodes.shape[0] >= n
        if sample == 'naive':
            return nodes[:n], th.arange(0, n)
        elif sample == 'random':
            #random_values = random.sample(range(nodes.shape[0]), n)
            random_values = th.randint(0, nodes.shape[0], (n,))
            return nodes[random_values], random_values


    def cache_store(self, feats, nodes, policy='naive'):
        assert nodes.shape[0] < self.cache_size, 'Insufficient cache size.'

        tic = time.time()
        if policy == 'naive':
            clines = self.get_cache_lines(nodes.shape[0])

        elif policy == 'lru':
            ## lru
            val, ind = th.sort(self.wt)
            clines = ind[:nodes.shape[0]]
        else:
            print('Error: Incorrect cache store policy!')

        ## reset selected clines cachemap
        rptr = self.rptr[clines]
        ind =  th.nonzero(rptr != -1, as_tuple=True)[0]
        rptr = rptr[ind]
        self.cachemap[rptr] = -1

        if chk:
            assert (self.immutable[rptr] == -1).sum() == rptr.shape[0]
        #    print((self.cachemap != -1).sum(), ' ', self.n)
        #    assert (self.cachemap != -1).sum() == self.n

        ## set reverse pointer from clines to cachemap
        self.rptr[clines] = nodes
        self.wt[clines] = self.cur_itr
        self.cur_itr += 1

        ## set cachemap entries
        if self.fused:
            clines += self.base

        self.cachemap[nodes] = clines ##.long()
        if stats:
            self.cache_occ += (self.cachemap != -1).sum()
        if chk:
            assert (self.immutable[nodes] == -1).sum() == nodes.shape[0]
        #    assert (self.cachemap != -1).sum() == self.n + clines.shape[0]

        # scatter feats to the cache buffer
        gnn_utils.scatter_features(feats, clines.long(), self.buf_feats, 0)
        toc = time.time()
        self.store_tim += toc - tic

    def cache_load(self, nodes):
        tic = time.time()
        clines_all = self.cachemap[nodes.long()]
        ind = th.nonzero(clines_all != -1, as_tuple=True)[0]
        clines = clines_all[ind]
        feats = gnn_utils.gather_features(self.buf_feats, clines.to(th.long))

        if self.fused:
            clines -= self.base

        self.wt[clines] = self.cur_itr
        self.cur_itr += 1
        toc = time.time()
        self.load_tim += toc - tic
