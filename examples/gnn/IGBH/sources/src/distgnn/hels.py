## Historical Embedding Protocol Implementation
## Author: Md. Vasmiuddin <vasimuddin.md@intel.com>
##

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

from .communicate import *
from .hels_cache import hels_cache
from . import set_mode

from .queue import gqueue, gstack
from tpp_pytorch_extension.gnn.common import gnn_utils

## hist emb comms
heq_comm1 = gqueue()
heq_comm2 = gqueue()
heq_st1 = gqueue()
heq_st2 = gqueue()
## fresh emb comms
q_comm1 = gqueue()
q_comm2 = gqueue()
q_st1 = gqueue()
q_st2 = gqueue()
hels_stk_comm = gstack()

pq_comm1 = gqueue()
pq_comm2 = gqueue()
pq_st1 = gqueue()
pq_st2 = gqueue()


validation=False
debug = False
checks=False

class csr_main:
    def __init__(self):
        self.s = 0
        self.d = 0
        self.indptr = 0
        self.data = None

## new csr format (defined above) having same interface as csr in spmm
## for use in tppfying mapped_spmm_emb
class csr_main_v2:
    def __init__(self):
        self.indices = 0
        self.dst = 0
        self.indptr = 0
        self.num_rows = 0
        self.data = None


class csr_():
    def __init__(self):
        self.indptr, self.indices = [0], []
        self.dt = {}
        self.acc = 0

class cache_stats():
    def __init__(self, nlayers):
        self.hits = 0
        self.exact_hits = 0
        self.reqs = 0
        self.mb_nodes, self.mb_bn, self.nmb = 0, 0, 0
        self.nlayer = nlayers
        self.nodes_recv, self.nodes_send = 0, 0

    def reset(self):
        self.hits = 0
        self.exact_hits = 0
        self.reqs = 0
        self.mb_nodes, self.mb_bn, self.nmb = 0, 0, 0
        self.nodes_recv, self.nodes_send = 0, 0

    def display(self):
        print("\t| hitrate {:.2f} | he_hits: {} | exact_hits: {} | tot_reqs: {} | "
        "#send nodes: {} | #recv nodes: {} |".
              format(self.hits*1.0/(self.reqs + 1), self.hits, self.exact_hits, self.reqs,
                     self.nodes_send, self.nodes_recv))
        print('\t \t| Avg, #level nodes: {:.2f} | #level bn nodes: {:.2f} |'.
              format(self.mb_nodes*1.0/(self.nmb+1), self.mb_bn*1.0/(self.nmb+1)))


class timer():
    def __init__(self, rank, num_parts):
        self.rank, self.num_parts = rank, num_parts
        self.convert_tim = 0.0
        self.init_tim = 0.0
        self.hels_tim, self.hels_init_tim = 0.0, 0.0
        self.hels_wait_tim, self.cache_wait_tim = 0.0, 0.0
        self.local_agg_tim = 0.0
        self.total_agg_tim = 0.0

        self.cache_in_tim, self.cache_out_tim = 0.0, 0.0
        self.gather_tim, self.async_comm_tim = 0.0, 0.0
        self.cache_in_inner_tim, self.map_tim = 0.0, 0.0
        self.halo_tim, self.r1_tim, self.r2_tim = 0.0, 0.0, 0.0
        self.fn_tim, self.map_in_tim, self.map_in_tim2 = 0.0, 0.0, 0.0
        self.alltoall_sync, self.halo_tim = 0.0, 0.0
        self.cache_out_scatter_tim, self.cache_out_lu_tim = 0.0, 0.0
        self.he_store_tim = 0.0

    def reset(self):
        self.convert_tim = 0.0
        self.init_tim = 0.0
        self.hels_tim, self.hels_init_tim = 0.0, 0.0
        self.hels_wait_tim, self.cache_wait_tim = 0.0, 0.0
        self.local_agg_tim = 0.0
        self.total_agg_tim = 0.0
        self.cache_in_tim, self.cache_out_tim = 0.0, 0.0
        self.cache_in_inner_tim, self.gather_tim = 0.0, 0.0
        self.async_comm_tim, self.map_tim = 0.0, 0.0
        self.halo_tim, self.r1_tim, self.r2_tim = 0.0, 0.0, 0.0
        self.fn_tim, self.map_in_tim, self.map_in_tim2 = 0.0, 0.0, 0.0
        self.alltoall_sync, self.halo_tim = 0.0, 0.0
        self.cache_out_scatter_tim, self.cache_out_lu_tim = 0.0, 0.0
        self.he_store_tim = 0.0


    def display(self):
        print('\n\tconvert time: {:.4f}'.format(self.convert_tim))
        print('\thels init time: {:.4f}'.format(self.init_tim))
        print('\ths create halo db time: {:.4f}'.format(self.halo_tim))
        print()
        print('\t>> preproc: {:.4f}, hels comm time: {:.4f}'.format(self.hels_init_tim, self.hels_tim))
        print('\t>> hels r1 (scatter) time: {:.4f}, hels_wait_tim: {:.4f}'.
              format(self.r1_tim, self.hels_wait_tim))
        print('\tlocal agg time: {:.4f}'.format(self.local_agg_tim))
        print('\tfind halo time: {:.4f}'.format(self.halo_tim))
        print('\t>> he cache send agg time: \t{:.4f}'.format(self.cache_in_tim))
        print('\t>>> he cache (inner) mapping time: {:.4f}'.format(self.map_tim))
        print('\t>>>> he cache (inner) map_in time: {:.4f}, {:.4f}, {:.4f}'.format(self.fn_tim, self.map_in_tim, self.map_in_tim2))
        print('\t>>> he cache (inner) node sampling time: {:.4f}'.format(self.cache_in_inner_tim))
        print('\t>>> he cache (inner) alltoall barrier time: {:.4f}'.format(self.alltoall_sync))
        print('\t>>> he cache (inner) gather agg time: {:.4f}'.format(self.gather_tim))
        print('\t>>> he cache (inner) async comm time: {:.4f}'.format(self.async_comm_tim))
        print('\t>> he wait time: {:.4f}'.format(self.cache_wait_tim))
        print('\t>> he recv (scatter) time: {:.4f}'.format(self.r2_tim))
        print('\t>>> he store  time: {:.4f}'.format(self.he_store_tim))
        print('\t>> he cache out agg time: {:.4f}'.format(self.cache_out_tim))
        print('\t>>> he cache out lu time: {:.4f}'.format(self.cache_out_lu_tim))
        print('\t>>> he cache out scatter time: {:.4f}'.format(self.cache_out_scatter_tim))
        print('\ttotal agg time: {:.4f}'.format(self.total_agg_tim))

        print(flush=True)

    def dist_gather(self, my_tensor):
        tensor = th.tensor([self.rank], dtype=th.float32)
        output = [tensor.clone() for _ in range(self.num_parts)]
        if self.rank == 0:
            dist.gather(tensor=my_tensor, gather_list = output, dst=0)
        else:
            dist.gather(tensor=my_tensor, gather_list = [], dst=0)

        output = th.tensor(output)
        return output

    def dis_all(self):
        cache_out_tim = th.tensor(self.cache_out_tim)
        cache_in_tim = th.tensor(self.cache_in_tim)
        alltoall_sync = th.tensor(self.alltoall_sync)
        r2_tim = th.tensor(self.r2_tim)
        local_agg_tim = th.tensor(self.local_agg_tim)
        map_tim = th.tensor(self.map_tim)

        output = self.dist_gather(r2_tim)
        if self.rank == 0:
            print()
            print('he scatter (r2):  | min: {:.4f}, avg: {:.4f}, max: {:.4f}'.
                  format(float(output.min()), float(output.mean()), float(output.max()) ) )

        output = self.dist_gather(cache_out_tim)
        if self.rank == 0:
            print('cache_out_tim:    | min: {:.4f}, avg: {:.4f}, max: {:.4f}'.
                  format(float(output.min()), float(output.mean()), float(output.max()) ) )

        output = self.dist_gather(local_agg_tim)
        if self.rank == 0:
            print('local_agg_tim:    | min: {:.4f}, avg: {:.4f}, max: {:.4f}'.
                  format(float(output.min()), float(output.mean()), float(output.max()) ) )

        output = self.dist_gather(cache_in_tim)
        if self.rank == 0:
            print('cache_in_tim:     | min: {:.4f}, avg: {:.4f}, max: {:.4f}'.
                  format(float(output.min()), float(output.mean()), float(output.max()) ) )

        output = self.dist_gather(map_tim)
        if self.rank == 0:
            print('map_tim:          | min: {:.4f}, avg: {:.4f}, max: {:.4f}'.
                  format(float(output.min()), float(output.mean()), float(output.max()) ) )

        output = self.dist_gather(alltoall_sync)
        if self.rank == 0:
            print('alltoall_sync:    | min: {:.4f}, avg: {:.4f}, max: {:.4f}'.
                  format(float(output.min()), float(output.mean()), float(output.max()) ) )

            print()


################################################################
def hels_master_func(pb, args):
    hels_obj = hels_master(pb.graph._graph, pb.graph._ntypes, pb.graph._etypes, \
                                pb.graph._node_frames, pb.graph._edge_frames)

    hels_obj.hels_init(pb, args)
    return hels_obj


class hels_master(DGLHeteroGraph):
    def hels_init(self, pb, args):
        set_mode('hels')
        self.part_method = "metis"
        self.N = pb.g_orig.number_of_nodes()      ## total nodes in original graph
        self.rank      = args.rank
        self.num_parts = pb.num_parts
        self.nlayers   = args.n_layers
        self.timer = timer(self.rank, self.num_parts)

        self.feats = pb.node_feats['feat']
        self.feat_size = self.feats.shape[1]

        self.ncache = args.ncache  ##-5 means cache inf/all cacheable items
        self.min_life = 0   #min_cahe line life,  Note: comms delay/aync is a factor here
        self.max_life = args.max_life   ## max cache life
        self.dl = args.dl   ## delay in push model

        self.layer_itr = 0
        self.epoch = []
        self.o2l_map = []
        self.ocomms = []
        self.mb_nodes, self.mb_bn = [], []
        for i in range(self.nlayers):
            if args.b2b:
                self.ocomms.append(th.tensor([-100 for j in range(self.N)], dtype=th.int32))
                self.o2l_map.append(th.tensor([-100 for j in range(self.N)], dtype=th.int64))
            self.epoch.append(0)

        ## To maintain async history
        self.cache_size = args.cache_size
        ncache_lim = ceil(float(args.cache_size)/self.num_parts)
        assert self.ncache > 0, "ncache should be > 0"
        assert self.ncache <= ncache_lim, "ncache should be < cache_size/num_parts!"

        self.hels, self.hels_stats = [], []
        for i in range(self.nlayers):
            self.hels.append(hels_cache(self, i))  ## s/w managed cache
            self.hels_stats.append(cache_stats(self.nlayers))

        self.create_sn_bitmap()

        self.enable_hec = args.enable_hec
        self.b2b =args.b2b
        if args.enable_hec and self.rank == 0:
            print('######### Historical embedding usage ENABLED.')
        elif args.enable_hec == False and self.rank == 0:
            print('######### Historical embedding usage DISABLED.')
        if args.b2b and self.rank == 0:
            print('######### b2b usage ENABLED.')
        elif args.b2b == False and self.rank == 0:
            print('######### b2b usage DISABLED.')


    def create_sn_bitmap(self):
        """
        """
        self.ndata['degs'] = self.in_degrees()
        self.deg_max = self.ndata['degs'].max()
        self.deg_min = self.ndata['degs'].min()
        dist.all_reduce(self.deg_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(self.deg_min, op=dist.ReduceOp.MIN)
        self.deg_max_lg = int(th.log2(self.deg_max.float()))

        self.ndata['solid'] = th.zeros(self.number_of_nodes(), dtype=th.int32)  ## make it bool
        db = th.tensor([-100 for i in range(self.N)], dtype=th.int64)

        ninners = (self.ndata['inner_node'] == 1).sum()
        db[self.ndata['orig'][:ninners]] = th.arange(ninners)
        self.communicate_halo_nodes_orig()
        self.sn_db = []
        offset = 0
        for np in range(self.num_parts):
            try:
                nodes = self.recv_bn[offset : offset + self.recv_bn_count[np]]
            except:
                print("Error: recv_bn is not created in communicate_halo_nodes_orig(), exiting..")
                sys.exit(1)

            if np == self.rank:
                assert nodes.shape[0] == 0
            db_nodes = db[nodes.long()]
            ind = th.nonzero(db_nodes != -100, as_tuple=True)[0]
            db_nodes = db_nodes[ind]
            self.ndata['solid'][db_nodes.long()] = 1

            ## Create remote halo nodes database as marker in orig tensor
            if False:
                ten = th.zeros(self.N, dtype=th.bool)
                ten[nodes.long()] = True
            else:
                ten = th.zeros(self.N, dtype=th.int32)
                ten[nodes.long()] = 1
            self.sn_db.append(ten)  ## change the name to sn_db

        assert self.ndata['solid'][ninners:].sum() == 0
        self.recv_bn = None
        self.recv_bn_count = None


    def create_sn_bitmap_ext(self):
        self.communicate_halo_nodes_orig()
        orig2part = th.tensor([-100 for i in range(self.N)], dtype=th.int64)
        ninners = (self.ndata['inner_node'] == 1).sum()
        orig2part[self.ndata['orig'][:ninners]] = th.arange(ninners)

        self.sn_part_db = []
        offset = 0
        for np in range(self.num_parts):
            try:
                onodes = self.recv_bn[offset : offset + self.recv_bn_count[np]]
            except:
                print("Error: recv_bn is not created in communicate_halo_nodes_orig(), exiting..")
                sys.exit(1)

            if np == self.rank: assert onodes.shape[0] == 0
            pnodes = orig2part[onodes.long()]
            ind = th.nonzero(onodes != -100, as_tuple=True)[0]
            ten = th.tensor([-1 for i in range(self.number_of_nodes())], dtype=th.int32)
            ten[pnodes[ind]] = onodes[ind]
            self.sn_part_db.append(ten)

    def printname(self):
        print('hels file name:', os.path.basename(__file__))

    def reset(self):
        for i in range(self.nlayers):
            self.epoch[i] = 0
            self.hels[i].cache_reset()
            self.hels_stats[i].reset()

        self.layer_itr = 0
        self.timer.reset()
        print('comm size: ', pq_comm1.size())
        assert pq_comm1.size() == 0
        assert pq_comm2.size() == 0
        assert pq_st1.size() == 0
        assert pq_st1.size() == 0
        assert pq_st2.size() == 0
        assert pq_st2.size() == 0

        return

    def profile(self):
        ## timers
        if self.rank == 0:
            self.timer.display()
            ## stats
            for i in range(self.nlayers):
                self.hels_stats[i].display()

        #self.timer.dis_all()

    def finalize(self):
        if self.rank == 0:
            print('Cache settings: ')
            print('hs cache size: ', self.cache_size)
            print('hs limit on caching per iteration: ', self.ncache, flush=True)

        return

    def find_halo_nodes_orig(self):
        bn_inner = self.ndata['inner_node']
        bn_idx = th.nonzero(~(bn_inner.bool()), as_tuple=True)[0]

        bn_part = bn_idx    ## part node id for border nodes
        bn_orig = self.ndata['orig'][bn_part]    ## orig id for border nodes
        return bn_orig, bn_part


    def communicate_halo_nodes_orig(self):
        bn_orig, bn_part = self.find_halo_nodes_orig()

        vsize = bn_orig.shape[0]
        send_size = [vsize for i in range(self.num_parts)]
        send_size[self.rank] = 0
        send_sr, recv_sr = alltoall_s(send_size)

        scount = [0 for i in range(self.num_parts)]
        self.recv_bn_count = [0 for i in range(self.num_parts)]
        tsend, trecv = sum(send_sr), sum(recv_sr)  ##recv data

        send_bn = th.empty(tsend, dtype=th.int32)
        self.recv_bn = th.empty(trecv, dtype=th.int32)
        offset = 0
        for np in range(self.num_parts):
            if np != self.rank:
                send_bn[offset: offset + send_sr[np]] = bn_orig.to(th.int32)
            else:
                assert int(send_sr[np]) == 0
                assert int(recv_sr[np]) == 0

            scount[np] = int(send_sr[np])
            self.recv_bn_count[np] = int(recv_sr[np])
            offset += scount[np]

        req = dist.all_to_all_single(self.recv_bn, send_bn,
                                     self.recv_bn_count, scount,
                                     async_op=True)
        req.wait()
        return req


#############################################################################################

class hels_mini_batch(DGLHeteroGraph):
    def find_solid_nodes_orig(self):
        lnodes = self.level_nodes
        assert lnodes.shape[0] > 0
        #'''
        sn_solid = self.part.ndata['solid'][lnodes]

        sn_idx = th.nonzero(sn_solid == 1, as_tuple=True)[0]

        sn_part = lnodes[sn_idx]    ## part node id for halo nodes

        sn_orig = self.part.ndata['orig'][sn_part]    ## orig id for halo nodes
        # sn_batch = self.dstnodes()[sn_idx]
        sn_batch = self.srcnodes()[sn_idx]
        '''
        sn_orig, sn_batch, sn_part = gnn_utils.find_nodes(
            self.part.ndata['solid'],
            self.part.ndata['orig'],
            self.srcnodes(), lnodes, 'solid'
        )
        '''
        return sn_orig, sn_batch, sn_part

    ## Finds different node ids (batch, part-local, part-orig) of layered halo nodes
    ## Output:
    ## bn_orig - orig node id of halo nodes in this layer
    ## bn_part - local node id for halo nodes in this layer
    ## bn_batch - batch node id for halo nodes in this layer
    def find_halo_nodes_orig(self):
        #bn_map = th.zeros(self.part.N, dtype=th.int32) ## partition #nodes
        lnodes = self.level_nodes
        #'''
        bn_inner = self.part.ndata['inner_node'][lnodes]
        bn_idx = th.nonzero(bn_inner == 0, as_tuple=True)[0]

        bn_part = lnodes[bn_idx]    ## part node id for halo nodes
        bn_orig = self.part.ndata['orig'][bn_part]    ## orig id for halo nodes
        # bn_batch = self.dstnodes()[bn_idx]
        bn_batch = self.srcnodes()[bn_idx]
        '''
        bn_orig, bn_batch, bn_part = gnn_utils.find_nodes(
                self.part.ndata['inner_node'].to(th.int32),
                self.part.ndata['orig'],
                self.srcnodes(), lnodes, 'halo'
        )
        '''
        ## validation
        if validation:
            assert th.equal(bn_batch, bn_idx) == True

        return bn_orig, bn_part, bn_batch


    ## For a given level, bcast the orig id (not bitmap) of my halo nodes
    def communicate_halo_nodes_orig(self):
        bn_orig, bn_part, bn_batch = self.find_halo_nodes_orig()

        vsize = bn_orig.shape[0]
        assert vsize <= self.level_nodes.shape[0]
        send_size = [vsize for i in range(self.part.num_parts)]
        send_size[self.rank] = 0
        send_sr, recv_sr = alltoall_s(send_size)

        #tic = time.time()
        scount = [0 for i in range(self.part.num_parts)]
        self.recv_bn_count = [0 for i in range(self.part.num_parts)]
        tsend = sum(send_sr)
        trecv = sum(recv_sr)  ##recv data

        ## stats
        self.hels_stats.reqs += bn_orig.shape[0]

        send_bn = th.empty(tsend, dtype=th.int32)
        self.recv_bn = th.empty(trecv, dtype=th.int32)
        offset = 0
        for np in range(self.part.num_parts):
            if np != self.rank:
                send_bn[offset: offset + send_sr[np]] = bn_orig.to(th.int32)
                if checks:
                    oid = self.o2l_map[send_bn[offset: offset + send_sr[np]].long()]
                    ind = th.nonzero(oid == -100, as_tuple=True)[0]
                    assert ind.shape[0] == 0
            else:
                assert int(send_sr[np]) == 0
                assert int(recv_sr[np]) == 0

            scount[np] = int(send_sr[np])
            self.recv_bn_count[np] = int(recv_sr[np])
            offset += scount[np]

        req = dist.all_to_all_single(self.recv_bn, send_bn,
                                     self.recv_bn_count, scount,
                                     async_op=True)
        req.wait()
        #return req


    ## Finds xn of local bn (oid) w/ recv bn (oid)
    ## Here remote id is original node id
    def r2l_mapping_orig(self):
        self.xnbn = []; self.xrbn = []
        p_node_map = 0; offset = 0
        for np in range(self.part.num_parts):
            rbn_orig = self.recv_bn[offset : offset + self.recv_bn_count[np]]

            if False:  ## python code for C/C++ ext code in else
                l_lid = self.o2l_map[rbn_orig.long()]
                index = (l_lid != -100).nonzero(as_tuple=True)[0]
                r_lid2 = rbn_orig[index]
                l_lid2 = l_lid[index]
            else:
                r_lid2, l_lid2 = gnn_utils.r2l_map(self.o2l_map, rbn_orig.long())

            self.xrbn.append(r_lid2.long())  ## remote orig node ids
            self.xnbn.append(l_lid2.long())  ## corr. part node id
            offset += self.recv_bn_count[np]


    def db_r2l_mapping_orig(self):
        self.xnbn = []; self.xrbn = []; self.xlbn = []
        p_node_map = 0; offset = 0
        tic = time.time()
        sn_orig, sn_batch, sn_part = self.find_solid_nodes_orig()
        toc = time.time()
        self.part.timer.fn_tim += toc - tic

        timer = th.tensor([0], dtype=th.int64)

        for np in range(self.part.num_parts):
            db_t = self.part.sn_db[np]
            tic = time.time()
            r_lid2 = th.empty(0, dtype=th.int64)
            b_lid2 = th.empty(0, dtype=th.int64)
            l_lid2 = th.empty(0, dtype=th.int64)
            '''
            l_lid = db_t[sn_orig.long()]   #.to(th.device('cpu'))

            index = th.nonzero(l_lid == True, as_tuple=True)[0]
            r_lid2 = sn_orig[index]
            b_lid2 = sn_batch[index]
            l_lid2 = sn_part[index]
            '''
            if np != self.rank:
                r_lid2, b_lid2, l_lid2 = gnn_utils.db_r2l_map(db_t, sn_orig, sn_batch, sn_part)

            toc = time.time()
            self.part.timer.map_in_tim2 += toc - tic

            if debug and self.rank == np:
                assert r_lid2.shape[0] == 0

            self.xrbn.append(r_lid2.long())  ## remote local id
            self.xnbn.append(b_lid2.long())  ## corr. local batch node id
            self.xlbn.append(l_lid2.long())  ## corr. local node id

        self.part.timer.map_in_tim += (timer[0] *1.0)/2.4/1e9
        return self.xnbn, self.xrbn, self.xlbn


    def hels_to_remote_cache_send(self):
        h_emb = self.hels
        piggy_bag = 0
        #neigh = self.srcdata['h']
        neigh = self.feat
        feat_size = neigh.shape[1]

        scount = [0 for i in range(self.num_parts)]
        rcount = [0 for i in range(self.num_parts)]
        scount_nodes = [0 for i in range(self.num_parts)]
        rcount_nodes = [0 for i in range(self.num_parts)]

        tic = time.time()
        # xn of remote part nodes and current nodes orig
        self.db_r2l_mapping_orig()

        toc = time.time()
        self.part.timer.map_tim += toc - tic

        xnbn, xrbn, xlbn = self.xnbn, self.xrbn, self.xlbn
        hil, hi, lo = self.part.deg_max_lg, self.part.deg_max, self.part.deg_min
        thres = self.part.ncache

        timer = th.tensor([0], dtype=th.int64)

        tic = time.time()
        offset = 0
        send_size = []
        for np in range(self.num_parts):
            if xnbn[np].shape[0] > self.part.ncache:
                degs = self.part.ndata['degs'][self.xlbn[np]]
                x, y = gnn_utils.node_sampling(degs, xnbn[np].long(), xrbn[np].long(), hil, thres)
                xnbn[np] = x
                xrbn[np] = y

            send_size.append(xnbn[np].shape[0])

        toc = time.time()
        self.part.timer.cache_in_inner_tim += toc - tic

        tic = time.time()
        send_size[self.rank] = 0
        send_sr, recv_sr, sync_req = alltoall_s_exp(send_size)

        toc = time.time()
        self.part.timer.alltoall_sync += toc - tic

        tic = time.time()
        tsend= sum(send_sr)  ##recv data
        send_feat = th.empty(tsend * (feat_size + piggy_bag), dtype=neigh.dtype)
        send_nodes = th.empty(tsend, dtype=th.int64)

        index = th.tensor([], dtype=th.int64)
        offsetv, offseti = 0, 0
        for np in range(self.part.num_parts):
            offsetv += int(send_sr[np]) * (feat_size + piggy_bag)
            scount[np] = int(send_sr[np]) * (feat_size + piggy_bag)
            rcount[np] = int(recv_sr[np]) * (feat_size + piggy_bag)

            index = th.cat((index,xnbn[np].long()))
        assert scount[self.rank] == 0
        send_feat_ = gnn_utils.gather_features(neigh, index)
        send_feat = send_feat_.view(send_feat_.shape[0] * send_feat_.shape[1])

        piggy_bag = 0
        offseti, offsetv = 0, 0
        for np in range(self.part.num_parts):
            index = th.arange(xrbn[np].shape[0])
            if self.rank != np:
                gnn_utils.gather_n_store_offset(xrbn[np].long(), index, send_nodes, offseti, offsetv)
            offsetv += int(send_sr[np])
            scount_nodes[np] = int(send_sr[np])
            rcount_nodes[np] = int(recv_sr[np])
            self.hels_stats.nodes_send += index.shape[0]

        assert scount_nodes[self.rank] == 0
        toc = time.time()
        self.part.timer.gather_tim += toc - tic

        tic = time.time()
        sync_req.wait()
        trecv = sum(recv_sr)
        assert trecv >= 0
        recv_feat = th.empty(trecv * (feat_size + piggy_bag), dtype=neigh.dtype)
        recv_nodes = th.empty(trecv, dtype=th.int64)
        for np in range(self.part.num_parts):
            rcount[np] = int(recv_sr[np]) * (feat_size + piggy_bag)
            rcount_nodes[np] = int(recv_sr[np])

        req2 = dist.all_to_all_single(recv_feat, send_feat, rcount, scount, async_op=True)
        pq_comm2.push(req2)
        pq_st2.push(recv_feat)
        pq_st2.push(rcount)

        req1 = dist.all_to_all_single(recv_nodes, send_nodes, rcount_nodes, scount_nodes, async_op=True)
        pq_comm1.push(req1)
        pq_st1.push(recv_nodes)
        pq_st1.push(rcount_nodes)

        toc = time.time()
        self.part.timer.async_comm_tim += toc - tic


    def hels_to_remote_cache_recv(self):
        h_emb = self.hels
        piggy_bag = 0
        #neigh = self.srcdata['h']
        neigh = self.feat
        feat_size = neigh.shape[1]

        if pq_comm1.empty() or pq_comm2.empty():
            print(self.part.epoch[self.part.layer_itr], ' ', self.part.layer_itr,
                  ' Error: Empty queue: q_comm!', ' ', pq_comm1.empty(), ' ', pq_comm2.empty())
            sys.exit(1)

        tic = time.time()
        req = pq_comm2.pop()
        req.wait()
        req = pq_comm1.pop()
        req.wait()
        recv_feat  = pq_st2.pop()
        rcount  = pq_st2.pop()
        recv_nodes = pq_st1.pop()
        rcount_nodes = pq_st1.pop()
        self.part.timer.cache_wait_tim += time.time() - tic

        if validation:
            assert th.unique(recv_nodes).shape[0] == recv_nodes.shape[0]

        h_emb.inc_age()
        offsetn, offsetf = 0, 0
        for np in range(self.part.num_parts):
            if self.rank != np:
                ptr_nodes = recv_nodes[offsetn : offsetn + rcount_nodes[np]]
                offsetn += rcount_nodes[np]
                ptr_feat = recv_feat[offsetf : offsetf + rcount[np]]
                ptr_feat = ptr_feat.view(rcount_nodes[np], feat_size)
                offsetf += rcount[np]

                data = ptr_feat, ptr_nodes.long(), None
                tic = time.time()
                h_emb.cache_store(data)
                toc = time.time()
                self.part.timer.he_store_tim += toc - tic
                self.hels_stats.nodes_recv += rcount_nodes[np]
            else:
                assert rcount_nodes[np] == 0


    def hels_from_local_cache(self):
        h_emb = self.hels
        bn_oid, bn_lid, bn_bid = self.find_halo_nodes_orig()
        tic = time.time()
        hits_index, hits_feats, hits_feat_size = \
            h_emb.fused_lookup_load(bn_oid)  ## piggy_bag is enabled
        toc = time.time()
        self.part.timer.cache_out_lu_tim += toc - tic

        self.hels_stats.mb_bn += bn_bid.shape[0]
        self.hels_stats.reqs += bn_bid.shape[0]

        assert hits_feat_size != 0
        if hits_index.shape[0] == 0:
            return
        if hits_feats.shape[0] == 0:
            return
        bn_bid = bn_bid[hits_index]

        feats = hits_feats.view(hits_index.shape[0], hits_feat_size)

        reduction = 0
        index = bn_bid
        assert index.shape[0] == feats.shape[0]
        self.hels_stats.hits += index.shape[0]

        tic = time.time()
        #gnn_utils.scatter_features(feats, index, self.srcdata['h'], reduction)
        gnn_utils.scatter_features(feats, index, self.feat, reduction)
        toc = time.time()
        self.part.timer.cache_out_scatter_tim += toc - tic

        #self.hels_stats.mb_nodes += self.srcdata['h'].shape[0]
        self.hels_stats.mb_nodes += self.feat.shape[0]
        self.hels_stats.nmb += 1

    def o2l_reset(elf):
        if self.part.b2b:
            self.o2l_map[self.level_nodes_orig] = -100
            if checks:
                val = (self.o2l_map != -100).sum()
                assert val == 0, "o2l_map is not reset back to -100 completely."



    def apply_hs_func(self, max_steps, feat_src):
        self.feat = feat_src
        if self.part.enable_hec:
            if self.part.epoch[self.level] >= self.part.dl:
                assert self.part.layer_itr == self.level
                tic = time.time()
                self.hels_to_remote_cache_recv()  ## Async recv to fill local HECs
                toc = time.time()
                self.part.timer.r2_tim += toc - tic

                tic_ = time.time()
                self.hels_from_local_cache()   ## search and get from loca HEC
                toc_ = time.time()
                self.part.timer.cache_out_tim += toc_ - tic_

            if self.part.epoch[self.level] < max_steps - self.part.dl:
                tic_ = time.time()
                self.hels_to_remote_cache_send()  ## async send to remote HECs
                toc_ = time.time()
                self.part.timer.cache_in_tim += toc_ - tic_

        self.part.epoch[self.part.layer_itr] += 1
        self.part.layer_itr = (self.part.layer_itr + 1) % self.part.nlayers

## ---------------------------------------------------------------------------------------------
class DGLBlockPush(hels_mini_batch):
    """Subclass that signifies the graph is a block created from
    :func:`dgl.to_block`.
    """
    # (BarclayII) I'm making a subclass because I don't want to make another version of
    # serialization that contains the is_block flag.
    is_block = True

    def init(self, part, input_nodes, level):
        self.part = part
        self.input_nodes = input_nodes
        self.level = level
        self.rank = part.rank
        self.num_parts = part.num_parts

        self.level_nodes = input_nodes[self.srcnodes()]  ## level nodes represent lid
        self.level_nodes_orig = part.ndata['orig'][self.level_nodes]

        self.hels = part.hels[level]   ## hec
        self.hels_stats = part.hels_stats[level]  ## hec stats

        if self.part.b2b:
            self.o2l_map = part.o2l_map[level]       ## running hashmap, only one map would do all
            self.ocomms = part.ocomms[level]
            self.o2l_map[self.level_nodes_orig] = self.srcnodes()

            tic = time.time()
            self.create_halo_db_v2()   ## old csr for using coo for s and d
            self.part.timer.halo_tim += time.time() - tic


    def __repr__(self):
        if len(self.srctypes) == 1 and len(self.dsttypes) == 1 and len(self.etypes) == 1:
            ret = 'Block(num_src_nodes={srcnode}, num_dst_nodes={dstnode}, num_edges={edge})'
            return ret.format(
                srcnode=self.number_of_src_nodes(),
                dstnode=self.number_of_dst_nodes(),
                edge=self.number_of_edges())
        else:
            ret = ('Block(num_src_nodes={srcnode},\n'
                   '      num_dst_nodes={dstnode},\n'
                   '      num_edges={edge},\n'
                   '      metagraph={meta})')
            nsrcnode_dict = {ntype : self.number_of_src_nodes(ntype)
                             for ntype in self.srctypes}
            ndstnode_dict = {ntype : self.number_of_dst_nodes(ntype)
                             for ntype in self.dsttypes}
            nedge_dict = {etype : self.number_of_edges(etype)
                          for etype in self.canonical_etypes}
            meta = str(self.metagraph().edges(keys=True))
            return ret.format(
                srcnode=nsrcnode_dict, dstnode=ndstnode_dict, edge=nedge_dict, meta=meta)



    def create_halo_db(self):
        src = self.srcnodes()
        ind = th.nonzero(self.part.ndata['inner_node'][self.input_nodes[src]] == 0, as_tuple=True)[0]
        csr = csr_()
        pos, acc = 0, 0

        src = src[ind]
        dout = self.out_edges(src)
        out, count = th.unique_consecutive(dout[0], return_counts = True)
        assert out.shape[0] > 0
        oid = self.part.ndata['orig'][self.input_nodes[out]]
        csr.indices = dout[1]
        pos = 0
        for i in range(out.shape[0]):
            csr.dt[int(oid[i])] = i

        z = th.zeros(1, dtype=count.dtype)
        cum = th.cumsum(count, 0)
        csr.indptr = th.cat((z, cum) )

        self.csr = csr

    def create_halo_db_v2(self):
        src = self.srcnodes()
        ind = th.nonzero(self.part.ndata['inner_node'][self.input_nodes[src]] == 0, as_tuple=True)[0]
        csr = csr_main()
        src = src[ind]
        dout = self.out_edges(src)
        do, index = dout[1].sort()
        so = dout[0][index]
        so = self.part.ndata['orig'][self.input_nodes[so]]

        uniq, counts = do.unique_consecutive(return_counts=True)
        uniq2 = do.unique()
        assert uniq.shape[0] == uniq2.shape[0]

        indptr = [0]
        cum = th.cumsum(counts, 0)
        indptr += cum.tolist()
        csr.s, csr.d, csr.indptr = so, do, th.tensor(indptr, dtype=th.int64)

        self.csr = csr

    ### New version fo create_halo_db_v3
    ## in drpa code creates halo db for tppfied mapped_spmm_emb
    def create_halo_db_v3(self):
        src = self.srcnodes()
        ind = th.nonzero(self.part.ndata['inner_node'][self.input_nodes[src]] == 0, as_tuple=True)[0]
        csr = csr_main_v2()
        src = src[ind]
        dout = self.out_edges(src)
        do, index = dout[1].sort()
        so = dout[0][index]
        so = self.part.ndata['orig'][self.input_nodes[so]]

        uniq, counts = do.unique_consecutive(return_counts=True)
        if checks:
            uniq2 = do.unique()
            assert uniq.shape[0] == uniq2.shape[0]

        indptr = [0]
        cum = th.cumsum(counts, 0)
        indptr += cum.tolist()
        assert len(indptr) - 1 == uniq.shape[0]
        csr.s, csr.d, csr.indptr = so, uniq, th.tensor(indptr, dtype=th.int64)

        self.csrv2 = csr
