## Distributed  mini-batch sampling core codebase
## Author: Md. Vasmiuddin <vasimuddin.md@intel.com>
## Parallel Computing Lab, Intel Corporation
##

import os, psutil, sys, time, random
from math import ceil, floor
import numpy as np
import dgl
import torch as th
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from .. import function as fn
from dgl import DGLHeteroGraph

from .communicate import *

from .queue import gqueue, gstack
from tpp_pytorch_extension.gnn.common import gnn_utils
from .iels_cache import feat_cache

debug = False
stats = False

class emb_buffer():
    def __init__(self, feat_size, dtype):
        self.feat_size = feat_size
        self.dtype = dtype
        self.buf_size = 1000
        self.feat_buf = th.empty([buf_size, self.feat_size], dtype = self.dtype)
        self.sn_db_ptr = th.tensor([-1 for i in range(bsize)])

class timer():
    def __init__(self, rank, num_parts):
        self.rank, self.num_parts = rank, num_parts
        self.reset()

    def reset(self):
        self.ltl, self.cn, self.cf, self.ltr, self.bf = 0.0, 0.0, 0.0, 0.0, 0.0
        self.find_bn, self.cf_gf, self.imb = 0.0, 0.0, 0.0
        self.cn_comm, self.cn_wait, self.lgf, self.lsf = 0.0, 0.0, 0.0, 0.0
        self.cn_comm0, self.cn_wait0 = 0.0, 0.0
        self.cf_imb, self.cf_comm, self.cf_wait, self.setup = 0.0, 0.0, 0.0, 0.0
        self.lsetup = 0.0
        self.nbnodes, self.nmb = 0, 0
        self.cntypes = [0 for i in range(4)]

    def display(self):
        print('load tensor local time: {:.4f}'.format(self.ltl))
        print(' -local setup time: {:.4f}'.format(self.lsetup))
        print(' -local gf time: {:.4f}'.format(self.lgf))
        print(' -local sf time: {:.4f}'.format(self.lsf))
        print('communicate nodes time: {:.4f}'.format(self.cn))
        print(' -cn comm0 time:       {:.4f}'.format(self.cn_comm0))
        print(' -cn wait0 time:       {:.4f}'.format(self.cn_wait0))
        print(' -cn time (async):   {:.4f}'.format(self.cn_comm))
        print(' -cn_wait time: {:.4f}'.format(self.cn_wait))        
        print('communicate feat time: {:.4f}'.format(self.cf))
        print(' - cf_gf time: {:.4f}'.format(self.cf_gf))
        print(' - cf_imb time: {:.4f}'.format(self.cf_imb))
        print(' - cf_comm time: {:.4f}'.format(self.cf_comm))
        print(' - cf_wait time: {:.4f}'.format(self.cf_wait))
        print('load tensor remote time: {:.4f}'.format(self.ltr))
        print('buffering feat time: {:.4f}'.format(self.bf))
        print()
        print('Number of MB: ', self.nmb)
        print('MB nodes: ', self.nbnodes)

    def display_glob(self):
        if self.rank == 0:
            print('Global times: >>>>>>>>>>>>>>>>>>>>>>')
        
        mx, avg = mpi_allreduce(self.cn, 'max'), mpi_allreduce(self.cn)
        if self.rank == 0:
            print('comms time (bi, nodes/feats): avg: {:.2f}, max: {:.2f}'.
                  format(avg/self.num_parts, mx))
            
        mx, avg = mpi_allreduce(self.cf_imb, 'max'), mpi_allreduce(self.cf_imb)
        if self.rank == 0:
            print('comms time (bi, nodes/feats): avg: {:.2f}, max: {:.2f}'.
                  format(avg/self.num_parts, mx))

        if self.rank == 0:            
            print()
        if self.rank == 0:
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


def iels_master_func(pb, args):
    iels_obj = iels_master() 
    
    iels_obj.iels_init(pb, args)
    return iels_obj

class iels_master(): 
    def __init__(self):
        pass
    
    def iels_init(self, pb, args):
        self.N = pb.g_orig.number_of_nodes() ## total nodes in original graph
        self.rank      = args.rank
        self.num_parts = pb.num_parts
        self.ndata = pb.node_feats

        self.ndtypes = pb.g_orig.ntypes
        self.ntypes_id, self.acc_onodes, offset = {}, {}, 0
        for keys, nt in enumerate(pb.g_orig.ntypes):
            self.ntypes_id[nt] = keys
            self.acc_onodes[nt] = offset
            offset += pb.g_orig.num_nodes(nt)
        self.pn = (self.ndata['inner_node'] == 1).sum()
        self.feats = self.ndata['feat'][0:self.pn][:]
        self.scf = self.ndata['scf'][0:self.pn][:] if self.feats.dtype == th.int8 else None
        self.dtype = self.feats.dtype
        self.feat_size = self.feats.shape[1]
        if self.scf is not None:
            self.scf_size = self.scf.shape[1]
        self.part_sn_onid = self.ndata['orig'][:self.pn]
        self.part_sn_gnid = th.tensor(list(th.arange(self.pn)), dtype=th.int32)

        if debug and self.rank == 0:
            print("sn_onid size: ", self.part_sn_onid.size())
            print("sn_gnid size: ", self.part_sn_gnid.size())

        ## database to know in which partition the remote ndoes are residing
        rr = self.create_remote_sn_db_commence()
        ## local feats db - orig node id to index in the feat table
        self.create_local_sn_db(self.part_sn_onid, self.part_sn_gnid)  ## at partition level
        self.create_remote_sn_db(rr)

        ## remote embedding buffering data structures
        self.enable_iec = args.enable_iec
        if self.enable_iec:
            self.buf_thres = args.ncache
            self.buf_size = args.cache_size

        self.buf_offset, self.buf_base = 0, self.pn
        if self.enable_iec:
            self.cache = feat_cache(pb, self.local_sn_db, args)
            self.feats = self.cache.fuse_buf(self.feats)

        self.timer = timer(self.rank, self.num_parts)
        ##stats
        self.sdata, self.rdata, self.snodes, self.rnodes = 0, 0, 0, 0


    def printname(self):
        print('dms file name:', os.path.basename(__file__))

    def profile(self):
        self.timer.display_glob()
        if self.rank == 0:
            self.timer.display()
            print("#########################################")
            print("Below numbers are accumulate across all the batches")
            print('Total send data (elements): ', self.sdata)
            print('Total recv data (elements): ', self.rdata)
            print('Total solid  nodes: ', self.snodes)
            print('Total remote nodes: ', self.rnodes)
            print("#########################################")
            if self.enable_iec:
                self.cache.display()

        snodes_max, snodes_avg = mpi_allreduce(self.snodes, 'max'), mpi_allreduce(self.snodes)
        rnodes_max, rnodes_avg = mpi_allreduce(self.rnodes, 'max'), mpi_allreduce(self.rnodes)
        if self.rank == 0:
            print("################### Global ###")
            print('solid nodes:  avg: {:.4f}, max: {}'.format(snodes_avg/self.num_parts, snodes_max))
            print('remote nodes: avg: {:.4f}, max: {}'.format(rnodes_avg/self.num_parts, rnodes_max))
            print("#########################################")

    def reset(self):
        self.timer.reset()
        self.sdata, self.rdata, self.snodes, self.rnodes = 0, 0, 0, 0

    def create_local_sn_db(self, onid, gnid):
        self.local_sn_db = th.full([self.N], -1,  dtype=th.int32)
        self.local_sn_db[onid] = gnid

    def create_remote_sn_db_commence(self):
        return self.communicate_solid_nodes()

    def create_remote_sn_db(self, rr):
        self.onid_map = th.full([self.N], -1,  dtype=th.int32)
        self.pid_map = th.full([self.N], -1,  dtype=th.int32)

        rr.wait()
        offset = 0
        for np in range(self.num_parts):
            try:
                nodes_onid = self.recv_sn_onid[offset : offset + self.recv_sn_count[np]]
                nodes_gnid = self.recv_sn_gnid[offset : offset + self.recv_sn_count[np]]
            except:
                print("Error: recv_bn is not created in communicate_halo_nodes_orig(), exiting..")
                sys.exit(1)

            if np == self.rank:
                assert nodes_onid.shape[0] == 0

            self.onid_map[nodes_onid.long()] = nodes_gnid   ## stores gnid
            self.pid_map[nodes_onid.long()] = np
            offset += self.recv_sn_count[np]

    def find_solid_nodes(self):
        bn_part = self.part_sn_gnid  ## part node id for border nodes
        bn_orig = self.part_sn_onid  ## orig id for border nodes
        return bn_orig, bn_part

    def communicate_solid_nodes(self):
        sn_orig, sn_part = self.find_solid_nodes()

        send_size = [sn_part.shape[0] for i in range(self.num_parts)]
        send_size[self.rank] = 0
        req, send_sr, recv_sr = alltoall_s_async(send_size)

        send_sn_count = [0 for i in range(self.num_parts)]
        self.recv_sn_count = [0 for i in range(self.num_parts)]
        tsend = sum(send_sr)

        send_sn_onid = th.empty(tsend, dtype=th.int32)
        send_sn_gnid = th.empty(tsend, dtype=th.int32)

        offset = 0
        for np in range(self.num_parts):
            if np != self.rank:
                send_sn_onid[offset: offset + send_sr[np]] = sn_orig.to(th.int32)
                send_sn_gnid[offset: offset + send_sr[np]] = sn_part.to(th.int32)
            else:
                assert int(send_sr[np]) == 0

            send_sn_count[np] = int(send_sr[np])
            offset += send_sn_count[np]

        req.wait()  ## wait on alltoall_s_async

        for np in range(self.num_parts):
            self.recv_sn_count[np] = int(recv_sr[np])

        trecv = sum(recv_sr)
        self.recv_sn_onid = th.empty(trecv, dtype=th.int32)
        self.recv_sn_gnid = th.empty(trecv, dtype=th.int32)

        assert send_sn_gnid.max() < self.pn
        req = dist.all_to_all_single(self.recv_sn_gnid, send_sn_gnid,
                                     self.recv_sn_count, send_sn_count,
                                     async_op=True)
        req = dist.all_to_all_single(self.recv_sn_onid, send_sn_onid,
                                     self.recv_sn_count, send_sn_count,
                                     async_op=True)
        return req

    def finalize(self):
        pass

class ielspp_batch():
    def __init__(self, part, input_nodes):
        self.part = part
        self.input_nodes = input_nodes
        self.rank = part.rank
        if self.part.enable_iec:
            self.buf_thres = part.buf_thres
            self.buf_size = part.buf_size
        self.buf_offset = part.buf_offset
        self.buf_base = part.buf_base
        self.timer = part.timer
        self.acc_onodes = part.acc_onodes
        self.batch_feats = {}
        self.batch_scf = {}
        for nt in self.part.ndtypes:
            self.batch_feats[nt] = None
            self.batch_scf[nt] = None
        self.queue = gqueue()
        self.queue0 = gqueue()
        self.queue1 = gqueue()

    def dist_sampling(self):
        pass

    def finalize(self):
        assert self.queue.size() == 0
        assert self.queue0.size() == 0
        assert self.queue1.size() == 0

    def start_comm(self):
        self.queue.reset()
        self.queue0.reset()
        self.queue1.reset()

        tic = time.time()
        for nt in self.part.ndtypes:
            self.communicate_remote_bnodesv2(nt)
        toc = time.time()
        self.timer.cn += toc - tic
        
        for nt in self.part.ndtypes:
            self.start_comm_(nt)

    def end_comm(self):
        for nt in self.part.ndtypes:
            self.end_comm_(nt)

        self.finalize()

    def start_comm_(self, ntype):
        t0 = time.time()

        ## sandwiching
        t1 = time.time()
        self.load_tensor_local_one(ntype)
        t2 = time.time()
        self.timer.ltl += t2 - t1

        t0 = time.time()
        self.communicate_remote_bnodes_featsv2()
        t1 = time.time()
        self.timer.cf += t1 - t0

    def end_comm_(self, ntype):
        #if self.rank == 0 and debug:
        #    print('batch_feats:<< ', self.batch_feats)
        #    print('batch_scf:<< ', self.batch_scf)

        send_rn_bnid = self.queue1.pop()
        send_rn_onid = self.queue1.pop()
        recv_feats = self.queue.pop()
        recv_count = self.queue.pop()
        req = self.queue.pop()
        if self.part.scf is not None:
            recv_scfs = self.queue.pop()
            recv_scf_count = self.queue.pop()
            reqscf = self.queue.pop()
        tic = time.time()
        req.wait()
        if self.part.scf is not None:
            reqscf.wait()
        toc = time.time()
        self.timer.cf_wait += toc - tic
        
        self.part.rdata += sum(recv_count)
        if self.part.scf is not None:
            self.part.rdata += sum(recv_scf_count)
        t2 = time.time()
        if self.part.scf is None: recv_scfs = None
        self.load_tensor_remote(recv_feats, recv_scfs, send_rn_bnid, ntype)
        if self.rank == 0  and debug:
            print('send rn bnid:> ', send_rn_bnid)
            print('send rn onid:> ', send_rn_onid)
            print('batch_feats:> ', self.batch_feats)
            print('batch_scf:> ', self.batch_scf)
        t3 = time.time()
        self.timer.ltr += t3 - t2

        ## buffer remote embeddings for future use
        if self.part.enable_iec:
            self.buffer_remote_feats(recv_feats, send_rn_onid)
        t4 = time.time()
        self.timer.bf += t4 - t3

    def find_remote_bnodes(self, ntype):
        if isinstance(self.input_nodes, dict):
            offset = self.acc_onodes[ntype]
            rn_batch = th.nonzero(
                         self.part.local_sn_db[ \
                             offset + self.input_nodes[ntype] \
                         ] == -1, as_tuple=True)[0]
            rn_orig = offset + self.input_nodes[ntype][rn_batch]    ## part node id for halo nodes   
        else:
            rn_batch = th.nonzero(
                         self.part.local_sn_db[ \
                             self.input_nodes \
                         ] == -1, as_tuple=True)[0]
            rn_orig = self.input_nodes[rn_batch]    ## part node id for halo nodes

        return rn_orig, rn_batch

    def load_tensor_remote(self, recv_feats, recv_scfs, send_rn_bnid, ntype):
        if isinstance(self.input_nodes, dict):
            rf = recv_feats.view(-1, self.part.feat_size)
            gnn_utils.scatter_features(rf, send_rn_bnid, self.batch_feats[ntype], 0)
            if recv_scfs is not None:
                rscf = recv_scfs.view(-1, self.part.scf_size)
                gnn_utils.scatter_features(rscf, send_rn_bnid, self.batch_scf[ntype], 0)
        else:
            rf = recv_feats.view(-1, self.part.feat_size)
            gnn_utils.scatter_features(rf, send_rn_bnid, self.batch_feats[ntype], 0)
            if recv_scfs is not None:
                rscf = recv_scfs.view(-1, self.part.scf_size)
                gnn_utils.scatter_features(rscf, send_rn_bnid, self.batch_scf[ntype], 0)
            
            if self.rank == 0 and stats:
                print('recv_feats: ',  recv_feats)
                print('send_rn_bid: ', send_rn_bnid)
                print('batch_feats: ', self.batch_feats)
                print()
                assert self.lsize + send_rn_bnid.shape[0] == self.input_nodes.shape[0]
                self.lut[send_rn_bnid] = 1
                assert (self.lut == -1).sum() == 0

    def buffer_remote_feats(self, recv_feats, send_rn_onid):
        assert self.buf_thres < self.buf_size

        num_nodes = send_rn_onid.shape[0]
        ## plain thresholding
        if send_rn_onid.shape[0] > self.buf_thres:
            recv_feats = recv_feats[:self.buf_thres, :]
            send_rn_onid = send_rn_onid[ :self.buf_thres]
            num_nodes = self.buf_thres

        recv_feats = recv_feats.view(num_nodes, self.part.feat_size)
        self.part.cache.cache_store(recv_feats, send_rn_onid)

    def get_buffer_lines(self, num_nodes_to_buf):
        if self.buf_size - self.buf_offset > num_nodes_to_buf:
            poffset = self.buf_offset
            self.buf_offset += num_nodes_to_buf
            index = th.tensor([ i for i in range(poffset, self.buf_offset)], dtype=th.int32)
        else:
            index1 = th.tensor([ i for i in range(self.buf_offset, self.buf_size)], dtype=th.int32)
            self.buf_offset = num_nodes_to_buf - (self.buf_size - self.buf_offset)
            index2 = th.tensor([ i for i in range(self.buf_offset)], dtype=th.int32)
            index = th.cat((index1, index2), 0)

        return index

    def load_tensor_local_one(self, ntype):
        self.timer.nmb += 1
        if isinstance(self.input_nodes, dict):
            nnt = self.input_nodes[ntype].shape[0]
            self.timer.nbnodes += nnt
            self.batch_feats[ntype] = th.empty([nnt, self.part.feat_size], dtype=self.part.dtype)
            if self.part.scf is not None:
                self.batch_scf[ntype] = th.empty([nnt, self.part.scf_size], dtype=th.float32)
            
            tic = time.time()
            offset = self.acc_onodes[ntype]
            gnid = self.part.local_sn_db[offset + self.input_nodes[ntype]]
            index = th.nonzero(gnid != -1, as_tuple=True)[0]
            toc = time.time()
            self.timer.setup += toc - tic
            
            self.part.snodes += int(index.shape[0])
            tic = time.time()
            buff = gnn_utils.gather_features(self.part.feats, gnid[index].to(th.long))
            if self.part.scf is not None:
                scf = gnn_utils.gather_features(self.part.scf, gnid[index].to(th.long))
            toc = time.time()            
            self.timer.lgf += toc - tic
            
            tic = time.time()
            gnn_utils.scatter_features(buff, index.long(), self.batch_feats[ntype], 0)
            if self.part.scf is not None:
                gnn_utils.scatter_features(scf, index.long(), self.batch_scf[ntype], 0)
            toc = time.time()
            self.timer.lsf += toc - tic
        else:
            nnt = self.input_nodes.shape[0]
            self.timer.nbnodes += nnt
            self.batch_feats[ntype] = th.empty([nnt, self.part.feat_size], dtype=self.part.dtype)
            
            tic = time.time()
            offset = self.acc_onodes[ntype]
            assert offset == 0
            gnid = self.part.local_sn_db[offset + self.input_nodes]
            index = th.nonzero(gnid != -1, as_tuple=True)[0]
            toc = time.time()
            self.timer.setup += toc - tic
            
            self.part.snodes += int(index.shape[0])
            tic = time.time()
            buff = gnn_utils.gather_features(self.part.feats, gnid[index].to(th.long))
            if self.part.scf is not None:
                scf = gnn_utils.gather_features(self.part.scf, gnid[index].to(th.long))
            toc = time.time()            
            self.timer.lgf += toc - tic
            
            tic = time.time()
            gnn_utils.scatter_features(buff, index.long(), self.batch_feats[ntype], 0)
            if self.part.scf is not None:
                gnn_utils.scatter_features(scf, index.long(), self.batch_scf[ntype], 0)
            toc = time.time()
            self.timer.lsf += toc - tic

    def communicate_remote_bnodesv2(self, ntype):
        tic = time.time()
        rn_onid, rn_bnid = self.find_remote_bnodes(ntype)
        toc = time.time()
        self.timer.find_bn += toc - tic

        ## stats
        self.part.rnodes += int(rn_onid.shape[0])
        #if self.rank == 0 and debug:
        #    print("halo bid: ", rn_bnid.sort())

        rn_gnid = self.part.onid_map[rn_onid]
        rn_pid = self.part.pid_map[rn_onid]

        if debug:
            assert th.max(rn_pid) <= self.part.num_parts

        send_rn_count = []
        send_rn_gnid = th.empty(rn_gnid.shape[0], dtype=self.part.onid_map.dtype)
        send_rn_bnid = th.empty(rn_gnid.shape[0], dtype=rn_bnid.dtype)
        send_rn_onid = th.empty(rn_gnid.shape[0], dtype=rn_onid.dtype)
        mask_lst = []
        for np in range(self.part.num_parts):
            if self.rank != np:
                mask = th.nonzero(rn_pid == np, as_tuple=True)[0]
                ssize = int(mask.shape[0])
                send_rn_count.append(ssize)
                mask_lst.append(mask)
            else:
                mask = th.nonzero(rn_pid == np, as_tuple=True)[0]  ## put under debug
                assert mask.shape[0] == 0
                ssize = 0
                send_rn_count.append(ssize)
                mask_lst.append(mask)

        assert send_rn_gnid.shape[0] == sum(send_rn_count)
        req, send_sr, recv_sr = alltoall_s_async(send_rn_count)      

        offset, ssize = 0, 0
        for np in range(self.part.num_parts):
            if self.rank != np:
                mask = mask_lst[np] 
                ssize = int(mask.shape[0])

                send_rn_gnid[offset: offset + ssize] = rn_gnid[mask]
                send_rn_bnid[offset: offset + ssize] = rn_bnid[mask]
                send_rn_onid[offset: offset + ssize] = rn_onid[mask]
            else:
                mask = mask_lst[np] 
                assert mask.shape[0] == 0
                ssize = 0

            offset += ssize
            
        toc = time.time()
        self.timer.cn_comm0 += toc - tic

        tic = time.time()
        req.wait()
        tsend, trecv = sum(send_sr), sum(recv_sr)  ##recv data
        toc = time.time()
        self.timer.cn_wait0 += toc - tic

        recv_rn_count = []
        for np in range(self.part.num_parts):
            recv_rn_count.append(int(recv_sr[np]))

        recv_rn_gnid = th.empty(trecv, dtype=self.part.onid_map.dtype)

        tic = time.time()        
        req = dist.all_to_all_single(recv_rn_gnid, send_rn_gnid,
                                     recv_rn_count, send_rn_count,
                                     async_op=True)
        self.queue0.push(req)
        self.queue0.push(recv_rn_gnid)
        self.queue0.push(recv_rn_count)
        self.queue0.push(send_rn_count)
        toc = time.time()
        self.timer.cn_comm += toc - tic

        self.queue1.push(send_rn_bnid)
        self.queue1.push(send_rn_onid)

    def communicate_remote_bnodes_featsv2(self):

        tic = time.time()
        req  =  self.queue0.pop()
        req.wait()
        recv_rn_gnid  =  self.queue0.pop()
        recv_rn_count =  self.queue0.pop()
        send_rn_count =  self.queue0.pop()
        toc = time.time()
        self.timer.cn_wait += toc - tic
        
        tic = time.time()
        sf = gnn_utils.gather_features(self.part.feats, recv_rn_gnid.long())
        send_feats = sf.view(recv_rn_gnid.shape[0]* self.part.feat_size)
        if self.part.scf is not None:
            sscf = gnn_utils.gather_features(self.part.scf, recv_rn_gnid.long())
            send_scfs = sscf.view(recv_rn_gnid.shape[0]* self.part.scf_size)
        toc = time.time()
        self.timer.cf_gf += toc - tic

        tic = time.time()
        send_count = []
        recv_count = []
        send_scf_count = []
        recv_scf_count = []
        for np in range(self.part.num_parts):
            ele = recv_rn_count[np]
            send_count.append(ele * self.part.feat_size)
            recv_count.append(send_rn_count[np] * self.part.feat_size)

        if self.part.scf is not None:
            for np in range(self.part.num_parts):
                ele = recv_rn_count[np]
                send_scf_count.append(ele * self.part.scf_size)
                recv_scf_count.append(send_rn_count[np] * self.part.scf_size)

        recv_feats = th.empty(sum(recv_count), dtype=self.part.dtype)
        if self.part.scf is not None:
            recv_scfs = th.empty(sum(recv_scf_count), dtype=th.float32)

        req = dist.all_to_all_single(recv_feats, send_feats,
                                     recv_count, send_count,
                                     async_op=True)
        self.part.sdata += sum(send_count)
        recv_feats = recv_feats.view(sum(send_rn_count), self.part.feat_size)

        if self.part.scf is not None:
            reqscf = dist.all_to_all_single(recv_scfs, send_scfs,
                                         recv_scf_count, send_scf_count,
                                         async_op=True)
            self.part.sdata += sum(send_scf_count)
            recv_scfs = recv_scfs.view(sum(send_rn_count), self.part.scf_size)

        self.queue.push(recv_feats)
        self.queue.push(recv_count)
        self.queue.push(req)
        if self.part.scf is not None:
            self.queue.push(recv_scfs)
            self.queue.push(recv_scf_count)
            self.queue.push(reqscf)
        toc = time.time()
        self.timer.cf_comm += toc - tic
        
