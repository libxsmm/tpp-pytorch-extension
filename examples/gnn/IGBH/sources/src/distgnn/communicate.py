import os, psutil, sys, gc, time, random
from math import ceil, floor
import dgl
import torch as th
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

class communicate():
    def __init__(self, rank, num_parts):
        self.rank = rank
        self.num_parts = num_parts

    #########
    def all_to_all_s(self, send_list):
        send_sr = []
        recv_sr = [th.zeros(1, dtype=th.int64) for i in range(0, self.num_parts)]
        for i in range(self.num_parts):
            t_ = th.tensor([send_list[i]], dtype=th.int64)
            send_sr.append(t_)

        sync_req = dist.all_to_all(recv_sr, send_sr, async_op=True)   # make it async
        sync_req.wait()

        return send_sr, recv_sr

    ###########
    def all_to_all_v(self, vector):
        return


def mpi_allreduce(sten, ops=None):
    op = dist.ReduceOp.SUM
    if ops == 'max':
        op = dist.ReduceOp.MAX
    elif ops == 'min':
        op = dist.ReduceOp.MIN

    if th.is_tensor(sten):
        dist.all_reduce(sten, op)
    else:
        sten = th.tensor(sten)
        dist.all_reduce(sten, op)

    return sten.tolist()

def alltoall_s(send_tensor):  ## input is a tensor
    if not th.is_tensor(send_tensor):
        send_tensor = th.tensor(send_tensor, dtype=th.int64)
        recv_tensor = th.empty(send_tensor.shape[0], dtype=th.int64)
    else:
        recv_tensor = th.empty(send_tensor.shape[0], dtype=send_tensor.dtype)
    scount = [1 for i in range(send_tensor.shape[0])]
    rcount = [1 for i in range(send_tensor.shape[0])]
    sync_req = dist.all_to_all_single(recv_tensor, send_tensor, rcount, scount, async_op=True)
    sync_req.wait()

    return send_tensor, recv_tensor

def alltoall_s_async(send_tensor):  ## input is a tensor
    if not th.is_tensor(send_tensor):
        send_tensor = th.tensor(send_tensor, dtype=th.int64)
        recv_tensor = th.empty(send_tensor.shape[0], dtype=th.int64)
    else:
        recv_tensor = th.empty(send_tensor.shape[0], dtype=send_tensor.dtype)
    scount = [1 for i in range(send_tensor.shape[0])]
    rcount = [1 for i in range(recv_tensor.shape[0])]
    sync_req = dist.all_to_all_single(recv_tensor, send_tensor, rcount, scount, async_op=True)
    #sync_req.wait()

    return sync_req, send_tensor, recv_tensor


def alltoall_s_exp(send_tensor):  ## input is a tensor
    if not th.is_tensor(send_tensor):
        send_tensor = th.tensor(send_tensor, dtype=th.int64)
        recv_tensor = th.empty(send_tensor.shape[0], dtype=th.int64)
    else:
        recv_tensor = th.empty(send_tensor.shape[0], dtype=send_tensor.dtype)
    scount = [1 for i in range(send_tensor.shape[0])]
    rcount = [1 for i in range(send_tensor.shape[0])]
    sync_req = dist.all_to_all_single(recv_tensor, send_tensor, rcount, scount, async_op=True)

    return send_tensor, recv_tensor, sync_req

###########
def alltoall_v(rdata, sdata, rcount, scount):
    sync_req = dist.all_to_all_single(rdata, sdata, rcount, scount, async_op=True)
    sync_req.wait()

def alltoall_v_sync(ten, num_ranks):
    nsend = ten.shape[0]
    send_size = [nsend for i in range(num_ranks)]
    send_sr, recv_sr = alltoall_s(send_size)
 
    tsend, trecv = sum(send_sr), sum(recv_sr)  ##recv data
    assert trecv >= 0
 
    send_data = th.empty(tsend, dtype=ten.dtype)
    recv_data = th.empty(trecv, dtype=ten.dtype)
 
    offset = 0
    for i in range(num_ranks):
        send_data[offset:offset + nsend] = ten
        offset += nsend
 
    send_sr, recv_sr = send_sr.tolist(), recv_sr.tolist()
    dist.all_to_all_single(recv_data, send_data, recv_sr, send_sr)
    return recv_data

def alltoall_v_async(rdata, sdata, rcount, scount):
    sync_req = dist.all_to_all_single(rdata, sdata, rcount, scount, async_op=True)
    return sync_req


def barrier():
    dist.barrier()


def communicate_data(params):
    neigh, send_size, xnbn, xrbn,num_parts, rank = params

    piggy_bag = 0
    send_sr, recv_sr = alltoall_s(send_size)

    tsend, trecv = sum(send_sr), sum(recv_sr)  ##recv data
    assert trecv >= 0

    feat_size = neigh.shape[0]
    send_feat = th.empty(tsend * (feat_size + piggy_bag), dtype=neigh.dtype)
    recv_feat = th.empty(trecv * (feat_size + piggy_bag), dtype=neigh.dtype)
    send_nodes = th.empty(tsend, dtype=th.int64)
    recv_nodes = th.empty(trecv, dtype=th.int64)

    offsetv, offseti = 0, 0
    for np in range(num_parts):
        index = xnbn[np].int()
        if int(send_sr[np]) != index.shape[0]:
            print(np, ' ', int(send_sr[np]),' ' ,index.shape[0], ' ', xnbn[np].shape[0])
            sys.exit(1)

        assert send_feat.shape[0] >= index.shape[0] * (feat_size + piggy_bag) + offsetv
        if rank != np:
            send_feat = gnn_utils.gather_fetatures(neigh, index, offseti, send_feat, offsetv, None, 0)
        offsetv += int(send_sr[np]) * (feat_size + piggy_bag)
        scount[np] = int(send_sr[np]) * (feat_size + piggy_bag)
        rcount[np] = int(recv_sr[np]) * (feat_size + piggy_bag)

    piggy_bag = 0
    offseti, offsetv = 0, 0
    for np in range(num_parts):
        index = th.arange(xrbn[np].shape[0])
        if rank != np:
            send_nodes = gather_features(xrbn[np].long(), index, offseti, send_nodes, offsetv, None, piggy_bag)
        offsetv += int(send_sr[np])
        scount_nodes[np] = int(send_sr[np])
        rcount_nodes[np] = int(recv_sr[np])


    req2 = dist.all_to_all_single(recv_feat, send_feat, rcount, scount, async_op=True)
    req2.wait()

    req1 = dist.all_to_all_single(recv_nodes, send_nodes, rcount_nodes, scount_nodes, async_op=True)
    req1.wait()

    return recv_feat, rcount, recv_nodes, rcount_nodes
