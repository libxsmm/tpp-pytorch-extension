###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Sasikanth Avancha (Intel Corp.)                                     #
###############################################################################

import torch
import math
from tpp_pytorch_extension._C import _gnn_utils as gnn_utils_cpp
import numpy as np
import os
from ...qtypes import *
import tqdm

def affinitize_cores(nthreads, nworkers):
    gnn_utils_cpp.affinitize_cores(nthreads, nworkers)


def find_nodes(pnd_s_in, pnd_orig, srcnodes, lnodes, ntype):
    inputs = [pnd_s_in, pnd_orig, srcnodes, lnodes]
    orig, batch, part = gnn_utils_cpp.find_nodes(inputs, ntype)
    return orig, batch, part


def db_r2l_map(db_t, sn_orig, sn_batch, sn_part):
    inputs = [db_t, sn_orig, sn_batch, sn_part]
    r, b, l = gnn_utils_cpp.db_r2l_map(inputs)
    return r, b, l


def r2l_map(o2l_map, rbn_orig):
    inputs = [o2l_map, rbn_orig]
    r_lid2, l_lid2 = gnn_utils_cpp.r2l_map(inputs)
    return r_lid2, l_lid2


def find_n_map_nodes(db_t, pnd_solid, pnd_orig, srcnodes, lnodes):
    inputs = [db_t, pnd_solid, pnd_orig, srcnodes, lnodes]
    r, b, l = gnn_utils_cpp.find_n_map_nodes(inputs)
    return r, b, l


def set_cond_index_vals(inp, cval, idx, outp, oval):
    inputs = [inp, idx, outp]
    gnn_utils_cpp.set_cond_index_vals(inputs, cval, oval)


def set_n_store_cline_indices(rptr, cl, hmap, age, nids, cval, oval):
    inputs = [rptr, cl, hmap, age, nids]
    gnn_utils_cpp.set_n_store_cline_indices(inputs, cval, oval)


def inc_cache_fill(cache_fill, nodes):
    gnn_utils_cpp.inc_cache_fill(cache_fill, nodes)


def cache_load(hmap, oid, feats, age=None, level=0, min_life=0, life=0):
    inputs = [hmap, oid, feats]
    if age is not None:
        inputs.append(age)
    oid_idx, gat_data = gnn_utils_cpp.cache_load(inputs, level, min_life, life)

    return oid_idx, gat_data


def cache_store(cache_data):
    (
        hashmap,
        rptr,
        age,
        nodes,
        storage_feats,
        feats,
        sz_feats,
        feats_sz,
        cp,
        cs,
        hval,
        rval,
    ) = cache_data
    inputs = [hashmap, rptr, age, nodes, storage_feats, feats, sz_feats, feats_sz, cp]
    gnn_utils_cpp.cache_store(inputs, cs, hval, rval)


def node_sampling(degs, xnbn, xrbn, hil, thres):
    inputs = [degs, xnbn, xrbn]
    xnbn, xrbn = gnn_utils_cpp.node_sampling(inputs, hil, thres)
    return xnbn, xrbn


def gather_n_store_offset(inp, ind, out, offi, offv):
    inputs = [inp, ind, out]
    gnn_utils_cpp.gather_n_store_offset(inputs, offi, offv)


def gather_features(nfeat, indices):
    N = indices.shape[0]
    align = 32 if N >= 32 or N == 0 else N
    inputs = [nfeat, indices]

    out = gnn_utils_cpp.gather_features(align, inputs)
    return out


def scatter_features(feat_src, indices, feat_dst, reduction):
    N = indices.shape[0]
    align = 32 if N >= 32 or N == 0 else N
    inputs = [feat_src, indices, feat_dst]

    gnn_utils_cpp.scatter_features(align, reduction, inputs)


def mapped_spmm_copy_lhs_add(dest, indptr, dind, sind, comms, source, edge, soff):
    if edge is None:
        inputs = [dest, indptr, dind, sind, comms, source]
    else:
        inputs = [dest, indptr, dind, sind, comms, source, edge]

    gnn_utils_cpp.mapped_spmm_copy_lhs_add(inputs, rank, soff)

def quantize_dataset(in_name, out_name, out_scf_name, feat_dim=1024, block_size=32):
    '''
    out, out_scf = gnn_utils_cpp.quantize_dataset(in_name, feat_dim, block_size)

    if out.dim() == 1:
        rows = out.shape[0] // feat_dim
        out = out.reshape(rows, feat_dim)
    '''
    f = open(in_name)
    sz = f.seek(0, os.SEEK_END)
    f.close()
    elements = sz // 4 # assuming 4-bytes per element, i.e., float
    rows = elements // feat_dim
    row_blocks = 1000000
    rows_main = rows // row_blocks
    rows_rem = rows % row_blocks

    out = torch.empty(0,dtype=torch.int8)
    out_scf = torch.empty(0)

    for r in tqdm.tqdm(range(0, rows_main)):
        offset = r * row_blocks * feat_dim * 4
        t_in = torch.from_numpy(np.fromfile(in_name, dtype=np.float32, count=row_blocks*feat_dim, offset=offset))
        t_in = t_in.reshape(row_blocks, feat_dim)

        qt = quantize_int8sym(t_in, block_size, -1, False)
        val = get_qval(qt)
        scf = get_scales(qt)
        if r == 0:
            out = val
            out_scf = scf
        else:
            out = torch.cat((out, val), 0)
            out_scf = torch.cat((out_scf, scf), 0)

    print(f'out shape {out.shape}')
    print(f'out_scf shape {out_scf.shape}')
    
    rem_start = rows_main * row_blocks
    offset = rem_start * feat_dim * 4
    t_in = torch.from_numpy(np.fromfile(in_name, dtype=np.float32, count=rows_rem*feat_dim, offset=offset))
    t_in = t_in.reshape(rows_rem, feat_dim)
    qt = quantize_int8sym(t_in, block_size, -1, False)
    val = get_qval(qt)
    scf = get_scales(qt)
    out = torch.cat((out, val), 0)
    out_scf = torch.cat((out_scf, scf), 0)

    print(f'out shape {out.shape}')
    print(f'out_scf shape {out_scf.shape}')

    torch.save(out, out_name)
    torch.save(out_scf, out_scf_name)
    out = None
    out_scf = None

def quantize_dataset_feat(node_feats, out_name, out_scf_name, block_size=32):
    '''
    out, out_scf = gnn_utils_cpp.quantize_dataset_feat(node_feats, block_size)

    if out.dim() == 1:
        rows = out.shape[0] // feat_dim
        out = out.reshape(rows, feat_dim)
    '''
    qt = quantize_int8sym(node_feats, block_size, -1, False)
    out = get_qval(qt)
    out_scf = get_scales(qt)
    torch.save(out, out_name)
    torch.save(out_scf, out_scf_name)

def downconvert_dataset(in_name, out_name, data_type, feat_dim):
    out = gnn_utils_cpp.downconvert_dataset(in_name, data_type, feat_dim)

    if out.dim() == 1:
        rows = out.shape[0] // feat_dim
        out = out.reshape(rows, feat_dim)
    torch.save(out, out_name)


def write_tensor_to_binary_file(inp, out_name):
    gnn_utils_cpp.write_tensor_to_binary_file(inp, out_name)


def glorot_initializer(tensor: torch.Tensor):
    a = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    tensor.data.uniform_(-a, a)
