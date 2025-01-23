import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from contextlib import contextmanager
from tpp_pytorch_extension.gnn.gat_inference import fused_gat as tpp_gat
import dgl
from dgl import apply_each

global_layer_dtype = torch.float32
global_use_qint8_gemm = False

@contextmanager
def opt_impl(enable=True, use_qint8_gemm=False, use_bf16=False):
    try:
        global GATConv
        global linear
        global use_tpp
        global global_layer_dtype
        global global_use_qint8_gemm
        GATConv = dgl.nn.pytorch.GATConv
        linear = nn.Linear
        try:
            if enable:
                use_tpp = enable
                if use_bf16:
                    global_layer_dtype = torch.bfloat16
                if use_qint8_gemm:
                    global_use_qint8_gemm = use_qint8_gemm
                GATConv =  tpp_gat.GATConvOpt
                linear = tpp_gat.LinearOut
            yield
        finally:
            GATConv = dgl.nn.pytorch.GATConv
            linear = nn.Linear
    except ImportError as e:
        pass

class RGAT_DGL(nn.Module):
    def __init__(self, etypes, in_feats=1024, h_feats=512, num_classes=2983, num_layers=3, n_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(dgl.nn.pytorch.HeteroGraphConv({
            etype: GATConv(in_feats, h_feats // n_heads, n_heads, activation=F.leaky_relu,
                   layer_dtype=global_layer_dtype,
                   use_qint8_gemm=global_use_qint8_gemm,)
            for etype in etypes}))

        for _ in range(num_layers - 2):
            self.layers.append(dgl.nn.pytorch.HeteroGraphConv({
                etype: GATConv(h_feats, h_feats // n_heads, n_heads, activation=F.leaky_relu,
                   layer_dtype=global_layer_dtype,
                   use_qint8_gemm=global_use_qint8_gemm,)
                for etype in etypes}))
         # No relu for final GATConv
        self.layers.append(dgl.nn.pytorch.HeteroGraphConv({
            etype: GATConv(h_feats, h_feats // n_heads, n_heads,
                   layer_dtype=global_layer_dtype,
                   use_qint8_gemm=global_use_qint8_gemm,)
            for etype in etypes}))
        self.linear = linear(h_feats, num_classes, global_layer_dtype)

    def forward_gather(self, blocks, x, idx):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if l != 0:
                mod_kwargs={}
                for _, etype, _ in block.canonical_etypes:
                    mod_kwargs.update({etype: {'first_layer': False}})

                h = layer(block, h, mod_kwargs=mod_kwargs)
            else:
                mod_kwargs={}
                for stype, etype, dtype in block.canonical_etypes:
                    dst_idx = {
                       k: v[: block.number_of_dst_nodes(k)] for k, v in idx.items()
                    }
                    mod_kwargs.update({etype: {'first_layer': True, 'src_idx':idx[stype], 'dst_idx':dst_idx[dtype]}})

                h = layer(block, h, mod_kwargs=mod_kwargs)

            h = dgl.apply_each(h, lambda x: x.view( x.shape[0], x.shape[1] * x.shape[2]))

        return self.linear(h['paper'])

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            mod_kwargs={}
            for _, etype, _ in block.canonical_etypes:
                mod_kwargs.update({etype: {'first_layer': False}})

            h = layer(block, h, mod_kwargs=mod_kwargs)
            h = dgl.apply_each(h, lambda x: x.view( x.shape[0], x.shape[1] * x.shape[2]))

        return self.linear(h['paper'])

    def forward_graph(self, blocks):
        breakpoint()
        h = blocks[0].srcdata['feat']
        breakpoint()
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            mod_kwargs={}
            for _, etype, _ in block.canonical_etypes:
                mod_kwargs.update({etype: {'first_layer': False}})

            h = layer(block, h, mod_kwargs=mod_kwargs)
            h = dgl.apply_each(h, lambda x: x.view( x.shape[0], x.shape[1] * x.shape[2]))

        return self.linear(h['paper'])

class RGAT(torch.nn.Module):
    def __init__(self, etypes, use_tpp, use_qint8_gemm, use_bf16):
        super().__init__()
        with opt_impl(use_tpp, use_qint8_gemm, use_bf16):
            self.model = RGAT_DGL(etypes=etypes)
            self.layers = self.model.layers

    def forward_gather(self, batch, batch_inputs, batch_idx):
        return self.model.forward_gather(batch, batch_inputs, batch_idx)

    def forward(self, batch, batch_inputs):
        return self.model.forward(batch, batch_inputs)

    def forward_graph(self, batch):
        return self.model.forward_graph(batch)

def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()
