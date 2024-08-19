import os, sys, psutil, json, argparse, time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data.utils import load_tensors, load_graphs
from dgl.data import register_data_args, load_data

import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from dgl.distgnn.communicate import mpi_allreduce
import time
import os

debug = False

class partition_book:
    def __init__(self, p):
        g_orig, node_feats, node_map, num_parts, n_classes, dle, etypes = p
        self.g_orig = g_orig
        self.node_feats = node_feats
        self.node_map = node_map
        self.num_parts = num_parts
        self.n_classes = n_classes
        self.dle = dle
        self.etypes = etypes

def standardize_metis_parts(graph, node_feats, rank, resize=False):
    N = graph.number_of_nodes()
    E = graph.number_of_edges()

    if resize:
        nlocal = (graph.ndata['inner_node'] == 1).sum()
        try:
            feat = node_feats['_N/features']
        except:
            feat = node_feats['_N/feat']

        try:
            label = node_feats['_N/label'].clone().resize_(N)
        except:
            label = node_feats['_N/labels'].clone().resize_(N)
        train = node_feats['_N/train_mask'].clone().resize_(N)
        val = node_feats['_N/val_mask'].clone().resize_(N)
        test = node_feats['_N/test_mask'].clone().resize_(N)

        ten = th.zeros(N-feat.shape[0], feat.shape[1], dtype=feat.dtype)
        feat_ = th.cat((feat, ten), 0)

        train[nlocal: ] = 0
        test[nlocal: ] = 0
        val[nlocal: ] = 0
        node_feats['feat'] = feat_
        node_feats['label'] = label
        node_feats['train_mask'] = train
        node_feats['test_mask'] = test
        node_feats['val_mask'] = val

        graph.ndata['orig'] = graph.ndata['orig_id']
        del graph.ndata['orig_id']
        ntrain = (node_feats['train_mask'] == 1).sum()
        tot_train = mpi_allreduce(ntrain)

        ntest = (node_feats['test_mask'] == 1).sum()
        tot_test = mpi_allreduce(ntest)

        nval = (node_feats['val_mask'] == 1).sum()
        tot_val = mpi_allreduce(nval)

        tot_nodes = mpi_allreduce(N)
        tot_edges = mpi_allreduce(E)

        if rank == 0:
            print('tot_train nodes: ', tot_train)
            print('tot_test nodes: ', tot_test)
            print('tot_val nodes: ', tot_val)

    else:
        nlocal = (graph.ndata['inner_node'] == 1).sum()
        try:
            feat = node_feats['_N/features']
        except:
            feat = node_feats['_N/feat']

        try:
            label = node_feats['_N/label']
        except:
            label = node_feats['_N/labels']
        train = node_feats['_N/train_mask']
        val = node_feats['_N/val_mask']
        test = node_feats['_N/test_mask']

        node_feats['feat'] = feat
        node_feats['label'] = label
        node_feats['train_mask'] = train
        node_feats['test_mask'] = test
        node_feats['val_mask'] = val
        node_feats['orig'] = graph.ndata['orig_id']
        node_feats['inner_node'] = graph.ndata['inner_node']
        node_feats['orig'] = graph.ndata['orig_id']

        ntrain = (node_feats['train_mask'] == 1).sum()
        tot_train = mpi_allreduce(ntrain)

        ntest = (node_feats['test_mask'] == 1).sum()
        tot_test = mpi_allreduce(ntest)

        nval = (node_feats['val_mask'] == 1).sum()
        tot_val = mpi_allreduce(nval)

        tot_nodes = mpi_allreduce(N)
        tot_edges = mpi_allreduce(E)

        if rank == 0:
            print('tot_train nodes: ', tot_train)
            print('tot_test nodes: ', tot_test)
            print('tot_val nodes: ', tot_val)

    try:
        del node_feats['_N/feat']
    except:
        del node_feats['_N/features']
    try:
        del node_feats['_N/label']
    except:
        del node_feats['_N/labels']
    del node_feats['_N/train_mask']
    del node_feats['_N/test_mask'], node_feats['_N/val_mask']


def load_GNNdataset(args):
    dlist = ['ogbn-products', 'ogbn-papers100M']
    if args.dataset in dlist:
        assert os.path.isdir(args.path) == True
        filename = os.path.join(args.path, "struct.graph")
        tfilename = os.path.join(args.path, "tensor.pt")

        if os.path.isfile(filename) and os.path.isfile(tfilename):
            data, _ = dgl.load_graphs(filename)
            g_orig = data[0]
            n_classes = int(th.load(tfilename))
        else:
            def load_ogb(name):
                from ogb.nodeproppred import DglNodePropPredDataset
            
                data = DglNodePropPredDataset(name=name, root='./dataset')
                splitted_idx = data.get_idx_split()
                graph, labels = data[0]
                labels = labels[:, 0]
            
                graph.ndata['features'] = graph.ndata['feat']
                graph.ndata['labels'] = labels
                in_feats = graph.ndata['features'].shape[1]
                num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
            
                # Find the node IDs in the training, validation, and test set.
                train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
                train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
                train_mask[train_nid] = True
                val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
                val_mask[val_nid] = True
                test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
                test_mask[test_nid] = True
                graph.ndata['train_mask'] = train_mask
                graph.ndata['val_mask'] = val_mask
                graph.ndata['test_mask'] = test_mask
                return graph, num_labels

            try:
                g_orig, n_classes = load_ogb(args.dataset)
                if not debug:
                    try:
                        del g_orig.ndata['feat']
                        del g_orig.ndata['features']
                    except:
                        pass
                g_orig = dgl.add_reverse_edges(g_orig)
                if args.rank == 0 and not debug:
                    dgl.save_graphs(filename, [g_orig])
                    th.save(th.tensor(n_classes), tfilename)
            except Exception as e:
                print(e)
                n_classes = -1
                g_orig = None

    elif args.dataset == 'IGBH':
        path = os.path.join(args.path, args.data, args.dataset_size)
        filename = os.path.join(
                     path, 
                     str(args.world_size)+args.token, 
                     "struct.graph"
                   )
        assert(os.path.isfile(filename)) == True
        n_classes = args.n_classes
        g_orig = dgl.load_graphs(filename)[0][0]
    else:
        print(">>>>>>>>>> Error: dataset {} not found! exiting...".format(dataset))
        sys.exit(1)

    return g_orig, n_classes

def partition_book_random(args, part_config, category='', resize_data=False):

    num_parts = args.world_size

    dls = time.time()
    g_orig, n_classes = load_GNNdataset(args)
    ntypes = g_orig.ntypes
    g_orig = None

    part_config_g = part_config
    di = str(num_parts) + args.token
    part_config_g = os.path.join(part_config_g, di)
    fjson = args.dataset + '.json'
    part_config = os.path.join(part_config_g, fjson)

    try:
        with open(part_config) as conf_f:
            part_metadata = json.load(conf_f)
    except:
        print(">>>>> Error: Partition data for {} not found!! at {}".
              format(args.dataset, part_config))
        sys.exit(1)

    prefix = part_config_g + "/"
    part_files = part_metadata['part-{}'.format(args.rank)]

    dle = time.time() - dls

    graph = load_graphs(prefix + part_files['part_graph'])[0][0]
    nsn = int(graph.ndata['inner_node'].sum())

    train_mask = th.zeros(nsn, dtype=th.uint8)
    val_mask = th.zeros(nsn, dtype=th.uint8)
    labels = th.zeros(nsn, dtype=th.long)

    pnid = {}

    mask_key = None
    for k, nt in enumerate(ntypes):
        pnid[k] = (graph.ndata['_TYPE'][:nsn] == k).nonzero(as_tuple=True)[0]
        if nt == category:
            mask_key = k
        
    inner_node = graph.ndata['inner_node']
    orig_id = graph.ndata['orig_id']
    _TYPE = graph.ndata['_TYPE']
                
    graph = None

    if args.rank == 0:
        process = psutil.Process(os.getpid())
        print(f'Loaded graph partition data. Mem used: {process.memory_info().rss/1e9} GB', flush=True)

    acc, acc_labels = 0, 0
    part_files = part_metadata['part-{}'.format(args.rank)]
    node_feats = load_tensors(prefix + part_files['node_feats'])
    if len(ntypes) > 1:
        train_mask[pnid[mask_key]] = node_feats[category+'/train_mask']
        val_mask[pnid[mask_key]] = node_feats[category+'/val_mask']
        labels[pnid[mask_key]] = node_feats[category+'/label']
    else:
        train_mask[pnid[mask_key]] = node_feats['train_mask']
        val_mask[pnid[mask_key]] = node_feats['val_mask']
        labels[pnid[mask_key]] = node_feats['label']

    node_feats['train_mask'] = train_mask
    node_feats['val_mask'] = val_mask
    node_feats['labels'] = labels
    node_feats['inner_node'] = inner_node
    node_feats['orig'] = orig_id
    node_feats['_TYPE'] = _TYPE
    del node_feats[category+'/train_mask']
    del node_feats[category+'/val_mask']
    del node_feats[category+'/test_mask']
    del node_feats[category+'/label']

    node_feats['feat'] = None
    node_feats['scf'] = None
    if len(ntypes) > 1:
        for nt in ntypes:
            if node_feats['feat'] is None:
                node_feats['feat'] = node_feats[nt+'/feat']
                del node_feats[nt+'/feat']
            else:
                node_feats['feat'] = th.cat((node_feats['feat'], node_feats[nt+'/feat']),0)
                del node_feats[nt+'/feat']

            if args.use_int8:
                if node_feats['scf'] is None:
                    node_feats['scf'] = node_feats[nt+'/scf']
                    del node_feats[nt+'/scf']
                else:
                    node_feats['scf'] = th.cat((node_feats['scf'], node_feats[nt+'/scf']), 0)
                    del node_feats[nt+'/scf']

    if not args.use_int8:
        if args.use_bf16 and (not node_feats['feat'].dtype == th.bfloat16): 
            node_feats['feat'] = node_feats['feat'].to(th.bfloat16)
        elif not args.use_bf16 and node_feats['feat'].dtype == th.bfloat16:
            node_feats['feat'] = node_feats['feat'].to(th.float32)

    dls = time.time()
    if args.rank == 0:
        process = psutil.Process(os.getpid())
        print(f'Loaded graph partition data. Mem used: {process.memory_info().rss/1e9} GB')

    g_orig, n_classes = load_GNNdataset(args)
    dle = dle + (time.time() - dls)

    node_feats['train_samples'] = g_orig.ndata['train_mask']['paper'].sum()
    node_feats['eval_samples'] = g_orig.ndata['val_mask']['paper'].sum()
    g_orig.ndata['test_mask']['paper'] = None
    g_orig.ndata['train_mask']['paper'] = None
    g_orig.ndata['val_mask']['paper'] = None

    if args.rank == 0:
        process = psutil.Process(os.getpid())
        print(f'Loaded full graph struct. Mem used: {process.memory_info().rss/1e9} GB')
    etypes = g_orig.etypes

    #node_map = part_metadata['node_map']
    node_map = None
    d = g_orig, node_feats, node_map, num_parts, n_classes, dle, etypes
    pb = partition_book(d)
    return pb

def partition_book_metis(args, part_config, resize_ndata=False):

    num_parts = args.world_size

    dls = time.time()
    g_orig, n_classes = load_GNNdataset(args)
    ntypes = g_orig.ntypes
    etypes = g_orig.canonical_etypes

    part_config_g = part_config

    di = args.dataset + "-" + str(num_parts) + args.token + "-balance-train"
    part_config_g = os.path.join(part_config_g, di)
    fjson = args.dataset + '.json'
    part_config = os.path.join(part_config_g, fjson)

    #if args.rank == 0:
    #    print("Dataset/partition location: ", part_config)

    try:
        with open(part_config) as conf_f:
            part_metadata = json.load(conf_f)
    except:
        print(">>>>> Error: Partition data for {} not found!! at {}".
              format(args.dataset, part_config))
        sys.exit(1)

    prefix = part_config_g + "/"
    part_files = part_metadata['part-{}'.format(args.rank)]
    assert 'node_feats' in part_files, "the partition does not contain node features."
    assert 'edge_feats' in part_files, "the partition does not contain edge feature."
    assert 'part_graph' in part_files, "the partition does not contain graph structure."
    node_feats = load_tensors(prefix + part_files['node_feats'])
    graph = load_graphs(prefix + part_files['part_graph'])[0][0]

    dle = time.time() - dls

    #num_parts = part_metadata['num_parts']
    #node_map_  = part_metadata['node_map']

    standardize_metis_parts(graph, node_feats, args.rank, resize_ndata)
    del graph
    if args.use_bf16: node_feats['feat'] = node_feats['feat'].to(th.bfloat16)

    #node_map = []   ## this block should go in standardize_metis_parts
    #for nm in node_map_['_N']:
    #    node_map.append(nm[1])

    node_map = None
    if args.rank == 0:
        print("n_classes: ", n_classes, flush=True)

    etypes = None
    d = g_orig, node_feats, node_map, num_parts, n_classes, dle, etypes
    pb = partition_book(d)
    return pb

def create_partition_book(args, part_config, resize=False):
    if args.part_method == 'metis':
        pb = partition_book_metis(args, part_config, resize)
    elif args.part_method == 'random':
        pb = partition_book_random(args, part_config, resize)

    return pb

class args_:
    def __init__(self, dataset):
        self.dataset = dataset
        print("dataset set to: ", self.dataset)
