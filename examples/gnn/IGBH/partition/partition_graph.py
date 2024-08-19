import argparse
import os
import sys
import time

import dgl
from dgl.data import DGLDataset

import numpy as np
import torch 
import os.path as osp
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

class IGBHeteroDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        self.process_graph()

    def process_graph(self):
        paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
        'paper__cites__paper', 'edge_index.npy')))
        author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
        'paper__written_by__author', 'edge_index.npy')))
        affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
        'author__affiliated_to__institute', 'edge_index.npy')))
        paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
        'paper__topic__fos', 'edge_index.npy')))

        if self.args.dataset_size in ['large', 'full']:
            paper_published_journal = torch.from_numpy(np.load(osp.join(self.dir, args.dataset_size, 'processed',
            'paper__published__journal', 'edge_index.npy')))
            paper_venue_conference = torch.from_numpy(np.load(osp.join(self.dir, args.dataset_size, 'processed',
            'paper__venue__conference', 'edge_index.npy')))

        self.edge_dict = {
            ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
            ('author', 'rev_written_by', 'paper'): (author_paper_edges[:, 1], author_paper_edges[:, 0]),
            ('fos', 'rev_topic', 'paper'): (paper_fos_edges[:, 1], paper_fos_edges[:, 0]),
            ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
            ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
            ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1]),
            ('institute', 'rev_affiliated_to', 'author'): (affiliation_author_edges[:, 1], affiliation_author_edges[:, 0]),
        }
        if self.args.dataset_size in ['large', 'full']:
            self.edge_dict[('conference', 'rev_venue', 'paper')] = (paper_venue_conference[:, 1], paper_venue_conference[:, 0])
            self.edge_dict[('journal', 'rev_published', 'paper')] = (paper_published_journal[:, 1], paper_published_journal[:, 0])
            self.edge_dict[('paper', 'published', 'journal')] = (paper_published_journal[:, 0], paper_published_journal[:, 1])
            self.edge_dict[('paper', 'venue', 'conference')] = (paper_venue_conference[:, 0], paper_venue_conference[:, 1])
            self.etypes = list(self.edge_dict.keys())

        self.graph = dgl.heterograph(self.edge_dict)     
        self.graph.predict = 'paper'
        print(self.graph)

        graph_paper_nodes = self.graph.num_nodes('paper')
        self.graph = dgl.remove_self_loop(self.graph, etype='cites')
        self.graph = dgl.add_self_loop(self.graph, etype='cites')

        if self.args.dataset_size == 'full':
            n_nodes = 157675969
        else:
            n_nodes = graph_paper_nodes
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
  
        label_file = 'node_label_2K.npy'
        paper_lbl_path = osp.join(args.path, args.dataset_size, 'processed', 'paper', label_file)
        if args.dataset_size in ['large', 'full']:
            paper_node_labels = torch.from_numpy(np.fromfile(paper_lbl_path, dtype=np.float32)).to(torch.long)
        else:
            paper_node_labels = torch.from_numpy(np.load(paper_lbl_path)).to(torch.long)

        shuffled_index = torch.randperm(n_nodes)      
        self.train_idx = shuffled_index[:n_train]
        self.val_idx = shuffled_index[n_train : n_train + n_val]
        self.test_idx = shuffled_index[n_train + n_val:]
        del shuffled_index

        train_mask = torch.zeros(graph_paper_nodes, dtype=torch.bool)
        val_mask = torch.zeros(graph_paper_nodes, dtype=torch.bool)
        test_mask = torch.zeros(graph_paper_nodes, dtype=torch.bool)
        
        train_mask[self.train_idx] = True
        val_mask[self.val_idx] = True
        test_mask[self.test_idx] = True

        self.graph.nodes['paper'].data['train_mask'] = train_mask
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask
        self.graph.nodes['paper'].data['label'] = paper_node_labels[:graph_paper_nodes]


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--path",
        type=str,
        help="dataset path",
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        help="dataset name",
    )    
    argparser.add_argument(
        "--num_parts", type=int, default=4, help="number of partitions"
    )
    argparser.add_argument(
        "--token", type=str, default="p", help="token to identify partition"
    )
    argparser.add_argument(
        "--feat_part_only",
        action="store_true",
        help="split only graph nodes across partitions",
    )
    argparser.add_argument(
        "--graph_struct_only",
        action="store_true",
        help="split only features across partitions",
    )
    argparser.add_argument(
        "--output",
        type=str,
        help="Output path of partitioned graph.",
    )
    argparser.add_argument(
        "--dataset_size",
        type=str,
        default='full',
        help="Output path of partitioned graph.",
    )
    argparser.add_argument(
        "--data_type",
        type=str,
        default='bf16',
        help="Feature data type.",
    )
    
    args = argparser.parse_args()
    output = args.output + args.token

    if args.graph_struct_only:
        filename = osp.join(args.path, args.dataset_size, 'struct.graph')
        print(f'Graph path {filename}')
        if not osp.isfile(filename):
            seed = int(datetime.datetime.now().timestamp())
            seed = seed >> (seed.bit_length() - 31)
            torch.manual_seed(seed)
            dgl.seed(seed)
            
            g = IGBHeteroDGLDataset(args)[0]
            dgl.save_graphs(filename, g)

            print(f'Stored full graph structure in {filename}', flush=True)
            print(
                "train: {}, valid: {}, test: {}".format(
                    torch.sum(g.ndata["train_mask"][g.predict]),
                    torch.sum(g.ndata["val_mask"][g.predict]),
                    torch.sum(g.ndata["test_mask"][g.predict]),
                )
            )
        else:
            g = dgl.load_graphs(filename)[0][0]
            print(f'Loaded full graph strcuture from {filename}')
            seed = int(datetime.datetime.now().timestamp())
            seed = seed >> (seed.bit_length() - 31)
            torch.manual_seed(seed)
            dgl.seed(seed)

    elif args.feat_part_only:
        filename = osp.join(args.path, args.dataset_size, 'struct.graph')
        g = dgl.load_graphs(filename)[0][0]

        if args.data_type == 'bf16':
            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'author', 'node_feat.pt')
            g.nodes['author'].data['feat'] = torch.load(feat_path)

            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'fos', 'node_feat.pt')
            g.nodes['fos'].data['feat'] = torch.load(feat_path)
            
            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'institute', 'node_feat.pt')
            g.nodes['institute'].data['feat'] = torch.load(feat_path)
            
            if args.dataset_size in ['large', 'full']:
                feat_path = osp.join(args.path, args.dataset_size, 'processed', 'conference', 'node_feat.pt')
                g.nodes['conference'].data['feat'] = torch.load(feat_path)
                
                feat_path = osp.join(args.path, args.dataset_size, 'processed', 'journal', 'node_feat.pt')
                g.nodes['journal'].data['feat'] = torch.load(feat_path)

            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'paper', 'node_feat.pt')
            graph_paper_nodes = g.num_nodes('paper')
            g.nodes['paper'].data['feat'] = torch.load(feat_path)[:graph_paper_nodes]

        elif args.data_type == 'int8':

            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'author', 'node_feat_int8.pt')
            scf_path = osp.join(args.path, args.dataset_size, 'processed', 'author', 'node_feat_scf.pt')
            g.nodes['author'].data['feat'] = torch.load(feat_path)
            g.nodes['author'].data['scf'] = torch.load(scf_path)

            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'fos', 'node_feat_int8.pt')
            scf_path = osp.join(args.path, args.dataset_size, 'processed', 'fos', 'node_feat_scf.pt')
            g.nodes['fos'].data['feat'] = torch.load(feat_path)
            g.nodes['fos'].data['scf'] = torch.load(scf_path)
            
            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'institute', 'node_feat_int8.pt')
            scf_path = osp.join(args.path, args.dataset_size, 'processed', 'institute', 'node_feat_scf.pt')
            g.nodes['institute'].data['feat'] = torch.load(feat_path)
            g.nodes['institute'].data['scf'] = torch.load(scf_path)
            
            if args.dataset_size in ['large', 'full']:
                feat_path = osp.join(args.path, args.dataset_size, 'processed', 'conference', 'node_feat_int8.pt')
                scf_path = osp.join(args.path, args.dataset_size, 'processed', 'conference', 'node_feat_scf.pt')
                g.nodes['conference'].data['feat'] = torch.load(feat_path)
                g.nodes['conference'].data['scf'] = torch.load(scf_path)
                
                feat_path = osp.join(args.path, args.dataset_size, 'processed', 'journal', 'node_feat_int8.pt')
                scf_path = osp.join(args.path, args.dataset_size, 'processed', 'journal', 'node_feat_scf.pt')
                g.nodes['journal'].data['feat'] = torch.load(feat_path)
                g.nodes['journal'].data['scf'] = torch.load(scf_path)

            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'paper', 'node_feat_int8.pt')
            scf_path = osp.join(args.path, args.dataset_size, 'processed', 'paper', 'node_feat_scf.pt')
            graph_paper_nodes = g.num_nodes('paper')
            g.nodes['paper'].data['feat'] = torch.load(feat_path)[:graph_paper_nodes]
            g.nodes['paper'].data['scf'] = torch.load(scf_path)[:graph_paper_nodes]

    dgl.distributed.partition_graph(
        g,
        args.dataset,
        args.num_parts,
        output,
        num_hops=1,
        part_method='random',
        balance_ntypes=None,
        balance_edges=False,
        return_mapping=False,
        num_trainers_per_machine=1,
        feat_part_only=args.feat_part_only
    )
    print(f'Stored graph partitions and features in {output}', flush=True)

