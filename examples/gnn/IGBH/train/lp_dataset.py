# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import os.path as osp
import os, psutil

import dgl
from dgl.data import DGLDataset

class IGBHeteroDGLDataset(DGLDataset):
  def __init__(self,
               path,
               dataset_size='tiny',
               in_memory=False,
               use_label_2K=False,
               data_type='bf16'):
    self.dir = path
    self.dataset_size = dataset_size
    self.in_memory = in_memory
    self.use_label_2K = use_label_2K
    self.data_type = data_type

    self.ntypes = ['paper', 'author', 'institute', 'fos']
    self.etypes = None
    self.edge_dict = {}
    self.paper_nodes_num = {'tiny':100000, 'small':1000000, 'medium':10000000, 'large':100000000, 'full':269346174}
    self.author_nodes_num = {'tiny':357041, 'small':1926066, 'medium':15544654, 'large':116959896, 'full':277220883}
    self.process()

  def process(self):
      path = osp.join(self.dir, self.dataset_size, 'struct.graph')
      try:
          self.graph = dgl.data.utils.load_graphs(path)[0][0]
      except:
          print(f'Could not load graph from {path}')
      print(self.graph)

      label_file = 'node_label_19.npy' if not self.use_label_2K else 'node_label_2K.npy'

      if self.data_type == 'int8':
          paper_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', 'node_feat_int8.pt')
          paper_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', 'node_feat_scf.pt')
          paper_lbl_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', label_file)

          paper_node_features = torch.load(paper_feat_path)
          paper_feat_scf = torch.load(paper_scf_path)
          if self.dataset_size in ['large', 'full']:
              paper_node_labels = torch.from_numpy(np.fromfile(paper_lbl_path, dtype=np.float32)).to(torch.long)
          else:
              paper_node_labels = torch.from_numpy(np.load(paper_lbl_path)).long()

          author_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'author', 'node_feat_int8.pt')
          author_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'author', 'node_feat_scf.pt')
          author_node_features = torch.load(author_feat_path)
          author_feat_scf = torch.load(author_scf_path)

          institute_feat_path = osp.join(self.dir, self.dataset_size, 'processed','institute', 'node_feat_int8.pt')
          institute_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'institute', 'node_feat_scf.pt')
          institute_node_features = torch.load(institute_feat_path)
          institute_feat_scf = torch.load(institute_scf_path)

          fos_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'fos', 'node_feat_int8.pt')
          fos_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'fos', 'node_feat_scf.pt')
          fos_node_features = torch.load(fos_feat_path)
          fos_feat_scf = torch.load(fos_scf_path)

          if self.dataset_size in ['large', 'full']:
              conference_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'conference', 'node_feat_int8.pt')
              conference_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'conference', 'node_feat_scf.pt')
              conference_node_features = torch.load(conference_feat_path)
              conference_feat_scf = torch.load(conference_scf_path)

              journal_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'journal', 'node_feat_int8.pt')
              journal_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'journal', 'node_feat_scf.pt')
              journal_node_features = torch.load(journal_feat_path)
              journal_feat_scf = torch.load(journal_scf_path)

      elif self.data_type == 'bf16':
          paper_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', 'node_feat.pt')
          paper_lbl_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', label_file)

          paper_node_features = torch.load(paper_feat_path, mmap=True) if self.dataset_size in ['large', 'full'] else torch.load(paper_feat_path)
          if self.dataset_size in ['large', 'full']:
              paper_node_labels = torch.from_numpy(np.fromfile(paper_lbl_path, dtype=np.float32)).to(torch.long)
          else:
              paper_node_labels = torch.from_numpy(np.load(paper_lbl_path)).long()

          author_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'author', 'node_feat.pt')
          author_node_features = torch.load(author_feat_path, mmap=True) if self.dataset_size in ['large', 'full'] else torch.load(paper_feat_path)

          institute_node_path = osp.join(self.dir, self.dataset_size, 'processed', 'institute', 'node_feat.pt')
          institute_node_features = torch.load(institute_node_path)

          fos_node_path = osp.join(self.dir, self.dataset_size, 'processed', 'fos', 'node_feat.pt')
          fos_node_features = torch.load(fos_node_path)

          if self.dataset_size in ['large', 'full']:
              conference_node_path = osp.join(self.dir, self.dataset_size, 'processed', 'conference', 'node_feat.pt')
              conference_node_features = torch.load(conference_node_path)

              journal_node_path = osp.join(self.dir, self.dataset_size, 'processed', 'journal', 'node_feat.pt')
              journal_node_features = torch.load(journal_node_path)

      elif self.data_type in ['bf8', 'hf8']:
          dt = self.data_type
          paper_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', 'node_feat_'+dt+'.pt')
          paper_lbl_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', label_file)

          paper_node_features = torch.load(paper_feat_path)
          paper_node_labels = torch.from_numpy(np.fromfile(paper_lbl_path, dtype=np.float32)).to(torch.long)

          author_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'author', 'node_feat_'+dt+'.pt')
          author_node_features = torch.load(author_feat_path)

          institute_node_path = osp.join(self.dir, self.dataset_size, 'processed', 'institute', 'node_feat_'+dt+'.pt')
          institute_node_features = torch.load(institute_node_path)

          fos_node_path = osp.join(self.dir, self.dataset_size, 'processed', 'fos', 'node_feat_'+dt+'.pt')
          fos_node_features = torch.load(fos_node_path)

          if self.dataset_size in ['large', 'full']:
              conference_node_path = osp.join(self.dir, self.dataset_size, 'processed', 'conference', 'node_feat_'+dt+'.pt')
              conference_node_features = torch.load(conference_node_path)

              journal_node_path = osp.join(self.dir, self.dataset_size, 'processed', 'journal', 'node_feat_'+dt+'.pt')
              journal_node_features = torch.load(journal_node_path)

      num_paper_nodes = self.paper_nodes_num[self.dataset_size]
      graph_paper_nodes = self.graph.num_nodes('paper')
      if graph_paper_nodes < num_paper_nodes:
          self.graph.nodes['paper'].data['feat'] = paper_node_features[0:graph_paper_nodes,:]
          if self.data_type == 'int8':
              self.graph.nodes['paper'].data['scf'] = paper_feat_scf[0:graph_paper_nodes] 
          self.graph.num_paper_nodes = graph_paper_nodes
          self.graph.nodes['paper'].data['label'] = paper_node_labels[0:graph_paper_nodes]
      else:
          self.graph.nodes['paper'].data['feat'] = paper_node_features
          if self.data_type == 'int8':
              self.graph.nodes['paper'].data['scf'] = paper_feat_scf
          self.graph.num_paper_nodes = paper_node_features.shape[0]
          self.graph.nodes['paper'].data['label'] = paper_node_labels[0:graph_paper_nodes]
      self.graph.nodes['author'].data['feat'] = author_node_features
      if self.data_type == 'int8':
          self.graph.nodes['author'].data['scf'] = author_feat_scf
      self.graph.num_author_nodes = author_node_features.shape[0]

      self.graph.nodes['institute'].data['feat'] = institute_node_features
      if self.data_type == 'int8':
          self.graph.nodes['institute'].data['scf'] = institute_feat_scf
      self.graph.num_institute_nodes = institute_node_features.shape[0]
      
      self.graph.nodes['fos'].data['feat'] = fos_node_features
      if self.data_type == 'int8':
          self.graph.nodes['fos'].data['scf'] = fos_feat_scf
      self.graph.num_fos_nodes = fos_node_features.shape[0]
      
      if self.dataset_size in ['large', 'full']:
          self.graph.num_conference_nodes = conference_node_features.shape[0]
          self.graph.nodes['conference'].data['feat'] = conference_node_features
          if self.data_type == 'int8':
              self.graph.nodes['conference'].data['scf'] = conference_feat_scf

          self.graph.num_journal_nodes = journal_node_features.shape[0]
          self.graph.nodes['journal'].data['feat'] = journal_node_features
          if self.data_type == 'int8':
              self.graph.nodes['journal'].data['scf'] = journal_feat_scf
      
    
  def __getitem__(self, i):
      return self.graph

  def __len__(self):
      return 1
