#!/bin/bash

if [ "$#" != "1" ]; then
  "Usage: apply_patch.sh <path-to-dgl-local-dir>"
  exit
fi

cp src/distgnn/* $1/python/dgl/distgnn
cp src/partition.py $1/python/dgl
cp src/distributed/partition.py $1/python/dgl/distributed
cp src/dataloading/neighbor_sampler.py $1/python/dgl/dataloading
cp src/nn/pytorch/hetero.py $1/python/dgl/nn/pytorch

cp csrc/cpu/* $1/src/array/cpu
cp csrc/runtime/ndarray.cc $1/src/runtime
cp csrc/runtime/ndarray.h $1/include/dgl/runtime
