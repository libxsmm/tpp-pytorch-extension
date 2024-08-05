
# Steps for Distributed R-GAT Model Inference on IGBH Dataset

0. If you are reading this document, you have already cloned tpp-pytorch-extension in your working directory (WD).

1. Create a new conda environment containing required packages
```
$cd WD
$pushd <conda-install-dir>
$bash <path-to-WD/tpp-pytorch-extension>/utils/setup_conda.sh
$source env.sh
$popd
```

2. Install tpp-pytorch-extension.

```
$pushd <path-to-WD>/tpp-pytorch-extension
$git submodule update --init
$python setup.py install
$popd
```

3. Install torch-mpi

```
$source <path-to-Intel MPI>/env/vars.sh
$pushd <path-to-WD>/tpp-pytorch-extension/examples/gnn/IGBH/torch_mpi
$python setup.py install
$popd
```

4. Download and Quantize IGBH-Full dataset

```
$pushd <path-to-WD>/tpp-pytorch-extension/examples/gnn/IGBH/dataset
$bash download_igbh600m.sh <path-to-data-store>
$bash cvt.sh <path-to-data-store>/IGBH full int8 128
$popd
```

5. Install Deep Graph Library and apply Intel optimizations

```
$git clone --recursive https://github.com/dmlc/dgl <path-to-WD/dgl-local-dir>
$pushd <path-to-WD/dgl-local-dir>/third_party/libxsmm
$git checkout main
$popd
$pushd <path-to-WD>/tpp-pytorch-extension/examples/gnn/IGBH/sources
$bash apply_patch.sh <path-to-WD/dgl-local-dir>
$popd
$pushd <path-to-WD/dgl-local-dir>
$mkdir build
$cd build
$cmake -DBUILD_TYPE='release' -DUSE_LIBXSMM=ON -DUSE_OPENMP=ON -DBUILD_SPARSE=OFF -DUSE_EPOLL=OFF ..
# Recommend gcc version >= 11.3.0 for compilation
$export CC=gcc 
$export CXX=g++ 
$make -j
$cd ../python
$python setup.py install
$popd
```

6. Partition IGBH-Full graph 

Pre-requisite: This step requires a machine with at least 2TB RAM.

```
$pushd <path-to-WD>/tpp-pytorch-extension/examples/gnn/IGBH/partition
$bash partition_graph.sh <path-to-data-store> full <num-partitions> int8
$popd
```

7. Evaluate trained R-GAT Model

```
$source <path-to-Intel MPI>
```

Run eval script 

In below command line, "path-to-partitioned-dataset" is path printed by partitioning script in Step 6 above

```
$pushd <path-to-WD>/tpp-pytorch-extension/examples/gnn/IGBH/eval
$bash run_dist_eval.sh <path-to-partitioned-dataset> <number-of-partitions> <num-processes-per-node> <batch-size> <path-to-model-checkpoint> <fraction-of-validation-data> <fan-out>
$popd
```

