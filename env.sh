#!/bin/bash
#
# Environment for the Bonsai (ternary) runs.
#
# Requirements beyond the stock env:
#   pip install "transformers==5.15.1"      # Qwen3.5 (Bonsai 27B) support; the
#                                           # Bonsai 8B fused Qwen3 path also works
#   torchaudio 2.11 in this env is ABI-broken against torch 2.7.1 and transformers>=5
#   imports it eagerly, so it must be disabled:
#     mv site-packages/torchaudio{,_disabled_broken}
#     mv site-packages/torchaudio-*.dist-info{,.disabled}
#
# NOTE: setup.py compiles with -march=native, so `python setup.py install` must be
# run ON THE TARGET NODE. A build tree from a different machine will SIGILL.

source /data/nfs_home/egeorgan/test_scc26_ae/libxsmm/tpp-pytorch-extension/miniforge3/bin/activate pt271
torch_ccl_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))" 2> /dev/null | grep oneccl_bind_pt |tail -n 1)
if test -f $torch_ccl_path/env/setvars.sh ; then
  source $torch_ccl_path/env/setvars.sh
fi

NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk '{print $NF}')
ARCH=$(lscpu | grep "Architecture" | awk '{print $NF}')
export OMP_NUM_THREADS=${NUM_THREADS}
if [ $(uname -m) == "x86_64" ] ; then
  export KMP_AFFINITY=compact,1,granularity=fine
  export KMP_BLOCKTIME=1
  export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so
fi
