#!/bin/bash

###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

set -e
HERE=$(cd "$(dirname "$0")" && pwd -P)
CONDA_INSTALL_DIR=`realpath ./miniforge3`
ENV_NAME=pt251

while (( "$#" )); do
  case "$1" in
    -n)
      ENV_NAME=$2
      shift 2
      ;;
    -p)
      CONDA_INSTALL_DIR=$2
      CONDA_INSTALL_DIR=`realpath $CONDA_INSTALL_DIR`
      shift 2
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      echo "Error: Unsupported argument $1" >&2
      exit 1
      ;;
  esac
done

if ! test -f Miniforge3-Linux-$(uname -m).sh ; then
  wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
fi
if ! test -d ${CONDA_INSTALL_DIR} ; then
  bash ./Miniforge3-Linux-$(uname -m).sh -b -p ${CONDA_INSTALL_DIR}
fi
if ! test -d ${CONDA_INSTALL_DIR}/envs/${ENV_NAME} ; then
${CONDA_INSTALL_DIR}/bin/conda create -y -n ${ENV_NAME} python=3.9
fi
source ${CONDA_INSTALL_DIR}/bin/activate ${ENV_NAME}

# conda install -n base conda-libmamba-solver
# conda config --set solver libmamba

if [ $(uname -m) == "x86_64" ] ; then
  conda install -y ninja setuptools tqdm future cmake numpy pyyaml scikit-learn pydot -c conda-forge
  conda install -y gperftools -c conda-forge
  conda install -y pytorch==2.5.1 torchvision torchaudio cpuonly -c pytorch
elif [ $(uname -m) == "aarch64" ] ; then
# rust required on aarch64 for building tokenizer
conda install -y pytorch numpy gperftools ninja setuptools tqdm future cmake  pyyaml scikit-learn pydot -c conda-forge
conda install -y -c conda-forge openblas
else
  echo "Unknown architecture: $(uname -m)"
  exit 1
fi
# for bert / llm
conda install -y pybind11 -c conda-forge
conda install -y h5py onnx tensorboardx -c conda-forge

# for unbuffer
#conda install -y -c eumetsat expect

if [ $(uname -m) == "x86_64" ] ; then
  # for development (code formatting)
  conda install -y black=22.6.0 clang-format=5.0.1 -c sarcasm -c conda-forge
  # conda install -y intel-openmp -c intel
  pip install intel-openmp
  # Need to downgrad mkl from 2024.1 due to undefined symbol error
  # conda install -y mkl=2024.0 -c intel
  #pip install mkl==2024.0
fi

# ENV_FN_NAME=env_${ENV_NAME}.sh
ENV_FN_NAME=env.sh
echo "Writing an ${ENV_FN_NAME} file..."
cat <<EOF > ${ENV_FN_NAME}
#!/bin/bash

source ${CONDA_INSTALL_DIR}/bin/activate ${ENV_NAME}
torch_ccl_path=\$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))" 2> /dev/null | grep oneccl_bind_pt |tail -n 1)
if test -f \$torch_ccl_path/env/setvars.sh ; then
  source \$torch_ccl_path/env/setvars.sh
fi

NUM_THREADS=\$(lscpu | grep "Core(s) per socket" | awk '{print \$NF}')
ARCH=\$(lscpu | grep "Architecture" | awk '{print \$NF}')
export OMP_NUM_THREADS=\${NUM_THREADS}
if [ \$(uname -m) == "x86_64" ] ; then
  export KMP_AFFINITY=compact,1,granularity=fine
  export KMP_BLOCKTIME=1
  export LD_PRELOAD=\${CONDA_PREFIX}/lib/libtcmalloc.so:\${CONDA_PREFIX}/lib/libiomp5.so
fi
EOF
chmod +x ${ENV_FN_NAME}
echo "${ENV_FN_NAME} file created..."

