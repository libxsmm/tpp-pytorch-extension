#!/bin/bash

###############################################################################
# Copyright (c) 2024 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

function print_vars {
  for VAR in ${!CCL*} ${!I_MPI*} ${!i_mpi*} ${!PSM3*} ${!FI_*} ${!KMP_*} ${!OMP_*} ${!GOMP*} ${!ATL_*} LD_PRELOAD LD_LIBRARY_PATH ${!DLRM_*} ${!PYTORCH_*} ${!TPP_*} ${!LIBXSMM_*} ${!EMULATE_*} VIRTUAL_ENV ${!ARGS_*} $@ ; do
    if ! test -z ${!VAR} ; then
       echo "Using $VAR=${!VAR}"
    fi
  done
}

SINGLE_SOCKET_ONLY=0

while (( "$#" )); do
  case "$1" in
    -n|-np)
      ARGS_NTASKS=$2
      shift 2
      ;;
    -t|-nt)
      ARGS_NTHREADS=$2
      shift 2
      ;;
    -m)
      ARGS_LIST=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      break
      ;;
  esac
done

CORES_PER_SOCKET=`$PREFIX lscpu | grep "Core(s) per socket" | awk '{print $NF}'`
NUM_SOCKETS=`$PREFIX lscpu | grep "Socket(s):" | awk '{print $NF}'`
NUM_NUMA_NODES=`$PREFIX lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
THREADS_PER_CORE=`$PREFIX lscpu | grep "Thread(s) per core:" | awk '{print $NF}'`
CORES_PER_NUMA=$(( (CORES_PER_SOCKET * NUM_SOCKETS) / NUM_NUMA_NODES ))

NP=1

if ! test -z $ARGS_NTASKS && ! test -z $ARGS_LIST ; then
  echo "Either -n/-np or -m can be specified but not both at the same time"
  exit 1
fi
if ! test -z $ARGS_NTASKS ; then
  NP=$ARGS_NTASKS ;
  NUMA_NODE_LIST=$(seq 0 $((NP-1)) )
  I_MPI_PIN_DOMAIN=numa
fi
if ! test -z $ARGS_LIST ; then
  NUMA_NODE_LIST=$(echo $ARGS_LIST | tr "," " ")
  NP=$(echo $NUMA_NODE_LIST | awk '{print NF}')
  I_MPI_PIN_DOMAIN=node
fi


echo "Running $NP tasks"

function rank_cmd() {
  RANK=$1
  NCORES=$2
  # PREFIX="-np 1 numactl -m $RANK -C $(( NCORES * RANK ))-$(( NCORES * (RANK+1) - 1 )) "
  CORERANGE=`lscpu | grep "NUMA node$1 CPU" | awk '{print $NF}' | tr "," " " | awk '{print $1}'`
  PREFIX="-np 1 numactl -m $RANK -C $CORERANGE "
  echo $PREFIX
}

if [ $THREADS_PER_CORE -gt 1 ] ; then
HT_WORKER_OFFSET=$(( CORES_PER_SOCKET * NUM_SOCKETS ))
else
HT_WORKER_OFFSET=0
fi
NUM_THREADS=${CORES_PER_NUMA}

if [ $NP == 1 ] ; then
export CCL_WORKER_COUNT=0
else
if [ "x${CCL_WORKER_COUNT}" == "x" ] ; then
export CCL_WORKER_COUNT=1
fi
fi
CCL_WORKER_AFFINITY=""
PYTORCH_MPI_THREAD_AFFINITY=""

for I in $NUMA_NODE_LIST ; do
  for((P=0;P < CCL_WORKER_COUNT ; P++)); do CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY} $(( HT_WORKER_OFFSET + I * NUM_THREADS + P ))" ; done
  PYTORCH_MPI_THREAD_AFFINITY="${PYTORCH_MPI_THREAD_AFFINITY} $(( HT_WORKER_OFFSET + I * NUM_THREADS ))"
done
export CCL_WORKER_AFFINITY=`echo ${CCL_WORKER_AFFINITY} | tr " " ","`
if ! test -z $ARGS_NTHREADS ; then 
  export OMP_NUM_THREADS=$ARGS_NTHREADS
else
  export OMP_NUM_THREADS=$NUM_THREADS
fi
export PYTORCH_MPI_THREAD_AFFINITY=`echo ${PYTORCH_MPI_THREAD_AFFINITY} | tr " " ","`

which python icc gcc mpicc mpiexec.hydra 2> /dev/null

echo "#### INITIAL ENV ####"
print_vars
echo "#### INITIAL ENV ####"

echo "PyTorch version: `python -c "import torch; print(torch.__version__)" 2> /dev/null`"

export MASTER_ADDR="127.0.0.1"
if test -z $MASTER_PORT ; then
  export MASTER_PORT=29500
fi
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

CMD=$1
shift
ARGS="$@"

for I in $NUMA_NODE_LIST ; do
  RANK_CMD=`rank_cmd $I $NUM_THREADS`
  if test -z "$MPI_CMD" ; then
    MPI_CMD="${RANK_CMD} $CMD ${ARGS}"
  else
    MPI_CMD="${MPI_CMD} : ${RANK_CMD} $CMD ${ARGS} "
  fi
done
MPIEXE_ARGS="-l -genv I_MPI_PIN_DOMAIN=$I_MPI_PIN_DOMAIN -genv CCL_WORKER_AFFINITY=$CCL_WORKER_AFFINITY -genv CCL_WORKER_COUNT=$CCL_WORKER_COUNT -genv OMP_NUM_THREADS=$OMP_NUM_THREADS -genv PYTORCH_MPI_THREAD_AFFINITY=$PYTORCH_MPI_THREAD_AFFINITY "

eval set -- "${MPIEXE_ARGS} $MPI_CMD"
echo "Running mpiexec.hydra $@"
echo "Start Time:  `date`"
SECONDS=0
mpiexec.hydra $@
echo "End Time:    `date`"
duration=$SECONDS
echo "Total Time: $(($duration / 60)) min and $(($duration % 60)) sec"

