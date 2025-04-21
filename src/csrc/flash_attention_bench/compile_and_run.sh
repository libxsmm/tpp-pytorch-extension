source /swtools/intel/2025.1/oneapi-vars.sh --force
export LD_PRELOAD=$HOME/jemalloc/lib/libjemalloc.so:$LD_PRELOAD
export LD_PRELOAD=/usr/lib64/libomp.so:$LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../../libxsmm/lib/
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

hyper=$1
if [ -z "$hyper" ]; then
  hyper=0
fi

# get number of cpu from lscpu command
cpu_count=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')
echo "CPU count: $cpu_count"

g++ -O2 MHA_attention_bench.cpp init.cpp -o mha.o -I ../ -I ../../../libxsmm/include/ -DLIBXSMM_DEFAULT_CONFIG -L ../../../libxsmm/lib/ -lxsmm -fopenmp -mavx512f

# KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=4 perf stat -e cycles,cpu/event=0xc2,umask=0x2,name=UOPS_RETIRED.RETIRE_SLOTS/,cpu/event=0xc1,umask=0x02,cmask=0x1,name=FP_ASSIST.ANY/,cpu/event=0xc1,umask=0x08,cmask=0x1,name=ASSISTS.PAGE_FAULT/,cpu/event=0xc1,umask=0x10,cmask=0x1,name=ASSISTS.SSE_AVX_MIX/,cpu/event=0x79,umask=0x30,name=IDQ.MS_UOPS/ ./mha.o 32 3072 8 8 0 0 0 1 0

# <batch_size> <seq_len> <num_heads> <head_size> <bias_flag> <nbbias_flag> <BF16> <num_layer> <num_iter>
if [ "$hyper" != "1" ]; then
    threads=$cpu_count
    echo "Running without hyperthreading, threads: $threads"
    KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=$threads numactl -m 0 -N 0 ./mha.o 512 4096 8 8 0 0 1 3 3
else
    threads=$((cpu_count * 2))
    echo "Running with hyperthreading, threads: $threads"
    KMP_AFFINITY=granularity=fine,compact,0,0 OMP_NUM_THREADS=$threads numactl -m 0 -N 0 ./mha.o 512 4096 8 8 0 0 1 3 3
fi





