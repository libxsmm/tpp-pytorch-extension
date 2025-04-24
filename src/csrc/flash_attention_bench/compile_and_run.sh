source /swtools/intel/2025.1/oneapi-vars.sh --force > /dev/null 2>&1
export LD_PRELOAD=$HOME/jemalloc/lib/libjemalloc.so:$LD_PRELOAD
export LD_PRELOAD=/usr/lib64/libomp.so:$LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../../libxsmm/lib/
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
# export LIBXSMM_TARGET=clx

# write help message
if [ "$1" == "--help" ]; then
  echo "Usage: $0 <llm> <batch_size> <seq_len> <hyper> <BF16> <num_layer> <num_iter> <nheads> <head_size> <bias_flag> <nbbias_flag>"
  echo ""
  echo "llm: name of the model (default: llama-7b), options: llama-7b, llama4, llama-3.1-8B, llama-3.2-3B, llama-3.2-1B,"
  echo "gpt2, gpt3-13b, gpt3-175b, alphafold2-h32, alphafold2-h16, alphafold2-h8"
  echo "batch_size: (default: 64)"
  echo "seq_len: (default: 4096)"
  echo "hyper: (default: 0, no hyperthreading)"
  echo "BF16: (default: 1, use BF16)"
  echo "num_layer: (default: 3)"
  echo "num_iter: (default: 3)"
  echo "nheads: (default: 32)"
  echo "head_size: (default: 128)"
  echo "bias_flag: (default: 0, no bias)"
  echo "nbbias_flag: (default: 0, no nbias)"
  echo "gate_flag: (default: 0, no gate)"
  exit 0
fi

llm=${1:-llama-7b}
batch_size=${2:-64}
seq_len=${3:-4096}
hyper=${4:-0}
BF16=${5:-1}
num_layer=${6:-3}
num_iter=${7:-3}
nheads=${8:-32}
head_size=${9:-128}
bias_flag=${10:-0}
nbbias_flag=${11:-0}
gate_flag=${12:-0}

# echo "Running $llm model"
if [ "$llm" == "llama-7b" -o "$llm" == "llama4" -o "$llm" == "llama-3.1-8B" ]; then
  embed_size=4096; nheads=32; head_size=128; bias_flag=0; nbbias_flag=0; gate_flag=0
elif [ "$llm" == "llama-70b" ]; then
  embed_size=8192; nheads=64; head_size=128; bias_flag=0; nbbias_flag=0; gate_flag=0
elif [ "$llm" == "llama-3.2-3B" ]; then
  embed_size=3072; nheads=24; head_size=128; bias_flag=0; nbbias_flag=0; gate_flag=0
elif [ "$llm" == "llama-3.2-1B" ]; then
  embed_size=2048; nheads=32; head_size=64; bias_flag=0; nbbias_flag=0; gate_flag=0
elif [ "$llm" == "gpt2" ]; then
  embed_size=768; nheads=12; head_size=64; bias_flag=0; nbbias_flag=0; gate_flag=0
elif [ "$llm" == "gpt3-13b" ]; then
  embed_size=5140; nheads=40; head_size=128; bias_flag=0; nbbias_flag=0; gate_flag=0
elif [ "$llm" == "gpt3-175b" ]; then
  embed_size=12288; nheads=96; head_size=128; bias_flag=0; nbbias_flag=0; gate_flag=0
elif [ "$llm" == "alphafold2-h32" ]; then
  embed_size=256; nheads=8; head_size=32; bias_flag=1; nbbias_flag=1; gate_flag=1
elif [ "$llm" == "alphafold2-h16" ]; then
  embed_size=64; nheads=4; head_size=16; bias_flag=1; nbbias_flag=1; gate_flag=1
elif [ "$llm" == "alphafold2-h8" ]; then
  embed_size=64; nheads=8; head_size=8; bias_flag=1; nbbias_flag=1; gate_flag=1
else
  echo "Custom model: $llm, please manually set the parameters"
fi

# print all the parameters in one line
echo "llm: $llm, batch_size: $batch_size, seq_len: $seq_len, hyper: $hyper, BF16: $BF16, num_layer: $num_layer, num_iter: $num_iter, nheads: $nheads, head_size: $head_size, bias_flag: $bias_flag, nbbias_flag: $nbbias_flag, gate_flag: $gate_flag"


echo "Compiling MHA"
g++ -O2 MHA_attention_bench.cpp init.cpp -o mha.o -I ../ -I ../../../libxsmm/include/ -DLIBXSMM_DEFAULT_CONFIG -L ../../../libxsmm/lib/ -lxsmm -fopenmp -mavx512f > /dev/null 2>&1
# KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=4 perf stat -e cycles,cpu/event=0xc2,umask=0x2,name=UOPS_RETIRED.RETIRE_SLOTS/,cpu/event=0xc1,umask=0x02,cmask=0x1,name=FP_ASSIST.ANY/,cpu/event=0xc1,umask=0x08,cmask=0x1,name=ASSISTS.PAGE_FAULT/,cpu/event=0xc1,umask=0x10,cmask=0x1,name=ASSISTS.SSE_AVX_MIX/,cpu/event=0x79,umask=0x30,name=IDQ.MS_UOPS/ ./mha.o 32 3072 8 8 0 0 0 1 0

# get number of cpu from lscpu command
cpu_count=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')


# <batch_size> <seq_len> <num_heads> <head_size> <bias_flag> <nbbias_flag> <gate_flag> <BF16> <num_layer> <num_iter>
if [ "$hyper" != "1" ]; then
    threads=$cpu_count
    echo "CPU count: $cpu_count, threads: $threads"
    KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=$threads numactl -m 0 -N 0 ./mha.o $batch_size $seq_len $nheads $head_size $bias_flag $nbbias_flag $gate_flag $BF16 $num_layer $num_iter
else
    threads=$((cpu_count * 2))
    echo "CPU count: $cpu_count, threads: $threads"
    KMP_AFFINITY=granularity=fine,compact,0,0 OMP_NUM_THREADS=$threads numactl -m 0 -N 0 ./mha.o $batch_size $seq_len $nheads $head_size $bias_flag $nbbias_flag $gate_flag $BF16 $num_layer $num_iter
fi