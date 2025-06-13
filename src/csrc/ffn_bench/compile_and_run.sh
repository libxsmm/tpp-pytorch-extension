source /swtools/intel/2025.1/oneapi-vars.sh --force > /dev/null 2>&1
export LD_PRELOAD=$HOME/jemalloc/lib/libjemalloc.so:$LD_PRELOAD
export LD_PRELOAD=/usr/lib64/libomp.so:$LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../../libxsmm/lib/
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
# export LIBXSMM_TARGET=clx

# write help message
if [ "$1" == "--help" ]; then
  echo "Usage: $0 <llm> <batch_size> <seq_len> <hyper> <BF16> <num_layer> <num_iter> <embedding_dim> <intermediate_dim> <gate_flag>"
  echo ""
  echo "llm: name of the model (default: llama-7b), options: llama-7b, llama4, llama-3.1-8B, llama-3.2-3B, llama-3.2-1B,"
  echo "gpt2, gpt3-13b, gpt3-175b, alphafold2-h32, alphafold2-h16, alphafold2-h8"
  echo "batch_size: (default: 64)"
  echo "seq_len: (default: 4096)"
  echo "hyper: (default: 0, no hyperthreading)"
  echo "BF16: (default: 1, use BF16)"
  echo "num_layer: (default: 3)"
  echo "num_iter: (default: 3)"
  echo "embedding_dim: (default: 4096)"
  echo "intermediate_dim: (default: 11008)"
  echo "gate_flag: (default: 1, gate)"
  exit 0
fi

llm=${1:-llama-7b}
batch_size=${2:-64}
seq_len=${3:-4096}
hyper=${4:-0}
BF16=${5:-0}
num_layer=${6:-3}
num_iter=${7:-3}
embedding_dim=${8:-4096}
intermediate_dim=${9:-11008}
gate_flag=${10:-1}


# echo "Running $llm model"
if [ "$llm" == "llama-7b" ]; then
  embedding_dim=4096; intermediate_dim=11008; gate_flag=1
elif [ "$llm" == "llama-3.1-8B" ]; then
  embedding_dim=4096; intermediate_dim=14336; gate_flag=1
elif [ "$llm" == "llama-70b" ]; then
  embedding_dim=8192; intermediate_dim=28672; gate_flag=1
elif [ "$llm" == "llama-3.2-3B" ]; then
  embedding_dim=3072; intermediate_dim=8192; gate_flag=1
elif [ "$llm" == "llama-3.2-1B" ]; then
  embedding_dim=2048; intermediate_dim=8192; gate_flag=1
elif [ "$llm" == "gpt2" ]; then
  embedding_dim=768; intermediate_dim=4*embedding_dim; gate_flag=0
elif [ "$llm" == "gpt3-13b" ]; then
  embedding_dim=5120; intermediate_dim=4*embedding_dim; gate_flag=0
elif [ "$llm" == "gpt3-175b" ]; then
  embedding_dim=12288; intermediate_dim=4*embedding_dim; gate_flag=0
else
  echo "Custom model: $llm, please manually set the parameters"
fi

# print all the parameters in one line
echo "llm: $llm, batch_size: $batch_size, seq_len: $seq_len, hyper: $hyper, BF16: $BF16, num_layer: $num_layer, num_iter: $num_iter, embedding_dim: $embedding_dim, intermediate_dim: $intermediate_dim, gate_flag: $gate_flag"


# echo "Compiling FFN benchmark code"
g++ -O2 FFN_bench.cpp init.cpp -o ffn.o -I ../ -I ../../../libxsmm/include/ -DLIBXSMM_DEFAULT_CONFIG -L ../../../libxsmm/lib/ -lxsmm -fopenmp -mavx512f

# get number of cpu from lscpu command
cpu_count=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')

if [ "$hyper" != "1" ]; then
    threads=$cpu_count
    echo "CPU count: $cpu_count, threads: $threads"
    KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=$threads numactl -m 0 -N 0 ./ffn.o $batch_size $seq_len $BF16 $num_layer $num_iter $embedding_dim $intermediate_dim $gate_flag 
else
    threads=$((cpu_count * 2))
    echo "CPU count: $cpu_count, threads: $threads"
    KMP_AFFINITY=granularity=fine,compact,0,0 OMP_NUM_THREADS=$threads numactl -m 0 -N 0 ./mha.o $batch_size $seq_len $BF16 $num_layer $num_iter $embedding_dim $intermediate_dim $gate_flag
fi