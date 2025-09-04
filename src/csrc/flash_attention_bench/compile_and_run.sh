if [[ $(uname -m) == "riscv64" ]] 
then
  echo "Running on RISC-V architecture"
else
  source /swtools/intel/2025.1/oneapi-vars.sh --force > /dev/null
fi
export LD_PRELOAD=$HOME/jemalloc/lib/libjemalloc.so:$LD_PRELOAD
export LIBXSMM_PATH=../../../libxsmm
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBXSMM_PATH/lib/
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

# write help message
if [ "$1" == "--help" ]; then
  echo "Usage: $0 <llm> <batch_size> <seq_len> <hyper> <BF16> <b_vnni> <num_layer> <num_iter> <nheads> <head_size> <bias_flag> <nbbias_flag> <gate_flag> <self_attention_flag>"
  echo ""
  echo "llm: name of the model (default: llama-7b), options: llama-7b, llama4, llama-3.1-8B, llama-3.2-3B, llama-3.2-1B,"
  echo "gpt2, gpt3-13b, gpt3-175b, alphafold2-h32, alphafold2-h16, alphafold2-h8"
  echo "batch_size: (default: 64)"
  echo "seq_len: (default: 4096)"
  echo "hyper: (default: 0, no hyperthreading)"
  echo "BF16: (default: 1, use BF16)"
  echo "blocked_layout: (default: 2, 0=flat_layout, 1=blocked_layout, 2=blocked_weight_layout)"
  echo "b_vnni: (default: 1, b is in vnni format)"
  echo "num_layer: (default: 3)"
  echo "num_iter: (default: 3)"
  echo "nheads: (default: 32)"
  echo "head_size: (default: 128)"
  echo "bias_flag: (default: 0, no bias)"
  echo "nbbias_flag: (default: 0, no nbias)"
  echo "gate_flag: (default: 0, no gate)"
  echo "correctness_check: (default: 0, no check)"
  exit 0
fi

if [[ $(uname -m) == "riscv64" ]] 
then
  llm=${1:-gpt2}
  batch_size=${2:-64}
  seq_len=${3:-256}
  hyper=${4:-0}
  BF16=${5:-0}
  blocked_layout=${6:-2}
  b_vnni=${7:-1}
  num_layer=${8:-3}
  num_iter=${9:-3}
  nheads=${10-32}
  head_size=${11:-128}
  bias_flag=${12:-0}
  nbbias_flag=${13:-0}
  gate_flag=${14:-0}
  self_attention_flag=${15:-1}
  correctness_check=${16:-0}
else
  llm=${1:-llama-7b}
  batch_size=${2:-64}
  seq_len=${3:-4096}
  hyper=${4:-0}
  BF16=${5:-1}
  blocked_layout=${6:-2}
  b_vnni=${7:-1}
  num_layer=${8:-3}
  num_iter=${9:-3}
  nheads=${10-32}
  head_size=${11:-128}
  bias_flag=${12:-0}
  nbbias_flag=${13:-0}
  gate_flag=${14:-0}
  self_attention_flag=${15:-1}
  correctness_check=${16:-0}
fi

# echo "Running $llm model"
if [ "$llm" == "llama-7b" -o "$llm" == "llama-3.1-8B" ]; then
  embed_size=4096; nheads=32; head_size=128; bias_flag=0; nbbias_flag=0; gate_flag=0 self_attention_flag=1
elif [ "$llm" == "llama-70b" ]; then
  embed_size=8192; nheads=64; head_size=128; bias_flag=0; nbbias_flag=0; gate_flag=0 self_attention_flag=1
elif [ "$llm" == "llama-3.2-3B" ]; then
  embed_size=3072; nheads=24; head_size=128; bias_flag=0; nbbias_flag=0; gate_flag=0 self_attention_flag=1
elif [ "$llm" == "llama-3.2-1B" ]; then
  embed_size=2048; nheads=32; head_size=64; bias_flag=0; nbbias_flag=0; gate_flag=0 self_attention_flag=1
elif [ "$llm" == "gpt2" ]; then
  embed_size=768; nheads=12; head_size=64; bias_flag=0; nbbias_flag=0; gate_flag=0 self_attention_flag=1
elif [ "$llm" == "gpt3-13b" -o "$llm" == "llama4" ]; then
  embed_size=5120; nheads=40; head_size=128; bias_flag=0; nbbias_flag=0; gate_flag=0 self_attention_flag=1
elif [ "$llm" == "gpt3-175b" ]; then
  embed_size=12288; nheads=96; head_size=128; bias_flag=0; nbbias_flag=0; gate_flag=0 self_attention_flag=1
elif [ "$llm" == "alphafold2-h32" ]; then
  embed_size=256; nheads=8; head_size=32; bias_flag=1; nbbias_flag=1; gate_flag=1 self_attention_flag=0
elif [ "$llm" == "alphafold2-h16" ]; then
  embed_size=64; nheads=4; head_size=16; bias_flag=1; nbbias_flag=1; gate_flag=1 self_attention_flag=0
elif [ "$llm" == "alphafold2-h8" ]; then
  embed_size=64; nheads=8; head_size=8; bias_flag=1; nbbias_flag=1; gate_flag=1 self_attention_flag=0
else
  echo "Custom model: $llm, please manually set the parameters"
fi

# print all the parameters in one line
echo "llm: $llm, batch_size: $batch_size, seq_len: $seq_len, hyper: $hyper, BF16: $BF16, blocked_layout: $blocked_layout, b_vnni: $b_vnni, num_layer: $num_layer, num_iter: $num_iter, nheads: $nheads, head_size: $head_size, bias_flag: $bias_flag, nbbias_flag: $nbbias_flag, gate_flag: $gate_flag, self_attention_flag: $self_attention_flag, correctness_check: $correctness_check"


echo "Compiling MHA"
if [[ $(uname -m) == "riscv64" ]] 
then
  # Get number of cpu from lscpu command
  cpu_count=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')

  threads=$cpu_count
  echo "CPU count: $cpu_count, threads: $threads"

  g++ -O2 MHA_attention_bench.cpp init.cpp -o mha.o -I ../ -I $LIBXSMM_PATH/include/ -DLIBXSMM_DEFAULT_CONFIG -L $LIBXSMM_PATH/lib/ -lxsmm -fopenmp
  KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=$threads taskset -c 0-$threads ./mha.o $batch_size $seq_len $nheads $head_size $bias_flag $nbbias_flag $gate_flag $BF16 $blocked_layout $b_vnni $num_layer $num_iter $self_attention_flag $correctness_check
else
  g++ -O2 MHA_attention_bench.cpp init.cpp -o mha.o -I ../ -I $LIBXSMM_PATH/include/ -DLIBXSMM_DEFAULT_CONFIG -L $LIBXSMM_PATH/lib/ -lxsmm -fopenmp -mavx512f

  # Get number of cpu from lscpu command
  cpu_count=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')

  # <batch_size> <seq_len> <num_heads> <head_size> <bias_flag> <nbbias_flag> <gate_flag> <BF16> <blocked_layout> <b_vnni> <num_layer> <num_iter> <self_attention_flag>
  if [ "$hyper" != "1" ]; then
      threads=$cpu_count
      echo "CPU count: $cpu_count, threads: $threads"
      KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=$threads numactl -m 0 -N 0 ./mha.o $batch_size $seq_len $nheads $head_size $bias_flag $nbbias_flag $gate_flag $BF16 $blocked_layout $b_vnni $num_layer $num_iter $self_attention_flag $correctness_check
  else
      threads=$((cpu_count * 2))
      echo "CPU count: $cpu_count, threads: $threads"
      KMP_AFFINITY=granularity=fine,compact,0,0 OMP_NUM_THREADS=$threads numactl -m 0 -N 0 ./mha.o $batch_size $seq_len $nheads $head_size $bias_flag $nbbias_flag $gate_flag $BF16 $blocked_layout $b_vnni $num_layer $num_iter $self_attention_flag $correctness_check
  fi
fi
