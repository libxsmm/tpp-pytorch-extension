source /swtools/intel/2025.2.0/setvars.sh --force > /dev/null 2>&1
export LD_PRELOAD=/data/swtools/intel/2025.2.0/2025.2/lib/libiomp5.so:$LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../../libxsmm/lib/

# export LD_PRELOAD=$TBBROOT/lib/libtbbmalloc_proxy.so:$LD_PRELOAD
# export TBB_MALLOC_USE_HUGE_PAGES=1

# export LD_PRELOAD=$HOME/jemalloc/lib/libjemalloc.so:$LD_PRELOAD
# export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

export LD_PRELOAD=$HOME/lib/lib/libtcmalloc.so.4:/usr/lib64/libtbbmalloc.so.2:$LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/nfs_home/nchaudh1/lib/lib
# export PROF=1
# export USE_TBB=1

# write help message
if [ "$1" == "--help" ]; then
  echo "Usage: $0 <llm> <batch_size> <seq_len> <hyper> <BF16> <b_vnni> <blocked> <num_layer> <num_iter> <embedding_dim> <intermediate_dim> <gate_flag>"
  echo ""
  echo "llm: name of the model (default: deepseek-r1), options: llama-7b, llama4, llama-3.1-8B, llama-3.2-3B, llama-3.2-1B,"
  echo "gpt2, gpt3-13b, gpt3-175b, alphafold2-h32, alphafold2-h16, alphafold2-h8"
  echo "batch_size: (default: 64)"
  echo "seq_len: (default: 4096)"
  echo "hyper: (default: 0, no hyperthreading)"
  echo "BF16: (default: 1, use BF16)"
  echo "b_vnni: (default: 0, not b_vnni)"
  echo "blocked: (default: 0, not blocked)"
  echo "num_layer: (default: 3)"
  echo "num_iter: (default: 3)"
  echo "embedding_dim: (default: 4096)"
  echo "intermediate_dim: (default: 11008)"
  echo "num_experts: (default: 256)"
  echo "num_experts_per_token: (default: 8)"
  echo "gate_flag: (default: 1, gate)"
  echo "correctness_check: (default: 1, check correctness)"
  exit 0
fi

llm=${1:-deepseek-r1}
batch_size=${2:-64}
seq_len=${3:-4096}
hyper=${4:-0}
BF16=${5:-0}
b_vnni=${6:-0}
blocked=${7:-0}
num_layer=${8:-1}
num_iter=${9:-1}
embedding_dim=${10:-7168}
intermediate_dim=${11:-2048}
num_expert=${12:-256}
num_experts_per_token=${13:-8}
gate_flag=${14:-1}
correctness_check=${15:-0}


# echo "Running $llm model"
if [ "$llm" == "deepseek-r1" ]; then
  embedding_dim=7168; intermediate_dim=2048; gate_flag=1 num_expert=256 num_experts_per_token=8
elif [ "$llm" == "qwen3-235B-A22B" ]; then
  embedding_dim=4096; intermediate_dim=1536; gate_flag=1 num_expert=128 num_experts_per_token=8
elif [ "$llm" == "qwen3-30B-A3B" ]; then
  embedding_dim=2048; intermediate_dim=768; gate_flag=1 num_expert=128 num_experts_per_token=8
elif [ "$llm" == "llama-7b" ]; then
  embedding_dim=4096; intermediate_dim=11008; gate_flag=1 num_expert=1 num_experts_per_token=1
elif [ "$llm" == "llama-3.1-8B" ]; then
  embedding_dim=4096; intermediate_dim=14336; gate_flag=1 num_expert=1 num_experts_per_token=1
elif [ "$llm" == "llama-70b" ]; then
  embedding_dim=8192; intermediate_dim=28672; gate_flag=1 num_expert=1 num_experts_per_token=1
elif [ "$llm" == "llama-3.2-3B" ]; then
  embedding_dim=3072; intermediate_dim=8192; gate_flag=1 num_expert=1 num_experts_per_token=1
elif [ "$llm" == "llama-3.2-1B" ]; then
  embedding_dim=2048; intermediate_dim=8192; gate_flag=1 num_expert=1 num_experts_per_token=1
elif [ "$llm" == "gpt2" ]; then
  embedding_dim=768; intermediate_dim=3072; gate_flag=0 num_expert=1 num_experts_per_token=1
elif [ "$llm" == "gpt3-13b" ]; then
  embedding_dim=5120; intermediate_dim=20480; gate_flag=0 num_expert=1 num_experts_per_token=1
elif [ "$llm" == "gpt3-175b" ]; then
  embedding_dim=12288; intermediate_dim=49152; gate_flag=0 num_expert=1 num_experts_per_token=1
else
  echo "Custom model: $llm, please manually set the parameters"
fi

# print all the parameters in one line
echo "llm: $llm, batch_size: $batch_size, seq_len: $seq_len, hyper: $hyper, BF16: $BF16, b_vnni: $b_vnni, blocked: $blocked, num_layer: $num_layer, num_iter: $num_iter, embedding_dim: $embedding_dim, intermediate_dim: $intermediate_dim, num_expert: $num_expert, num_experts_per_token: $num_experts_per_token  gate_flag: $gate_flag", "correctness_check: $correctness_check"

# get number of cpu from lscpu command
cpu_count=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')
if [ "$USE_PYTHON" == "1" ]; then
  threads=$((cpu_count))
  if [ "$BF16" == 1 ]; then
    dtype="bfloat16"
  else
    dtype="float32"
  fi
  # export SGLANG_CPU_OMP_THREADS_BIND="64-127"
  # export SGLANG_USE_CPU_ENGINE=1
  echo "Running with Python, CPU count: $cpu_count, threads: $threads"
  # python -m torch.backends.xeon.run_cpu --ninstances 1 --ncores-per-instance $threads --node-id 1 \
  KMP_BLOCKTIME=1 KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=$threads numactl -m 1 -N 1 \
        python py_FFN_bench.py --batch_size $batch_size --seq_len $seq_len \
          --hidden_size $embedding_dim --intermediate_size $intermediate_dim  \
          --num_layer $num_layer --num_iter $num_iter --dtype $dtype
  exit 0
fi

# echo "Compiling FFN benchmark code"
if [ "$PROF" == "1" ]; then
  echo "Profiling enabled"
  FLAGS="-O0 -pg"
  [ "$BF16" = "0" ] && FLAGS="$FLAGS -DUSE_FLOAT"
  [ "$BF16" = "0" ] && echo "Compiling with FP32" || echo "Compiling with BF16"
  [ "$USE_TBB" = "1" ] && FLAGS="$FLAGS -DUSE_TBB -ltbb" && echo "Compiling with TBB" || echo "Compiling with OpenMP"
  g++ $FLAGS FFN_bench.cpp init.cpp -o ffn.o -I ../ -I ../../../libxsmm/include/ -DLIBXSMM_DEFAULT_CONFIG -L ../../../libxsmm/lib/ -lxsmm -fopenmp -mavx512f -liomp5

else
  echo "Normal compilation"
  FLAGS="-O2"
  # FLAGS="$FLAGS -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free"
  [ "$BF16" = "0" ] && FLAGS="$FLAGS -DUSE_FLOAT"
  [ "$BF16" = "0" ] && echo "Compiling with FP32" || echo "Compiling with BF16"
  [ "$USE_TBB" = "1" ] && FLAGS="$FLAGS -DUSE_TBB -ltbb" && echo "Compiling with TBB" || echo "Compiling with OpenMP"
  g++ $FLAGS FFN_bench.cpp init.cpp -o ffn.o -I ../ -I ../../../libxsmm/include/ -DLIBXSMM_DEFAULT_CONFIG -L ../../../libxsmm/lib/ -lxsmm -fopenmp -mavx512f -liomp5
fi


PERF_CMD="perf stat -e cycles,"
PERF_CMD+="cpu/event=0xc2,umask=0x2,name=UOPS_RETIRED.RETIRE_SLOTS/,"
PERF_CMD+="cpu/event=0xe5,umask=0x03,name=MEM_UOP_RETIRED.ANY/,"
PERF_CMD+="cpu/event=0xd0,umask=0x81,name=MEM_INST_RETIRED.ALL_LOADS/,"
PERF_CMD+="cpu/event=0xd1,umask=0x01,name=MEM_LOAD_RETIRED.L1_HIT/,"
PERF_CMD+="cpu/event=0xd1,umask=0x08,name=MEM_LOAD_RETIRED.L1_MISS/,"
PERF_CMD+="cpu/event=0x47,umask=0x02,cmask=0x2,name=MEMORY_ACTIVITY.CYCLES_L1D_MISS/,"
PERF_CMD+="cpu/event=0x47,umask=0x03,cmask=0x03,name=MEMORY_ACTIVITY.STALLS_L1D_MISS/,"
PERF_CMD+="cpu/event=0xd1,umask=0x02,name=MEM_LOAD_RETIRED.L2_HIT/,"
PERF_CMD+="cpu/event=0xd1,umask=0x10,name=MEM_LOAD_RETIRED.L2_MISS/,"
PERF_CMD+="cpu/event=0x47,umask=0x05,cmask=0x05,name=MEMORY_ACTIVITY.STALLS_L2_MISS/,"
PERF_CMD+="cpu/event=0xd1,umask=0x04,name=MEM_LOAD_RETIRED.L3_HIT/,"
PERF_CMD+="cpu/event=0xd1,umask=0x20,name=MEM_LOAD_RETIRED.L3_MISS/,"
PERF_CMD+="cpu/event=0x47,umask=0x09,cmask=0x09,name=MEMORY_ACTIVITY.STALLS_L3_MISS/,"

PERF_CMD+="cpu/event=0x27,umask=0x20,name=CORE_SNOOP_RESPONSE.I_FWD_FE/,"
PERF_CMD+="cpu/event=0x27,umask=0x10,name=CORE_SNOOP_RESPONSE.I_FWD_M/,"
PERF_CMD+="cpu/event=0x27,umask=0x02,name=CORE_SNOOP_RESPONSE.I_HIT_FSE/,"
PERF_CMD+="cpu/event=0x27,umask=0x01,name=CORE_SNOOP_RESPONSE.MISS/,"
PERF_CMD+="cpu/event=0x27,umask=0x40,name=CORE_SNOOP_RESPONSE.S_FWD_FE/,"
PERF_CMD+="cpu/event=0x27,umask=0x08,name=CORE_SNOOP_RESPONSE.S_FWD_M/,"
PERF_CMD+="cpu/event=0x27,umask=0x04,name=CORE_SNOOP_RESPONSE.S_HIT_FSE/ "

#PERF_CMD="vtune -collect hpc-performance \
#      -knob collect-memory-bandwidth=true \
#      -result-dir ./omp_analysis "
# PERF_CMD="vtune -collect threading -knob sampling-and-waits=hw -knob enable-stack-collection=true -result-dir ./thread_results"
# vtune -report summary -result-dir ./omp_analysis -group-by omp-region
PERF_CMD=""
if [ "$hyper" != "1" ]; then
    threads=64
    echo "CPU count: $cpu_count, threads: $threads"
    if [ "$USE_TBB" == "1" ]; then
      OMP_NUM_THREADS=$threads numactl -m 1 -N 1 $PERF_CMD ./ffn.o $batch_size $seq_len $b_vnni $blocked $num_layer $num_iter $embedding_dim $intermediate_dim $num_expert $num_experts_per_token $gate_flag $correctness_check
    else
      KMP_BLOCKTIME=1 KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=$threads numactl -m 1 -N 1 $PERF_CMD ./ffn.o $batch_size $seq_len $b_vnni $blocked $num_layer $num_iter $embedding_dim $intermediate_dim $num_expert $num_experts_per_token $gate_flag $correctness_check
    fi
else
    threads=$((cpu_count * 2))
    echo "CPU count: $cpu_count, threads: $threads"
    if [ "$USE_TBB" == "1" ]; then
      OMP_NUM_THREADS=$threads numactl -m 1 -N 1 ./ffn.o $batch_size $seq_len $b_vnni $blocked $num_layer $num_iter $embedding_dim $intermediate_dim $num_expert $num_experts_per_token $gate_flag $correctness_check
    else
      KMP_BLOCKTIME=1 KMP_AFFINITY=granularity=fine,compact,0,0 OMP_NUM_THREADS=$threads numactl -m 1 -N 1 ./ffn.o $batch_size $seq_len $b_vnni $blocked $num_layer $num_iter $embedding_dim $intermediate_dim $num_expert $num_experts_per_token $gate_flag $correctness_check
    fi
fi

if [ "$PROF" == "1" ]; then
  gprof ffn.o gmon.out | gprof2dot -s -w | dot -Gdpi=200 -Tpng -o output.png
  echo "Profiled output saved to output.png"
fi