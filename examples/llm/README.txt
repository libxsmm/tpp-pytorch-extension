
Prerequsite:
-----------

Install and activate conda environment as described in README file in root folder.

Install LLM (GPT-J) task specific requirements (one time):
$pip install -r requirements.txt

To run reference code (HuggingFace Transformers+PyTorch code) GPT-J (default) on single socket simply run:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python -u run_generation.py --device cpu --dtype bfloat16 --max-new-tokens 32

For TPP Optimized code:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python -u run_generation.py --device cpu --dtype bfloat16 --max-new-tokens 32 --use-tpp

To use BFloat8 or HFloat8 weight in TPP Optimized code:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python -u run_generation.py --device cpu --dtype bfloat16 --max-new-tokens 32 --use-tpp --weight-dtype bfloat8

To enable autograd profiling add "--profile" to above command line.

To enable torchscript JIT tracing add "--jit" to above command line (effective only when using --use-tpp).

To enable printing 1st and 2nd token latencies add "--token-latency" to above command line (effective only when using --use-tpp).

Distributed Tensor Parallel implementation on multi-socket (or in SNC4 mode) requires Intel MPI and either MPI enabled PyTorch or Intel torch-ccl installed.
Dual socket run 2 NUMA domains:
run_dist_ht.sh -np 2 -ppn 2 python -u run_generation.py --device cpu --dtype bfloat16 --max-new-tokens 32 --use-tpp

Dual socket run on Xeon Max (Sapphire Rapids with HBM) in Quad mode with flat HBM:
run_dist_ht.sh -np 2 -ppn 2 bash numawrap.sh 2 python -u run_generation.py --device cpu --dtype bfloat16 --max-new-tokens 32 --use-tpp

Single socket run with SNC4 enabled:
run_dist_ht.sh -np 4 -ppn 4 -sso python -u run_generation.py --device cpu --dtype bfloat16 --max-new-tokens 32 --use-tpp

Dual socket run with SNC4 enabled:
run_dist_ht.sh -np 8 -ppn 8 python -u run_generation.py --device cpu --dtype bfloat16 --max-new-tokens 32 --use-tpp

Dual socket run from HBM memory with SNC4 enabled:
This runs out of memory while loading model. So, we first load model in DDR memory, shard it and save sharded model on disk. Then load sharded model when running from HBM.
# One time model sharding from DDR 
run_dist_ht.sh -np 8 -ppn 8 python -u run_generation.py --device cpu --dtype bfloat16 --max-new-tokens 32 --use-tpp --save-sharded-model

# Run from HBM with sharded model
run_dist_ht.sh -np 8 -ppn 8 bash numawrap.sh 8 python -u run_generation.py --device cpu --dtype bfloat16 --max-new-tokens 32 --use-tpp --load-sharded-model



# First token benchmark
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python -u run_first_token.py --input-tokens 1024  --use-tpp
