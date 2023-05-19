
Prerequsite:
-----------

Install and activate conda environment as described in README file in root folder.

Install GPT-J task specific requirements (one time):
$pip install -r requirements.txt

To run reference code (HuggingFace Transformers+PyTorch code) GPT-J on single socket simply run:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python -u run_gptj.py --device cpu --dtype bfloat16 --max-new-tokens 32

For TPP Optimized code:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python -u run_gptj.py --device cpu --dtype bfloat16 --max-new-tokens 32 --use_tpp

To enable autograd profiling add "--profile" to above command line.
