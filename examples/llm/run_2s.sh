SPARSE=0
if [ "x$1" != "x" ] ; then
  SPARSE=$1
fi

NC=56

OMP_NUM_THREADS=$NC SPARSE_KERNELS=${SPARSE} numactl -m 0 -C 0-$((NC-1)) python -u run_generation.py --use-tpp --num-iter 10 --num-warmup 3 --input-tokens 1024 --max-new-token 128 --token  -m 'llama_13b_sparse' & 
OMP_NUM_THREADS=$NC SPARSE_KERNELS=${SPARSE} numactl -m 1 -C 56-$((56+NC-1)) python -u run_generation.py --use-tpp --num-iter 10 --num-warmup 3 --input-tokens 1024 --max-new-token 128 --token  -m 'llama_13b_sparse'
