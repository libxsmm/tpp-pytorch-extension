#!/bin/bash
export OMP_NUM_THREADS=40 
echo "batch-size "$1
echo "model-name "$2
echo "fan-out "$3
echo "validation-fraction "$4
echo "data-type "$5
echo "dataset-size "$6
echo "path "$7

#Example bash command line:
#bash run_eval.sh 1 ./model_full.pt 45,30,15 0.025 int8 full
#python -u -W ignore eval_hetero.py \
#          --batch_size 1 --use_tpp \
#          --use_bf16 --use_qint8_gemm \
#          --checkpoint ./model_full.pt \
#          --fan_out 45,30,15 --validation_frac 0.025 \
#          --data_type int8 --dataset_size full

export QINT8_BLOCK_SIZE=128
python -u -W ignore eval_hetero.py --batch_size $1 --use_tpp --use_bf16 --use_qint8_gemm --checkpoint $2 --fan_out $3 --validation_frac $4 --data_type $5 --dataset_size $6 --path $7