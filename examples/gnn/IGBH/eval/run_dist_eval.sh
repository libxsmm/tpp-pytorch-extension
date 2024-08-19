#!/bin/bash

path=$1
dataset='IGBH'
dataset_size='large'

export QINT8_BLOCK_SIZE=128

run_dist.sh -n $2 -ppn $3 \
  python -u -W ignore dist_eval_rgat.py \
  --path $path --batch_size $4 \
  --use_tpp --use_bf16 --use_qint8_gemm --use_int8 \
  --checkpoint $5 --val_fraction $6 --profile \
  --fan_out $7 --n_classes 2983 --dataset $dataset \
  --dataset_size $dataset_size
#run_dist.sh -n $2 -ppn $3 \
#  python -u -W ignore dist_eval_rgat.py \
#  --path $path --batch_size $4 \
#  --use_tpp --use_bf16 --use_int8 \
#  --checkpoint $5 --val_fraction $6 \
#  --fan_out $7 --n_classes 2983 --dataset $dataset \
#  --dataset_size $dataset_size
