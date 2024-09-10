#!/bin/bash

if [ "$#" -lt 3 ]; then
  echo "Usage: cvt.sh <path-to-data-store>/<dataset> <dataset-size> target-data-type"
  exit
fi

#Example command-line:
#bash cvt.sh <path-to-data-store>/IGBH medium int8 128

echo "path = "$1
echo "dataset size = "$2
echo "tgt dtype = "$3
echo "block-size = "$4
echo "ntype = "$5

export OMP_NUM_THREADS=64
python -u -W ignore convert_features.py --path $1 --dataset_size $2 --target_dtype $3 --block_size $4 --ntype $5

