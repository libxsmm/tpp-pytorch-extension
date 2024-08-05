#!/bin/bash

if [ "$#" -lt 3 ]; then
  echo "Usage: partition_graph.sh <path-to-data-store> <dataset_size> <num-partitions> [<data-type>]"
  exit
fi
dataset_size=$2
dataset="IGBH"
in_path=$1/$dataset
graph_path=$1/$dataset/$dataset_size
num_parts=$3
out_path=$in_path"/"$dataset_size"/"$num_parts
token="p"

echo "Partitioning graph into "$num_parts" parts"
echo "Input graph from "$graph_path
echo "Output partitions into "$out_path$token

python -u -W ignore partition_graph.py --path $in_path --dataset $dataset --dataset_size $dataset_size --num_parts $num_parts --output $out_path --graph_struct_only
pushd $out_path$token
ln -s ../struct.graph
popd

python -u -W ignore post_process_part.py --part_path $out_path --num_parts $num_parts

echo "Partitioning node features into "$num_parts" parts"

python -u -W ignore partition_graph.py --path $in_path --dataset $dataset --dataset_size $dataset_size --num_parts $num_parts --output $out_path --feat_part_only --data_type $4

