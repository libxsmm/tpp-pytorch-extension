
MDL_PTH=( "mrm8488/bert-tiny-5-finetuned-squadv2" bert-base-uncased bert-large-uncased )
MDL_NM=( tiny base large )
MDL_FMS=( 128 768 1024 )
# echo ${MDL_PTH[0]}
# echo ${MDL_PTH[1]}
# echo ${MDL_PTH[2]}
# exit
#
NUM_EXAMPLES=1000
if [ "x$1" != "x" ] ; then
NUM_EXAMPLES=$1
fi
CACHE_DIR=cache_$NUM_EXAMPLES
mkdir -p $CACHE_DIR

( cd $CACHE_DIR && ln -s ../SQUAD1 . )

CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | awk '{print $NF}'`
NUM_THREADS=$CORES_PER_SOCKET
MB=$(( 2* NUM_THREADS ))
CPU=spr
LOG_DIR=vpmm_out_thp
mkdir -p $LOG_DIR
for MDL_NUM in 0 1 2 ; do
        for MAX_SEQ_LEN in 16 32 48 64 128 256 ; do
                FMB=64
                QUERY_LEN=$(( MAX_SEQ_LEN / 2 ))
                echo "NUM_THREADS=$NUM_THREADS  MB=$MB  MDL=${MDL_NM[$MDL_NUM]}  MAX_SEQ_LEN=$MAX_SEQ_LEN  FMB = $FMB" 
                OMP_NUM_THREADS=$NUM_THREADS model_path=${MDL_PTH[$MDL_NUM]} bash cmd_infer.sh --use_tpp --tpp_bf16 --per_gpu_eval_batch_size ${MB} --max_seq_length $MAX_SEQ_LEN --max_query_length ${QUERY_LEN} --opt_infer --features_block_size=$FMB --no_tqdm --data_dir=$CACHE_DIR --num_examples=$NUM_EXAMPLES
        done
done

