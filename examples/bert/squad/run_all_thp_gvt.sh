
MDL_PTH=( "mrm8488/bert-tiny-5-finetuned-squadv2" bert-base-uncased bert-large-uncased )
MDL_NM=( tiny base large )
MDL_FMS=( 128 768 1024 )
# echo ${MDL_PTH[0]}
# echo ${MDL_PTH[1]}
# echo ${MDL_PTH[2]}
# exit

CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | awk '{print $NF}'`
NUM_THREADS=$CORES_PER_SOCKET
MB=$(( 2* NUM_THREADS ))
CACHE_DIR=cache_1000
CPU=gvt3
LOG_DIR=vpmm_out_thp
mkdir -p $LOG_DIR
for MDL_NUM in 0 1 2 ; do
        for MAX_SEQ_LEN in 16 32 48 64 128 256 ; do
		for BFDOT in 0 1 ; do
		export LIBXSMM_AARCH64_USE_BFDOT=$BFDOT
		CPU=gvt3bfd$BFDOT
                if [ $MAX_SEQ_LEN -gt 64 ] ; then SB=64 ; else SB=$MAX_SEQ_LEN ; fi
                FMB=64
                QUERY_LEN=$(( MAX_SEQ_LEN / 2 ))
                OUT_FILE=${LOG_DIR}/vmm_${CPU}_mdl_${MDL_NM[$MDL_NUM]}_seq_${MAX_SEQ_LEN}_mb_${MB}_C_${NUM_THREADS}_fmb_${FMB}_tpp_bf16_pad.txt
                echo ${OUT_FILE}
                echo "NUM_THREADS=$NUM_THREADS  MB=$MB  MDL=${MDL_NM[$MDL_NUM]}  MAX_SEQ_LEN=$MAX_SEQ_LEN  FMB = $FMB  BFDOT=$BFDOT" | tee $OUT_FILE
                echo "CMDLINE: OMP_NUM_THREADS=$NUM_THREADS model_path=${MDL_PTH[$MDL_NUM]} bash cmd_infer.sh --use_tpp --tpp_bf16 --per_gpu_eval_batch_size ${MB} --max_seq_length $MAX_SEQ_LEN --max_query_length ${QUERY_LEN} --opt_infer --features_block_size=$FMB --no_tqdm --data_dir=$CACHE_DIR $@" | tee -a $OUT_FILE
                echo >> $OUT_FILE
                OMP_NUM_THREADS=$NUM_THREADS model_path=${MDL_PTH[$MDL_NUM]} bash cmd_infer.sh --use_tpp --tpp_bf16 --per_gpu_eval_batch_size ${MB} --max_seq_length $MAX_SEQ_LEN --max_query_length ${QUERY_LEN} --opt_infer --features_block_size=$FMB --no_tqdm --data_dir=$CACHE_DIR $@ &>> ${OUT_FILE}
                tail ${OUT_FILE}
        	done
        done
done

