
MDL_PTH=( "mrm8488/bert-tiny-5-finetuned-squadv2" bert-base-uncased bert-large-uncased )
MDL_NM=( tiny base large )
MDL_FMS=( 128 768 1024 )
# echo ${MDL_PTH[0]}
# echo ${MDL_PTH[1]}
# echo ${MDL_PTH[2]}
# exit
CPU=gvt3
LOG_DIR=vpmm_bf8out_lat
mkdir -p $LOG_DIR
#for NUM_THREADS in 16 8 4 ; do
#for MB in 1 2 4 ; do
for MB in 1 ; do
        for NUM_THREADS in 8 4 2 1; do
                for MDL_NUM in 0 1 2 ; do
                        #for MAX_SEQ_LEN in 16 32 48 64 128 256 ; do
                        for MAX_SEQ_LEN in 16 32 48 64 128 256 ; do
				for I8DOT in 0 1 ; do
				export LIBXSMM_AARCH64_USE_I8DOT=$I8DOT
				CPU=gvt3bfd$I8DOT
                                if [ $MAX_SEQ_LEN -gt 64 ] ; then SB=64 ; else SB=$MAX_SEQ_LEN ; fi
                                FMB=$(( ${MDL_FMS[$MDL_NUM]} * MAX_SEQ_LEN / SB / NUM_THREADS ))
                                if [ $FMB -lt 16 ] ; then FMB=16 ; fi
                                if [ $FMB -gt 64 ] ; then FMB=64 ; fi
                                if [ $MDL_NUM -eq 0 ] ; then CACHE_DIR=cache_1000 ; else CACHE_DIR=cache_200 ; fi
                                QUERY_LEN=$(( MAX_SEQ_LEN / 2 ))
                                OUT_FILE=${LOG_DIR}/vmm_${CPU}_mdl_${MDL_NM[$MDL_NUM]}_seq_${MAX_SEQ_LEN}_mb_${MB}_C_${NUM_THREADS}_fmb_${FMB}_tpp_bf8emu_pad.txt
                                echo ${OUT_FILE}
                                echo "NUM_THREADS=$NUM_THREADS  MB=$MB  MDL=${MDL_NM[$MDL_NUM]}  MAX_SEQ_LEN=$MAX_SEQ_LEN  FMB = $FMB I8DOT=$I8DOT" | tee $OUT_FILE
                                echo "CMDLINE: LIBXSMM_AARCH64_USE_I8DOT=$I8DOT OMP_NUM_THREADS=$NUM_THREADS model_path=${MDL_PTH[$MDL_NUM]} bash cmd_infer.sh --use_tpp --tpp_bf8 --per_gpu_eval_batch_size ${MB} --max_seq_length $MAX_SEQ_LEN --max_query_length ${QUERY_LEN} --opt_infer --features_block_size=$FMB --no_tqdm --data_dir=$CACHE_DIR $@" | tee -a $OUT_FILE
                                echo >> $OUT_FILE
                                LIBXSMM_AARCH64_USE_I8DOT=$I8DOT OMP_NUM_THREADS=$NUM_THREADS model_path=${MDL_PTH[$MDL_NUM]} bash cmd_infer.sh --use_tpp --tpp_bf8 --per_gpu_eval_batch_size ${MB} --max_seq_length $MAX_SEQ_LEN --max_query_length ${QUERY_LEN} --opt_infer --features_block_size=$FMB --no_tqdm --data_dir=$CACHE_DIR $@ &>> ${OUT_FILE}
                                tail ${OUT_FILE}
			        done
                        done
                done
        done
done

