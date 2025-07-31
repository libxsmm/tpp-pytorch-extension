EXECPATH=.
PLATFORM=arlh
CORES=14
PCORES=6
PROMPT="\"Tell me about Alexander the Great in 200 words.\""
QBS=1600
HC=400
MODEL=""
HYBRID=0

THREADS=""
COREBIND=""
OMPCONFIG=""
LMQUANTSIZE=""
LMQUANT=""
INTCONFIG=""
WOQ=""

source ../../env.sh

for MODEL in "tiiuae/Falcon3-1B-Instruct" "facebook/MobileLLM-ParetoQ-1.5B-1.58-bit" "meta-llama/Llama-3.1-8B"; do
  for PREC in 0 1 2 3; do
    for HYBRID in 0 1; do
      MODEL_USE=$MODEL
      LMQUANTSIZE=""
      LMQUANT=""
      INTCONFIG=""
      WOQ=""
      PRECNAME=""

      if [ "$PREC" -eq 0 ]; then
        PRECNAME="bf16"
      fi
      if [ "$PREC" -eq 1 ]; then
        PRECNAME="mxfp4"
      fi
      if [ "$PREC" -eq 2 ]; then
        PRECNAME="int2"
      fi
      if [ "$PREC" -eq 3 ]; then
        PRECNAME="int1"
      fi  
    
      if [ "$HYBRID" -eq 1 ]; then
        THREADS=$CORES
        COREBIND="0-$(($CORES - 1))"
        OMPCONFIG="OMP_NUM_THREADS=$THREADS GEMM_LOOP_SCHEME_REUSE=\"aCB@schedule(dynamic,1)\" GEMM_LOOP_SCHEME_STREAMING=\"aCb@schedule(dynamic,1)\""
      else
        THREADS=$PCORES
        COREBIND="0-$(($PCORES - 1))"
        OMPCONFIG="OMP_NUM_THREADS=$THREADS GEMM_LOOP_SCHEME_REUSE=\"aCB\" GEMM_LOOP_SCHEME_STREAMING=\"aCb\""
      fi

      if [[ "$MODEL" == *"ParetoQ"* ]]; then
        QBS=1600
        HC=400
      else
        QBS=2048
        HC=512
      fi

      if [ "$PREC" -gt 0 ]; then
        LMQUANTSIZE="QINT8_BLOCK_SIZE=1024"
        LMQUANT="--quantize-lm-head"
      fi

      if [ "$PREC" -eq 1 ]; then
        WOQ="--weight-dtype mxfp4"
        INTCONFIG="USE_MXFP4_INT8=1 "
      fi

      if [ "$PREC" -eq 2 ]; then
        WOQ="--weight-dtype qint2"     
        INTCONFIG="QINT2_BLOCK_SIZE=$QBS MAX_HC_SIZE=$HC"
      fi

      if [ "$PREC" -eq 3 ]; then
        WOQ="--weight-dtype qint2"       
        INTCONFIG="TPP_QINT2_AS_QINT1=1 QINT2_BLOCK_SIZE=$QBS MAX_HC_SIZE=$HC"
        if [[ "$MODEL == *"ParetoQ* ]]; then
          search_string="1.58-bit" 
          replace_string="1-bit"    
          MODEL_USE=$(echo "$MODEL" | sed "s|$search_string|$replace_string|g")
        fi
      fi

      OUTNAME=${PLATFORM}_${THREADS}t_hybrid_${HYBRID}_tpp_$PRECNAME.out
      CMD="$LMQUANTSIZE $INTCONFIG $OMPCONFIG numactl -m 0 -C $COREBIND python -u $EXECPATH/run_generation.py --prompt $PROMPT --greedy --use-tpp --token --max-new-tokens 128 $WOQ $LMQUANT -m $MODEL_USE >> $OUTNAME"
      echo $CMD
      eval $CMD
    done
  done
done


