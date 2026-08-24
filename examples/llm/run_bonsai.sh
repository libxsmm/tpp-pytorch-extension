#!/bin/bash
###############################################################################
# Reproduce the Bonsai (ternary) results on CPU.
#
#   ./run_bonsai.sh <8b|27b> <int2|bf16> [extra run_generation.py args]
#
# Assumes the standard install:
#   bash utils/setup_conda.sh && source env.sh
#   python setup.py install
#   cd examples/llm && pip install -r requirements.txt
#
# Models are pulled from the HF cache / hub by id; override with BONSAI_MODEL.
###############################################################################

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${HERE}/../.." && pwd)

if [ ! -f "${ROOT}/env.sh" ]; then
  echo "error: ${ROOT}/env.sh not found; run 'bash utils/setup_conda.sh' first" >&2
  exit 1
fi
# conda's activate/deactivate hooks are not nounset-clean, so source before set -u
# shellcheck disable=SC1091
source "${ROOT}/env.sh"
set -euo pipefail

SIZE=${1:-}; PREC=${2:-int2}
[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift

case "${SIZE}" in
  8b)  MODEL_ID=prism-ml/Ternary-Bonsai-8B-unpacked;  FUSED_ARGS="" ;;
  # no fused C++ block for Qwen3.5's gated-DeltaNet, so quantize the Linears only
  27b) MODEL_ID=prism-ml/Ternary-Bonsai-27B-unpacked; FUSED_ARGS="--tpp-quant-linear-only" ;;
  *)   echo "usage: $0 <8b|27b> <int2|bf16> [args...]" >&2; exit 1 ;;
esac

case "${PREC}" in
  int2) WEIGHT_DTYPE=qint2 ;;
  bf16) WEIGHT_DTYPE=bfloat16 ;;
  *)    echo "usage: $0 <8b|27b> <int2|bf16> [args...]" >&2; exit 1 ;;
esac

MODEL=${BONSAI_MODEL:-${MODEL_ID}}
CORES=${CORES:-$(lscpu | awk '/^Core\(s\) per socket:/ {print $NF}')}
LAST=$((CORES - 1))

# Bonsai is natively ternary at group size 128; larger blocks are lossy.
export QINT2_BLOCK_SIZE=${QINT2_BLOCK_SIZE:-128}
export QINT8_BLOCK_SIZE=${QINT8_BLOCK_SIZE:-512}
export MAX_HC_SIZE=${MAX_HC_SIZE:-64}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${CORES}}
# decode is a GEMV over BS=1, so the streaming scheme is the one that is used
export GEMM_LOOP_SCHEME_REUSE=${GEMM_LOOP_SCHEME_REUSE:-"aCB@schedule(guided)"}
export GEMM_LOOP_SCHEME_STREAMING=${GEMM_LOOP_SCHEME_STREAMING:-"aCb@schedule(guided)"}

set -x
numactl -m 0 -C "0-${LAST}" python -u "${HERE}/run_generation.py" \
  --greedy --use-tpp ${FUSED_ARGS} \
  --prompt "Tell me aboout photosythesis in 200 words" \
  --dtype bfloat16 --weight-dtype "${WEIGHT_DTYPE}" --quantize-lm-head \
  --token-latency --max-new-tokens 256 --num-iter 1 --num-warmup 0 \
  -m "${MODEL}" "$@"
