#!/usr/bin/env bash
set -euo pipefail
source configs/tensor_parallel.env

[[ "$MODEL" == REPLACE_* ]] && { echo "Set MODEL in configs/tensor_parallel.env"; exit 1; }

vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 --api-key local-token \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
