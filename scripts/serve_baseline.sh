#!/usr/bin/env bash
set -euo pipefail
source configs/baseline.env

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key local-token \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
