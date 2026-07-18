#!/usr/bin/env bash
set -euo pipefail
source configs/speculative.env

# vLLM's speculative configuration evolves. Verify the syntax against the installed version:
#   vllm serve --help | grep -i speculative
SPEC_CONFIG=$(printf '{"method":"draft_model","model":"%s","num_speculative_tokens":%s}' "$DRAFT_MODEL" "$NUM_SPECULATIVE_TOKENS")

vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 --api-key local-token \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --speculative-config "$SPEC_CONFIG"
