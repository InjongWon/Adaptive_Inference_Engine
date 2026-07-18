#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
for workload in smoke throughput long_context; do
  python -m benchmark.load_generator \
    --workload "$workload" \
    --output "results/${workload}.json"
done
python -m benchmark.plot_results results/smoke.json results/throughput.json results/long_context.json
