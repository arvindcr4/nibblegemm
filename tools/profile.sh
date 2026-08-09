#!/usr/bin/env bash
# Capture the ncu metrics quoted in docs/OPTIMIZATION_LOG.md.
#
# The metric set is deliberately small. A full `--set full` capture takes minutes
# per kernel and buries the two numbers that actually explain this kernel's
# behaviour: registers per thread (which caps blocks per SM, hence occupancy) and
# the ratio of total instructions to tensor instructions (which reveals whether a
# tensor-core kernel is doing math or bookkeeping).
#
# Writes docs/results/ncu.txt.
set -euo pipefail

METRICS="dram__bytes.sum.per_second,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
sm__inst_executed_pipe_tensor.sum,\
smsp__inst_executed.sum"

OUT=docs/results/ncu.txt
mkdir -p docs/results

# Build the extension before profiling so nvcc does not appear on the timeline.
python scripts/profile_target.py --target gemm -M 4096 >/dev/null 2>&1

{
  echo "# ncu metrics -- see docs/OPTIMIZATION_LOG.md"
  echo "# $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo
  echo "## prefill kernel, M=4096, K=N=4096"
  ncu --target-processes all --kernel-name regex:gemm_v6 --launch-count 1 \
      --metrics "$METRICS" python scripts/profile_target.py --target gemm -M 4096 2>&1 \
    | grep -E "Metric Name|^\s+(dram__|sm__|launch__|smsp__)"
  echo
  echo "## decode kernel, M=1, K=N=4096"
  ncu --target-processes all --kernel-name regex:gemv_v4 --launch-count 1 \
      --metrics "$METRICS" python scripts/profile_target.py --target gemv -M 1 2>&1 \
    | grep -E "Metric Name|^\s+(dram__|sm__|launch__|smsp__)"
} | tee "$OUT"

echo
echo "wrote $OUT"
