#!/usr/bin/env bash
# Capture the ncu metrics quoted in docs/OPTIMIZATION_LOG.md.
#
# The metric set is deliberately small. A full `--set full` capture takes minutes
# per kernel and buries the numbers that actually explain this kernel. Two groups
# matter:
#
#   * registers per thread and achieved occupancy -- registers cap blocks per SM,
#     which caps occupancy, which caps latency hiding;
#   * the per-pipe instruction breakdown -- on NVIDIA GPUs integer multiply-add
#     issues on the FMA pipe and integer division on XU, so the pipe mix
#     distinguishes "doing math" from "computing addresses". That distinction is
#     what located the runtime integer division in the weight-staging loop.
#
# Writes docs/results/ncu.txt.
set -euo pipefail

CORE="launch__registers_per_thread,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum.per_second"

PIPES="sm__inst_executed_pipe_tensor.sum,\
smsp__inst_executed.sum,\
smsp__inst_executed_pipe_alu.sum,\
smsp__inst_executed_pipe_fma.sum,\
smsp__inst_executed_pipe_lsu.sum,\
smsp__inst_executed_pipe_xu.sum"

OUT=docs/results/ncu.txt
mkdir -p docs/results
FILTER='Metric Name|^\s+(dram__|sm__|launch__|smsp__)'

# Build the extension before profiling so nvcc does not appear on the timeline.
python scripts/profile_target.py --target gemm -M 4096 >/dev/null 2>&1

{
  echo "# ncu metrics -- see docs/OPTIMIZATION_LOG.md"
  echo "# $(nvidia-smi --query-gpu=name --format=csv,noheader), CUDA $(nvcc --version | grep -oE 'release [0-9.]+' | cut -d' ' -f2)"
  echo "# Regenerate with: bash tools/profile.sh"
  echo
  for v in 6 7; do
    echo "## prefill v$v, M=4096, K=N=4096, group_size=128"
    ncu --target-processes all --kernel-name "regex:gemm_v$v" --launch-count 1 \
        --metrics "$CORE,$PIPES" \
        python scripts/profile_target.py --target gemm -M 4096 --gemm-version "$v" 2>&1 \
      | grep -E "$FILTER"
    echo
  done
  echo "## decode v4, M=1, K=N=4096, group_size=128"
  ncu --target-processes all --kernel-name regex:gemv_v4 --launch-count 1 \
      --metrics "$CORE,$PIPES" python scripts/profile_target.py --target gemv -M 1 2>&1 \
    | grep -E "$FILTER"
} | tee "$OUT"

echo
echo "wrote $OUT"
