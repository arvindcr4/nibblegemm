"""Measurement harness.

Microbenchmarking a memory-bound kernel is easy to get wrong in a way that
flatters the result, so the choices here are deliberate.

**The L2 trap.** An A100 has 40 MB of L2. A 4096x4096 INT4 weight matrix is
8 MB. Time the same buffer in a loop and after the first iteration the weights
are resident in L2, so the kernel is served at L2 bandwidth (several TB/s)
rather than HBM bandwidth (~1.5 TB/s). The kernel looks 3x faster than it is,
and -- worse -- the INT4 kernel benefits more than the fp16 baseline it is
being compared against, because 8 MB fits in L2 and 32 MB does not. That single
mistake can manufacture most of a speedup out of nothing.

The fix used here is **buffer rotation**: allocate enough distinct copies of the
weights to overflow L2 and walk them round-robin, so each timed repetition
reads memory that has been evicted. This is also what actually happens during
inference -- a forward pass touches every layer once and never revisits a
weight -- so the rotating measurement is the realistic one, not merely the
conservative one. `flush_l2` is offered as an alternative for cases where
rotation is impractical.

**Reported statistics.** Colab GPUs are shared and clock behaviour drifts, so a
single mean is not trustworthy. Every measurement reports median with 5th/95th
percentiles across trials, and SM clocks are sampled before and after so a
throttled run is visible rather than silently averaged in.

**Peak bandwidth.** Percentages of peak are computed against a *measured*
achievable bandwidth from a large copy on this machine, not the 1555 GB/s from
the spec sheet. No real kernel reaches the spec number, so quoting it inflates
nothing and understates everything.
"""
from __future__ import annotations

import statistics
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# Device facts
# ---------------------------------------------------------------------------
def l2_bytes() -> int:
    return torch.cuda.get_device_properties(0).L2_cache_size


def sm_clock_mhz() -> Optional[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def device_summary() -> str:
    p = torch.cuda.get_device_properties(0)
    return (f"{p.name} | sm_{p.major}{p.minor} | {p.multi_processor_count} SMs | "
            f"L2 {p.L2_cache_size / 2**20:.0f} MiB | {p.total_memory / 2**30:.0f} GiB HBM")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@dataclass
class Timing:
    median_ms: float
    p05_ms: float
    p95_ms: float
    trials: int
    reps: int
    clock_before: Optional[int] = None
    clock_after: Optional[int] = None
    samples: Sequence[float] = field(default_factory=list, repr=False)

    @property
    def spread_pct(self) -> float:
        """p95-p05 as a percentage of the median. Large values mean untrustworthy."""
        return 100.0 * (self.p95_ms - self.p05_ms) / self.median_ms if self.median_ms else 0.0

    @property
    def throttled(self) -> bool:
        if self.clock_before is None or self.clock_after is None:
            return False
        return abs(self.clock_after - self.clock_before) > 0.05 * self.clock_before


_flush_buf: Optional[torch.Tensor] = None


def flush_l2() -> None:
    """Evict L2 by writing a buffer twice its size."""
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(2 * l2_bytes(), dtype=torch.uint8, device="cuda")
    _flush_buf.zero_()


def bench(fn: Callable[[int], object], *, reps: int = 32, trials: int = 30,
          warmup: int = 10, flush: bool = False) -> Timing:
    """Time ``fn(rep_index)``; the callee is expected to rotate its own buffers.

    One event pair spans ``reps`` back-to-back launches so that per-launch event
    overhead (a few microseconds, non-trivial against a 30 us kernel) is
    amortised rather than counted.
    """
    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize()

    clock_before = sm_clock_mhz()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []

    for _ in range(trials):
        if flush:
            flush_l2()
        torch.cuda.synchronize()
        start.record()
        for r in range(reps):
            fn(r)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / reps)

    samples.sort()
    return Timing(
        median_ms=statistics.median(samples),
        p05_ms=samples[max(0, int(0.05 * len(samples)) - 1)],
        p95_ms=samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        trials=trials, reps=reps,
        clock_before=clock_before, clock_after=sm_clock_mhz(),
        samples=samples,
    )


def bench_graph(fn: Callable[[int], object], *, reps: int = 32, trials: int = 30,
                warmup: int = 10) -> Timing:
    """Time ``reps`` calls captured into a CUDA graph and replayed.

    Same work as :func:`bench`, minus per-launch CPU overhead. Comparing the two
    isolates how much of a short kernel's wall-clock is the launch rather than
    the kernel: a split-K decode kernel issues two launches and can finish in
    under 10 us, at which point launch cost stops being a rounding error. If the
    graph number is materially better, the kernel is latency bound, not
    bandwidth bound, and further memory tuning is wasted effort.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(warmup):
            fn(i)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for r in range(reps):
            fn(r)

    clock_before = sm_clock_mhz()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / reps)

    samples.sort()
    return Timing(
        median_ms=statistics.median(samples),
        p05_ms=samples[max(0, int(0.05 * len(samples)) - 1)],
        p95_ms=samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        trials=trials, reps=reps,
        clock_before=clock_before, clock_after=sm_clock_mhz(),
        samples=samples,
    )


def measured_peak_read_gbps(size_mb: int = 512) -> float:
    """Achievable HBM bandwidth on this machine, via a large device-to-device copy.

    A copy moves ``2 * size`` bytes (one read, one write). This lands well under
    the spec sheet figure, which is the point: it is the number a real streaming
    kernel can actually be held to.
    """
    n = size_mb * 2**20 // 2
    src = torch.empty(n, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    t = bench(lambda _: dst.copy_(src), reps=8, trials=15, warmup=5)
    return (2 * src.numel() * 2) / (t.median_ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# Rotating buffers
# ---------------------------------------------------------------------------
class Rotation:
    """N copies of a tensor, sized so that cycling through them overflows L2."""

    def __init__(self, proto: torch.Tensor, min_total_bytes: Optional[int] = None,
                 cap: int = 32):
        if min_total_bytes is None:
            min_total_bytes = int(1.5 * l2_bytes())
        nbytes = proto.numel() * proto.element_size()
        count = max(1, min(cap, -(-min_total_bytes // max(nbytes, 1))))
        self.buffers = [proto] + [proto.clone() for _ in range(count - 1)]
        self.count = len(self.buffers)
        self.total_bytes = self.count * nbytes

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.buffers[i % self.count]

    def defeats_l2(self) -> bool:
        return self.total_bytes > l2_bytes()


class Cycle:
    """Modulo-indexed view over a list, so objects derived from a Rotation
    (a QuantizedWeight wrapping each buffer, say) rotate in step with it."""

    def __init__(self, items):
        self.items = list(items)

    def __getitem__(self, i: int):
        return self.items[i % len(self.items)]

    def __len__(self) -> int:
        return len(self.items)


# ---------------------------------------------------------------------------
# Traffic model
# ---------------------------------------------------------------------------
def weight_bytes(K: int, N: int, group_size: int, bits: int) -> int:
    w = K * N * bits // 8
    s = (K // group_size) * N * 2 if bits < 16 else 0
    return w + s


def total_bytes(M: int, K: int, N: int, group_size: int, bits: int) -> int:
    """Compulsory traffic: weights once, activations in, outputs out."""
    return weight_bytes(K, N, group_size, bits) + M * K * 2 + M * N * 2


def achieved_gbps(M: int, K: int, N: int, group_size: int, bits: int, ms: float) -> float:
    return total_bytes(M, K, N, group_size, bits) / (ms * 1e-3) / 1e9


def tflops(M: int, K: int, N: int, ms: float) -> float:
    return (2.0 * M * K * N) / (ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def markdown_table(rows: Sequence[dict], columns: Sequence[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def write_csv(rows: Sequence[dict], path: str, columns: Sequence[str]) -> None:
    import csv, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
