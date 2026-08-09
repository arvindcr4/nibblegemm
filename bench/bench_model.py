"""End-to-end: does the kernel actually make a model faster, and is it still the same model?

A microbenchmark can be fast and useless. Two things have to hold before a
quantised kernel is worth shipping, and they pull in opposite directions:

* **Speed at realistic scale.** A single 4096x14336 projection is 31 us, but a
  decode step is ~224 of them and the win only survives if per-layer overhead
  does not eat it. `layers` mode builds a Llama-3-8B-shaped stack of decoder
  projections and times a full decode step. Random weights are fine here --
  throughput does not depend on what the numbers are -- and it avoids a 16 GB
  download to measure something that is purely a memory-traffic property.

* **Quality.** INT4 that halves latency and ruins the model is worthless.
  `model` mode quantises a real checkpoint and reports perplexity next to the
  fp16 baseline, which is the only measurement that says the kernel produces a
  usable model rather than merely a numerically plausible tensor.

The split is deliberate: speed is measured where scale matters and quality where
real weights matter, instead of compromising on one model that does neither well.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

import nibblegemm as ng  # noqa: E402

# Llama-3-8B decoder geometry. GQA means k/v are narrow (1024, not 4096).
LLAMA3_8B = dict(hidden=4096, kv=1024, intermediate=14336, layers=32)


# ---------------------------------------------------------------------------
# layers mode: a decode step at 8B scale
# ---------------------------------------------------------------------------
def build_stack(cfg, quantized, group_size, device="cuda"):
    """Materialise one layer's worth of projections.

    Every layer of a decoder has identical shapes, so one set of weights is
    replayed `layers` times rather than building 32 distinct copies, which would
    cost 14 GB in fp16.

    That reuse would be a measurement bug if the weights fitted in cache, so:
    one layer is 218M parameters, which is 436 MB in fp16 and 109 MB packed to
    INT4. Both are far larger than the A100's 40 MB L2, so by the time the loop
    returns to a given projection it has long been evicted and the access
    pattern is genuine streaming -- the same property bench/harness.py achieves
    by rotating buffers.
    """
    h, kv, inter = cfg["hidden"], cfg["kv"], cfg["intermediate"]
    # q, k, v, o, gate, up, down
    shapes = [(h, h), (h, kv), (h, kv), (h, h), (h, inter), (h, inter), (inter, h)]
    mods = []
    for k, n in shapes:
        W = torch.randn(k, n, dtype=torch.float32) * 0.02
        if quantized:
            mods.append(("q", ng.quantize(W, group_size=group_size).to(device), k, n))
        else:
            mods.append(("f", W.to(torch.float16).to(device), k, n))
    return mods


def decode_step(mods, device="cuda"):
    """One batch-1 decode step through every projection of every layer.

    Activations are allocated once, outside the timed region: at ~30 us per
    projection, per-call allocator work would be a visible share of the result.
    """
    acts = {k: torch.randn(1, k, dtype=torch.float16, device=device)
            for _, _, k, _ in mods}

    def run(layers):
        for _ in range(layers):
            for kind, W, k, _ in mods:
                if kind == "q":
                    ng.matmul(acts[k], W)
                else:
                    torch.mm(acts[k], W)
    return run


def layers_mode(args):
    cfg = LLAMA3_8B
    layers = args.layers
    print(f"Llama-3-8B geometry: hidden={cfg['hidden']} kv={cfg['kv']} "
          f"intermediate={cfg['intermediate']} x {layers} layers")

    rows = []
    for label, quant in (("fp16 (cuBLAS)", False), ("nibblegemm INT4", True)):
        mods = build_stack(cfg, quant, args.group_size)
        bytes_per_step = sum(
            (W.qweight.numel() * 4 + W.scales.numel() * 2) if kind == "q" else W.numel() * 2
            for kind, W, _, _ in mods
        ) * layers
        run = decode_step(mods)

        t = H.bench(lambda _: run(layers), reps=1, trials=15, warmup=3)
        tg = H.bench_graph(lambda _: run(layers), reps=1, trials=15, warmup=3)
        rows.append({
            "impl": label,
            "weights GiB": round(bytes_per_step / 2**30, 2),
            "step ms": round(t.median_ms, 3),
            "graph ms": round(tg.median_ms, 3),
            "tok/s": round(1000.0 / t.median_ms, 1),
            "graph tok/s": round(1000.0 / tg.median_ms, 1),
            "GB/s": round(bytes_per_step / (tg.median_ms * 1e-3) / 1e9, 1),
            "_ms": t.median_ms, "_g": tg.median_ms,
        })
        del mods
        torch.cuda.empty_cache()

    cols = ["impl", "weights GiB", "step ms", "graph ms", "tok/s", "graph tok/s", "GB/s"]
    print(H.markdown_table(rows, cols))
    f, q = rows[0], rows[1]
    print(f"\nspeedup: {f['_ms']/q['_ms']:.2f}x eager, {f['_g']/q['_g']:.2f}x under CUDA graph")
    print(f"weight footprint: {f['weights GiB']:.2f} -> {q['weights GiB']:.2f} GiB "
          f"({f['weights GiB']/q['weights GiB']:.2f}x smaller)")
    H.write_csv(rows, "docs/results/model_layers.csv", cols)


# ---------------------------------------------------------------------------
# model mode: real checkpoint, perplexity and generation
# ---------------------------------------------------------------------------
def swap_linears(model, group_size, clip_search=True):
    """Replace decoder nn.Linear layers with QuantLinear.

    lm_head and embeddings are left in fp16, which is standard practice: the
    output projection is disproportionately sensitive to quantisation error and
    is a single layer, so quantising it costs accuracy for almost no memory.
    """
    swapped, skipped = 0, []
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if "lm_head" in full:
                continue
            K, N = child.in_features, child.out_features
            if K % group_size or N % 4 or K % 32:
                skipped.append((full, K, N))
                continue
            qw = ng.quantize(child.weight.data.t().contiguous().float(), group_size,
                             clip_search=clip_search)
            bias = child.bias.data.clone() if child.bias is not None else None
            setattr(module, child_name,
                    ng.QuantLinear(qw.to(child.weight.device), bias))
            swapped += 1
    return swapped, skipped


@torch.no_grad()
def perplexity(model, ids, window, device="cuda"):
    """Standard sliding-window perplexity over a fixed token stream."""
    nll, count = 0.0, 0
    for start in range(0, ids.size(1) - 1, window):
        chunk = ids[:, start:start + window + 1].to(device)
        if chunk.size(1) < 2:
            break
        out = model(chunk[:, :-1]).logits.float()
        loss = torch.nn.functional.cross_entropy(
            out.reshape(-1, out.size(-1)), chunk[:, 1:].reshape(-1), reduction="sum")
        nll += loss.item()
        count += chunk.size(1) - 1
    return float(torch.exp(torch.tensor(nll / count)))


@torch.no_grad()
def decode_tokens_per_s(model, ids, n_new, device="cuda"):
    """Manual greedy decode with KV cache, timing only the forward passes."""
    out = model(ids.to(device), use_cache=True)
    past, tok = out.past_key_values, out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_new):
        out = model(tok, past_key_values=past, use_cache=True)
        past, tok = out.past_key_values, out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    return n_new / (time.perf_counter() - t0)


def model_mode(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    text, source = load_eval_text(args.tokens * 6)
    if source is None:
        print("\nREFUSING to report perplexity: no real evaluation text could be\n"
              "obtained, and repeated filler gives ~1.0 perplexity for any model,\n"
              "which would hide arbitrarily bad quantisation. Re-run with network\n"
              "access, or pass --skip-perplexity to measure throughput only.",
              file=sys.stderr)
        if not args.skip_perplexity:
            return 1
    ids = tok(text, return_tensors="pt").input_ids[:, : args.tokens]
    print(f"eval stream: {ids.size(1)} tokens from {source or 'synthetic filler'}")

    # Calibration comes from a slice after the evaluation window. Calibrating on
    # the text perplexity is then measured on would leak, and GPTQ would look
    # better than it is.
    all_ids = tok(text, return_tensors="pt").input_ids
    calib_start = args.tokens
    need = args.calib_seqs * args.calib_seqlen
    calib_pool = all_ids[:, calib_start:calib_start + need]
    calib = [calib_pool[:, i:i + args.calib_seqlen]
             for i in range(0, calib_pool.size(1) - args.calib_seqlen + 1, args.calib_seqlen)]
    print(f"calibration: {len(calib)} x {args.calib_seqlen} tokens, "
          f"disjoint from the eval window")

    # Every variant is evaluated so the table separates the kernel's error
    # (nil -- it is bit-exact) from the quantiser's.
    variants = [
        ("fp16", None),
        ("INT4 max-abs RTN", "rtn"),
        ("INT4 + clip search", "clip"),
        ("INT4 + GPTQ", "gptq"),
    ]

    rows = []
    for label, kind in variants:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16).to("cuda").eval()
        note = ""
        if kind == "gptq":
            if not calib:
                print("  [no calibration text available, skipping GPTQ]", file=sys.stderr)
                del model
                continue
            n, skipped = ng.quantize_model_gptq(
                model, calib, group_size=args.group_size, log=lambda _m: None)
            note = f"{n} layers quantised"
            if skipped:
                note += f", {len(skipped)} skipped (shape)"
            torch.cuda.empty_cache()
        elif kind is not None:
            n, skipped = swap_linears(model, args.group_size, clip_search=(kind == "clip"))
            note = f"{n} layers quantised"
            if skipped:
                note += f", {len(skipped)} skipped (shape)"
            torch.cuda.empty_cache()
        mem = sum(p.numel() * p.element_size() for p in model.parameters()) \
            + sum(b.numel() * b.element_size() for b in model.buffers())
        ppl = None if args.skip_perplexity else perplexity(model, ids, args.window)
        tps = decode_tokens_per_s(model, ids[:, :64], args.new_tokens)
        rows.append({"impl": label, "weights GiB": round(mem / 2**30, 2),
                     "perplexity": "n/a" if ppl is None else round(ppl, 3),
                     "decode tok/s": round(tps, 1), "notes": note})
        print(f"  {label}: ppl {ppl if ppl is None else round(ppl, 3)}, "
              f"{tps:.1f} tok/s, {mem/2**30:.2f} GiB {note}")
        del model
        torch.cuda.empty_cache()

    cols = ["impl", "weights GiB", "perplexity", "decode tok/s", "notes"]
    print()
    print(H.markdown_table(rows, cols))
    base = rows[0]
    if not args.skip_perplexity:
        print(f"\nperplexity vs fp16 ({source}):")
        for r in rows[1:]:
            print(f"  {r['impl']:22s} {r['perplexity']:.3f}  "
                  f"({100*(r['perplexity']/base['perplexity'] - 1):+.2f}%)")
    print(f"weights: {base['weights GiB']:.2f} -> {rows[-1]['weights GiB']:.2f} GiB")
    print(f"decode: {base['decode tok/s']:.1f} -> {rows[-1]['decode tok/s']:.1f} tok/s "
          f"({rows[-1]['decode tok/s']/base['decode tok/s']:.2f}x)")
    H.write_csv(rows, "docs/results/model_quality.csv", cols)
    return 0


def load_eval_text(min_chars: int):
    """Return (text, source). `source` is None when only synthetic filler exists.

    Perplexity is only ever compared against the fp16 baseline on the *same*
    text, so the absolute value matters less than the delta -- but the text must
    still be real language, or both sides collapse toward 1.0 and the comparison
    stops being able to detect damage.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
        if len(text) >= min_chars:
            return text, "wikitext-2-raw test"
    except Exception as exc:
        print(f"  [wikitext unavailable: {type(exc).__name__}]", file=sys.stderr)

    import urllib.request
    FALLBACK_TEXT_URLS = [
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/11/11-0.txt",           # Alice in Wonderland
    ]
    for url in FALLBACK_TEXT_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as fh:
                text = fh.read().decode("utf-8", errors="ignore")
            if len(text) >= min_chars:
                return text, url.rsplit("/", 1)[-1]
        except Exception as exc:
            print(f"  [{url} unavailable: {type(exc).__name__}]", file=sys.stderr)

    seed = ("The quantisation of neural network weights to four bits reduces memory "
            "bandwidth by roughly a factor of four, which is the dominant cost of "
            "autoregressive decoding at small batch sizes. ")
    return seed * (min_chars // len(seed) + 1), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["layers", "model"])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--calib-seqs", type=int, default=32)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--skip-perplexity", action="store_true",
                    help="throughput only; use when no real evaluation text is reachable")
    args = ap.parse_args()

    print(H.device_summary())
    ng.extension()
    print()
    return (layers_mode if args.mode == "layers" else model_mode)(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
