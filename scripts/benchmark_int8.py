"""
Benchmark: torchao INT8 weight-only quantization on HeartMuLa backbone.

Measures:
  - Latency per generate_frame call (baseline vs INT8)
  - Peak memory usage
  - Whether MPS ops fall back to CPU (via op profiling)

Usage:
    .venv/bin/python scripts/benchmark_int8.py --model_path ./ckpt --version 3B --n_frames 20
"""

import argparse
import time
import gc
import torch
import statistics


def str2device(v):
    return torch.device(v.lower())


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--version", type=str, default="3B")
    p.add_argument("--device", type=str2device, default=None,
                   help="Device to run on (default: mps if available, else cpu)")
    p.add_argument("--dtype", type=str, default="float16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--n_frames", type=int, default=20,
                   help="Number of generate_frame calls to benchmark")
    p.add_argument("--warmup", type=int, default=3,
                   help="Warmup calls before timing")
    return p.parse_args()


def _default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _str_to_dtype(s):
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[s]


def get_mps_mem_gb():
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**3
    return None


def get_cuda_mem_gb(device):
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024**3
    return None


def clear_cache(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def load_model(model_path, version, device, dtype):
    import os
    from heartlib.heartmula.modeling_heartmula import HeartMuLa
    mula_path = os.path.join(model_path, f"HeartMuLa-oss-{version}")
    print(f"  Loading HeartMuLa from {mula_path} ...")
    model = HeartMuLa.from_pretrained(mula_path, device_map=device, dtype=dtype)
    model.eval()
    return model


def make_synthetic_inputs(model, device, dtype, prompt_len=80, batch_size=1):
    """Build fake prompt tensors matching the shapes generate_frame expects."""
    parallel_n = 9  # 8 audio codebooks + 1 text
    tokens = torch.zeros([batch_size, prompt_len, parallel_n], dtype=torch.long, device=device)
    tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
    tokens_mask[:, :, -1] = True
    pos = torch.arange(prompt_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    muq_embed = torch.zeros([batch_size, 512], dtype=dtype, device=device)
    starts = [0] * batch_size
    return tokens, tokens_mask, pos, muq_embed, starts


def run_benchmark(model, device, dtype, n_frames, warmup, label):
    """Run generate_frame n_frames times and return per-call latencies in ms."""
    model.setup_caches(1)
    tokens, tokens_mask, pos, muq_embed, starts = make_synthetic_inputs(model, device, dtype)

    print(f"\n  [{label}] Warming up ({warmup} calls)...")
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=dtype):
            for _ in range(warmup):
                curr = model.generate_frame(
                    tokens=tokens, tokens_mask=tokens_mask, input_pos=pos,
                    temperature=1.0, topk=50, cfg_scale=1.0,
                    continuous_segments=muq_embed, starts=starts,
                )

    # Build padded token for subsequent steps
    def pad(token):
        padded = torch.ones((1, 9), device=device, dtype=torch.long)
        padded[:, :-1] = token
        padded = padded.unsqueeze(1)
        mask = torch.ones_like(padded, dtype=torch.bool)
        mask[..., -1] = False
        return padded, mask

    print(f"  [{label}] Timing {n_frames} frames...")
    latencies = []
    curr = curr  # last warmup output

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=dtype):
            for i in range(n_frames):
                curr_t, curr_m = pad(curr)

                # Synchronise before timing
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elif device.type == "mps":
                    torch.mps.synchronize()

                t0 = time.perf_counter()
                curr = model.generate_frame(
                    tokens=curr_t, tokens_mask=curr_m,
                    input_pos=pos[..., -1:] + i + 1,
                    temperature=1.0, topk=50, cfg_scale=1.0,
                    continuous_segments=None, starts=None,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elif device.type == "mps":
                    torch.mps.synchronize()

                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

    return latencies


def print_stats(latencies, label):
    mean = statistics.mean(latencies)
    med = statistics.median(latencies)
    p90 = sorted(latencies)[int(len(latencies) * 0.9)]
    print(f"\n  [{label}]")
    print(f"    Mean latency : {mean:.1f} ms/frame")
    print(f"    Median       : {med:.1f} ms/frame")
    print(f"    P90          : {p90:.1f} ms/frame")
    print(f"    RTF (×)      : {mean / 80:.2f}×  (80 ms = 1 frame @ 12.5 Hz)")
    return mean


def count_linear_layers(model):
    return sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))


def estimate_model_size_gb(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / 1024**3


def main():
    args = parse_args()
    device = args.device or _default_device()
    dtype = _str_to_dtype(args.dtype)

    print("=" * 60)
    print(f"HeartMuLa INT8 Quantization Benchmark")
    print(f"  Device : {device}")
    print(f"  Dtype  : {dtype}")
    print(f"  Frames : {args.n_frames} (+ {args.warmup} warmup)")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Baseline: FP16
    # ----------------------------------------------------------------
    print("\n[1/3] Loading baseline model (no quantization)...")
    model_baseline = load_model(args.model_path, args.version, device, dtype)
    n_linears = count_linear_layers(model_baseline)
    size_gb = estimate_model_size_gb(model_baseline)
    print(f"  Linear layers : {n_linears}")
    print(f"  Model size    : {size_gb:.2f} GB")

    mem_before = get_mps_mem_gb() or get_cuda_mem_gb(device) or 0
    lat_baseline = run_benchmark(model_baseline, device, dtype,
                                  args.n_frames, args.warmup, "Baseline FP16")
    mean_baseline = print_stats(lat_baseline, "Baseline FP16")

    del model_baseline
    clear_cache(device)

    # ----------------------------------------------------------------
    # INT8 weight-only via torchao
    # ----------------------------------------------------------------
    print("\n[2/3] Loading INT8 quantized model (torchao int8_weight_only)...")
    model_int8 = load_model(args.model_path, args.version, device, dtype)

    try:
        from torchao.quantization import quantize_, int8_weight_only
        print("  Applying torchao int8_weight_only quantization...")
        t_quant_start = time.perf_counter()
        quantize_(model_int8, int8_weight_only())
        t_quant_end = time.perf_counter()
        print(f"  Quantization applied in {t_quant_end - t_quant_start:.1f} s")

        size_int8_gb = estimate_model_size_gb(model_int8)
        print(f"  Model size after INT8 : {size_int8_gb:.2f} GB")

        # Detect CPU fallback: run one frame and check if any op went to CPU
        print("\n[2b/3] Checking for CPU fallbacks via torch profiler...")
        model_int8.setup_caches(1)
        tokens, tokens_mask, pos, muq_embed, starts = make_synthetic_inputs(
            model_int8, device, dtype
        )
        cpu_ops = []
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                *(
                    [torch.profiler.ProfilerActivity.CUDA]
                    if device.type == "cuda" else []
                ),
            ],
            record_shapes=False,
        ) as prof:
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=dtype):
                    _ = model_int8.generate_frame(
                        tokens=tokens, tokens_mask=tokens_mask, input_pos=pos,
                        temperature=1.0, topk=50, cfg_scale=1.0,
                        continuous_segments=muq_embed, starts=starts,
                    )

        # Look for matmul/mm ops that executed on CPU
        for evt in prof.key_averages():
            name_lower = evt.key.lower()
            if any(k in name_lower for k in ["mm", "matmul", "linear", "addmm"]):
                if evt.cpu_time_total > 0 and device.type != "cpu":
                    cpu_ops.append((evt.key, evt.cpu_time_total / 1000))

        if cpu_ops and device.type != "cpu":
            print(f"\n  ⚠️  DETECTED {len(cpu_ops)} matmul-related ops with CPU time on {device}:")
            for name, cpu_ms in sorted(cpu_ops, key=lambda x: -x[1])[:10]:
                print(f"    {name:50s}  cpu_time={cpu_ms:.1f} ms")
            print("\n  This suggests INT8 dequantization is falling back to CPU.")
            cpu_fallback_detected = True
        else:
            print(f"  ✅ No obvious CPU fallback detected for matmul ops.")
            cpu_fallback_detected = False

        lat_int8 = run_benchmark(model_int8, device, dtype,
                                   args.n_frames, args.warmup, "INT8 torchao")
        mean_int8 = print_stats(lat_int8, "INT8 torchao")

    except Exception as e:
        print(f"  ❌ INT8 quantization failed: {e}")
        mean_int8 = None
        cpu_fallback_detected = None

    del model_int8
    clear_cache(device)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Device               : {device}")
    print(f"  Baseline FP16 mean   : {mean_baseline:.1f} ms/frame")
    if mean_int8 is not None:
        speedup = mean_baseline / mean_int8
        print(f"  INT8 mean            : {mean_int8:.1f} ms/frame")
        print(f"  Speedup              : {speedup:.2f}×")
        print(f"  CPU fallback detected: {cpu_fallback_detected}")
        print()
        if speedup > 1.05:
            print("  ✅ INT8 quantization is BENEFICIAL on this device.")
        elif speedup < 0.95:
            print("  ❌ INT8 quantization is SLOWER — likely CPU fallback for dequantization.")
            print("     Recommendation: skip INT8 on MPS, explore MLX backend instead.")
        else:
            print("  ≈  INT8 quantization has NEUTRAL impact (within noise).")
    else:
        print("  INT8 quantization    : FAILED (see error above)")

    print()
    # Extrapolate to full 30s generation
    frames_30s = 375
    if mean_int8 is not None:
        baseline_30s = mean_baseline * frames_30s / 1000
        int8_30s = mean_int8 * frames_30s / 1000
        print(f"  Estimated time for 30s audio ({frames_30s} frames):")
        print(f"    Baseline : {baseline_30s:.0f} s  ({baseline_30s/30:.1f}× RTF)")
        print(f"    INT8     : {int8_30s:.0f} s  ({int8_30s/30:.1f}× RTF)")
    print("=" * 60)


if __name__ == "__main__":
    main()
