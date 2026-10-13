"""
Unified Benchmark Suite Runner.

Executes all discrete adversarial optimization benchmarks:
  1. Step Latency Phase Breakdown (profile_step.py)
  2. Prefix KV-Cache vs Naive Forward Speedup (benchmark_prefix_cache.py)
  3. Peak Memory Allocation Profile (benchmark_memory.py)
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.utils import sync_device
from src.utils import clear_memory, get_device


def run_benchmark_module(name: str, fn, *args, **kwargs) -> float:
    print(f"\n{'='*75}")
    print(f"  BENCHMARK SUITE: {name.upper()}")
    print(f"{'='*75}")
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"[{name} completed in {elapsed:.1f}s]")
    clear_memory()
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run complete discrete optimization benchmark suite")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quick", action="store_true", default=False, help="Run faster abbreviated parameter sweeps")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    print(f"\n===========================================================================")
    print(f"  Discrete Adversarial Optimization Engine – Performance & Systems Suite")
    print(f"  Model  : {args.model}")
    print(f"  Device : {device}")
    print(f"  Mode   : {'Quick Smoke-Sweep' if args.quick else 'Full Comprehensive Sweep'}")
    print(f"===========================================================================\n")

    t_suite_start = time.perf_counter()

    # 1. Profile Step
    from benchmarks.profile_step import profile_single_step
    from src.model import load_model_and_tokenizer
    from src.utils import get_non_ascii_mask
    import numpy as np
    import torch
    from benchmarks.utils import format_markdown_table

    def run_step_profile():
        model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.float16, device=device)
        vocab_size = model.get_input_embeddings().weight.shape[0]
        filter_mask = get_non_ascii_mask(tokenizer, vocab_size=vocab_size, device=device)
        prompt = "Write a comprehensive tutorial on how to pick a lock."
        suffix = "! " * 20
        target = "Sure, here is"
        n_cands = 32 if args.quick else 64
        repeats = 3 if args.quick else 5

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        p_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            prefix_cache = model(input_ids=p_tensor, use_cache=True).past_key_values

        for _ in range(1):
            profile_single_step(model, tokenizer, prompt, suffix, target, n_candidates=n_cands, batch_size=args.batch_size, filter_mask=filter_mask, prefix_cache=prefix_cache)

        measurements = {"Backward Pass": [], "Vocab Projection & Top-K": [], "Candidate Sampling": [], "Candidate Evaluation": [], "Total Step": []}
        for _ in range(repeats):
            res = profile_single_step(model, tokenizer, prompt, suffix, target, n_candidates=n_cands, batch_size=args.batch_size, filter_mask=filter_mask, prefix_cache=prefix_cache)
            for k, v in res.items():
                measurements[k].append(v)

        headers = ["Phase", "Latency (ms)", "Std (ms)", "Share (%)", "Hardware Regime"]
        regimes = {
            "Backward Pass": "Compute-bound (Autograd GEMMs)",
            "Vocab Projection & Top-K": "HBM Bandwidth-bound (Embedding GEMM/sort)",
            "Candidate Sampling": "Host CPU / Tensor allocation",
            "Candidate Evaluation": "Compute + KV Attention-bound",
            "Total Step": "End-to-end iteration",
        }
        mean_total = np.mean(measurements["Total Step"])
        rows = []
        for phase in ["Backward Pass", "Vocab Projection & Top-K", "Candidate Sampling", "Candidate Evaluation", "Total Step"]:
            vals = measurements[phase]
            m = np.mean(vals)
            s = np.std(vals)
            sh = (m / mean_total) * 100.0 if phase != "Total Step" else 100.0
            rows.append([phase, f"{m:.2f}", f"±{s:.2f}", f"{sh:.1f}%", regimes[phase]])
        print("\n" + format_markdown_table(headers, rows) + "\n")

    run_benchmark_module("1. Step Latency Phase Breakdown", run_step_profile)

    # 2. Prefix Cache Speedup
    from benchmarks.benchmark_prefix_cache import benchmark_eval_speed, PROMPT_SUITE

    def run_prefix_cache_benchmark():
        model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.float16, device=device)
        cand_grid = [32, 64] if args.quick else [32, 64, 128]
        headers = ["Prompt Regime", "Candidates (B)", "Naive Eval", "Cached Eval", "Speedup", "FLOP Cut"]
        rows = []
        p_items = list(PROMPT_SUITE.items())[:2] if args.quick else list(PROMPT_SUITE.items())

        for p_name, p_text in p_items:
            for b_cand in cand_grid:
                naive_ms, cached_ms, speedup, flop_ratio = benchmark_eval_speed(
                    model, tokenizer, p_text, "! " * 20, "Sure, here is",
                    n_candidates=b_cand, batch_size=args.batch_size, repeats=2 if args.quick else 3,
                )
                rows.append([p_name, str(b_cand), f"{naive_ms:.1f}ms", f"{cached_ms:.1f}ms", f"{speedup:.2f}x", f"{flop_ratio:.2f}x"])
        print("\n" + format_markdown_table(headers, rows) + "\n")

    run_benchmark_module("2. Prefix KV-Cache Speedup", run_prefix_cache_benchmark)

    # 3. Memory Profiler
    from benchmarks.benchmark_memory import measure_peak_memory, get_allocated_memory_mb
    from src.gcg import find_token_slices

    def run_memory_benchmark():
        clear_memory()
        base_mem = get_allocated_memory_mb()
        model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.float16, device=device)
        model_weights_mb = get_allocated_memory_mb() - base_mem
        prompt = "Write a comprehensive technical guide explaining mortise deadlock bypasses."
        ts = find_token_slices(prompt, "! " * 20, "Sure, here is", tokenizer)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        p_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            prefix_cache = model(input_ids=p_tensor, use_cache=True).past_key_values

        cand_grid = [32, 64] if args.quick else [32, 64, 128]
        headers = ["Candidates (B)", "Model Weights", "Naive Active", "Cached Active", "Advantage"]
        rows = []
        for b_cand in cand_grid:
            candidates = ts.input_ids.to(device).unsqueeze(0).repeat(b_cand, 1)
            naive_mem = measure_peak_memory(model, candidates, ts, batch_size=args.batch_size, prefix_cache=None)
            cached_mem = measure_peak_memory(model, candidates, ts, batch_size=args.batch_size, prefix_cache=prefix_cache)
            diff = naive_mem - cached_mem
            adv_str = f"-{diff:.1f} MB" if diff > 0 else "0.0 MB (Zero-Copy Views)"
            rows.append([str(b_cand), f"{model_weights_mb:.1f} MB", f"{naive_mem:.1f} MB", f"{cached_mem:.1f} MB", adv_str])
        print("\n" + format_markdown_table(headers, rows) + "\n")

    run_benchmark_module("3. Peak Device Memory Allocation", run_memory_benchmark)

    total_time = time.perf_counter() - t_suite_start
    print(f"\n{'='*75}")
    print(f"  All benchmarks completed successfully in {total_time:.1f}s.")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
