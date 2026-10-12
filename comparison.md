# GCG Implementation Comparison

**`gcg_imp`** (minimal) vs **`white_boxed_red`** (advanced)

---

## Project Structure

| Aspect | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Total source files | 4 (`gcg.py`, `model.py`, `utils.py`, `run.py`) | ~20 across `src/`, `configs/`, `experiments/` |
| Architecture | Flat functions + one orchestrator class | ABC hierarchy, dataclass configs, modular packages |
| Entry point | `run.py` (argparse) | `experiments/run_optimization.py` + YAML configs |
| Dependencies beyond torch/transformers | None | `rich`, `pyyaml`, `numpy`, `psutil` |

---

## Core Algorithm: Gradient Computation

Both compute first-order Taylor gradients to score vocabulary substitutions at each mutable position. The mathematical operation is the same — project the loss gradient onto the embedding space — but the implementation differs.

| Detail | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Gradient method | One-hot × embedding matrix; backprop through `one_hot @ W_e` | Detached embedding clone with `requires_grad=True`; backprop through `inputs_embeds`, then project via `grad @ W_e.T` |
| Gradient output | Direct `one_hot.grad` ∈ `[m, V]` — already in vocab space | `embed_grad @ embedding_matrix.T` ∈ `[m, V]` — explicit Taylor projection step |
| Token filtering | None — all vocab tokens eligible | Boolean mask filters non-ASCII and special tokens to `+inf` gradient |
| Loss returned | Not returned from gradient fn | Returns `(gradients, current_loss)` tuple |

**Note:** The one-hot method (`gcg_imp`) and the embedding-grad-projection method (`white_boxed_red`) are mathematically equivalent under the first-order Taylor expansion. The one-hot approach computes `∂L/∂one_hot` which equals `∂L/∂embed @ W_e.T` by chain rule, but does so implicitly via autograd.

---

## Candidate Sampling

| Detail | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Candidates per step | Configurable `n_candidates` (default 256) | Configurable `batch_size` (default 512) |
| Coordinates mutated | 1 per candidate | `num_coordinates_per_step` (default 1, configurable) |
| Identity candidate | Not guaranteed — all candidates are random mutations | Candidate 0 is always the unmodified sequence (monotonicity guarantee) |
| Top-k selection | `(-grads).topk(k)` | Same: `torch.topk(-grads, k)` |
| Surrogate scores | Not computed | Placeholder (`torch.zeros`) — scaffolded but unused |

The identity-candidate guarantee in `white_boxed_red` ensures loss never increases between steps — a subtle but important property for convergence.

---

## Candidate Evaluation

| Detail | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Batching | Simple loop with configurable `batch_size` | Micro-batched with explicit `micro_batch_size` and periodic cache clearing |
| Loss computation | `F.cross_entropy` on sliced logits vs expanded targets | Shifted causal logits (`logits[:, :-1]` vs `ids[:, 1:]`), then slice — explicit causal alignment |
| Loss aggregation | Mean CE per candidate | Mean CE per candidate, or Mellowmax (configurable) |
| Memory management | None | `del` intermediates + `clear_device_cache()` every 4 micro-batches |
| Output | `(best_idx, best_loss)` tuple | `CandidateEvaluationBatch` dataclass with all losses, best index, batch size |

---

## Loss Functions

| Feature | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Cross-entropy | `F.cross_entropy` (mean) | `nn.CrossEntropyLoss(reduction="none")` → manual aggregation |
| Mellowmax | ❌ | ✅ — `(1/α) · log((1/|Y|) · Σ exp(α·Lₜ))`, numerically stable via `logsumexp` |
| Perplexity regularization | ❌ | ✅ — `PerplexityRegularizer` adds `λ · max(0, log(PPL) - threshold)` to loss |
| Composite objectives | ❌ | ✅ — `CompositeObjective` combines target loss + regularizers |

Mellowmax smoothly interpolates between mean CE (α→0) and max-token CE (α→∞), addressing token-level loss plateaus where one token dominates. This is absent in the minimal implementation.

---

## Optimization Loop

| Detail | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Class | `GCGAttack` — concrete, no inheritance | `CoordinateGradientOptimizer(DiscreteOptimizer)` — inherits from ABC |
| Config | Constructor kwargs | `OptimizerConfig` dataclass (20+ fields) |
| Early stopping | Generative check — runs `model.generate()`, compares output prefix to target string | Loss threshold — stops when `loss ≤ early_stopping_loss` |
| Trajectory tracking | Print statements only | `OptimizationTrajectory` dataclass recording every step's loss, gradient norm, tokens, timing, memory |
| Per-step metadata | None | `OptimizationStepResult` with substituted position, original/new token IDs, gradient norm, candidate count, time |
| Callbacks | None | Optional `callback: Callable[[StepResult], None]` per step |
| Logging | `print()` with simple table formatting | `rich` tables + `logging` module |
| Initial loss | Not computed | Computed and recorded before loop starts |
| Reproducibility | No seed control | `torch.manual_seed(config.seed)` |

---

## Token Slicing / Prompt Construction

| Detail | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Method | `find_token_slices()` — tokenize 3 segments independently, concatenate | `CausalLMModelWrapper.build_prompt_slices()` — sentinel-based boundary detection |
| Chat templates | Not used in slicing (applied separately in `run.py`) | First-class — sentinels injected into chat template to locate exact opt boundaries |
| Slice validation | None | `TokenSlices.validate()` enforces `loss_slice.stop ≤ total_len`, contiguity, and `loss_slice == slice(target-1, ...)` |
| BPE boundary handling | Acknowledged as a "small mismatch" in docstring | Sentinel markers (`<<<OPT_START>>>`) avoid BPE merge distortion |
| Slice dataclass | 4 fields: `input_ids`, `suffix_slice`, `target_slice`, `loss_slice` | 5+ fields: `prefix_slice`, `opt_slice`, `target_slice`, `loss_slice`, `delim_slice`; frozen dataclass |

---

## Model Handling

| Detail | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Loading | `load_model_and_tokenizer()` — direct `AutoModel` + device move | `CausalLMModelWrapper` — wraps model with cached embedding matrix, vocab masks, generation helpers |
| Embedding access | `model.get_input_embeddings().weight` inline | `_resolve_embedding_layer()` with fallback chain (`embed_tokens`, `wte`, etc.) |
| Vocab filtering | None | `build_non_ascii_mask()` — iterates vocab, flags non-ASCII + special tokens |
| Generation | Done in `run.py` directly | `wrapper.generate_text()` method |
| Synthetic model | None — requires HF model download | `build_synthetic_model_and_tokenizer()` — random-init GPT-2 for testing without downloads |

---

## Device / Memory Management

| Detail | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Device selection | `get_device()` — MPS or CPU | `get_best_device()` — MPS → CUDA → CPU with validation |
| Memory tracking | None | `MPSMemoryTracker` context manager, `MPSMemorySnapshot` dataclass, `psutil`-based system memory |
| Cache clearing | `clear_memory()` — manual call | `clear_device_cache()` called automatically during eval micro-batches |
| Micro-batch sizing | Fixed | `estimate_safe_micro_batch_size()` heuristic based on available memory |
| Memory profiling | None | Per-step `memory_snapshot` in step results when `profile_memory=True` |

---

## Draft Model Filtering

| Feature | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Multi-fidelity screening | ❌ | ✅ — `DraftModelFilter` evaluates candidates on a small model first |
| Filtering ratio | — | Top `B/r` candidates retained (default `512/8 = 64`) |
| Identity preservation | — | Always retains candidate 0 (unmodified sequence) through filtering |
| Telemetry | — | `DraftScreeningResult` with timing, batch sizes, effective ratio |

This is a speculative-decoding-inspired acceleration: evaluate 512 candidates on a 0.5B model, keep the 64 best, then run only those through the full target model.

---

## Evaluation / Analysis

| Feature | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Sensitivity profiling | ❌ | ✅ — `DiscreteSensitivityProfiler` applies random substitutions/swaps/insertions at varying noise rates |
| Perturbation types | — | `RANDOM_SUBSTITUTION`, `RANDOM_SWAP`, `RANDOM_INSERTION` |
| Loss landscape analysis | — | Measures loss degradation curve under perturbation to quantify solution sharpness |
| Benchmarking | — | `benchmark_suite.py`, `plot_dynamics.py` for convergence visualization |

---

## Configuration System

| Detail | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| Config format | CLI `argparse` flags | Nested dataclasses (`ModelTargetConfig`, `OptimizerHyperparameters`, `EvaluationSuiteConfig`) |
| Persistence | None | YAML load/save via `ExperimentConfig.from_yaml()` / `.to_yaml()` |
| Per-model configs | None | `configs/qwen_2_5_1_5b.yaml`, `configs/llama_3_2_1b.yaml` |

---

## Summary

| Dimension | `gcg_imp` | `white_boxed_red` |
|---|---|---|
| **Purpose** | Educational / reference GCG implementation | Research-grade framework with production concerns |
| **Lines of core logic** | ~400 | ~2500+ |
| **Algorithmic fidelity** | Correct GCG with all 4 canonical steps | Same core + Mellowmax, draft filtering, multi-coordinate mutation, perplexity regularization |
| **Convergence guarantee** | None (loss can increase between steps) | Identity candidate ensures monotonic loss descent |
| **Memory safety** | Manual, best-effort | Systematic micro-batching, cache clearing, memory profiling |
| **Extensibility** | Modify functions directly | ABC + config dataclasses; swap optimizers, objectives, models |
| **Reproducibility** | No seed management | Seeded RNG, full trajectory serialization |

`gcg_imp` is the algorithm distilled to its essence — ~400 lines, zero abstraction overhead, easy to read and modify. `white_boxed_red` wraps the same core math in a framework designed for systematic experimentation: configurable objectives, memory-aware execution on Apple Silicon, draft-model acceleration, and post-hoc sensitivity analysis.
