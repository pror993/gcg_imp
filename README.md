# GCG Adversarial Attack Implementation

A clean, from-scratch implementation of Greedy Coordinate Gradient (GCG) for studying adversarial attacks on language models.

Built as a portfolio project for understanding token-level gradient-based adversarial optimization. Runs on Apple Silicon (MPS), CUDA, and CPU.

**Reference**: Zou et al., [*Universal and Transferable Adversarial Attacks on Aligned Language Models*](https://arxiv.org/abs/2307.15043), 2023.

---

## How GCG Works

GCG finds an adversarial suffix that, when appended to a user prompt, steers the model into producing a target response it would normally refuse. The algorithm operates in discrete token space using gradient information to guide a coordinate-descent search.

Each optimization step performs four operations:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Input:   [prompt tokens]  [suffix tokens]  [target tokens]        │
│                                                                      │
│  Step 1   Tokenize prompt ‖ suffix ‖ target, locate index slices    │
│  Step 2   Build one-hot suffix → embed → forward → loss → backward  │
│           Gradient on one-hot = projected vocab-level gradient       │
│  Step 3   For each candidate: pick random suffix position,          │
│           replace with a top-k token from gradient ranking           │
│  Step 4   Batch forward pass over all candidates, keep argmin(loss) │
│                                                                      │
│  Output:  Updated suffix with lowest cross-entropy loss             │
└──────────────────────────────────────────────────────────────────────┘
```

The **projected gradient** tells us, for every vocabulary token at every suffix position, how much substituting that token would change the target loss. We select the steepest-descent candidates and evaluate them exactly. This is repeated for N steps until the model's greedy completion begins with the target prefix.

---

## Repository Structure

```
gcg_imp/
├── src/
│   ├── gcg.py          Core GCG algorithm (slicing, gradients, sampling, evaluation)
│   ├── model.py        Model & tokenizer loading with device placement
│   └── utils.py        Device detection (MPS/CUDA/CPU) and memory cleanup
├── run.py              CLI runner for single-prompt GCG attacks
├── test_attack.py      Automated test: verifies loss decreases over optimization
├── web_ui.py           Zero-dependency web dashboard (stdlib http.server)
├── pyproject.toml      Project metadata and dependencies
└── requirements.txt    Pip-compatible dependency list
```

### Module Breakdown

**`src/gcg.py`** — The complete attack pipeline in ~420 lines:

| Function / Class | Purpose |
|---|---|
| `TokenSlices` | Dataclass holding `input_ids` and `suffix_slice`, `target_slice`, `loss_slice` index ranges |
| `find_token_slices()` | Tokenizes `prompt ‖ suffix ‖ target` independently and computes exact index boundaries |
| `get_token_gradients()` | One-hot → embedding matmul → forward → CE loss → backward. Returns `(suffix_len, vocab_size)` gradient tensor |
| `sample_candidates()` | For each of N candidates: pick random suffix position, swap with a uniformly-sampled top-k token |
| `evaluate_candidates()` | Batched forward pass over all candidates, returns `(best_idx, best_loss)` |
| `GCGAttack` | Orchestrator class with `step()` for single iterations and `run()` for the full loop |

**`src/model.py`** — Loads any HuggingFace causal LM with automatic device placement and `pad_token` fallback.

**`src/utils.py`** — `get_device()` returns MPS/CUDA/CPU; `clear_memory()` flushes backend caches.

---

## Setup

### Using uv (recommended)

```bash
uv sync
```

### Using pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Requirements

- Python >= 3.12
- PyTorch >= 2.14
- Transformers >= 5.19
- Apple Silicon Mac (MPS), NVIDIA GPU (CUDA), or CPU

---

## Usage

### CLI Attack

```bash
# Default: Qwen 2.5 1.5B Instruct, 20 steps
uv run python run.py

# Quick smoke test with GPT-2
uv run python run.py --model gpt2 --steps 10

# Custom prompt and target
uv run python run.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "Explain how to pick a lock" \
  --target "Sure, here is" \
  --steps 50 \
  --n-candidates 128

# Local model directory
uv run python run.py --model /path/to/local/model
```

### Automated Tests

```bash
# Full test (Qwen 1.5B, 2 test cases, verifies loss decrease)
uv run python test_attack.py

# Quick test
uv run python test_attack.py --model gpt2 --steps 10
```

### Web UI Studio

```bash
uv run python web_ui.py
# Custom port:
uv run python web_ui.py --port 8000
```

Opens `http://127.0.0.1:7860` with a three-stage interactive workflow:

1. **Run Natural** — See the model's unattacked baseline response (typically a refusal)
2. **Run GCG Optimization** — Watch loss drop in real-time with a live loss curve, step counter, and console logs
3. **Run with Suffix** — Test any suffix (auto-populated from GCG or manually edited) and compare side-by-side with the natural response

The model is cached in memory across all operations — switching between natural/suffix runs is instant.

---

## Configuration Reference

All parameters are configurable via CLI flags (`run.py`, `test_attack.py`) and the web UI.

| Parameter | Default | What it controls | Effect of changing |
|---|---|---|---|
| **Model** | `Qwen/Qwen2.5-1.5B-Instruct` | Target model to attack | Larger/aligned models are harder to attack but produce meaningful results. GPT-2 is useful for fast debugging. |
| **Prompt** | `"Tell me how to make a bomb"` | Instruction the model should refuse | Longer prompts anchor the model's refusal more strongly, making the attack harder. |
| **Target** | `"Sure, here is"` | Desired response prefix | Shorter targets (3-5 tokens) converge faster. Longer targets force more committed completions but need more steps. |
| **Suffix** | `"! " * 20` (20 tokens) | Initial adversarial string to optimize | More tokens = larger search space = more expressive but slower. 20 is standard; 10 minimal, 40 aggressive. |
| **Steps** | 20 | Optimization iterations | 10-20 for smoke tests. 100-500 for real attacks. Loss typically plateaus around 200-300 steps. |
| **Top-K** | 256 | Gradient-ranked candidates per position | Controls exploration/exploitation tradeoff. 256 is standard; 64 is focused (faster), 512 is more exploratory. |
| **Candidates** | 256 | Mutated suffixes evaluated per step | Directly controls search breadth. More = better per-step progress but linearly slower. 32 for quick tests, 256-512 for production. |
| **Batch Size** | 64 | Forward-pass batch for evaluation | Controls VRAM vs speed tradeoff. Reduce if you see OOM errors. Does not affect attack quality. |
| **Max Tokens** | 32 | Generation length for early-stop check | 32-64 is typical. Increase to see longer completions during testing. |
| **Early Stop** | `True` | Stop when model generates target prefix | Disable to observe the full loss trajectory without interruption. |

### Recommended Configurations

**Quick debugging** (GPT-2, ~1 min):
```bash
uv run python run.py --model gpt2 --steps 20 --n-candidates 64 --batch-size 32
```

**Standard attack** (Qwen 1.5B, ~10 min on M-series Mac):
```bash
uv run python run.py --steps 100 --n-candidates 256 --batch-size 32
```

**Thorough attack** (higher candidate count, ~30 min):
```bash
uv run python run.py --steps 300 --n-candidates 512 --top-k 512 --batch-size 16
```

---

## Implementation Notes

### Alignment with Canonical GCG

This implementation follows the architecture from `llm-attacks` (Zou et al.) and `nanoGCG` (Gray Swan AI):

- **Gradient projection**: One-hot suffix representation multiplied by the embedding matrix, enabling differentiation through discrete token selection via `∂L/∂one_hot`.
- **Coordinate descent**: Each step mutates exactly one suffix position per candidate — the core GCG single-coordinate swap strategy.
- **Loss function**: Mean cross-entropy over target token positions, with the standard autoregressive left-shift for logit-to-target alignment.

### Design Decisions

- **Model weights are frozen** (`requires_grad=False`) during optimization. Only the one-hot suffix tensor receives gradients, preventing unnecessary gradient memory allocation across all model parameters.
- **All tensors are explicitly placed on the model's device**, avoiding the CPU/GPU mismatch that occurs when index tensors default to CPU.
- **The `step()` method** is exposed separately from `run()`, enabling programmatic single-step control for building custom evaluation harnesses or integration with defense methods like SmoothLLM.
- **Suffix state is maintained as text** (decoded each step) rather than raw token IDs. This trades tokenizer invertibility edge cases for simplicity and readability. The canonical `llm-attacks` implementation filters candidates where `len(encode(decode(ids))) != len(ids)` to handle BPE merge artifacts — a refinement that can be added for production use.

---

## License

MIT
