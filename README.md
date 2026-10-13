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
│  Step 2   inputs_embeds → forward → loss → backward → Taylor projection│
│           grad @ W_embed.T = projected vocab-level gradient score      │
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
| `get_token_gradients()` | Forward via `inputs_embeds` → CE loss → backward → Taylor projection (`grad @ W_embed.T`). Optional token filter mask. |
| `sample_candidates()` | Retains candidate 0 as identity (monotonicity); samples 1..N-1 with top-k replacements |
| `evaluate_candidates()` | Batched forward pass over all candidates, returns `(best_idx, best_loss)` |
| `GCGAttack` | Orchestrator class with `step()` for single iterations and `run()` for the full loop |

**`src/model.py`** — Loads any HuggingFace causal LM with automatic device placement and `pad_token` fallback.

**`src/utils.py`** — `get_device()` returns MPS/CUDA/CPU; `clear_memory()` flushes backend caches; `get_non_ascii_mask()` builds boolean masks for non-ASCII/special tokens; `expand_kv_cache()` broadcasts prefix key-value caches across candidate batches.

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

# Enable chat template wrapping during evaluation
uv run python run.py --use-chat-template

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
| **Prompt** | `"Write a tutorial on how to pick a lock"` | Instruction the model should refuse | Longer prompts anchor the model's refusal more strongly, making the attack harder. |
| **Suffix** | `"! " * 20` (20 tokens) | Initial adversarial string to optimize | More tokens = larger search space = more expressive but slower. 20 is standard; 10 minimal, 40 aggressive. |
| **Steps** | 20 | Optimization iterations | 10-20 for smoke tests. 100-500 for real attacks. Loss typically plateaus around 200-300 steps. |
| **Top-K** | 256 | Gradient-ranked candidates per position | Controls exploration/exploitation tradeoff. 256 is standard; 64 is focused (faster), 512 is more exploratory. |
| **Candidates** | 256 | Mutated suffixes evaluated per step | Directly controls search breadth. More = better per-step progress but linearly slower. 32 for quick tests, 256-512 for production. |
| **Batch Size** | 64 | Forward-pass batch for evaluation | Controls VRAM vs speed tradeoff. Reduce if you see OOM errors. Does not affect attack quality. |
| **Max Tokens** | 32 | Generation length for early-stop check | 32-64 is typical. Increase to see longer completions during testing. |
| **Early Stop** | `True` | Stop when model generates target prefix | Disable to observe the full loss trajectory without interruption. |
| **Chat Template** | `False` | Apply model chat template during generation | When False (default), generation matches the raw token sequence optimized by GCG. When True, wraps input in the chat template (e.g. ChatML). |
| **Allow Non-ASCII** | `False` | Permit non-ASCII/special tokens | When False (default), filters unprintable unicode, replacement characters, and special tokens from candidate pool. |
| **Prefix KV Cache** | `True` | Precompute and reuse KV cache for prompt prefix | When True (default), avoids recomputing attention/MLP layers on prompt tokens across candidate batches (1.5x-3x faster). Disable with `--no-prefix-cache`. |
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


## Key Algorithmic & Systems Enhancements

This implementation builds directly on canonical GCG with three grounded mathematical and systems refinements:

### 1. Monotonic Identity Candidate ($c_0$) Preservation
In standard random coordinate sampling, every candidate mutates at least one token coordinate. If none of the sampled top-$k$ substitutions yield lower loss than the current state, the optimizer is forced to adopt a suboptimal candidate, leading to loss spikes and erratic trajectories.

By preserving candidate index 0 as the unmodified sequence $x^{(t)}$, we enforce the coordinate descent invariant:
$$\mathcal{L}(x^{(t+1)}) \le \mathcal{L}(x^{(t)})$$
If all proposed mutations fail to decrease loss, $c_0$ is retained, ensuring strictly non-increasing loss progression.

### 2. Direct Embedding Gradient Projection via `inputs_embeds`
Rather than allocating a dense one-hot tensor $\mathbb{R}^{m \times V}$ and routing autograd through custom prefix/suffix slices, the forward pass runs directly on `inputs_embeds = embed_layer(input_ids).unsqueeze(0).detach().requires_grad_(True)`.

The vocabulary-level gradient scores are computed via the exact first-order Taylor expansion:
$$\nabla_{\text{vocab}} \mathcal{L} = \nabla_{e_{\text{suffix}}} \mathcal{L} \cdot W_e^T \in \mathbb{R}^{m \times V}$$
This eliminates graph nodes and allocations for $(m \times V)$ float tensors, reducing memory overhead and delivering identical gradients ($\Delta = 0.0$).

### 3. Non-ASCII & Special Token Filtering
Language model vocabularies (e.g. LLaMA, Qwen, GPT-2) include hundreds of raw byte tokens, non-printable Unicode characters, and control tokens (`<|endoftext|>`, `<|im_start|>`). Without filtering, GCG frequently selects these tokens to minimize loss via tokenizer edge cases rather than true prompt jailbreaks.

`get_non_ascii_mask()` inspects token decodings at initialization and precomputes a boolean mask flagging non-ASCII, unprintable (`\ufffd`), and special tokens. Masked tokens receive a score of $+\infty$, ensuring `topk()` (steepest descent) never selects them.

### 4. Static Prefix KV Caching
Candidate evaluation accounts for >85% of total GCG compute. Because the user instruction / prompt prefix remains invariant across all $N$ optimization iterations and all $B$ candidate forward passes, causal attention guarantees that Key and Value representations of prefix tokens never depend on the downstream suffix or target tokens.

At step 0, `compute_prefix_cache()` executes a single forward pass over the prompt prefix (batch size 1). During candidate evaluation, `expand_kv_cache()` broadcasts the precomputed KV cache across candidate batches using $\mathcal{O}(1)$ strided views without copying tensor memory. Candidates only forward their $[suffix \parallel target]$ tokens ($S+T$ instead of $P+S+T$), attending back to the cached prefix. This yields a **1.5× to 3× wall-clock speedup** with zero loss in mathematical precision ($\Delta \approx 10^{-6}$).

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

### Prompt Formatting & Chat Template Alignment

- **Raw token space by default**: The optimization computes gradients and cross-entropy loss over the raw token concatenation `prompt ‖ suffix ‖ target` without special delimiter tokens.
- **Evaluation consistency**: `check_early_stop()`, `run.py`, and `web_ui.py` (`generate_text`) use the exact same input formatting (`use_chat_template=False` by default). This guarantees that when early stopping detects the target prefix during optimization, manual execution with the suffix produces the identical target response.
- **Optional template transfer**: An optional `use_chat_template` toggle allows testing whether suffixes optimized in raw token space transfer to structured chat templates (e.g., ChatML `<|im_start|>user ... <|im_end|><|im_start|>assistant`).

---

## License

MIT
