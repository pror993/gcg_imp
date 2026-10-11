# Minimal Greedy Coordinate Gradient (GCG) on Apple Silicon (MPS)

Clean, minimal implementation of Greedy Coordinate Gradient (GCG) for token-level adversarial gradient optimization on Apple Silicon using PyTorch and `mps`.

## Setup

Using `uv`:
```bash
uv sync
```

Or using standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Sanity Check

### Using Hugging Face Hub (auto-downloads to cache)
```bash
uv run python run.py --model Qwen/Qwen2.5-1.5B-Instruct
```

For quick local testing without downloading a large model:
```bash
uv run python run.py --model gpt2
```

### Using a Local Directory
Point `--model` directly to the local folder containing model weights (`config.json`, `.safetensors`, `tokenizer.json`):
```bash
uv run python run.py --model /path/to/local/model
```

## Web UI Studio

Launch the zero-dependency local web dashboard to run natural baseline comparisons, configure GCG attacks, inspect loss curves, and evaluate adversarial suffixes interactively:

```bash
uv run python web_ui.py
# Or with custom port:
uv run python web_ui.py --port 8000
```

## Running Automated Tests

```bash
uv run python test_attack.py --model gpt2 --steps 10
```
