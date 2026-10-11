import argparse
import torch
from src.model import load_model_and_tokenizer
from src.utils import clear_memory, get_device


def main():
    parser = argparse.ArgumentParser(description="GCG environment sanity check")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face repo ID or path to local model directory",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model,
        dtype=torch.float16,
        device=device,
    )

    prompt = "Hello, world! Testing MPS inference."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    print(f"Input: {prompt!r}")
    print(f"Logits shape: {tuple(logits.shape)} on {logits.device}")

    clear_memory()
    print("Setup ready")


if __name__ == "__main__":
    main()
