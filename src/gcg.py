"""
Greedy Coordinate Gradient (GCG) attack implementation.

Reference: Zou et al., "Universal and Transferable Adversarial Attacks on
Aligned Language Models", 2023.

The algorithm optimises an adversarial suffix so that the model is steered
towards producing a specific target string.  At each step we:
  1. Tokenise  prompt ‖ suffix ‖ target  and locate the suffix / target slices.
  2. Compute per-token gradients of the target-token cross-entropy loss
     w.r.t. the suffix embeddings, then project onto the full vocabulary.
  3. Sample candidate replacements by swapping one suffix token at a time
     with a top-k candidate from the gradient ranking.
  4. Evaluate all candidates in a batched forward pass and keep the winner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Data container returned by find_token_slices
# ---------------------------------------------------------------------------
@dataclass
class TokenSlices:
    """Holds the full token sequence and the index ranges for each segment."""
    input_ids: torch.Tensor       # (seq_len,)  – full tokenised sequence
    suffix_slice: slice           # index range of the suffix tokens
    target_slice: slice           # index range of the target tokens
    # For the loss we predict the *next* token, so the logit positions that
    # correspond to target tokens are shifted left by one.
    loss_slice: slice             # logit positions used for the CE loss


# ---------------------------------------------------------------------------
# Step 1 – Slice the tokens
# ---------------------------------------------------------------------------
def find_token_slices(
    prompt: str,
    suffix: str,
    target: str,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenSlices:
    """Tokenise  prompt ‖ suffix ‖ target  and find the exact index ranges.

    We tokenise each piece individually and concatenate so that we know the
    exact boundaries, avoiding any ambiguity from merges at the boundaries
    (a known tokeniser quirk – we accept the small mismatch for clarity).

    Returns a ``TokenSlices`` with the full ``input_ids`` tensor (1-D, on CPU)
    and the three slices.
    """
    # Tokenise each segment independently (no special tokens).
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    full_ids = prompt_ids + suffix_ids + target_ids

    # Compute index boundaries.
    p_len = len(prompt_ids)
    s_len = len(suffix_ids)
    t_len = len(target_ids)

    suffix_slice = slice(p_len, p_len + s_len)
    target_slice = slice(p_len + s_len, p_len + s_len + t_len)

    # For auto-regressive loss the logit at position i predicts token i+1,
    # so the logit positions that predict the target tokens are one step
    # earlier: [target_start - 1, target_end - 1).
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)

    input_ids = torch.tensor(full_ids, dtype=torch.long)

    return TokenSlices(
        input_ids=input_ids,
        suffix_slice=suffix_slice,
        target_slice=target_slice,
        loss_slice=loss_slice,
    )


# ---------------------------------------------------------------------------
# Step 2 – Get gradients on suffix embeddings projected to vocab
# ---------------------------------------------------------------------------
@torch.enable_grad()
def get_token_gradients(
    model: PreTrainedModel,
    token_slices: TokenSlices,
) -> torch.Tensor:
    """Compute the gradient-based score for every vocab token at each suffix position.

    Returns
    -------
    token_gradients : Tensor of shape ``(suffix_len, vocab_size)``
        Lower (more negative) values indicate tokens that would *decrease*
        the loss most if substituted in.
    """
    embed_layer = model.get_input_embeddings()          # nn.Embedding
    embed_weights = embed_layer.weight                  # (vocab_size, d_model)
    device = embed_weights.device

    input_ids = token_slices.input_ids.to(device)       # (seq_len,)

    # --- build the one-hot → embedding manually for the suffix so we can
    #     differentiate through it. ------------------------------------------
    # Convert ids to one-hot for the suffix positions only.
    suffix_ids = input_ids[token_slices.suffix_slice]    # (suffix_len,)
    suffix_len = suffix_ids.shape[0]
    vocab_size = embed_weights.shape[0]

    # One-hot representation of the suffix tokens – requires grad.
    one_hot = torch.zeros(
        suffix_len, vocab_size, device=device, dtype=embed_weights.dtype
    )
    one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0)
    one_hot.requires_grad_(True)

    # Suffix embeddings via matmul with embedding table.
    suffix_embeds = one_hot @ embed_weights              # (suffix_len, d_model)

    # Embeddings for the rest of the sequence (no grad needed).
    with torch.no_grad():
        prefix_embeds = embed_layer(input_ids[:token_slices.suffix_slice.start])
        postfix_embeds = embed_layer(input_ids[token_slices.suffix_slice.stop:])

    # Concatenate: prefix ‖ suffix ‖ postfix
    full_embeds = torch.cat(
        [prefix_embeds, suffix_embeds, postfix_embeds], dim=0
    ).unsqueeze(0)  # (1, seq_len, d_model)

    # --- forward pass -------------------------------------------------------
    logits = model(inputs_embeds=full_embeds).logits      # (1, seq_len, vocab)

    # Slice out the logits that predict the target tokens.
    loss_logits = logits[0, token_slices.loss_slice, :]   # (target_len, vocab)
    target_ids = input_ids[token_slices.target_slice]     # (target_len,)

    loss = F.cross_entropy(loss_logits, target_ids)

    # --- backward pass -------------------------------------------------------
    loss.backward()

    # The gradient on one_hot has shape (suffix_len, vocab_size).  Each entry
    # tells us how much the loss would change if we increased the weight of
    # that vocab token at that position – i.e. the projected gradient.
    token_gradients = one_hot.grad.clone()               # (suffix_len, vocab_size)

    # Clean up the computation graph.
    model.zero_grad(set_to_none=True)

    return token_gradients


# ---------------------------------------------------------------------------
# Step 3 – Sample candidate suffixes
# ---------------------------------------------------------------------------
def sample_candidates(
    current_ids: torch.Tensor,
    token_gradients: torch.Tensor,
    suffix_slice: slice,
    top_k: int = 256,
    n_candidates: int = 256,
) -> torch.Tensor:
    """Sample candidate suffix replacements using top-k gradient info.

    For each candidate we:
      1. Pick a random position in the suffix.
      2. Replace the token at that position with one of the top-k tokens
         (chosen uniformly at random from the k best per that position).

    Parameters
    ----------
    current_ids : Tensor (seq_len,)
        The current full token sequence.
    token_gradients : Tensor (suffix_len, vocab_size)
        Gradient scores from ``get_token_gradients``.
    suffix_slice : slice
        Index range of the suffix inside ``current_ids``.
    top_k : int
        Number of lowest-gradient candidates to consider per position.
    n_candidates : int
        How many candidate sequences to generate.

    Returns
    -------
    candidates : Tensor (n_candidates, seq_len)
        Each row is a full token sequence with exactly one suffix token swapped.
    """
    device = token_gradients.device
    suffix_len = token_gradients.shape[0]

    # For each suffix position, find top_k tokens with LOWEST gradient
    # (steepest descent direction).
    top_k_ids = (-token_gradients).topk(top_k, dim=1).indices  # (suffix_len, top_k)

    # Tile the current sequence on device into (n_candidates, seq_len).
    candidates = current_ids.to(device).unsqueeze(0).repeat(n_candidates, 1)

    # Pick a random suffix position and top-k replacement token for each candidate.
    rand_pos = torch.randint(0, suffix_len, (n_candidates,), device=device)
    rand_k = torch.randint(0, top_k, (n_candidates,), device=device)
    replacements = top_k_ids[rand_pos, rand_k]

    # Mutate selected positions in-place.
    candidates[torch.arange(n_candidates, device=device), rand_pos + suffix_slice.start] = replacements
    return candidates


# ---------------------------------------------------------------------------
# Step 4 – Evaluate candidates and pick the best
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_candidates(
    model: PreTrainedModel,
    candidates: torch.Tensor,
    token_slices: TokenSlices,
    batch_size: int = 64,
) -> Tuple[int, float]:
    """Run a batched forward pass over candidates and return the best one.

    Parameters
    ----------
    model : PreTrainedModel
    candidates : Tensor (n_candidates, seq_len)
    token_slices : TokenSlices
        Provides ``loss_slice`` and ``target_slice`` (index ranges are the
        same for every candidate since sequence length is fixed).
    batch_size : int
        Process candidates in chunks to avoid OOM.

    Returns
    -------
    best_idx : int
        Index into ``candidates`` of the lowest-loss candidate.
    best_loss : float
        The cross-entropy loss achieved by the best candidate.
    """
    device = next(model.parameters()).device
    n_candidates = candidates.shape[0]
    target_ids = token_slices.input_ids[token_slices.target_slice].to(device)

    all_losses = []

    for start in range(0, n_candidates, batch_size):
        end = min(start + batch_size, n_candidates)
        batch_ids = candidates[start:end].to(device)       # (B, seq_len)

        logits = model(input_ids=batch_ids).logits          # (B, seq_len, vocab)

        # Slice out the logits that predict target tokens.
        loss_logits = logits[:, token_slices.loss_slice, :]  # (B, target_len, vocab)

        # Mean cross-entropy loss over target tokens per candidate.
        targets = target_ids.unsqueeze(0).expand(batch_ids.size(0), -1)
        loss = F.cross_entropy(loss_logits.transpose(1, 2), targets, reduction="none").mean(dim=1)
        all_losses.append(loss)
    all_losses = torch.cat(all_losses)                      # (n_candidates,)
    best_idx = all_losses.argmin().item()
    best_loss = all_losses[best_idx].item()

    return best_idx, best_loss


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------
class GCGAttack:
    """Greedy Coordinate Gradient (GCG) attack.

    Coordinates token-level gradient calculation and candidate substitution
    to find adversarial suffixes that elicit target model outputs.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        top_k: int = 256,
        n_candidates: int = 256,
        batch_size: int = 64,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.n_candidates = n_candidates
        self.batch_size = batch_size

        # Freeze model weights so backward only computes gradients for input tokens.
        for p in self.model.parameters():
            p.requires_grad = False

    # ---- helpers -----------------------------------------------------------

    @torch.no_grad()
    def check_early_stop(
        self,
        prompt: str,
        suffix: str,
        target: str,
        max_new_tokens: int = 32,
        use_chat_template: bool = False,
    ) -> Tuple[bool, str]:
        """Generate greedily from  prompt ‖ suffix  and check for the target.

        Returns ``(success, generated_text)``.  ``success`` is True when the
        generated text starts with ``target`` (case-insensitive, stripped).
        """
        device = next(self.model.parameters()).device
        full_prompt = prompt + suffix
        if use_chat_template and getattr(self.tokenizer, "chat_template", None) is not None:
            messages = [{"role": "user", "content": full_prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(formatted, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
        else:
            input_ids = self.tokenizer.encode(
                full_prompt, return_tensors="pt", add_special_tokens=False
            ).to(device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the newly generated tokens.
        generated = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        success = generated.lower().startswith(target.lower().strip())
        return success, generated

    # ---- public API --------------------------------------------------------
    def step(
        self,
        prompt: str,
        suffix: str,
        target: str,
    ) -> Tuple[str, float]:
        """Execute one GCG iteration and return ``(updated_suffix, best_loss)``."""
        ts = find_token_slices(prompt, suffix, target, self.tokenizer)
        token_grads = get_token_gradients(self.model, ts)
        candidates = sample_candidates(
            current_ids=ts.input_ids,
            token_gradients=token_grads,
            suffix_slice=ts.suffix_slice,
            top_k=self.top_k,
            n_candidates=self.n_candidates,
        )
        best_idx, best_loss = evaluate_candidates(
            self.model, candidates, ts, batch_size=self.batch_size,
        )
        suffix_ids = candidates[best_idx, ts.suffix_slice]
        new_suffix = self.tokenizer.decode(suffix_ids, skip_special_tokens=True)
        return new_suffix, best_loss

    def run(
        self,
        prompt: str,
        target: str,
        suffix: str | None = None,
        n_steps: int = 20,
        early_stop: bool = True,
        use_chat_template: bool = False,
    ) -> str:
        """Run the GCG optimisation loop.

        Parameters
        ----------
        prompt : str
            The user prompt (e.g. an instruction the model would normally refuse).
        target : str
            The desired model output prefix (e.g. "Sure, here is").
        suffix : str | None
            Initial adversarial suffix.  Defaults to 20 ``"!"`` characters.
        n_steps : int
            Number of optimisation iterations.
        early_stop : bool
            If True, stop as soon as the model's greedy output starts with ``target``.

        Returns
        -------
        suffix : str
            The optimised adversarial suffix.
        """
        if suffix is None:
            suffix = "! " * 20

        print(f"\n{'='*70}")
        print(f"  GCG Attack")
        print(f"  steps={n_steps}  top_k={self.top_k}  "
              f"candidates={self.n_candidates}  batch_size={self.batch_size}")
        print(f"  Prompt : {prompt}")
        print(f"  Target : {target}")
        print(f"  Suffix : {suffix!r}")
        print(f"{'='*70}")
        print(f"{'Step':>6}  {'Loss':>8}  {'Time':>7}  Suffix")
        print(f"{'-'*6}  {'-'*8}  {'-'*7}  {'-'*40}")

        for step_num in range(1, n_steps + 1):
            step_start = time.perf_counter()
            suffix, best_loss = self.step(prompt, suffix, target)
            elapsed = time.perf_counter() - step_start

            print(f"{step_num:>4}/{n_steps:<2}  {best_loss:>8.4f}  {elapsed:>6.1f}s  {suffix!r}")

            if early_stop:
                success, generated = self.check_early_stop(
                    prompt, suffix, target, use_chat_template=use_chat_template
                )
                if success:
                    print(f"\n✅  Early stop at step {step_num}!\n    Generated: {generated!r}")
                    return suffix

        print(f"\n{'='*70}\n  Finished {n_steps} steps (no early stop triggered)\n{'='*70}")
        return suffix

