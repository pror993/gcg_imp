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
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from src.utils import expand_kv_cache, get_non_ascii_mask


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
    filter_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the first-order Taylor gradient scores for every vocab token at each suffix position.

    Differentiates loss directly with respect to full sequence embeddings and projects
    the gradient onto the vocabulary embedding table via matrix multiplication (``grad @ W_embed.T``).

    Parameters
    ----------
    model : PreTrainedModel
        Target causal language model.
    token_slices : TokenSlices
        Data container with input IDs and slice boundaries.
    filter_mask : Tensor, optional
        Boolean tensor of shape ``(vocab_size,)`` where ``True`` marks tokens to
        exclude (e.g. non-ASCII or special tokens). Masked tokens receive ``+inf``.

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

    # 1. Forward embeddings with gradient tracking enabled on inputs_embeds
    # Avoids constructing and differentiating through dense (suffix_len, vocab_size) one-hot matrices.
    inputs_embeds = embed_layer(input_ids).unsqueeze(0).detach().requires_grad_(True)

    # 2. Forward pass through causal transformer
    logits = model(inputs_embeds=inputs_embeds).logits  # (1, seq_len, vocab_size)

    # 3. Compute cross-entropy loss over target token slice
    loss_logits = logits[0, token_slices.loss_slice, :] # (target_len, vocab_size)
    target_ids = input_ids[token_slices.target_slice]   # (target_len,)

    loss = F.cross_entropy(loss_logits, target_ids)

    # 4. Backward pass to extract embedding gradients
    loss.backward()

    # 5. First-order Taylor projection: dot product with embedding matrix
    # token_gradients[i, v] = <grad_i, e(v)>
    # (suffix_len, d_model) @ (d_model, vocab_size) -> (suffix_len, vocab_size)
    suffix_grad = inputs_embeds.grad[0, token_slices.suffix_slice, :]
    token_gradients = suffix_grad @ embed_weights.T

    # 6. Mask filtered tokens (non-ASCII, special) with +inf so top-k never selects them
    if filter_mask is not None:
        token_gradients[:, filter_mask.to(device)] = float("inf")

    # Clean up computation graph
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

    Candidate 0 is always retained as the unmodified current sequence to guarantee
    monotonic (non-increasing) loss under coordinate descent. The remaining
    ``n_candidates - 1`` candidates each substitute one random suffix position
    with a random token from the top-k gradient pool.
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

    # Candidate 0 is preserved as the current sequence (guarantees monotonic non-increasing loss).
    # Mutate only candidates 1..n_candidates - 1.
    n_mutations = n_candidates - 1
    if n_mutations > 0:
        rand_pos = torch.randint(0, suffix_len, (n_mutations,), device=device)
        rand_k = torch.randint(0, top_k, (n_mutations,), device=device)
        replacements = top_k_ids[rand_pos, rand_k]

        # Mutate selected positions in-place for candidates 1 to N-1
        candidates[1 + torch.arange(n_mutations, device=device), rand_pos + suffix_slice.start] = replacements
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
    prefix_cache: Any = None,
) -> Tuple[int, float]:
    """Run a batched forward pass over candidates and return the best one.

    When ``prefix_cache`` is provided (precomputed KV cache for the static prompt prefix),
    the model only forwards candidate ``[suffix ‖ target]`` tokens, attending back to the
    cached prefix representations. This eliminates redundant attention and MLP projections
    over the prefix across all candidates.

    Parameters
    ----------
    model : PreTrainedModel
    candidates : Tensor (n_candidates, seq_len)
    token_slices : TokenSlices
        Provides ``loss_slice`` and ``target_slice``.
    batch_size : int
        Process candidates in chunks to avoid OOM.
    prefix_cache : DynamicCache or tuple, optional
        Precomputed batch-1 KV cache for the static prompt prefix tokens.

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

    if prefix_cache is not None:
        p_len = token_slices.suffix_slice.start
        s_len = token_slices.suffix_slice.stop - p_len
        t_len = token_slices.target_slice.stop - token_slices.target_slice.start
        # Suffix tokens are at [0, s_len); target tokens are at [s_len, s_len + t_len).
        # Causal prediction: logit at (s_len - 1) predicts target token 0.
        cache_loss_slice = slice(s_len - 1, s_len + t_len - 1)

        for start in range(0, n_candidates, batch_size):
            end = min(start + batch_size, n_candidates)
            cur_b = end - start

            # Forward only the mutable suffix and target tokens
            batch_suffix_target = candidates[start:end, p_len:].to(device)
            batch_kv = expand_kv_cache(prefix_cache, cur_b)

            logits = model(
                input_ids=batch_suffix_target,
                past_key_values=batch_kv,
                use_cache=True,
            ).logits

            loss_logits = logits[:, cache_loss_slice, :]
            targets = target_ids.unsqueeze(0).expand(cur_b, -1)
            loss = F.cross_entropy(loss_logits.transpose(1, 2), targets, reduction="none").mean(dim=1)
            all_losses.append(loss)
    else:
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
        filter_non_ascii: bool = True,
        use_prefix_cache: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.n_candidates = n_candidates
        self.batch_size = batch_size
        self.filter_non_ascii = filter_non_ascii
        self.use_prefix_cache = use_prefix_cache

        # Freeze model weights so backward only computes gradients for input tokens.
        for p in self.model.parameters():
            p.requires_grad = False

        # Precompute vocabulary mask to filter non-ASCII and special tokens
        device = next(self.model.parameters()).device
        vocab_size = self.model.get_input_embeddings().weight.shape[0]
        self.filter_mask = (
            get_non_ascii_mask(self.tokenizer, vocab_size=vocab_size, device=device)
            if self.filter_non_ascii
            else None
        )

    @torch.no_grad()
    def compute_prefix_cache(self, prompt: str) -> Any:
        """Precompute the key-value cache for the static prompt prefix tokens."""
        device = next(self.model.parameters()).device
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            return None
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = self.model(input_ids=input_ids, use_cache=True)
        return out.past_key_values

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
        prefix_cache: Any = None,
    ) -> Tuple[str, float]:
        """Execute one GCG iteration and return ``(updated_suffix, best_loss)``."""
        if prefix_cache is None and self.use_prefix_cache:
            prefix_cache = self.compute_prefix_cache(prompt)

        ts = find_token_slices(prompt, suffix, target, self.tokenizer)
        token_grads = get_token_gradients(self.model, ts, filter_mask=self.filter_mask)
        candidates = sample_candidates(
            current_ids=ts.input_ids,
            token_gradients=token_grads,
            suffix_slice=ts.suffix_slice,
            top_k=self.top_k,
            n_candidates=self.n_candidates,
        )
        best_idx, best_loss = evaluate_candidates(
            self.model,
            candidates,
            ts,
            batch_size=self.batch_size,
            prefix_cache=prefix_cache,
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

        prefix_cache = self.compute_prefix_cache(prompt) if self.use_prefix_cache else None

        for step_num in range(1, n_steps + 1):
            step_start = time.perf_counter()
            suffix, best_loss = self.step(prompt, suffix, target, prefix_cache=prefix_cache)
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

