"""
HotFlip adversarial attack implementation.

Reference: Ebrahimi et al., "HotFlip: White-Box Adversarial Examples for
Text Classification", ACL 2018.

HotFlip uses directional derivatives with respect to the one-hot input
representations to greedily identify the single token substitution (coordinate)
that yields the steepest first-order Taylor approximation decrease in loss:

    Delta L_{i, j} ~ <grad_{e_i} L, e_j - e_i> = <grad_{e_i} L, e_j> - <grad_{e_i} L, e_i>

This module supports two optimization targets:
  1. Sequence classification flip (e.g. guardrail/sentiment models).
  2. Autoregressive causal LM suffix forcing (1st-order HotFlip baseline vs GCG).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.utils import get_non_ascii_mask


# ---------------------------------------------------------------------------
# Classification Gradient & Projection
# ---------------------------------------------------------------------------
@torch.enable_grad()
def get_classifier_gradients(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    target_label: int,
    filter_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Compute 1st-order Taylor delta for all token replacements on a classifier.

    Parameters
    ----------
    model : PreTrainedModel
        Sequence classification model.
    input_ids : Tensor of shape (seq_len,)
        Input token sequence.
    target_label : int
        Target class index to steer the prediction toward.
    filter_mask : Tensor, optional
        Boolean tensor of shape (vocab_size,) marking tokens to exclude (+inf).

    Returns
    -------
    delta_loss : Tensor of shape (seq_len, vocab_size)
        First-order estimated change in loss for substituting token j at position i.
    logits : Tensor of shape (num_classes,)
        Pre-softmax classification logits before perturbation.
    current_loss : float
        Current cross-entropy loss against target_label.
    """
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()
    embed_weights = embed_layer.weight                  # (vocab_size, d_model)

    ids = input_ids.to(device)
    inputs_embeds = embed_layer(ids.unsqueeze(0)).detach().requires_grad_(True)

    outputs = model(inputs_embeds=inputs_embeds)
    logits = outputs.logits[0]                          # (num_classes,)

    target_tensor = torch.tensor([target_label], dtype=torch.long, device=device)
    loss = F.cross_entropy(logits.unsqueeze(0), target_tensor)
    loss_val = loss.item()
    loss.backward()

    # grad: (seq_len, d_model)
    grad = inputs_embeds.grad[0]

    # Projections:
    # 1. <grad_i, e_j> for all vocab j -> (seq_len, vocab_size)
    proj = grad @ embed_weights.T

    # 2. <grad_i, e_{x_i}> for the current token at position i -> (seq_len, 1)
    current_tokens = ids.unsqueeze(1)                   # (seq_len, 1)
    current_embeds = embed_weights[ids]                 # (seq_len, d_model)
    current_proj = (grad * current_embeds).sum(dim=-1, keepdim=True)

    # First-order Taylor approximation: Delta L = <grad, e_j> - <grad, e_{x_i}>
    delta_loss = proj - current_proj

    # Mask self-tokens: HotFlip must replace x_i with a different token (j != x_i)
    delta_loss[torch.arange(ids.shape[0], device=device), ids] = float("inf")

    if filter_mask is not None:
        delta_loss[:, filter_mask.to(device)] = float("inf")

    model.zero_grad(set_to_none=True)
    return delta_loss, logits.detach(), loss_val


# ---------------------------------------------------------------------------
# HotFlip Attack Class
# ---------------------------------------------------------------------------
class HotFlipAttack:
    """HotFlip discrete gradient optimization attack.

    Executes 1st-order greedy token flips for sequence classifiers
    and causal language models.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        filter_non_ascii: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.filter_non_ascii = filter_non_ascii

        # Freeze model weights
        for p in self.model.parameters():
            p.requires_grad = False

        device = next(self.model.parameters()).device
        vocab_size = self.model.get_input_embeddings().weight.shape[0]
        self.filter_mask = (
            get_non_ascii_mask(self.tokenizer, vocab_size=vocab_size, device=device)
            if self.filter_non_ascii
            else None
        )

    # ---- Classifier Attack -------------------------------------------------

    def step_classifier(
        self,
        input_ids: torch.Tensor,
        target_label: int,
        mutable_slice: Optional[slice] = None,
    ) -> Tuple[torch.Tensor, int, float, Tuple[int, int, int]]:
        """Perform one greedy token flip on a sequence classifier.

        Parameters
        ----------
        input_ids : Tensor of shape (seq_len,)
        target_label : int
        mutable_slice : slice, optional
            Restricts positions that can be mutated (e.g. exclude CLS/SEP).
            Defaults to slice(1, -1) if length > 2, else full slice.

        Returns
        -------
        updated_ids : Tensor of shape (seq_len,)
        pred_label : int
        current_loss : float
        flip_info : Tuple[int, int, int]
            (position, old_token_id, new_token_id)
        """
        seq_len = input_ids.shape[0]
        if mutable_slice is None:
            mutable_slice = slice(1, max(1, seq_len - 1)) if seq_len > 2 else slice(0, seq_len)

        delta_loss, logits, current_loss = get_classifier_gradients(
            self.model, input_ids, target_label, filter_mask=self.filter_mask
        )

        # Slice mutable positions
        sub_delta = delta_loss[mutable_slice]           # (num_mutable, vocab_size)

        # Find global minimum substitution across all mutable positions and vocab
        flat_idx = sub_delta.argmin().item()
        vocab_size = delta_loss.shape[1]
        rel_pos = flat_idx // vocab_size
        new_token_id = flat_idx % vocab_size

        # Map relative position back to full sequence index
        start_pos = mutable_slice.start if mutable_slice.start is not None else 0
        best_pos = start_pos + rel_pos
        old_token_id = input_ids[best_pos].item()

        updated_ids = input_ids.clone()
        updated_ids[best_pos] = new_token_id

        pred_label = logits.argmax().item()
        return updated_ids, pred_label, current_loss, (best_pos, old_token_id, new_token_id)

    def run_classifier(
        self,
        text: str,
        target_label: int,
        max_steps: int = 10,
    ) -> Dict[str, Any]:
        """Iteratively apply HotFlip to flip the classification label of text.

        Parameters
        ----------
        text : str
            Input prompt or sentence to perturb.
        target_label : int
            Target class index to achieve.
        max_steps : int
            Maximum number of token flips.

        Returns
        -------
        result : Dict[str, Any]
            Dictionary containing original/perturbed texts, label history, and status.
        """
        device = next(self.model.parameters()).device
        encoding = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoding["input_ids"][0].to(device)

        # Initial prediction
        with torch.no_grad():
            init_logits = self.model(input_ids.unsqueeze(0)).logits[0]
            init_pred = init_logits.argmax().item()

        id2label = getattr(self.model.config, "id2label", {})
        init_label_name = id2label.get(init_pred, str(init_pred))
        target_label_name = id2label.get(target_label, str(target_label))

        print(f"\n[HotFlip Classifier Attack]")
        print(f"Original Text  : {text!r}")
        print(f"Initial Class  : {init_label_name} ({init_pred})")
        print(f"Target Class   : {target_label_name} ({target_label})")
        print(f"Max Token Flips: {max_steps}\n")

        current_ids = input_ids
        flips = []
        success = False

        for step in range(1, max_steps + 1):
            t0 = time.perf_counter()
            current_ids, _, _, (pos, old_tok, new_tok) = self.step_classifier(
                current_ids, target_label
            )
            step_time = (time.perf_counter() - t0) * 1000

            # Evaluate post-flip prediction and loss
            with torch.no_grad():
                post_logits = self.model(current_ids.unsqueeze(0)).logits[0]
                target_tensor = torch.tensor([target_label], dtype=torch.long, device=device)
                post_loss = F.cross_entropy(post_logits.unsqueeze(0), target_tensor).item()
                post_pred = post_logits.argmax().item()

            old_str = self.tokenizer.decode([old_tok])
            new_str = self.tokenizer.decode([new_tok])
            post_label_name = id2label.get(post_pred, str(post_pred))

            print(
                f"Step {step:2d} | Post-Loss: {post_loss:.4f} | Pred: {post_label_name} "
                f"| Flip @ {pos}: {old_str!r} -> {new_str!r} ({step_time:.1f}ms)"
            )
            flips.append((pos, old_str, new_str))

            if post_pred == target_label:
                print(f"\n[Success] Classification flipped to {target_label_name} at step {step}!")
                success = True
                break

        perturbed_text = self.tokenizer.decode(current_ids, skip_special_tokens=True)
        final_label = post_pred if 'post_pred' in locals() else init_pred
        return {
            "success": success,
            "original_text": text,
            "perturbed_text": perturbed_text,
            "initial_label": init_pred,
            "final_label": final_label,
            "flips": flips,
            "steps": len(flips),
        }

    # ---- Causal LM Attack --------------------------------------------------

    def step_causal_lm(
        self,
        prompt: str,
        suffix: str,
        target: str,
        suffix_ids: torch.Tensor | None = None,
        return_ids: bool = False,
    ) -> Tuple[str, float] | Tuple[str, float, torch.Tensor]:
        """Perform one 1st-order greedy HotFlip step on a causal LM suffix.

        Unlike GCG, which samples candidates and evaluates batched forward passes,
        HotFlip directly swaps the single suffix token with the steepest
        first-order gradient projection score (0 candidate forward passes).
        """
        from src.gcg import TokenSlices, find_token_slices, get_token_gradients

        device = next(self.model.parameters()).device
        if suffix_ids is not None:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            p_len = len(prompt_ids)
            s_len = len(suffix_ids)
            t_len = len(target_ids)
            full_ids = torch.tensor(prompt_ids + suffix_ids.tolist() + target_ids, dtype=torch.long)
            ts = TokenSlices(
                input_ids=full_ids,
                suffix_slice=slice(p_len, p_len + s_len),
                target_slice=slice(p_len + s_len, p_len + s_len + t_len),
                loss_slice=slice(p_len + s_len - 1, p_len + s_len + t_len - 1),
            )
            s_ids = suffix_ids.to(device)
        else:
            ts = find_token_slices(prompt, suffix, target, self.tokenizer)
            s_ids = ts.input_ids[ts.suffix_slice].to(device)

        token_grads = get_token_gradients(self.model, ts, filter_mask=self.filter_mask)
        suffix_len = s_ids.shape[0]

        # Baseline subtraction: Delta L ~ <grad_i, e_j - e_{x_i}>
        current_scores = token_grads[torch.arange(suffix_len, device=device), s_ids].unsqueeze(1)
        delta_scores = token_grads - current_scores

        # Mask current tokens: j != x_i (force an actual token substitution)
        delta_scores[torch.arange(suffix_len, device=device), s_ids] = float("inf")

        flat_argmin = delta_scores.argmin().item()
        vocab_size = token_grads.shape[1]
        best_suffix_pos = flat_argmin // vocab_size
        best_new_token = flat_argmin % vocab_size

        new_suffix_ids = s_ids.clone()
        new_suffix_ids[best_suffix_pos] = best_new_token

        new_suffix = self.tokenizer.decode(new_suffix_ids, skip_special_tokens=True)
        est_delta = delta_scores[best_suffix_pos, best_new_token].item()
        if return_ids:
            return new_suffix, est_delta, new_suffix_ids.cpu()
        return new_suffix, est_delta
