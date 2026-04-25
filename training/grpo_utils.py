"""Manual GRPO gradient update for custom environment rollouts.

TRL's GRPOTrainer expects a HuggingFace Dataset of prompts and handles
generation internally — that design conflicts with our multi-turn environment
rollout where the environment must be stepped between generations.

This module implements the GRPO objective directly so training scripts can
collect (prompt, completion, reward) tuples from any rollout function and
apply a weight update in one call.

GRPO objective (Shao et al., 2024):
    For each prompt p with G sampled completions {c_1..c_G}:
        advantage_i = (r_i - mean(r)) / (std(r) + eps)
    Loss = -mean( advantage_i * log_prob(c_i | p) )

Usage:
    optimizer = grpo_setup_optimizer(model, lr=5e-6)
    for step in training_loop:
        prompts, completions, rewards = collect_batch(...)
        loss = grpo_step(model, tokenizer, optimizer, prompts, completions, rewards)
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch


def grpo_setup_optimizer(model, lr: float = 5e-6):
    """Return an AdamW optimizer over the model's trainable parameters."""
    from torch.optim import AdamW
    trainable = [p for p in model.parameters() if p.requires_grad]
    return AdamW(trainable, lr=lr, eps=1e-8)


def grpo_step(
    model,
    tokenizer,
    optimizer,
    prompts: List[str],
    completions: List[str],
    rewards: List[float],
    num_generations: int = 4,
    eps: float = 1e-8,
    max_grad_norm: float = 1.0,
    kl_coeff: float = 0.0,
) -> float:
    """Perform one GRPO update step.

    Args:
        model:           The trainable LLM (LoRA adapter active).
        tokenizer:       Corresponding tokenizer.
        optimizer:       AdamW optimizer wrapping model parameters.
        prompts:         List of prompt strings (length B).
        completions:     List of completion strings (length B).
        rewards:         List of scalar rewards (length B).
        num_generations: Group size for advantage normalisation.
                         Samples are grouped in windows of this size.
                         Pass 1 to normalise over the entire batch.
        eps:             Denominator epsilon for advantage normalisation.
        max_grad_norm:   Gradient clipping norm.
        kl_coeff:        KL penalty coefficient (0 = pure GRPO, no ref model).

    Returns:
        loss_value (float) — the scalar loss for logging.
    """
    if not prompts:
        return 0.0

    model.train()
    device = next(model.parameters()).device

    rewards_t = torch.tensor(rewards, dtype=torch.float32)

    # --- Advantage normalisation (group-relative) ---
    # Group samples into windows of num_generations; normalise within each group.
    advantages = torch.zeros_like(rewards_t)
    group_size = max(num_generations, 1)
    for start in range(0, len(rewards_t), group_size):
        end  = min(start + group_size, len(rewards_t))
        grp  = rewards_t[start:end]
        mean = grp.mean()
        std  = grp.std() if len(grp) > 1 else torch.tensor(1.0)
        advantages[start:end] = (grp - mean) / (std + eps)

    advantages = advantages.to(device)

    # --- Compute log-probabilities of completions under the policy ---
    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for prompt, completion, adv in zip(prompts, completions, advantages):
        full_text = prompt + completion
        enc = tokenizer(full_text, return_tensors="pt", truncation=True,
                        max_length=512).to(device)
        prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=512).to(device)
        prompt_len = prompt_enc["input_ids"].shape[1]

        if enc["input_ids"].shape[1] <= prompt_len:
            continue  # empty completion after tokenisation

        logits = model(**enc).logits  # (1, seq_len, vocab)

        # Log-prob of completion tokens only (everything after prompt_len)
        completion_ids   = enc["input_ids"][0, prompt_len:]
        completion_logits= logits[0, prompt_len - 1 : -1, :]   # shifted by 1

        log_probs = torch.nn.functional.log_softmax(completion_logits, dim=-1)
        token_log_probs = log_probs[
            torch.arange(len(completion_ids), device=device), completion_ids
        ]
        mean_log_prob = token_log_probs.mean()

        total_loss = total_loss + (-adv * mean_log_prob)
        n_valid   += 1

    if n_valid == 0:
        return 0.0

    loss = total_loss / n_valid

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_grad_norm
    )
    optimizer.step()

    return loss.item()
