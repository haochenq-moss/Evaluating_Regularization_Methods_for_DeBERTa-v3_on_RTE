"""Regularizers behind one shared loss interface.

All four methods operate on token embeddings / soft targets / logits, so they compose:

    ids -> embeddings -> [mixup] -> [NEFTune noise] -> encoder (1 or 2 dropout passes)
        -> soft-target CE (label smoothing applied to the soft targets)
        -> [+ R-Drop symmetric KL]
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def smooth_targets(targets: torch.Tensor, eps: float) -> torch.Tensor:
    """(1-eps)*y + eps/K for (possibly already soft) targets of shape [B, K]."""
    if eps <= 0:
        return targets
    return (1.0 - eps) * targets + eps / targets.size(-1)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean over batch of -sum_k t_k log softmax(logits)_k."""
    return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


def mix_batch(emb, targets, mask, lam: float, perm: torch.Tensor):
    """Embedding mixup. Row i is mixed with row perm[i]; embeddings, soft labels and
    attention masks all use the SAME `perm` and `lam`, which guarantees alignment.
    The mixed mask is the union so that tokens from either source stay visible."""
    emb_m = lam * emb + (1.0 - lam) * emb[perm]
    tgt_m = lam * targets + (1.0 - lam) * targets[perm]
    mask_m = torch.maximum(mask, mask[perm])
    return emb_m, tgt_m, mask_m


def neftune_noise(emb: torch.Tensor, mask: torch.Tensor, alpha: float) -> torch.Tensor:
    """Add U(-1,1) * alpha / sqrt(L*d) noise, L = real (unpadded) length of each sequence,
    d = embedding size. Padding positions receive no noise. Training-only (caller's job)."""
    d = emb.size(-1)
    length = mask.sum(dim=1).clamp(min=1).to(emb.dtype)            # [B]
    scale = alpha / torch.sqrt(length * d)                         # [B]
    noise = (torch.rand_like(emb) * 2.0 - 1.0) * scale[:, None, None]
    return emb + noise * mask[..., None].to(emb.dtype)


def symmetric_kl(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """0.5 * (KL(p1 || p2) + KL(p2 || p1)); each KL is summed over classes and averaged
    over the batch ('batchmean'). KL(P||Q) = sum_k P_k (log P_k - log Q_k)."""
    lp1 = F.log_softmax(logits1.float(), dim=-1)
    lp2 = F.log_softmax(logits2.float(), dim=-1)
    # F.kl_div(input=log Q, target=log P, log_target=True) == KL(P || Q)
    kl_12 = F.kl_div(lp2, lp1, reduction="batchmean", log_target=True)
    kl_21 = F.kl_div(lp1, lp2, reduction="batchmean", log_target=True)
    return 0.5 * (kl_12 + kl_21)


def training_loss(model, batch: dict, reg: dict[str, float], num_labels: int = 2):
    """Compute the regularised training loss.

    reg keys (all optional): label_smoothing (eps), mixup (alpha), rdrop (coef), neftune (alpha).
    Returns (loss, logits_of_first_pass, n_forward_passes).
    """
    input_ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    emb = model.get_input_embeddings()(input_ids)
    targets = F.one_hot(labels, num_labels).to(emb.dtype)

    if reg.get("mixup", 0) > 0:
        a = float(reg["mixup"])
        lam = float(torch.distributions.Beta(a, a).sample())
        perm = torch.randperm(input_ids.size(0), device=input_ids.device)
        emb, targets, mask = mix_batch(emb, targets, mask, lam, perm)

    if reg.get("neftune", 0) > 0:
        # sampled once; both R-Drop passes see identical inputs, differing only in dropout
        emb = neftune_noise(emb, mask, float(reg["neftune"]))

    targets = smooth_targets(targets, float(reg.get("label_smoothing", 0.0)))

    def forward():
        return model(inputs_embeds=emb, attention_mask=mask).logits

    logits1 = forward()
    coef = float(reg.get("rdrop", 0.0))
    if coef > 0:
        logits2 = forward()  # independent dropout masks
        ce = 0.5 * (soft_cross_entropy(logits1, targets) + soft_cross_entropy(logits2, targets))
        loss = ce + coef * symmetric_kl(logits1, logits2)
        return loss, logits1, 2
    return soft_cross_entropy(logits1, targets), logits1, 1
