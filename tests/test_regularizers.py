import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from regbench.regularizers import (mix_batch, neftune_noise, smooth_targets, soft_cross_entropy,
                                   symmetric_kl, training_loss)


class Fake(nn.Module):
    """Embedding -> mean-pool -> dropout -> linear; mimics HF interface."""

    def __init__(self, p=0.0, v=50, d=16, k=3):
        super().__init__()
        self.emb, self.drop, self.head = nn.Embedding(v, d), nn.Dropout(p), nn.Linear(d, k)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds, attention_mask):
        m = attention_mask[..., None].float()
        pooled = (inputs_embeds * m).sum(1) / m.sum(1)
        return type("O", (), {"logits": self.head(self.drop(pooled))})


def batch(B=6, L=7, v=50):
    ids = torch.randint(1, v, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    mask[0, 4:] = 0
    return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor([0, 1, 2, 0, 1, 2][:B])}


def test_mixup_alignment():
    torch.manual_seed(0)
    emb = torch.randn(6, 7, 16)
    y = F.one_hot(torch.tensor([0, 1, 2, 0, 1, 2]), 3).float()
    mask = batch()["attention_mask"]
    perm = torch.tensor([3, 2, 1, 0, 5, 4])
    lam = 0.3
    e, t, m = mix_batch(emb, y, mask, lam, perm)
    for i in range(6):
        assert torch.allclose(e[i], lam * emb[i] + (1 - lam) * emb[perm[i]])
        assert torch.allclose(t[i], lam * y[i] + (1 - lam) * y[perm[i]])
        assert torch.equal(m[i], torch.maximum(mask[i], mask[perm[i]]))
    assert torch.allclose(t.sum(-1), torch.ones(6))


def test_mixup_loss_equals_lambda_weighted_ce():
    # Dropout off, so loss == lam*CE(logits,y) + (1-lam)*CE(logits,y[perm]) with the same perm.
    model, b = Fake(0.0), batch()
    torch.manual_seed(1)
    loss, _, _ = training_loss(model, b, {"mixup": 0.4}, num_labels=3)
    torch.manual_seed(1)
    lam = float(torch.distributions.Beta(0.4, 0.4).sample())
    perm = torch.randperm(6)
    emb = model.emb(b["input_ids"])
    e, _, m = mix_batch(emb, F.one_hot(b["labels"], 3).float(), b["attention_mask"], lam, perm)
    logits = model(e, m).logits
    ref = lam * F.cross_entropy(logits, b["labels"]) + (1 - lam) * F.cross_entropy(logits, b["labels"][perm])
    assert torch.allclose(loss, ref, atol=1e-5)


def test_label_smoothing_matches_torch():
    logits, y = torch.randn(8, 3), torch.randint(0, 3, (8,))
    ours = soft_cross_entropy(logits, smooth_targets(F.one_hot(y, 3).float(), 0.1))
    assert torch.allclose(ours, F.cross_entropy(logits, y, label_smoothing=0.1), atol=1e-6)


def test_rdrop_kl_direction_and_reduction():
    a, b = torch.randn(5, 3), torch.randn(5, 3)
    p, q = F.softmax(a, -1), F.softmax(b, -1)
    kl_pq = (p * (p.log() - q.log())).sum(-1).mean()      # KL(p||q), mean over batch (not over classes)
    kl_qp = (q * (q.log() - p.log())).sum(-1).mean()
    assert not torch.allclose(kl_pq, kl_qp)               # asymmetric, so direction is testable
    assert torch.allclose(symmetric_kl(a, b), 0.5 * (kl_pq + kl_qp), atol=1e-6)
    assert torch.allclose(symmetric_kl(a, b), symmetric_kl(b, a), atol=1e-6)
    assert torch.allclose(symmetric_kl(a, a), torch.tensor(0.0), atol=1e-7)


def test_rdrop_loss_composition():
    model, b = Fake(0.5), batch()
    model.train()
    torch.manual_seed(3)
    loss, _, passes = training_loss(model, b, {"rdrop": 4.0}, num_labels=3)
    torch.manual_seed(3)
    emb = model.emb(b["input_ids"])
    l1, l2 = model(emb, b["attention_mask"]).logits, model(emb, b["attention_mask"]).logits
    assert not torch.allclose(l1, l2)  # passes use independent dropout masks
    ref = 0.5 * (F.cross_entropy(l1, b["labels"]) + F.cross_entropy(l2, b["labels"])) + 4.0 * symmetric_kl(l1, l2)
    assert passes == 2 and torch.allclose(loss, ref, atol=1e-5)


def test_neftune_scale_and_padding():
    emb, mask = torch.zeros(4, 10, 64), torch.ones(4, 10, dtype=torch.long)
    mask[0, 5:] = 0
    out = neftune_noise(emb, mask, 10.0)
    assert torch.all(out[0, 5:] == 0)
    for i, L in enumerate([5, 10, 10, 10]):
        bound = 10.0 / math.sqrt(L * 64)
        assert out[i].abs().max() <= bound + 1e-7 and out[i].abs().max() > 0.5 * bound


def test_neftune_training_only_effect():
    model, b = Fake(0.0), batch()
    torch.manual_seed(0)
    l_noise, _, _ = training_loss(model, b, {"neftune": 15.0}, num_labels=3)
    l_clean, _, _ = training_loss(model, b, {}, num_labels=3)
    assert not torch.allclose(l_noise, l_clean)  # eval path (trainer.predict) never calls training_loss


def test_nonfinite_logits_are_not_silently_reported():
    import numpy as np
    import pytest

    from regbench.metrics import compute_metrics

    with pytest.raises(ValueError, match="NaN or infinity"):
        compute_metrics(np.array([[float("nan"), 0.0, 0.0]]), np.array([0]))


def test_stacked_all_methods_backward():
    model, b = Fake(0.1), batch()
    loss, _, passes = training_loss(
        model, b,
        {"label_smoothing": 0.1, "mixup": 0.2, "rdrop": 1.0, "neftune": 5.0},
        num_labels=3,
    )
    loss.backward()
    assert passes == 2 and torch.isfinite(loss) and model.emb.weight.grad is not None
