from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    f1s = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    """Top-label confidence bins. Returns (bin_conf, bin_acc, bin_count) arrays."""
    conf = probs.max(-1)
    correct = (probs.argmax(-1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(float)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    sum_acc = np.bincount(idx, weights=correct, minlength=n_bins)
    nz = count > 0
    bc, ba = np.zeros(n_bins), np.zeros(n_bins)
    bc[nz], ba[nz] = sum_conf[nz] / count[nz], sum_acc[nz] / count[nz]
    return bc, ba, count


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    bc, ba, count = reliability_bins(probs, y, n_bins)
    return float(np.sum(count / count.sum() * np.abs(ba - bc)))


def compute_metrics(logits: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    if not np.isfinite(logits).all():
        raise ValueError("Cannot compute metrics: logits contain NaN or infinity.")
    probs = softmax(logits.astype(np.float64))
    pred = probs.argmax(-1)
    nll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": macro_f1(y, pred, logits.shape[-1]),
        "ece": ece(probs, y, n_bins),
        "loss": float(nll),
        "n": int(len(y)),
    }
