import sys
from pathlib import Path

import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_jachess_pilot as pilot


def test_reduced_jachess_estimator_returns_finite_layer_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "OUT", tmp_path)
    config = DebertaV2Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, vocab_size=32, num_labels=2,
        max_position_embeddings=32,
    )
    model = DebertaV2ForSequenceClassification(config).eval()
    examples = [
        {"input_ids": [1, 2, 3], "labels": 0},
        {"input_ids": [4, 5, 6, 7, 8], "labels": 1},
    ]
    embeddings, attention_mask = pilot.prepare_embeddings(
        model, examples, pad_id=0, device=torch.device("cpu")
    )
    assert embeddings.shape[:2] == (2, 5)
    assert attention_mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

    rows = pilot.estimate_layer_norms(model, embeddings, attention_mask, 1, 1)

    assert len(rows) == 1
    assert rows[0]["layer"] == 1
    assert rows[0]["jacobian_frobenius_sq_estimate"] >= 0
    assert rows[0]["hessian_frobenius_sq_estimate"] >= 0
    assert rows[0]["elapsed_seconds"] > 0