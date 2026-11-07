import pytest

from scripts.run_epoch_study import SEEDS, SETTINGS, freeze_selection


def _validation():
    return {
        name: {
            str(seed): {"dev": {"accuracy": score, "loss": 0.4}}
            for seed in SEEDS
        }
        for name, score in zip(SETTINGS, (0.8, 0.81, 0.82, 0.83, 0.825))
    }


def test_selection_uses_all_matched_validation_seeds():
    selection = freeze_selection(_validation())
    assert selection["stack_epoch"] == 6
    assert selection["seeds"] == list(SEEDS)


def test_selection_rejects_missing_seeds():
    validation = _validation()
    del validation["stack8"]["42"]
    with pytest.raises(ValueError, match="seeds"):
        freeze_selection(validation)
