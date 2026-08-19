import pytest

from src.label_utils import classify_five, transition_class, trimmed_mean


def test_transition_class_handles_loss_direction_and_sign_changes():
    assert transition_class(-100, -60, "loss", "loss", None) == "loss_narrowing"
    assert transition_class(-100, -130, "loss", "loss", None) == "loss_widening"
    assert transition_class(-100, 50, "loss", "profit", None) == "turn_to_profit"
    assert transition_class(100, -50, "profit", "loss", None) == "turn_to_loss"


def test_classify_five_validates_threshold_order():
    with pytest.raises(ValueError):
        classify_five(0.1, flat_threshold=0.2, strong_threshold=0.1)


def test_trimmed_mean_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        trimmed_mean([1, 2, 3], fraction=0.5)
