from pathlib import Path

import pytest
import torch

from src.models.checkpoint import (
    load_state_dict_safe,
    state_dict_diff_metrics,
    strip_prefix,
)


def test_load_state_dict_safe_unwraps_common_checkpoint(tmp_path: Path):
    path = tmp_path / "weights.pt"
    torch.save({"state_dict": {"module.weight": torch.tensor([1.0])}}, path)
    state = load_state_dict_safe(str(path))
    assert list(state) == ["module.weight"]


def test_strip_prefix_handles_nested_prefixes_and_rejects_collisions():
    tensor = torch.tensor([1.0])
    assert list(strip_prefix({"module.model.weight": tensor})) == ["weight"]

    with pytest.raises(ValueError, match="键冲突"):
        strip_prefix({"weight": tensor, "module.weight": tensor})


def test_state_dict_diff_metrics_distinguishes_zero_from_nonzero():
    zero = {"w": torch.zeros(2)}
    one = {"w": torch.ones(2)}
    metrics = state_dict_diff_metrics(zero, one)
    assert metrics["mean_cosine"] == 0.0
    assert metrics["mae"] == 1.0

    identical = state_dict_diff_metrics(zero, zero)
    assert identical["mean_cosine"] == 1.0
    assert identical["delta_l2"] == 0.0


def test_state_dict_diff_metrics_rejects_missing_keys():
    with pytest.raises(KeyError):
        state_dict_diff_metrics({"a": torch.ones(1)}, {}, keys=["a"])

