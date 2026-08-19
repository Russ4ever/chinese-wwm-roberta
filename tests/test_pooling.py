import pytest
import torch

from src.models.pooling import apply_pooling, masked_mean


def test_masked_mean_excludes_padding_and_handles_all_padding():
    hidden = torch.tensor([[[1.0], [3.0]], [[9.0], [7.0]]])
    mask = torch.tensor([[1, 0], [0, 0]])
    actual = masked_mean(hidden, mask)
    assert torch.equal(actual, torch.tensor([[1.0], [0.0]]))


def test_apply_pooling_rejects_missing_mask_and_unknown_mode():
    hidden = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="attention_mask"):
        apply_pooling("masked_mean", hidden)
    with pytest.raises(ValueError, match="未知 pooling"):
        apply_pooling("invalid", hidden)

