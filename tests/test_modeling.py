from pathlib import Path

import torch
from src.models.modeling import build_candidate
from transformers import BertConfig, BertModel


def test_build_candidate_loads_complete_checkpoint_and_runs_forward(tmp_path: Path):
    config = BertConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
    base_dir = tmp_path / "base"
    config.save_pretrained(base_dir)

    backbone = BertModel(config)
    state = {f"bert.{key}": value for key, value in backbone.state_dict().items()}
    state["fc.weight"] = torch.randn(2, config.hidden_size)
    state["fc.bias"] = torch.randn(2)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": state}, checkpoint)

    model = build_candidate(
        str(base_dir),
        str(checkpoint),
        pooling="cls",
        model_hash="test-hash",
    )
    output = model(
        torch.tensor([[1, 2, 0]]),
        attention_mask=torch.tensor([[1, 1, 0]]),
    )

    assert output.logits.shape == (1, 2)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(1))
    assert not any(parameter.requires_grad for parameter in model.bert.parameters())
