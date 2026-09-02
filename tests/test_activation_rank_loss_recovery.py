from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.activation_rank_loss_recovery import (
    SITE_KINDS,
    SingleSiteProjection,
    load_loss_recovery_policy,
    select_disjoint_evaluation_reports,
    summarize_loss_recovery,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_policy():
    policy = deepcopy(
        load_loss_recovery_policy(
            ROOT / "configs" / "activation_rank_loss_recovery.yaml"
        )
    )
    policy["evaluation"]["layers"] = [1, 2]
    policy["evaluation"]["kinds"] = ["attention_output", "mlp_output"]
    policy["projection"]["components"] = [0, 384, 768]
    policy["identifiability"]["bootstrap_samples"] = 300
    return policy


def _synthetic_distances(policy):
    """Generate synthetic per-report CLS distances for two layers × two kinds."""
    rows = []
    report_ids = ["r1", "r2", "r3", "r4"]
    layers = [int(v) for v in policy["evaluation"]["layers"]]
    kinds = [str(v) for v in policy["evaluation"]["kinds"]]
    components = [int(v) for v in policy["projection"]["components"]]

    for layer in layers:
        for kind in kinds:
            site = f"{kind}_{layer:02d}"
            # mlp_output has a near-zero ablation delta (non-identifiable)
            delta = 0.0001 if kind == "mlp_output" else 1.0
            for index, report_id in enumerate(report_ids):
                base = 0.5 + index * 0.1
                for n_components, recovered in ((0, 0.0), (384, 1.0), (768, 1.0)):
                    dist = delta * (1.0 - recovered) + base * 0.0001
                    rows.append(
                        {
                            "condition": "projection",
                            "site": site,
                            "layer": int(layer),
                            "kind": kind,
                            "n_components": int(n_components),
                            "report_id": str(report_id),
                            "representation_distance": float(dist),
                        }
                    )
    return pd.DataFrame(rows)


def test_policy_is_multi_layer_and_has_identifiability_gate():
    policy = load_loss_recovery_policy(
        ROOT / "configs" / "activation_rank_loss_recovery.yaml"
    )
    assert len(policy["evaluation"]["layers"]) == 12
    assert policy["evaluation"]["layers"][0] == 1
    assert policy["evaluation"]["layers"][-1] == 12
    assert "attention_output" in policy["evaluation"]["kinds"]
    assert policy["projection"]["components"][0] == 0
    assert policy["projection"]["components"][-1] == 768
    assert policy["identifiability"]["minimum_ablation_delta"] >= 0
    assert "mask_seeds" not in policy["evaluation"]
    assert "mask_probability" not in policy["evaluation"]


def test_evaluation_selection_is_deterministic_and_disjoint_by_id_and_text_hash():
    reports = pd.DataFrame(
        {
            "report_id": ["a", "b", "c", "d", "e"],
            "text": ["A", "B", "C", "D", "E"],
            "text_sha256": ["ha", "hb", "hc", "hd", "he"],
        }
    )
    sample = pd.DataFrame({"report_id": ["a", "x"], "text_sha256": ["unused", "hb"]})
    first = select_disjoint_evaluation_reports(
        reports, sample, report_count=2, order_seed="fixed"
    )
    second = select_disjoint_evaluation_reports(
        reports, sample, report_count=2, order_seed="fixed"
    )
    pd.testing.assert_frame_equal(first, second)
    assert not set(first["report_id"]).intersection({"a", "x"})
    assert not set(first["text_sha256"]).intersection({"unused", "hb"})


def test_small_denominator_is_non_identifiable_and_never_clipped():
    policy = _small_policy()
    metrics, summary = summarize_loss_recovery(_synthetic_distances(policy), policy)

    # mlp_output at any layer should be non-identifiable (tiny delta)
    for layer in policy["evaluation"]["layers"]:
        site = f"mlp_output_{layer:02d}"
        mlp_summary = summary.set_index("site").loc[site]
        assert mlp_summary["status"] == "non_identifiable"
        mlp_metrics = metrics.loc[metrics["site"].eq(site)]
        assert mlp_metrics["recovery_fraction"].isna().all()

    # attention_output should be identifiable
    for layer in policy["evaluation"]["layers"]:
        site = f"attention_output_{layer:02d}"
        attn_summary = summary.set_index("site").loc[site]
        assert attn_summary["status"] == "identifiable"
        assert attn_summary["n_sustained_recovery"] == 384

    assert bool(summary["full_projection_passed"].all())


def test_projection_hook_is_single_site_and_removed_after_exception():
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.eye(2))
    values = torch.tensor([[2.0, -1.0]])
    basis = np.eye(2, dtype=np.float32)
    with pytest.raises(RuntimeError, match="boom"):
        with SingleSiteProjection(module, basis, 0):
            assert torch.equal(module(values), torch.zeros_like(values))
            raise RuntimeError("boom")
    torch.testing.assert_close(module(values), values)
