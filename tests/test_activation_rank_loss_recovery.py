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
    policy["evaluation"]["mask_seeds"] = [11, 13]
    policy["projection"]["components"] = [0, 384, 768]
    policy["identifiability"]["bootstrap_samples"] = 300
    return policy


def _synthetic_losses(policy):
    rows = []
    report_ids = ["r1", "r2", "r3", "r4"]
    seeds = policy["evaluation"]["mask_seeds"]
    for seed in seeds:
        for index, report_id in enumerate(report_ids):
            baseline = 2.0 + index * 0.05 + (seed - 11) * 0.001
            rows.append(
                {
                    "condition": "original",
                    "site": "original",
                    "n_components": -1,
                    "mask_seed": seed,
                    "report_id": report_id,
                    "ce_sum": baseline * 10,
                    "mask_count": 10,
                }
            )
            for kind in SITE_KINDS:
                site = f"{kind}_06"
                delta = 0.0005 if kind == "mlp_output" else 1.0
                for n_components, recovered in ((0, 0.0), (384, 1.0), (768, 1.0)):
                    loss = baseline + delta * (1.0 - recovered)
                    rows.append(
                        {
                            "condition": "projection",
                            "site": site,
                            "n_components": n_components,
                            "mask_seed": seed,
                            "report_id": report_id,
                            "ce_sum": loss * 10,
                            "mask_count": 10,
                        }
                    )
    return pd.DataFrame(rows)


def test_policy_is_single_layer_multi_seed_and_has_identifiability_gate():
    policy = load_loss_recovery_policy(
        ROOT / "configs" / "activation_rank_loss_recovery.yaml"
    )
    assert policy["evaluation"]["layer"] == 6
    assert len(policy["evaluation"]["mask_seeds"]) >= 2
    assert policy["projection"]["components"][0] == 0
    assert policy["projection"]["components"][-1] == 768
    assert policy["identifiability"]["minimum_ablation_delta"] > 0


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
    metrics, summary = summarize_loss_recovery(_synthetic_losses(policy), policy)
    mlp = summary.set_index("site").loc["mlp_output_06"]
    assert mlp["ablation_delta"] == pytest.approx(0.0005)
    assert mlp["status"] == "non_identifiable"
    mlp_metrics = metrics.loc[metrics["site"].eq("mlp_output_06")]
    assert mlp_metrics["recovery_fraction"].isna().all()
    attention = summary.set_index("site").loc["attention_output_06"]
    assert attention["status"] == "identifiable"
    assert attention["n_sustained_recovery"] == 384
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
