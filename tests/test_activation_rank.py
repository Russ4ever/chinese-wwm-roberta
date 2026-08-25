import ast
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

from src.activation_rank import (
    AUXILIARY_STREAMS,
    CORE_SITES,
    PRIMARY_STREAM,
    ActivationMomentConsumer,
    BertActivationHooks,
    OnlineMoments,
    ScalarMoments,
    _atomic_npz,
    _length_bucketed_batches,
    _load_auxiliary_state,
    _norm_audit,
    _precision_comparison,
    _publish_auxiliary_state,
    _publish_primary_shard,
    _stable_hash,
    _validate_primary_shard,
    activation_sites,
    arrays_to_moments,
    collapse_evidence,
    compute_dtype_epsilon,
    covariance_eigendecomposition,
    load_activation_rank_config,
    moments_to_arrays,
    output_projection_covariance,
    paired_bootstrap_log_ratio_ci,
    rank_metrics_from_eigenvalues,
    relative_frobenius_error,
    scalar_moment_maps_to_state,
    select_activation_batch_size,
    select_compute_dtype,
    select_norm_calibration,
    wo_mechanism_records,
)


ROOT = Path(__file__).resolve().parents[1]


def _site_moments(dimension: int = 3, seed: int = 11):
    rng = np.random.default_rng(seed)
    return {
        site: OnlineMoments.from_array(rng.normal(size=(12, dimension)))
        for site in CORE_SITES
    }


def _empty_norm_state():
    return scalar_moment_maps_to_state(
        {
            population: {site: ScalarMoments() for site in CORE_SITES}
            for population in ("token", "cls")
        }
    )


def test_protocol_config_and_49_sites_are_label_free():
    config = load_activation_rank_config(ROOT / "configs" / "activation_rank.yaml")
    assert len(CORE_SITES) == 49
    assert CORE_SITES == activation_sites(12)
    assert CORE_SITES[0] == "residual_00"
    assert CORE_SITES[-1] == "mlp_output_12"
    assert config["sampling"]["token_checkpoints"] == [
        500000,
        1000000,
        2000000,
        5000000,
        10000000,
    ]
    assert config["sampling"]["hash_seed"] == "checkpoint_activation_rank_v1"
    assert config["experiment"]["run_id"] == "financial_reports_v2"
    assert config["output"]["run_directory"].endswith("financial_reports_v2")
    serialized = json.dumps(config).lower()
    for forbidden in (
        "sentiment_label",
        "continuous_targets",
        "return_probe",
        "strict_test",
    ):
        assert forbidden not in serialized


def test_online_centered_moments_match_direct_covariance_and_shard_merge():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(301, 9)) + np.linspace(-3, 4, 9)
    direct = OnlineMoments.from_array(values)
    merged = OnlineMoments.empty(9)
    for piece in np.array_split(values, 8):
        merged.merge(OnlineMoments.from_array(piece))
    np.testing.assert_allclose(merged.mean, direct.mean, rtol=0, atol=1e-13)
    np.testing.assert_allclose(merged.m2, direct.m2, rtol=2e-14, atol=2e-12)
    np.testing.assert_allclose(
        merged.covariance(),
        np.cov(values, rowvar=False, ddof=1),
        rtol=2e-14,
        atol=2e-14,
    )


def test_moment_npz_roundtrip_and_known_low_rank_spectrum(tmp_path: Path):
    rng = np.random.default_rng(19)
    values = rng.normal(size=(400, 3)) @ rng.normal(size=(3, 10))
    moment = OnlineMoments.from_array(values)
    arrays = moments_to_arrays({PRIMARY_STREAM: {"residual_00": moment}})
    path = tmp_path / "moments.npz"
    _atomic_npz(path, arrays)
    with np.load(path, allow_pickle=False) as archive:
        restored = arrays_to_moments({name: archive[name] for name in archive.files})
    recovered = restored[PRIMARY_STREAM]["residual_00"]
    np.testing.assert_allclose(recovered.m2, moment.m2)

    eigenvalues, _ = covariance_eigendecomposition(moment)
    metrics = rank_metrics_from_eigenvalues(eigenvalues)
    direct_singular = np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
    probabilities = direct_singular / direct_singular.sum()
    direct_erank = np.exp(-(probabilities * np.log(probabilities)).sum())
    assert metrics["effective_rank"] == pytest.approx(direct_erank, rel=1e-10)
    assert metrics["k99_variance"] <= 3


def test_special_tokens_are_excluded_and_cls_and_unique_are_separate():
    thresholds = {
        f"{population}__{site}": 1e9
        for population in ("token", "cls")
        for site in CORE_SITES
    }
    consumer = ActivationMomentConsumer(
        hidden_size=3,
        device="cpu",
        thresholds=thresholds,
    )
    # Ordinary positions: row 0 -> 1,2; row 1 -> 1,2. CLS(0), SEP(3), PAD excluded.
    token_mask = torch.tensor([[False, True, True, False], [False, True, True, False]])
    consumer.begin_batch(token_mask=token_mask, unique_rows=torch.tensor([True, False]))
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    consumer("residual_00", tensor)
    assert consumer.streams[PRIMARY_STREAM]["residual_00"].count == 4
    assert consumer.streams["token_natural_unfiltered"]["residual_00"].count == 4
    assert consumer.streams["token_unique_filtered"]["residual_00"].count == 2
    assert consumer.streams["cls_natural_filtered"]["residual_00"].count == 2
    assert consumer.streams["cls_natural_unfiltered"]["residual_00"].count == 2


class _ToySelf(torch.nn.Module):
    def forward(self, hidden_states, **_kwargs):
        return (hidden_states * 1.01,)


class _ToyAttentionOutput(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.dense = torch.nn.Linear(hidden, hidden)
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, hidden_states, residual):
        return self.norm(self.dense(hidden_states) + residual)


class _ToyAttention(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self = _ToySelf()
        self.output = _ToyAttentionOutput(hidden)

    def forward(self, hidden_states):
        context = self.self(hidden_states)[0]
        return (self.output(context, hidden_states),)


class _ToyOutput(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.dense = torch.nn.Linear(hidden, hidden)
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, intermediate, residual):
        return self.norm(self.dense(intermediate) + residual)


class _ToyLayer(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attention = _ToyAttention(hidden)
        self.intermediate = torch.nn.Linear(hidden, hidden)
        self.output = _ToyOutput(hidden)

    def forward(self, hidden_states):
        attention = self.attention(hidden_states)[0]
        intermediate = torch.tanh(self.intermediate(attention))
        return (self.output(intermediate, attention),)


class _ToyEmbeddings(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, hidden)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class _ToyBert(torch.nn.Module):
    def __init__(self, layers=2, hidden=4):
        super().__init__()
        self.embeddings = _ToyEmbeddings(hidden)
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList(
            [_ToyLayer(hidden) for _ in range(layers)]
        )

    def forward(self, input_ids):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)[0]
        return hidden


def test_toy_bert_hooks_capture_exact_sites_once_with_correct_shapes():
    model = _ToyBert(layers=2, hidden=4).eval().requires_grad_(False)
    seen = {}

    def consume(site, tensor):
        assert site not in seen
        seen[site] = tuple(tensor.shape)

    with BertActivationHooks(model, consume, expected_layers=2) as hooks:
        output = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))
        hooks.assert_complete_forward()
    assert output.shape == (2, 3, 4)
    assert set(seen) == set(activation_sites(2))
    assert set(seen.values()) == {(2, 3, 4)}


def test_toy_hook_values_are_pre_residual_z_attention_and_mlp_writes():
    model = _ToyBert(layers=1, hidden=4).eval().requires_grad_(False)
    captured = {}

    def consume(site, tensor):
        captured[site] = tensor.detach().clone()

    with BertActivationHooks(model, consume, expected_layers=1) as hooks:
        model(torch.tensor([[1, 2, 3]]))
        hooks.assert_complete_forward()
    layer = model.encoder.layer[0]
    np.testing.assert_allclose(
        captured["attention_output_01"],
        layer.attention.output.dense(captured["z_01"]).detach(),
        rtol=1e-6,
        atol=1e-6,
    )
    attention_residual = layer.attention.output.norm(
        captured["attention_output_01"] + captured["residual_00"]
    )
    intermediate = torch.tanh(layer.intermediate(attention_residual))
    np.testing.assert_allclose(
        captured["mlp_output_01"],
        layer.output.dense(intermediate).detach(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        captured["residual_01"],
        layer.output.norm(captured["mlp_output_01"] + attention_residual).detach(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_output_projection_covariance_identity_and_direction_decomposition():
    rng = np.random.default_rng(31)
    z = rng.normal(size=(3000, 5)) @ np.diag([3.0, 2.0, 1.0, 0.5, 0.2])
    weight = rng.normal(size=(5, 5))
    bias = rng.normal(size=5)
    o = z @ weight.T + bias
    z_moment = OnlineMoments.from_array(z)
    o_moment = OnlineMoments.from_array(o)
    predicted = output_projection_covariance(weight, z_moment.covariance())
    assert relative_frobenius_error(o_moment.covariance(), predicted) < 1e-12
    records, summary = wo_mechanism_records(
        layer=1,
        weight=weight,
        z_moment=z_moment,
        o_moment=o_moment,
        seed=9,
    )
    assert summary["covariance_relative_frobenius_error"] < 1e-12
    np.testing.assert_allclose(
        [record["factor_product"] for record in records],
        [record["attention_output_eigenvalue"] for record in records],
        rtol=1e-10,
        atol=1e-10,
    )


def test_rank_collapse_decision_boundaries():
    assert (
        collapse_evidence(
            ratio_ci_upper_z=0.89,
            ratio_ci_upper_mlp=0.88,
            ratio_ci_upper_residual=0.87,
            attention_k99=45,
            z_k99=55,
            mlp_k99=60,
            residual_k99=70,
        )
        == "clear_collapse"
    )
    assert (
        collapse_evidence(
            ratio_ci_upper_z=0.95,
            ratio_ci_upper_mlp=0.96,
            ratio_ci_upper_residual=0.97,
            attention_k99=54,
            z_k99=55,
            mlp_k99=60,
            residual_k99=70,
        )
        == "mild_compression"
    )
    assert (
        collapse_evidence(
            ratio_ci_upper_z=1.01,
            ratio_ci_upper_mlp=0.90,
            ratio_ci_upper_residual=0.90,
            attention_k99=40,
            z_k99=55,
            mlp_k99=60,
            residual_k99=70,
        )
        == "not_found_or_inconclusive"
    )


def test_paired_log_ratio_bootstrap_uses_shards_and_is_deterministic():
    numerator = np.arange(1, 9, dtype=float)
    denominator = numerator * 2
    first = paired_bootstrap_log_ratio_ci(numerator, denominator, samples=1000, seed=8)
    second = paired_bootstrap_log_ratio_ci(numerator, denominator, samples=1000, seed=8)
    assert first == second
    assert first == pytest.approx((0.5, 0.5, 0.5))


def test_dtype_and_batch_selection_obey_accuracy_and_headroom():
    dtype_records = [
        {"dtype": "float32", "passed": True, "valid_tokens_per_second": 100},
        {"dtype": "float16", "passed": False, "valid_tokens_per_second": 300},
        {"dtype": "bfloat16", "passed": True, "valid_tokens_per_second": 180},
    ]
    assert select_compute_dtype(dtype_records) == "bfloat16"
    assert (
        select_activation_batch_size(
            [
                {
                    "batch_size": 64,
                    "status": "ok",
                    "headroom_fraction": 0.30,
                    "valid_tokens_per_second": 100,
                },
                {
                    "batch_size": 128,
                    "status": "ok",
                    "headroom_fraction": 0.16,
                    "valid_tokens_per_second": 130,
                },
                {
                    "batch_size": 256,
                    "status": "insufficient_headroom",
                    "headroom_fraction": 0.10,
                    "valid_tokens_per_second": 160,
                },
            ]
        )
        == 128
    )


def test_selected_compute_dtype_also_controls_norm_calibration():
    fp32 = {
        population: {
            site: ScalarMoments.from_array(np.array([1.0, 1.0])) for site in CORE_SITES
        }
        for population in ("token", "cls")
    }
    fp16 = {
        population: {
            site: ScalarMoments.from_array(np.array([2.0, 2.1])) for site in CORE_SITES
        }
        for population in ("token", "cls")
    }
    selected = select_norm_calibration({"float32": fp32, "float16": fp16}, "float16")
    assert selected is fp16
    assert compute_dtype_epsilon("float16") == pytest.approx(2.0**-10)
    assert compute_dtype_epsilon("bfloat16") == pytest.approx(2.0**-7)
    with pytest.raises(KeyError, match="selected dtype"):
        select_norm_calibration({"float32": fp32}, "float16")


def _scalar_with_mean_std(mean: float, std: float, count: int) -> ScalarMoments:
    return ScalarMoments(
        count=count,
        mean=mean,
        m2=std * std * (count - 1),
        minimum=mean - std,
        maximum=mean + std,
    )


def test_norm_audit_is_scale_aware_per_site_and_cls_is_nonblocking():
    norms = {
        population: {
            site: _scalar_with_mean_std(10.0, 0.5, 10000) for site in CORE_SITES
        }
        for population in ("token", "cls")
    }
    pilot = {
        f"{population}__{site}": {
            "count": 2000,
            "mean": 10.0,
            "std": 0.5,
        }
        for population in ("token", "cls")
        for site in CORE_SITES
    }
    filtered = {
        f"{population}__{site}": 0
        for population in ("token", "cls")
        for site in CORE_SITES
    }
    policy = {
        "blocking_populations": ["token"],
        "mean_relative_shift_tolerance": 0.01,
        "std_relative_shift_tolerance": 0.01,
        "maximum_filtered_fraction": 0.001,
        "std_denominator_floor": "compute_dtype_epsilon_times_abs_mean",
    }

    pilot["token__residual_00"]["std"] = 1e-8
    norms["token"]["residual_00"] = _scalar_with_mean_std(10.0, 8e-5, 10000)
    filtered["cls__residual_01"] = 5000
    summary, table = _norm_audit(
        norms,
        filtered,
        pilot,
        compute_dtype="float16",
        policy=policy,
    )
    assert len(table) == 2 * len(CORE_SITES)
    assert summary["passed"] is True
    assert summary["token_passed"] is True
    assert summary["cls_passed"] is False
    near_constant = table[
        table["population"].eq("token") & table["site"].eq("residual_00")
    ].iloc[0]
    assert near_constant["std_denominator"] == pytest.approx(
        compute_dtype_epsilon("float16") * 10.0
    )
    assert bool(near_constant["std_passed"])

    filtered["token__residual_02"] = 100
    failed, failed_table = _norm_audit(
        norms,
        filtered,
        pilot,
        compute_dtype="float16",
        policy=policy,
    )
    assert failed["passed"] is False
    assert failed["failed_blocking_position_count"] == 1
    assert failed["failed_blocking_positions"][0]["site"] == "residual_02"
    failed_row = failed_table[
        failed_table["population"].eq("token") & failed_table["site"].eq("residual_02")
    ].iloc[0]
    assert failed_row["filtered_fraction"] == pytest.approx(0.01)


def test_precision_comparison_detects_matching_and_distorted_spectra():
    reference = _site_moments(dimension=4)
    matching = {site: value.copy() for site, value in reference.items()}
    assert _precision_comparison(reference, matching)["passed"] is True
    distorted = {}
    for site, value in reference.items():
        changed = value.copy()
        changed.m2 = changed.m2.copy()
        changed.m2[0] *= 100
        changed.m2[:, 0] *= 100
        distorted[site] = changed
    assert _precision_comparison(reference, distorted)["passed"] is False


def test_hash_sharding_and_length_bucket_batches_are_reproducible():
    ids = ["r3", "r1", "r2", "r4"]
    first = [_stable_hash(value, "seed") for value in ids]
    second = [_stable_hash(value, "seed") for value in ids]
    assert first == second
    assert first != [_stable_hash(value, "other") for value in ids]
    frame = pd.DataFrame(
        {
            "report_id": ids,
            "text": ids,
            "stable_order": first,
            "token_count": [500, 20, 120, 250],
            "is_unique_text": [True] * 4,
        }
    )
    batches = list(_length_bucketed_batches(frame, batch_size=2, max_length=512))
    assert sum(len(batch) for batch in batches) == 4
    assert all(
        batch["token_count"].max() <= 2 * max(batch["token_count"].min(), 64)
        for batch in batches
    )


def test_primary_atomic_publication_fingerprint_and_corruption_detection(
    tmp_path: Path,
):
    identity = {"run_fingerprint": "run-a", "checkpoint_sha256": "ckpt"}
    execution = {"execution_fingerprint": "exec-a", "selected_compute_dtype": "float32"}
    primary = _site_moments(dimension=3)
    filtered = {
        f"{population}__{site}": 0
        for population in ("token", "cls")
        for site in CORE_SITES
    }
    output = _publish_primary_shard(
        tmp_path,
        identity,
        execution,
        shard=0,
        primary=primary,
        processed_rows=12,
        processed_tokens=100,
        checkpoints=[{"checkpoint_tokens": 100, "site": "residual_00"}],
        norm_audit={"passed": True},
        norm_states=_empty_norm_state(),
        filtered_counts=filtered,
    )
    manifest = _validate_primary_shard(tmp_path, identity, 0, execution)
    assert manifest["processed_tokens"] == 100
    assert (tmp_path / "moments" / "shard_00.npz").samefile(
        output / "primary_moments.npz"
    )
    with pytest.raises(RuntimeError, match="fingerprint"):
        _validate_primary_shard(
            tmp_path, identity, 0, {"execution_fingerprint": "other"}
        )
    public = tmp_path / "moments" / "shard_00.npz"
    public.unlink()
    public.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="hardlink"):
        _validate_primary_shard(tmp_path, identity, 0, execution)


def test_auxiliary_segment_ledger_makes_retry_idempotent(tmp_path: Path):
    identity = {"run_fingerprint": "run-a"}
    execution = {"execution_fingerprint": "exec-a", "selected_compute_dtype": "float32"}
    delta = {
        stream: _site_moments(dimension=2, seed=index + 1)
        for index, stream in enumerate(AUXILIARY_STREAMS)
    }
    _publish_auxiliary_state(
        tmp_path, identity, execution, {}, delta, segment="shard_00_through_10"
    )
    existing, manifest = _load_auxiliary_state(tmp_path, identity, execution)
    before = existing[AUXILIARY_STREAMS[0]]["residual_00"].count
    _publish_auxiliary_state(
        tmp_path,
        identity,
        execution,
        existing,
        delta,
        segment="shard_00_through_10",
    )
    after, retry_manifest = _load_auxiliary_state(tmp_path, identity, execution)
    assert after[AUXILIARY_STREAMS[0]]["residual_00"].count == before
    assert manifest["included_segments"] == retry_manifest["included_segments"]


def test_notebook_has_exactly_eight_orchestration_cells_and_is_ast_valid():
    notebook_path = ROOT / "notebooks" / "activation_rank_pipeline.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    assert len(notebook.cells) == 8
    assert [cell.cell_type for cell in notebook.cells] == ["code"] * 8
    trees = [ast.parse(cell.source) for cell in notebook.cells]
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for tree in trees
        for node in ast.walk(tree)
    )
    assert "run_rank_shards_stage(config)" in notebook.cells[4].source
    assert "run_wo_mechanism_stage(config)" in notebook.cells[6].source
    assert all(
        cell.execution_count is None and not cell.outputs for cell in notebook.cells
    )
