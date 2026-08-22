from pathlib import Path

import pytest

from src.config import load_yaml_config, resolve_env


def test_resolve_env_uses_default_for_missing_and_empty_values(monkeypatch):
    monkeypatch.delenv("PROJECT_VALUE", raising=False)
    assert resolve_env("${PROJECT_VALUE:-fallback}") == "fallback"

    monkeypatch.setenv("PROJECT_VALUE", "")
    assert resolve_env("${PROJECT_VALUE:-fallback}") == "fallback"

    monkeypatch.setenv("PROJECT_VALUE", "configured")
    assert resolve_env({"x": ["${PROJECT_VALUE:-fallback}"]}) == {"x": ["configured"]}


def test_load_yaml_config_rejects_non_mapping(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("- item\n", encoding="utf-8")
    with pytest.raises(TypeError, match="顶层必须是映射"):
        load_yaml_config(path)


def test_shared_server_thread_budget_is_conservative_and_overridable(monkeypatch):
    config_path = Path(__file__).parents[1] / "configs" / "report_labels.yaml"
    monkeypatch.delenv("SHARED_SERVER_MAX_THREADS", raising=False)
    assert load_yaml_config(config_path)["performance"]["max_threads"] == "8"

    monkeypatch.setenv("SHARED_SERVER_MAX_THREADS", "16")
    assert load_yaml_config(config_path)["performance"]["max_threads"] == "16"
