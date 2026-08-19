"""配置加载：YAML + ${ENV:-default} 环境变量覆盖（避免写死机器绝对路径）。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def resolve_env(value: Any) -> Any:
    """递归解析 ${VAR:-default} 形式的值。"""
    if isinstance(value, dict):
        return {k: resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env(v) for v in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda m: os.environ.get(m.group(1)) or m.group(2),
            value,
        )
    return value


def load_yaml_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """加载 YAML 配置并递归解析环境变量覆盖。"""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"YAML 顶层必须是映射，实际为 {type(raw).__name__}: {path}")
    return resolve_env(raw)
