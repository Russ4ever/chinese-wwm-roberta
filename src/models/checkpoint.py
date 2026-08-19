"""checkpoint 安全解包、前缀处理与键报告。

本模块只包含纯函数与安全的加载/比较工具，不依赖 notebook 隐藏状态。

设计要点
--------
- ``unwrap_state_dict`` / ``strip_prefix`` 均为纯函数，返回新对象，不修改入参。
- ``load_state_dict_safe`` 优先 ``weights_only=True`` 反序列化，避免任意 pickle 执行。
- ``match_state_dicts`` 输出 matched / missing / unexpected / shape_mismatch，
  并给出 tensor 覆盖率与参数覆盖率，便于验收。
"""
from __future__ import annotations

import hashlib
import inspect
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch


# --------------------------------------------------------------------------- #
# 安全加载
# --------------------------------------------------------------------------- #
def load_state_dict_safe(path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """加载 state dict，优先 weights_only=True（PyTorch 版本允许时）。

    只应加载可信来源的 checkpoint 文件；weights_only 禁止反序列化任意 Python 对象。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 文件不存在: {path}")
    if "weights_only" in inspect.signature(torch.load).parameters:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    else:  # pragma: no cover - 仅兼容不支持 weights_only 的旧版 PyTorch
        obj = torch.load(path, map_location=map_location)

    state = unwrap_state_dict(obj)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint 解包后不是 state dict: {type(state).__name__}")
    invalid = [
        key for key, value in state.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if invalid:
        raise TypeError(f"checkpoint 含非张量 state 项: {invalid[:5]}")
    return dict(state)


# --------------------------------------------------------------------------- #
# 解包与前缀处理（纯函数）
# --------------------------------------------------------------------------- #
def unwrap_state_dict(obj: Mapping) -> Mapping:
    """解开常见的 state dict 包装层（如 {"state_dict": {...}}）。

    循环内重复解包以应对嵌套包装；不修改入参。
    """
    while isinstance(obj, Mapping) and any(k in obj for k in ("state_dict", "model_state_dict")):
        for key in ("state_dict", "model_state_dict"):
            if isinstance(obj, Mapping) and key in obj:
                obj = obj[key]
    return obj


def strip_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefixes: Sequence[str] = ("module.", "model."),
) -> Dict[str, torch.Tensor]:
    """移除键的常见前缀（如 DDP 的 ``module.``、训练代码可能加的 ``model.``）。

    返回**新 dict**，不修改入参。只删除出现在键首的前缀；剩余键按原顺序保留。
    """
    if any(not prefix for prefix in prefixes):
        raise ValueError("prefixes 不得包含空字符串")

    out: Dict[str, torch.Tensor] = {}
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break
        if key in out:
            raise ValueError(
                f"移除前缀后键冲突: {original_key!r} 与其他键都映射到 {key!r}"
            )
        out[key] = value
    return out


# --------------------------------------------------------------------------- #
# 键匹配报告
# --------------------------------------------------------------------------- #
def match_state_dicts(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    """比较 candidate 与 reference 两组键，输出详细匹配报告。

    - matched: 键存在且 shape 一致
    - missing: 在 reference 但不在 candidate
    - unexpected: 在 candidate 但不在 reference
    - shape_mismatch: 键都在但 shape 不一致

    tensor 覆盖率 = len(matched) / len(reference)
    参数覆盖率   = sum(numel(matched reference 张量)) / sum(numel(reference))

    返回的字典可直接序列化为 JSON/CSV 落盘。
    """
    candidate_keys = set(candidate.keys())
    reference_keys = set(reference.keys())

    matched_keys: List[str] = []
    missing_keys: List[str] = sorted(reference_keys - candidate_keys)
    unexpected_keys: List[str] = sorted(candidate_keys - reference_keys)
    shape_mismatch: List[Dict[str, object]] = []

    ref_numel_total = 0
    ref_numel_matched = 0
    for key in reference_keys:
        ref_t = reference[key]
        ref_numel_total += ref_t.numel()
        if key in candidate_keys:
            cand_t = candidate[key]
            if tuple(ref_t.shape) == tuple(cand_t.shape):
                matched_keys.append(key)
                ref_numel_matched += ref_t.numel()
            else:
                shape_mismatch.append(
                    {
                        "key": key,
                        "reference_shape": list(ref_t.shape),
                        "candidate_shape": list(cand_t.shape),
                    }
                )

    matched_keys.sort()
    tensor_ratio = len(matched_keys) / len(reference_keys) if reference_keys else 1.0
    param_ratio = ref_numel_matched / ref_numel_total if ref_numel_total else 1.0

    return {
        "candidate_key_count": len(candidate_keys),
        "reference_key_count": len(reference_keys),
        "matched_key_count": len(matched_keys),
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "shape_mismatch_count": len(shape_mismatch),
        "matched_keys": matched_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch": shape_mismatch,
        "matched_tensor_ratio": tensor_ratio,
        "matched_param_ratio": param_ratio,
        "matched_param_numel": ref_numel_matched,
        "reference_param_numel": ref_numel_total,
    }


def report_fc(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    """明确记录 fc.weight / fc.bias 的存在性与 shape，而非默认假设。"""
    out: Dict[str, object] = {}
    for key, expected in (("fc.weight", (2, 768)), ("fc.bias", (2,))):
        if key in state_dict:
            t = state_dict[key]
            out[key] = {
                "present": True,
                "shape": list(t.shape),
                "expected_shape": list(expected),
                "shape_matches": tuple(t.shape) == expected,
                "numel": t.numel(),
            }
        else:
            out[key] = {
                "present": False,
                "shape": None,
                "expected_shape": list(expected),
                "shape_matches": False,
                "numel": 0,
            }
    return out


# --------------------------------------------------------------------------- #
# 哈希工具
# --------------------------------------------------------------------------- #
def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 SHA-256，适用于 >400MB 的权重文件。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_tensor(t: torch.Tensor) -> str:
    """对张量字节计算 SHA-256（便于自比较/审计）。

    bfloat16 无 numpy 标量类型，先转 float32 再哈希。
    """
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def param_count(state_dict: Mapping[str, torch.Tensor]) -> int:
    """state dict 的总参数量。"""
    return sum(t.numel() for t in state_dict.values())


def state_dict_diff_metrics(
    a: Mapping[str, torch.Tensor],
    b: Mapping[str, torch.Tensor],
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """对给定键集合计算 base/delta/relative L2 与 cosine。

    相同 state dict 自比较时应得到：所有 delta=0，cos=1。

    Raises:
        ValueError: 没有任何键被实际比较时（keys 为空或全部缺失），避免"空比较
            却返回完美一致"的虚假结果。
    """
    key_list = list(keys) if keys is not None else list(a.keys())
    total_base_l2 = 0.0
    total_delta_l2 = 0.0
    total_abs_delta = 0.0
    total_numel = 0
    max_abs_delta = 0.0
    cosine_sum = 0.0
    counted = 0
    for k in key_list:
        if k not in a or k not in b:
            raise KeyError(f"比较键 {k!r} 未同时出现在两组 state dict 中")
        if tuple(a[k].shape) != tuple(b[k].shape):
            raise ValueError(
                f"比较键 {k!r} shape 不一致: {tuple(a[k].shape)} != {tuple(b[k].shape)}"
            )
        ta = a[k].flatten().double()
        tb = b[k].flatten().double()
        if ta.numel() == 0:
            continue
        delta = ta - tb
        total_delta_l2 += (delta * delta).sum().item()
        total_base_l2 += (tb * tb).sum().item()
        total_abs_delta += delta.abs().sum().item()
        total_numel += tb.numel()
        max_abs_delta = max(max_abs_delta, delta.abs().max().item())
        norm_a = ta.norm().item()
        norm_b = tb.norm().item()
        denom = norm_a * norm_b
        if denom > 0:
            cosine_sum += float((ta @ tb).item() / denom)
        else:
            cosine_sum += 1.0 if norm_a == 0 and norm_b == 0 else 0.0
        counted += 1
    if counted == 0:
        raise ValueError(
            f"state_dict_diff_metrics 没有比较任何非空张量（请求 {len(key_list)} 个）")
    return {
        "delta_l2": total_delta_l2 ** 0.5,
        "base_l2": total_base_l2 ** 0.5,
        "relative_l2": (total_delta_l2 ** 0.5) / (total_base_l2 ** 0.5) if total_base_l2 else float("nan"),
        "mae": total_abs_delta / total_numel if total_numel else float("nan"),
        "max_abs_delta": max_abs_delta,
        "mean_cosine": cosine_sum / counted,
    }
