"""三种 pooling 的统一接口（P1）。

实现 CLS / pooler / masked-mean 三种 pooling，均可对 batch 输出 ``[batch, 768]``。

- ``cls``:        ``hidden_states[:, 0]``
- ``pooler``:     BERT 的 ``pooler_output``（``tanh(dense(CLS))``），可作用任意层 CLS
- ``masked_mean``: 只对 ``attention_mask == 1`` 的 token 求均值

``make_pooling_fn`` 返回 ``(hidden_states, attention_mask) -> [batch, hidden]``
的可调用对象，供 modeling / heads / early_exit 复用。
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


def cls_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    """取 [CLS] 位置的表示。不受右侧 padding 影响。"""
    return hidden_states[:, 0]


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """只对 attention_mask == 1 的 token 求均值。

    reference (TODO §3.1):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def pooler_pool(hidden_states: torch.Tensor, pooler: nn.Module) -> torch.Tensor:
    """用模型自带的 BERT pooler（dense + tanh）对给定层 hidden state 池化。

    说明：BERT pooler 本身定义为 ``tanh(dense(hidden[:, 0]))``，因此可以作用在
    任意层的 hidden state 上；它是在最后一层 CLS 上训练的，作用到浅层属于
    零样本诊断，不代表浅层 pooler 语义。
    """
    return pooler(hidden_states)


def apply_pooling(
    pooling: str,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pooler: Optional[nn.Module] = None,
) -> torch.Tensor:
    """按名称应用 pooling，输出 ``[batch, hidden]``。

    Raises:
        ValueError: pooling 名称未知或缺少必需输入（pooler / attention_mask）。
    """
    if pooling == "cls":
        return cls_pool(hidden_states)
    if pooling == "pooler":
        if pooler is None:
            raise ValueError("pooling='pooler' 需要传入 pooler 模块")
        return pooler_pool(hidden_states, pooler)
    if pooling == "masked_mean":
        if attention_mask is None:
            raise ValueError("pooling='masked_mean' 需要传入 attention_mask")
        return masked_mean(hidden_states, attention_mask)
    raise ValueError(f"未知 pooling: {pooling!r}（可选 cls / pooler / masked_mean）")


def make_pooling_fn(
    pooling: str, pooler: Optional[nn.Module] = None
) -> Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    """构造 ``(hidden_states, attention_mask) -> [batch, hidden]`` 的池化函数。"""
    def fn(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return apply_pooling(pooling, hidden_states, attention_mask=attention_mask, pooler=pooler)

    return fn
