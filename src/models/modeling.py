"""完整二分类推理候选模型。

由于原模型类和训练代码暂缺，本模块实现的是"可枚举、可替换、明确标注为候选"
的前向，而不是武断认定某种 pooling。标签映射未知时，输出字段固定为
``class_0_* / class_1_*``，禁止提前命名为 negative/positive。

该候选由冻结的 BERT backbone + 从 checkpoint 加载的原 ``fc`` 头组成，
可通过 ``pooling`` 参数选择 CLS / pooler / masked-mean。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .checkpoint import load_state_dict_safe, sha256_file, strip_prefix
from .pooling import apply_pooling

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

if TYPE_CHECKING:
    from transformers import BertModel

NUM_CLASSES = 2


def checkpoint_hash(path: str, manifest_path: Optional[str] = None) -> str:
    """返回 checkpoint 文件的 SHA-256，优先复用 manifest 中已记录的哈希。

    只在**当前文件与 manifest 记录一致**时复用（大小 + 修改时间都匹配），
    否则重新计算。避免 checkpoint 被替换后仍用旧的 manifest 哈希去标注
    新加载的权重（audit trail 与缓存版本键会因此失真）。

    manifest 缺省路径为 ``metadata/model_manifest.json``。
    """
    if manifest_path is None:
        manifest_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "metadata", "model_manifest.json",
        )
    if os.path.isfile(path) and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            checkpoint_meta = meta.get("checkpoint", {})
            stat = os.stat(path)
            h = checkpoint_meta.get("sha256")
            size = checkpoint_meta.get("size_bytes")
            mtime_ns = checkpoint_meta.get("mtime_ns")
            if h and size == stat.st_size and mtime_ns == stat.st_mtime_ns:
                return str(h)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return sha256_file(path)


@dataclass
class CandidateOutput:
    """完整推理候选的统一输出。

    字段固定为 class_0/1（标签映射未知时不得改名为 positive/negative）。
    """

    logits: torch.Tensor               # [batch, 2]
    probabilities: torch.Tensor        # [batch, 2]
    pooled_feature: torch.Tensor       # [batch, hidden]
    pooling: str
    pooling_confirmed: bool
    model_hash: str
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    text_ids: Optional[List[str]] = None

    # ---- 固定命名输出（禁止 positive/negative）----
    @property
    def class_0_logit(self) -> torch.Tensor:
        return self.logits[:, 0]

    @property
    def class_1_logit(self) -> torch.Tensor:
        return self.logits[:, 1]

    @property
    def class_0_prob(self) -> torch.Tensor:
        return self.probabilities[:, 0]

    @property
    def class_1_prob(self) -> torch.Tensor:
        return self.probabilities[:, 1]

    @property
    def logit_margin_1_minus_0(self) -> torch.Tensor:
        return self.logits[:, 1] - self.logits[:, 0]


class BinaryClassificationCandidate(nn.Module):
    """冻结 backbone + 原 fc 头的二分类推理候选。

    参数来源：
    - backbone：checkpoint 中 ``bert.*`` 键，去掉前缀后 strict 加载进 ``BertModel``；
    - ``fc``：checkpoint 中的 ``fc.weight [2,hidden]`` 与 ``fc.bias [2]``。
    """

    def __init__(
        self,
        bert: "BertModel",
        fc: nn.Linear,
        pooling: str,
        pooling_confirmed: bool = False,
        model_hash: str = "unknown",
    ):
        super().__init__()
        self.bert = bert
        self.fc = fc
        self.pooling = pooling
        self.pooling_confirmed = pooling_confirmed
        self.model_hash = model_hash
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """backbone 恒冻结（无数据阶段不允许更新 backbone 权重）。"""
        self.bert.requires_grad_(False)

    def apply_frozen_head(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对预先计算的单层 hidden state 复用当前冻结 pooling/fc 路径。

        Layer Probe 主协议把 ``pooling`` 锁定为 ``cls``。保留统一方法是为了让
        normal forward 与逐层 projection 共享完全相同的实现，并使 Layer 12
        数值一致性成为真正的代码路径检查。
        """

        pooled = apply_pooling(
            self.pooling,
            hidden_states,
            attention_mask=attention_mask,
            pooler=self.bert.pooler,
        )
        return pooled, self.fc(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> CandidateOutput:
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
        )
        last_hidden = out.last_hidden_state
        pooled, logits = self.apply_frozen_head(
            last_hidden, attention_mask=attention_mask
        )
        probabilities = torch.softmax(logits, dim=-1)
        return CandidateOutput(
            logits=logits,
            probabilities=probabilities,
            pooled_feature=pooled,
            pooling=self.pooling,
            pooling_confirmed=self.pooling_confirmed,
            model_hash=self.model_hash,
            hidden_states=out.hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


def load_backbone(base_model_dir: str, attn_implementation: str = "eager") -> "BertModel":
    """从本地配置构造 BertModel；权重随后由项目 checkpoint 严格加载。

    attn_implementation: "eager"(默认, hook 兼容) | "sdpa"(纯前向加速, A100 走 flash attention)
    """
    from transformers import BertConfig, BertModel

    config = BertConfig.from_pretrained(base_model_dir, local_files_only=True)
    config._attn_implementation = attn_implementation
    config.return_dict = True
    return BertModel(config)


def build_candidate(
    base_model_dir: str,
    checkpoint_path: str,
    pooling: str = "cls",
    pooling_confirmed: bool = False,
    model_hash: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    attn_implementation: str = "eager",
) -> BinaryClassificationCandidate:
    """构建完整二分类推理候选。

    - backbone 以 strict=True 从 ckpt 的 ``bert.*`` 键加载；
    - fc 从 ckpt 的 ``fc.weight/fc.bias`` 加载。
    - attn_implementation="sdpa" 用于纯前向 CLS 提取(flash attention 加速, 不需要 hook)。
    """
    bert = load_backbone(base_model_dir, attn_implementation=attn_implementation)
    state = strip_prefix(load_state_dict_safe(checkpoint_path, map_location="cpu"))
    backbone = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
    if not backbone:
        raise KeyError("checkpoint 中没有 bert.* backbone 权重")
    bert.load_state_dict(backbone, strict=True)

    if "fc.weight" not in state or "fc.bias" not in state:
        raise KeyError("checkpoint 缺少 fc.weight / fc.bias，无法构建二分类候选")
    weight = state["fc.weight"]
    bias = state["fc.bias"]
    expected_weight_shape = (NUM_CLASSES, bert.config.hidden_size)
    if tuple(weight.shape) != expected_weight_shape or tuple(bias.shape) != (NUM_CLASSES,):
        raise ValueError(
            "checkpoint 分类头 shape 不匹配: "
            f"fc.weight={tuple(weight.shape)}（期望 {expected_weight_shape}）, "
            f"fc.bias={tuple(bias.shape)}（期望 {(NUM_CLASSES,)}）"
        )
    fc = nn.Linear(bert.config.hidden_size, NUM_CLASSES)
    with torch.no_grad():
        fc.weight.copy_(weight)
        fc.bias.copy_(bias)

    if model_hash is None:
        model_hash = checkpoint_hash(checkpoint_path)

    model = BinaryClassificationCandidate(
        bert, fc, pooling=pooling, pooling_confirmed=pooling_confirmed, model_hash=model_hash,
    )
    return model.to(device=device, dtype=dtype).eval()
