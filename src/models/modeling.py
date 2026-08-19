"""完整二分类推理候选模型（P1）。

由于原模型类和训练代码暂缺，本模块实现的是"可枚举、可替换、明确标注为候选"
的前向，而不是武断认定某种 pooling。标签映射未知时，输出字段固定为
``class_0_* / class_1_*``，禁止提前命名为 negative/positive。

该候选由冻结的 BERT backbone + 从 checkpoint 加载的原 ``fc`` 头组成，
可通过 ``pooling`` 参数选择 CLS / pooler / masked-mean。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BertModel

from .checkpoint import load_state_dict_safe
from .pooling import apply_pooling

P2_IN = 768
P2_OUT = 2


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
            h = meta.get("checkpoint", {}).get("sha256")
            size = meta.get("checkpoint", {}).get("size_bytes")
            if h and size == os.path.getsize(path):
                # 文件大小一致且 manifest 未改动（用 mtime 做二次校验）
                manifest_mtime = os.path.getmtime(manifest_path)
                file_mtime = os.path.getmtime(path)
                if manifest_mtime >= file_mtime:
                    return h
        except Exception:  # noqa: BLE001
            pass
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@dataclass
class CandidateOutput:
    """完整推理候选的统一输出。

    字段固定为 class_0/1（标签映射未知时不得改名为 positive/negative）。
    """

    logits: torch.Tensor               # [batch, 2]
    probabilities: torch.Tensor        # [batch, 2]
    pooled_feature: torch.Tensor       # [batch, 768]
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
    - ``fc``：checkpoint 中的 ``fc.weight [2,768]`` 与 ``fc.bias [2]``。
    """

    def __init__(
        self,
        bert: BertModel,
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
        pooled = apply_pooling(self.pooling, last_hidden, attention_mask=attention_mask,
                               pooler=self.bert.pooler)
        logits = self.fc(pooled)
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


def load_backbone(base_model_dir: str) -> BertModel:
    """从本地底座目录构造 BertModel（local_files_only，eager attention）。"""
    return BertModel.from_pretrained(base_model_dir, local_files_only=True, attn_implementation="eager")


def build_candidate(
    base_model_dir: str,
    checkpoint_path: str,
    pooling: str = "cls",
    pooling_confirmed: bool = False,
    model_hash: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BinaryClassificationCandidate:
    """构建完整二分类推理候选。

    - backbone 以 strict=True 从 ckpt 的 ``bert.*`` 键加载（P0 已证明 100% 覆盖）；
    - fc 从 ckpt 的 ``fc.weight/fc.bias`` 加载。
    """
    bert = load_backbone(base_model_dir)
    state = load_state_dict_safe(checkpoint_path, map_location="cpu")
    backbone = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
    bert.load_state_dict(backbone, strict=True)

    if "fc.weight" not in state or "fc.bias" not in state:
        raise KeyError("checkpoint 缺少 fc.weight / fc.bias，无法构建二分类候选")
    fc = nn.Linear(P2_IN, P2_OUT)
    with torch.no_grad():
        fc.weight.copy_(state["fc.weight"].to(dtype))
        fc.bias.copy_(state["fc.bias"].to(dtype))

    if model_hash is None:
        model_hash = checkpoint_hash(checkpoint_path)

    model = BinaryClassificationCandidate(
        bert, fc, pooling=pooling, pooling_confirmed=pooling_confirmed, model_hash=model_hash,
    )
    return model.to(device=device, dtype=dtype).eval()


def tokenize_texts(
    texts: List[str],
    tokenizer,
    max_length: int = 128,
    truncation: bool = True,
    padding: bool = True,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """组批 tokenize，返回可直接喂给候选模型的 dict。"""
    return tokenizer(
        texts,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )


def load_tokenizer(base_model_dir: str):
    """加载本地 tokenizer。

    注意：transformers 5.x 中本目录无 fast tokenizer，``BertTokenizerFast`` 会静默
    回退为 slow ``BertTokenizer``，行为等价，这里显式使用 slow 类。
    """
    from transformers import BertTokenizer

    return BertTokenizer.from_pretrained(base_model_dir, local_files_only=True)
