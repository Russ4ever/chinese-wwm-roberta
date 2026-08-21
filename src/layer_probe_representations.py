"""Layer 0~12 CLS representation 提取与可审计存储。

Notebook 只调用本模块的公开函数；模型加载、冻结检查、批量前向、原分类头输出、
原子写盘和产物校验全部在 Python 中完成。隐藏表示按
``[representation_row, layer, hidden]`` 保存为可 mmap 的 NPY，避免一次载入内存。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .inference_cache import tokenizer_signature
from .models.modeling import build_candidate, checkpoint_hash
from .report_label_economics import canonical_stock_code


REPRESENTATION_FILE = "cls_representations.npy"
METADATA_FILE = "text_metadata.parquet"
HEAD_OUTPUT_FILE = "original_head_outputs.parquet"
MANIFEST_FILE = "representation_manifest.json"


@dataclass(frozen=True)
class RepresentationArtifacts:
    directory: Path
    representations: Path
    metadata: Path
    head_outputs: Path
    manifest: Path


def representation_artifacts(directory: str | Path) -> RepresentationArtifacts:
    root = Path(directory).expanduser().resolve()
    return RepresentationArtifacts(
        directory=root,
        representations=root / REPRESENTATION_FILE,
        metadata=root / METADATA_FILE,
        head_outputs=root / HEAD_OUTPUT_FILE,
        manifest=root / MANIFEST_FILE,
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name}缺少字段: {', '.join(missing)}")


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"文本数据不存在: {source}")
    suffixes = "".join(source.suffixes).lower()
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffixes.endswith(".csv.gz") or source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"仅支持CSV/CSV.GZ/Parquet文本表: {source}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_text_dataset(
    path: str | Path,
    *,
    id_column: str,
    text_column: str,
    symbol_column: str,
    date_column: str,
    sentiment_label_column: str | None = None,
    sentiment_labels_path: str | Path | None = None,
    sentiment_label_id_column: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """读取并规范化 representation 输入，输出固定字段名。

    情绪标签可以已经在文本表中，也可以通过独立表按 text_id 一对一连接。这里不做
    任何 train/test 划分，避免把数据划分隐藏在输入预处理里。
    """

    frame = _read_table(path)
    _require_columns(
        frame,
        [id_column, text_column, symbol_column, date_column],
        "文本输入",
    )
    rename = {
        id_column: "text_id",
        text_column: "text",
        symbol_column: "symbol",
        date_column: "trading_date",
    }
    out = frame.rename(columns=rename).copy()
    out["text_id"] = out["text_id"].astype("string").str.strip()
    if out["text_id"].isna().any() or out["text_id"].eq("").any():
        raise ValueError("文本输入含空text_id")
    if out["text_id"].duplicated().any():
        raise ValueError("文本输入的text_id不唯一")
    out["text"] = out["text"].fillna("").astype(str).str.strip()
    if out["text"].eq("").any():
        raise ValueError(f"文本输入含{int(out['text'].eq('').sum())}条空文本")
    out["symbol"] = canonical_stock_code(out["symbol"])
    if out["symbol"].isna().any():
        raise ValueError(f"文本输入含{int(out['symbol'].isna().sum())}个无效股票代码")
    out["trading_date"] = pd.to_datetime(
        out["trading_date"], errors="coerce"
    ).dt.normalize()
    if out["trading_date"].isna().any():
        raise ValueError(
            f"文本输入含{int(out['trading_date'].isna().sum())}个无效交易日"
        )

    if sentiment_labels_path:
        if not sentiment_label_column:
            raise ValueError(
                "配置sentiment_labels_path时必须配置sentiment_label_column"
            )
        labels = _read_table(sentiment_labels_path)
        label_id = sentiment_label_id_column or id_column
        _require_columns(labels, [label_id, sentiment_label_column], "情绪标签表")
        labels = labels[[label_id, sentiment_label_column]].rename(
            columns={label_id: "text_id", sentiment_label_column: "sentiment_label"}
        )
        labels["text_id"] = labels["text_id"].astype("string").str.strip()
        if labels["text_id"].duplicated().any():
            raise ValueError("情绪标签表的text_id不唯一")
        out = out.drop(columns=["sentiment_label"], errors="ignore").merge(
            labels,
            on="text_id",
            how="left",
            validate="one_to_one",
        )
    elif sentiment_label_column and sentiment_label_column in out.columns:
        out = out.rename(columns={sentiment_label_column: "sentiment_label"})

    if "sentiment_label" in out.columns:
        labels = pd.to_numeric(out["sentiment_label"], errors="coerce")
        labeled = labels.notna()
        if not labels.loc[labeled].isin([0, 1]).all():
            raise ValueError("sentiment_label的非空值必须是0/1")
        out["sentiment_label"] = labels.astype("Int64")

    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit必须为正整数")
        out = out.iloc[: int(limit)].copy()
    if out.empty:
        raise ValueError("representation输入为空")
    out = out.reset_index(drop=True)
    out.insert(0, "representation_row", np.arange(len(out), dtype=np.int64))
    out["text_sha256"] = out["text"].map(_sha256_text)
    return out


def freeze_and_validate_inference_model(
    model: Any, *, expected_hidden_layers: int | None = None
) -> dict[str, object]:
    """冻结整个候选模型、关闭dropout并返回可写入manifest的检查结果。"""

    import torch

    model.eval()
    model.requires_grad_(False)
    parameters = list(model.parameters())
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    dropout_training = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Dropout) and module.training
    ]
    if model.training or trainable or dropout_training:
        raise RuntimeError(
            "representation模型未处于严格推理状态: "
            f"model.training={model.training}, trainable={trainable}, "
            f"dropout_training={dropout_training[:5]}"
        )
    configured_layers = getattr(getattr(model, "bert", None), "config", None)
    configured_layers = getattr(configured_layers, "num_hidden_layers", None)
    if expected_hidden_layers is not None and configured_layers is not None:
        if int(configured_layers) + 1 != int(expected_hidden_layers):
            raise ValueError(
                f"模型配置应输出{expected_hidden_layers}层（含embedding），"
                f"实际为{int(configured_layers) + 1}层"
            )
    return {
        "model_training": bool(model.training),
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameter_count": int(trainable),
        "dropout_modules_in_training_mode": dropout_training,
        "forward_context": "torch.inference_mode",
    }


def stack_layer_cls(hidden_states: Sequence[Any], *, expected_layers: int) -> Any:
    """从一次forward返回的hidden states提取所有层CLS，形状[B,L,H]。"""

    import torch

    if hidden_states is None or len(hidden_states) != expected_layers:
        actual = None if hidden_states is None else len(hidden_states)
        raise ValueError(f"hidden_states层数错误: {actual} != {expected_layers}")
    if any(state.ndim != 3 for state in hidden_states):
        raise ValueError("每层hidden state必须是[batch, tokens, hidden]")
    shapes = {tuple(state.shape) for state in hidden_states}
    if len(shapes) != 1:
        raise ValueError(f"各层hidden state形状不一致: {sorted(shapes)}")
    return torch.stack([state[:, 0, :] for state in hidden_states], dim=1)


def _atomic_replace_directory(temporary: Path, output: Path) -> None:
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    moved = False
    try:
        if output.exists():
            os.replace(output, backup)
            moved = True
        try:
            os.replace(temporary, output)
        except BaseException:
            if moved and backup.exists() and not output.exists():
                os.replace(backup, output)
                moved = False
            raise
        if moved:
            shutil.rmtree(backup)
            moved = False
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if moved and backup.exists() and not output.exists():
            os.replace(backup, output)


def extract_cls_representations(
    texts: pd.DataFrame,
    *,
    model: Any,
    tokenizer: Any,
    output_directory: str | Path,
    batch_size: int,
    max_length: int,
    expected_layers: int = 13,
    storage_dtype: str = "float16",
    device: str = "cpu",
    model_identity: Mapping[str, object] | None = None,
) -> RepresentationArtifacts:
    """一次forward提取全部层CLS并原子写出representation数据集。"""

    import torch

    _require_columns(
        texts,
        ["representation_row", "text_id", "text", "symbol", "trading_date"],
        "representation文本表",
    )
    if not texts["representation_row"].equals(pd.Series(np.arange(len(texts)))):
        raise ValueError("representation_row必须从0连续递增")
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size和max_length必须为正整数")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype必须是float16或float32")

    validation = freeze_and_validate_inference_model(
        model, expected_hidden_layers=expected_layers
    )
    model = model.to(device).eval()
    output = Path(output_directory).expanduser().resolve()
    if output == Path(output.anchor) or len(output.parts) < 3:
        raise ValueError(f"拒绝写入过宽目录: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))

    n_rows = len(texts)
    representation_map: np.memmap | None = None
    head_records: list[pd.DataFrame] = []
    hidden_size: int | None = None
    try:
        with torch.inference_mode():
            for start in range(0, n_rows, batch_size):
                end = min(n_rows, start + batch_size)
                encoded = tokenizer(
                    texts.iloc[start:end]["text"].tolist(),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_attention_mask=True,
                    return_token_type_ids=True,
                )
                inputs = {
                    key: value.to(device)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask", "token_type_ids"}
                }
                candidate = model(**inputs, output_hidden_states=True)
                layer_cls = stack_layer_cls(
                    candidate.hidden_states, expected_layers=expected_layers
                )
                if layer_cls.requires_grad or candidate.logits.requires_grad:
                    raise RuntimeError("inference_mode下输出仍携带梯度")
                if hidden_size is None:
                    hidden_size = int(layer_cls.shape[-1])
                    representation_map = np.lib.format.open_memmap(
                        temporary / REPRESENTATION_FILE,
                        mode="w+",
                        dtype=np.dtype(storage_dtype),
                        shape=(n_rows, expected_layers, hidden_size),
                    )
                elif int(layer_cls.shape[-1]) != hidden_size:
                    raise RuntimeError("不同batch的hidden size不一致")
                assert representation_map is not None
                representation_map[start:end] = (
                    layer_cls.float().cpu().numpy().astype(storage_dtype, copy=False)
                )
                logits = candidate.logits.float().cpu().numpy()
                probabilities = candidate.probabilities.float().cpu().numpy()
                head_records.append(
                    pd.DataFrame(
                        {
                            "representation_row": np.arange(start, end, dtype=np.int64),
                            "text_id": texts.iloc[start:end]["text_id"]
                            .astype(str)
                            .to_numpy(),
                            "class_0_logit": logits[:, 0],
                            "class_1_logit": logits[:, 1],
                            "logit_margin_1_minus_0": logits[:, 1] - logits[:, 0],
                            "class_0_prob": probabilities[:, 0],
                            "class_1_prob": probabilities[:, 1],
                            "token_count": inputs["attention_mask"]
                            .sum(dim=1)
                            .cpu()
                            .numpy(),
                        }
                    )
                )
        if representation_map is None or hidden_size is None:
            raise RuntimeError("没有生成任何representation")
        representation_map.flush()
        del representation_map

        metadata = texts.copy()
        if "text_sha256" not in metadata.columns:
            metadata["text_sha256"] = metadata["text"].map(_sha256_text)
        metadata.to_parquet(temporary / METADATA_FILE, index=False, compression="zstd")
        head = pd.concat(head_records, ignore_index=True)
        if not head["representation_row"].equals(pd.Series(np.arange(n_rows))):
            raise RuntimeError("分类头输出与representation行号不一致")
        head.to_parquet(temporary / HEAD_OUTPUT_FILE, index=False, compression="zstd")
        fingerprint_columns = [
            "text_id",
            "text_sha256",
            "symbol",
            "trading_date",
        ]
        if "sentiment_label" in metadata.columns:
            fingerprint_columns.append("sentiment_label")
        text_fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(metadata[fingerprint_columns], index=False)
            .to_numpy()
            .tobytes()
        ).hexdigest()
        manifest = {
            "schema_version": "layer_cls_representation_v1.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "shape": [n_rows, expected_layers, hidden_size],
            "layer_indices": list(range(expected_layers)),
            "pooling": "raw_cls_token",
            "storage_dtype": storage_dtype,
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "text_dataset_fingerprint": text_fingerprint,
            "row_alignment": "representation_row is zero-based and identical across files",
            "model_validation": validation,
            "model_identity": dict(model_identity or {}),
            "files": {
                "representations": REPRESENTATION_FILE,
                "metadata": METADATA_FILE,
                "original_head_outputs": HEAD_OUTPUT_FILE,
            },
        }
        (temporary / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _atomic_replace_directory(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    artifacts = representation_artifacts(output)
    validate_representation_artifacts(artifacts.directory)
    return artifacts


def validate_representation_artifacts(
    directory: str | Path,
) -> dict[str, object]:
    """验证四个representation产物存在、shape和行号完全一致。"""

    artifacts = representation_artifacts(directory)
    for path in (
        artifacts.representations,
        artifacts.metadata,
        artifacts.head_outputs,
        artifacts.manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"representation产物缺失: {path}")
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    shape = tuple(int(value) for value in manifest.get("shape", []))
    array = np.load(artifacts.representations, mmap_mode="r")
    if array.shape != shape:
        raise ValueError(f"representation shape错误: {array.shape} != {shape}")
    if not np.isfinite(array).all():
        raise ValueError("representation含NaN或Inf")
    metadata = pd.read_parquet(artifacts.metadata)
    head = pd.read_parquet(artifacts.head_outputs)
    expected_rows = pd.Series(np.arange(shape[0]), name="representation_row")
    if not metadata["representation_row"].reset_index(drop=True).equals(expected_rows):
        raise ValueError("metadata的representation_row不连续")
    if not head["representation_row"].reset_index(drop=True).equals(expected_rows):
        raise ValueError("分类头输出的representation_row不连续")
    if (
        not metadata["text_id"]
        .astype(str)
        .reset_index(drop=True)
        .equals(head["text_id"].astype(str).reset_index(drop=True))
    ):
        raise ValueError("metadata与分类头输出的text_id顺序不一致")
    fingerprint_columns = ["text_id", "text_sha256", "symbol", "trading_date"]
    if "sentiment_label" in metadata.columns:
        fingerprint_columns.append("sentiment_label")
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(metadata[fingerprint_columns], index=False)
        .to_numpy()
        .tobytes()
    ).hexdigest()
    if fingerprint != manifest.get("text_dataset_fingerprint"):
        raise ValueError("文本metadata指纹与manifest不一致")
    return {
        "rows": shape[0],
        "layers": shape[1],
        "hidden_size": shape[2],
        "dtype": str(array.dtype),
        "metadata_rows": len(metadata),
        "head_rows": len(head),
    }


def run_representation_stage(config: Mapping[str, object]) -> RepresentationArtifacts:
    """按配置加载固定checkpoint/tokenizer并执行阶段1。"""

    from transformers import BertTokenizerFast

    text_cfg = config.get("text", {})
    model_cfg = config.get("model", {})
    output_cfg = config.get("output", {})
    if not all(
        isinstance(value, Mapping) for value in (text_cfg, model_cfg, output_cfg)
    ):
        raise ValueError("text/model/output配置必须是对象")
    texts = load_text_dataset(
        text_cfg.get("path", ""),
        id_column=str(text_cfg.get("id_column", "text_id")),
        text_column=str(text_cfg.get("text_column", "text")),
        symbol_column=str(text_cfg.get("symbol_column", "symbol")),
        date_column=str(text_cfg.get("date_column", "trading_date")),
        sentiment_label_column=(
            str(text_cfg["sentiment_label_column"])
            if text_cfg.get("sentiment_label_column")
            else None
        ),
        sentiment_labels_path=text_cfg.get("sentiment_labels_path") or None,
        sentiment_label_id_column=(
            str(text_cfg["sentiment_label_id_column"])
            if text_cfg.get("sentiment_label_id_column")
            else None
        ),
        limit=int(text_cfg["limit"]) if text_cfg.get("limit") else None,
    )
    base_model = Path(str(model_cfg.get("base_model_dir", ""))).expanduser().resolve()
    checkpoint = Path(str(model_cfg.get("checkpoint", ""))).expanduser().resolve()
    if not base_model.is_dir():
        raise FileNotFoundError(f"tokenizer/base model目录不存在: {base_model}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint不存在: {checkpoint}")
    device = str(model_cfg.get("device", "cuda"))
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但当前没有可用GPU")
    compute_dtype = str(model_cfg.get("compute_dtype", "float16"))
    dtype = (
        torch.float16
        if compute_dtype == "float16" and device.startswith("cuda")
        else torch.float32
    )
    head_pooling = str(model_cfg.get("head_pooling", "cls"))
    model_hash = checkpoint_hash(str(checkpoint))
    model = build_candidate(
        str(base_model),
        str(checkpoint),
        pooling=head_pooling,
        pooling_confirmed=bool(model_cfg.get("head_pooling_confirmed", False)),
        model_hash=model_hash,
        device=device,
        dtype=dtype,
    )
    tokenizer = BertTokenizerFast.from_pretrained(base_model, local_files_only=True)
    identity = {
        "text_source_path": str(
            Path(str(text_cfg.get("path", ""))).expanduser().resolve()
        ),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": model_hash,
        "tokenizer_path": str(base_model),
        "tokenizer_sha256": tokenizer_signature(base_model),
        "head_pooling": head_pooling,
        "head_pooling_confirmed": bool(model_cfg.get("head_pooling_confirmed", False)),
    }
    return extract_cls_representations(
        texts,
        model=model,
        tokenizer=tokenizer,
        output_directory=output_cfg.get(
            "representations", "artifacts/layer_probe/representations"
        ),
        batch_size=int(model_cfg.get("batch_size", 128)),
        max_length=int(model_cfg.get("max_length", 512)),
        expected_layers=13,
        storage_dtype=str(model_cfg.get("storage_dtype", "float16")),
        device=device,
        model_identity=identity,
    )
