"""批量推理脚本共享的 tokenizer 缓存工具。"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# 本项目仅使用 PyTorch；避免环境中不兼容的可选 TensorFlow 安装影响 tokenizer/model 导入。
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

_TOKENIZER = None
_MAX_LENGTH = 0
_TOKENIZER_FILES = (
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def file_signature(path: str | os.PathLike[str]) -> dict[str, int]:
    """返回足以发现本地源文件变化的纳秒级签名。"""
    stat = Path(path).stat()
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def tokenizer_signature(base_dir: str | os.PathLike[str]) -> str:
    """对影响分词结果的本地 tokenizer 文件计算组合 SHA-256。"""
    base = Path(base_dir)
    h = hashlib.sha256()
    found = False
    for name in _TOKENIZER_FILES:
        path = base / name
        if not path.is_file():
            continue
        found = True
        h.update(name.encode("utf-8"))
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    if not found:
        raise FileNotFoundError(f"tokenizer 目录中没有可识别的配置文件: {base}")
    return h.hexdigest()


def load_json_object(path: str | os.PathLike[str]) -> dict[str, Any]:
    """读取 JSON 对象，并拒绝数组或标量顶层。"""
    with Path(path).open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return value


def save_json_object(path: str | os.PathLike[str], value: dict[str, Any]) -> None:
    """原子写入 JSON，避免中断后留下看似可复用的半成品。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
    os.replace(tmp, target)


def save_npy(path: str | os.PathLike[str], value: np.ndarray) -> None:
    """原子写入 NPY 缓存。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, value)
    os.replace(tmp, target)


def save_csv(path: str | os.PathLike[str], frame: Any) -> None:
    """原子写入未压缩 CSV；``frame`` 需提供 pandas 风格的 ``to_csv``。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, target)


def validate_cached_ids(
    path: str | os.PathLike[str],
    *,
    n_rows: int,
    max_length: int,
) -> None:
    """验证缓存 shape 与整数 dtype，异常时由调用方决定重建。"""
    ids = np.load(path, mmap_mode="r")
    if ids.shape != (n_rows, max_length):
        raise ValueError(
            f"token 缓存 shape 错误: {ids.shape} != {(n_rows, max_length)}"
        )
    if ids.dtype.kind not in "iu":
        raise TypeError(f"token 缓存必须是整数 dtype，实际为 {ids.dtype}")


def _tokenizer_init(base_dir: str, max_length: int) -> None:
    global _TOKENIZER, _MAX_LENGTH
    from transformers import BertTokenizerFast

    _TOKENIZER = BertTokenizerFast.from_pretrained(base_dir, local_files_only=True)
    _MAX_LENGTH = max_length


def _tokenize_chunk(texts: Sequence[str]) -> np.ndarray:
    if _TOKENIZER is None:
        raise RuntimeError("tokenizer worker 尚未初始化")
    encoded = _TOKENIZER(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=_MAX_LENGTH,
        return_tensors="np",
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return encoded["input_ids"].astype("int32", copy=False)


def tokenize_texts_parallel(
    texts: Sequence[str],
    *,
    base_dir: str,
    max_length: int,
    workers: int,
) -> tuple[np.ndarray, int]:
    """把文本批量分词为定长 int32 数组，并返回 pad token id。"""
    if not texts:
        raise ValueError("输入文本为空，无法建立 token 缓存")
    if max_length <= 0:
        raise ValueError("max_length 必须大于 0")
    if workers <= 0:
        raise ValueError("token workers 必须大于 0")

    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(base_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer 未定义 pad_token_id")
    pad_token_id = int(tokenizer.pad_token_id)
    del tokenizer

    slots = max(1, workers * 4)
    step = max(1, (len(texts) + slots - 1) // slots)
    chunks = [texts[i:i + step] for i in range(0, len(texts), step)]
    workers = min(workers, len(chunks))
    if workers == 1:
        _tokenizer_init(base_dir, max_length)
        arrays = [_tokenize_chunk(chunk) for chunk in chunks]
    else:
        # Rust fast-tokenizer 在 fork 后可能继承线程池状态并死锁；spawn 更可预测。
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            workers,
            initializer=_tokenizer_init,
            initargs=(base_dir, max_length),
        ) as pool:
            arrays = pool.map(_tokenize_chunk, chunks)
    return np.concatenate(arrays, axis=0), pad_token_id
