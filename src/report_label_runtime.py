"""报告级 Label 构建的共享服务器资源控制。"""

from __future__ import annotations

import math
import os
import resource
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SHARED_SERVER_THREAD_CAP = 32
_NICE_APPLIED = False
_THREADPOOL_GUARD: object | None = None


@dataclass(frozen=True)
class RuntimeResources:
    configured_threads: int
    effective_threads: int
    shared_server: bool
    affinity_threads: int | None
    scheduler_threads: int | None
    cgroup_threads: int | None
    blas_threads: int
    nice_increment: int
    nice_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _affinity_limit() -> int | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return len(getter(0))
    except OSError:
        return None


def _cgroup_limit() -> int | None:
    """读取 cgroup v2/v1 CPU quota；无限额时返回 None。"""
    v2 = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota_text, period_text = v2.read_text(encoding="utf-8").strip().split()[:2]
        if quota_text != "max":
            quota = int(quota_text)
            period = int(period_text)
            if quota > 0 and period > 0:
                return max(1, math.floor(quota / period))
    except (OSError, ValueError):
        pass

    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        quota = int(quota_path.read_text(encoding="utf-8").strip())
        period = int(period_path.read_text(encoding="utf-8").strip())
        if quota > 0 and period > 0:
            return max(1, math.floor(quota / period))
    except (OSError, ValueError):
        pass
    return None


def resolve_runtime_resources(config: Mapping[str, object]) -> RuntimeResources:
    shared = bool(config.get("shared_server", True))
    configured = _positive_int(config.get("max_threads", 8))
    if configured is None:
        raise ValueError("performance.max_threads 必须为正整数")
    if shared:
        configured = min(configured, SHARED_SERVER_THREAD_CAP)

    affinity = _affinity_limit()
    scheduler = _positive_int(os.environ.get("SLURM_CPUS_PER_TASK"))
    cgroup = _cgroup_limit()
    candidates = [configured]
    candidates.extend(x for x in (affinity, scheduler, cgroup) if x is not None)
    effective = max(1, min(candidates))
    blas = _positive_int(config.get("blas_threads", 1))
    if blas is None:
        raise ValueError("performance.blas_threads 必须为正整数")
    nice_increment = int(config.get("nice_increment", 5))
    if nice_increment < 0:
        raise ValueError("performance.nice_increment 不能为负")
    return RuntimeResources(
        configured_threads=configured,
        effective_threads=effective,
        shared_server=shared,
        affinity_threads=affinity,
        scheduler_threads=scheduler,
        cgroup_threads=cgroup,
        blas_threads=blas,
        nice_increment=nice_increment,
        nice_applied=False,
    )


def configure_runtime(config: Mapping[str, object]) -> RuntimeResources:
    """在导入 Polars/Numba 前设置线程池，并抑制嵌套 BLAS 并行。"""
    global _NICE_APPLIED, _THREADPOOL_GUARD
    resolved = resolve_runtime_resources(config)
    threads = str(resolved.effective_threads)
    os.environ["POLARS_MAX_THREADS"] = threads
    os.environ["NUMBA_NUM_THREADS"] = threads
    # 不设置 OMP_NUM_THREADS；Numba 可能使用 OpenMP。BLAS 由 threadpoolctl 单独限制。
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(resolved.blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(resolved.blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(resolved.blas_threads)
    os.environ["BLIS_NUM_THREADS"] = str(resolved.blas_threads)

    from threadpoolctl import threadpool_limits

    _THREADPOOL_GUARD = threadpool_limits(limits=resolved.blas_threads, user_api="blas")
    try:
        import numba

        numba.set_num_threads(resolved.effective_threads)
    except ImportError as exc:
        raise RuntimeError("报告 Label 高性能路径要求安装 numba") from exc

    nice_applied = False
    if (
        sys.platform.startswith("linux")
        and resolved.nice_increment > 0
        and not _NICE_APPLIED
    ):
        try:
            os.nice(resolved.nice_increment)
            nice_applied = True
            _NICE_APPLIED = True
        except OSError:
            nice_applied = False
    return RuntimeResources(**{**resolved.to_dict(), "nice_applied": nice_applied})


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux 返回 KiB，macOS 返回 bytes。
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return value / divisor
