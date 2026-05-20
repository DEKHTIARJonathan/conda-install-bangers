from __future__ import annotations

import asyncio
import csv
import io
import os
import re
import subprocess
import sys
import time
from typing import Any

from bangers.models.common import GpuDeviceStats, GpuStatsResponse


_NVIDIA_SMI_QUERY = (
    "index,name,uuid,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,power.draw,power.limit"
)
_IOREG_UTILIZATION_KEYS = {
    "gpu_utilization_percent": "Device Utilization %",
    "renderer_utilization_percent": "Renderer Utilization %",
    "tiler_utilization_percent": "Tiler Utilization %",
}
_MLX_TOTAL_MEMORY_KEYS = (
    "max_recommended_working_set_size",
    "recommended_max_working_set_size",
    "memory_size",
    "physical_memory",
    "max_buffer_length",
)
_MLX_METAL_AVAILABLE_CACHE: bool | None = None
_MLX_DEVICE_INFO_CACHE: dict[str, Any] | None = None


def _float_or_none(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned in {"[N/A]", "N/A"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _int_or_zero(value: str) -> int:
    try:
        return int(value.strip())
    except ValueError:
        return 0


def _bytes_to_mb(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / (1024 * 1024), 1)
    except (TypeError, ValueError):
        return None


def _bounded_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, min(100.0, value)), 1)


def _memory_percent(used_mb: float | None, total_mb: float | None) -> float | None:
    if used_mb is None or not total_mb:
        return None
    return _bounded_percent(used_mb / total_mb * 100.0)


def parse_nvidia_smi_csv(
    text: str,
    *,
    node_id: str,
    node_role: str,
    busy: bool = False,
    holder: str | None = None,
) -> list[GpuDeviceStats]:
    """Parse the narrow `nvidia-smi --query-gpu` CSV used by the monitor."""
    gpus: list[GpuDeviceStats] = []
    reader = csv.reader(io.StringIO(text.strip()))
    for row in reader:
        if len(row) < 9:
            continue
        used = _float_or_none(row[5])
        total = _float_or_none(row[6])
        vram_percent = None
        if used is not None and total:
            vram_percent = round(used / total * 100.0, 1)
        index = _int_or_zero(row[0])
        label = node_id or f"{node_role or 'node'}-{index}"
        gpus.append(
            GpuDeviceStats(
                node_id=node_id,
                node_role=node_role,
                label=label,
                device_index=index,
                name=row[1].strip(),
                uuid=row[2].strip(),
                provider="nvidia-smi",
                memory_type="vram",
                utilization_gpu_percent=_float_or_none(row[3]),
                utilization_memory_percent=_float_or_none(row[4]),
                vram_used_mb=used,
                vram_total_mb=total,
                vram_percent=vram_percent,
                power_draw_w=_float_or_none(row[7]),
                power_limit_w=_float_or_none(row[8]),
                busy=busy,
                holder=holder,
            )
        )
    return gpus


def parse_apple_ioreg_performance_statistics(text: str) -> dict[str, float]:
    """Extract Apple GPU utilization counters from `ioreg` text output."""
    metrics: dict[str, float] = {}
    for field, key in _IOREG_UTILIZATION_KEYS.items():
        match = re.search(rf'"{re.escape(key)}"\s*=\s*(-?\d+(?:\.\d+)?)', text)
        if match is None:
            continue
        value = _float_or_none(match.group(1))
        bounded = _bounded_percent(value)
        if bounded is not None:
            metrics[field] = bounded
    return metrics


def _with_legacy_first_device_fields(response: GpuStatsResponse) -> GpuStatsResponse:
    if not response.gpus:
        return response
    first = response.gpus[0]
    response.provider = first.provider
    response.memory_type = first.memory_type
    response.gpu_utilization_percent = first.utilization_gpu_percent
    response.renderer_utilization_percent = first.renderer_utilization_percent
    response.tiler_utilization_percent = first.tiler_utilization_percent
    response.vram_used_mb = first.vram_used_mb
    response.vram_total_mb = first.vram_total_mb
    response.vram_percent = first.vram_percent
    response.memory_cache_mb = first.memory_cache_mb
    response.memory_peak_mb = first.memory_peak_mb
    return response


def merge_gpu_stats(
    responses: list[GpuStatsResponse],
    *,
    device: str,
) -> GpuStatsResponse:
    gpus: list[GpuDeviceStats] = []
    errors: list[str] = []
    updated_at = time.time()
    for response in responses:
        gpus.extend(response.gpus)
        if response.error:
            errors.append(response.error)
        if response.updated_at:
            updated_at = max(updated_at, response.updated_at)
    merged = GpuStatsResponse(
        device=device,
        gpus=gpus,
        updated_at=updated_at,
        error="; ".join(errors),
    )
    return _with_legacy_first_device_fields(merged)


def _read_nvidia_smi_stats_sync(
    *,
    node_id: str,
    node_role: str,
    device: str,
    busy: bool = False,
    holder: str | None = None,
) -> GpuStatsResponse | None:
    updated_at = time.time()
    visible_devices = _cuda_visible_devices()
    if visible_devices == ():
        return GpuStatsResponse(
            device=device,
            provider="nvidia-smi",
            memory_type="vram",
            updated_at=updated_at,
        )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={_NVIDIA_SMI_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    if visible_devices is not None:
        cmd.append(f"--id={','.join(visible_devices)}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return GpuStatsResponse(
            device=device,
            provider="nvidia-smi",
            memory_type="vram",
            updated_at=updated_at,
            error="nvidia-smi timed out",
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        return GpuStatsResponse(
            device=device,
            provider="nvidia-smi",
            memory_type="vram",
            updated_at=updated_at,
            error=detail or "nvidia-smi failed",
        )

    gpus = parse_nvidia_smi_csv(
        result.stdout,
        node_id=node_id,
        node_role=node_role,
        busy=busy,
        holder=holder,
    )
    response = GpuStatsResponse(
        device=device,
        provider="nvidia-smi",
        memory_type="vram",
        gpus=gpus,
        updated_at=updated_at,
    )
    return _with_legacy_first_device_fields(response)


def _cuda_visible_devices() -> tuple[str, ...] | None:
    """Return CUDA_VISIBLE_DEVICES as nvidia-smi ids, or None for all devices."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.lower() == "all":
        return None
    if value.lower() in {"none", "void"} or value == "-1":
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _mlx_total_memory_bytes(device_info: dict[str, Any]) -> float | None:
    for key in _MLX_TOTAL_MEMORY_KEYS:
        value = device_info.get(key)
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _mlx_memory_bytes(mx: Any, name: str) -> float | None:
    fn = getattr(mx, name, None)
    if not callable(fn):
        return None
    try:
        value = fn()
    except Exception:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mlx_metal_available(metal: Any) -> bool:
    global _MLX_METAL_AVAILABLE_CACHE

    if _MLX_METAL_AVAILABLE_CACHE is not None:
        return _MLX_METAL_AVAILABLE_CACHE
    is_available = getattr(metal, "is_available", None)
    if not callable(is_available):
        _MLX_METAL_AVAILABLE_CACHE = True
        return True
    try:
        _MLX_METAL_AVAILABLE_CACHE = is_available() is True
    except Exception:
        _MLX_METAL_AVAILABLE_CACHE = False
    return _MLX_METAL_AVAILABLE_CACHE


def _mlx_device_info(metal: Any) -> dict[str, Any]:
    global _MLX_DEVICE_INFO_CACHE

    if _MLX_DEVICE_INFO_CACHE is not None:
        return _MLX_DEVICE_INFO_CACHE
    try:
        raw_device_info = getattr(metal, "device_info", lambda: {})()
    except Exception:
        raw_device_info = {}
    _MLX_DEVICE_INFO_CACHE = (
        dict(raw_device_info) if isinstance(raw_device_info, dict) else {}
    )
    return _MLX_DEVICE_INFO_CACHE


def _read_apple_ioreg_utilization_sync() -> dict[str, float]:
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1", "-w0"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    return parse_apple_ioreg_performance_statistics(result.stdout)


def _read_apple_mlx_stats_sync(
    *,
    node_id: str,
    node_role: str,
    device: str,
    busy: bool = False,
    holder: str | None = None,
) -> GpuStatsResponse | None:
    if sys.platform != "darwin":
        return None

    updated_at = time.time()
    try:
        import mlx.core as mx  # type: ignore[import-untyped]
    except Exception:
        return None

    metal = getattr(mx, "metal", None)
    if not _mlx_metal_available(metal):
        return None
    device_info = _mlx_device_info(metal)

    active_mb = _bytes_to_mb(_mlx_memory_bytes(mx, "get_active_memory"))
    cache_mb = _bytes_to_mb(_mlx_memory_bytes(mx, "get_cache_memory"))
    peak_mb = _bytes_to_mb(_mlx_memory_bytes(mx, "get_peak_memory"))
    total_mb = _bytes_to_mb(_mlx_total_memory_bytes(device_info))
    vram_percent = _memory_percent(active_mb, total_mb)
    utilization = _read_apple_ioreg_utilization_sync()

    name = str(
        device_info.get("device_name")
        or device_info.get("name")
        or device_info.get("architecture")
        or "Apple GPU"
    )
    label = node_id or name
    gpu = GpuDeviceStats(
        node_id=node_id,
        node_role=node_role,
        label=label,
        device_index=0,
        name=name,
        provider="mlx",
        memory_type="unified",
        utilization_gpu_percent=utilization.get("gpu_utilization_percent"),
        renderer_utilization_percent=utilization.get("renderer_utilization_percent"),
        tiler_utilization_percent=utilization.get("tiler_utilization_percent"),
        vram_used_mb=active_mb,
        vram_total_mb=total_mb,
        vram_percent=vram_percent,
        memory_cache_mb=cache_mb,
        memory_peak_mb=peak_mb,
        busy=busy,
        holder=holder,
    )
    response = GpuStatsResponse(
        device=device,
        provider="mlx",
        memory_type="unified",
        gpus=[gpu],
        updated_at=updated_at,
    )
    return _with_legacy_first_device_fields(response)


def _read_torch_cuda_stats_sync(
    *,
    node_id: str,
    node_role: str,
    device: str,
    busy: bool = False,
    holder: str | None = None,
) -> GpuStatsResponse | None:
    try:
        import torch
    except Exception:
        return None

    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available) or is_available() is not True:
        return None

    try:
        used_mb = round(cuda.memory_allocated() / (1024 * 1024), 1)
        total_mb = round(cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
        name = str(cuda.get_device_name(0)) if hasattr(cuda, "get_device_name") else "CUDA GPU"
    except Exception:
        return None

    gpu = GpuDeviceStats(
        node_id=node_id,
        node_role=node_role,
        label=node_id or name,
        device_index=0,
        name=name,
        provider="torch.cuda",
        memory_type="vram",
        vram_used_mb=used_mb,
        vram_total_mb=total_mb,
        vram_percent=_memory_percent(used_mb, total_mb),
        busy=busy,
        holder=holder,
    )
    response = GpuStatsResponse(
        device=device,
        provider="torch.cuda",
        memory_type="vram",
        gpus=[gpu],
        updated_at=time.time(),
    )
    return _with_legacy_first_device_fields(response)


def _read_torch_mps_stats_sync(
    *,
    node_id: str,
    node_role: str,
    device: str,
    busy: bool = False,
    holder: str | None = None,
) -> GpuStatsResponse | None:
    try:
        import torch
    except Exception:
        return None

    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    is_available = getattr(mps_backend, "is_available", None)
    if not callable(is_available) or is_available() is not True:
        return None

    torch_mps = getattr(torch, "mps", None)
    if torch_mps is None:
        return None
    allocated = None
    for method in ("current_allocated_memory", "driver_allocated_size"):
        fn = getattr(torch_mps, method, None)
        if callable(fn):
            try:
                allocated = fn()
                break
            except Exception:
                continue
    used_mb = _bytes_to_mb(allocated)
    if used_mb is None:
        return None

    gpu = GpuDeviceStats(
        node_id=node_id,
        node_role=node_role,
        label=node_id or "Apple GPU",
        device_index=0,
        name="Apple GPU",
        provider="torch.mps",
        memory_type="allocated",
        vram_used_mb=used_mb,
        busy=busy,
        holder=holder,
    )
    response = GpuStatsResponse(
        device=device,
        provider="torch.mps",
        memory_type="allocated",
        gpus=[gpu],
        updated_at=time.time(),
    )
    return _with_legacy_first_device_fields(response)


def _read_local_gpu_stats_sync(
    *,
    node_id: str,
    node_role: str,
    device: str,
    busy: bool = False,
    holder: str | None = None,
) -> GpuStatsResponse:
    apple = _read_apple_mlx_stats_sync(
        node_id=node_id,
        node_role=node_role,
        device=device,
        busy=busy,
        holder=holder,
    )
    if apple is not None:
        return apple

    nvidia = _read_nvidia_smi_stats_sync(
        node_id=node_id,
        node_role=node_role,
        device=device,
        busy=busy,
        holder=holder,
    )
    if nvidia is not None and nvidia.gpus:
        return nvidia

    cuda = _read_torch_cuda_stats_sync(
        node_id=node_id,
        node_role=node_role,
        device=device,
        busy=busy,
        holder=holder,
    )
    if cuda is not None:
        return cuda

    mps = _read_torch_mps_stats_sync(
        node_id=node_id,
        node_role=node_role,
        device=device,
        busy=busy,
        holder=holder,
    )
    if mps is not None:
        return mps

    if nvidia is not None and nvidia.error:
        return nvidia

    return GpuStatsResponse(device=device, updated_at=time.time())


async def read_local_gpu_stats(
    *,
    node_id: str,
    node_role: str,
    device: str,
    busy: bool = False,
    holder: str | None = None,
) -> GpuStatsResponse:
    try:
        return await asyncio.to_thread(
            _read_local_gpu_stats_sync,
            node_id=node_id,
            node_role=node_role,
            device=device,
            busy=busy,
            holder=holder,
        )
    except Exception as exc:
        return GpuStatsResponse(
            device=device,
            updated_at=time.time(),
            error=str(exc),
        )
