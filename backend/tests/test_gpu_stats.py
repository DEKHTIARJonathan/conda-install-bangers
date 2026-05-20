import pytest

from bangers.models.common import GpuDeviceStats, GpuStatsResponse
from bangers.services import gpu_stats
from bangers.services.gpu_stats import (
    parse_apple_ioreg_performance_statistics,
    parse_nvidia_smi_csv,
)


def test_parse_nvidia_smi_csv_handles_gb10_na_memory() -> None:
    stats = parse_nvidia_smi_csv(
        "0, NVIDIA GB10, GPU-test, 95, 0, [N/A], [N/A], 47.28, [N/A]\n",
        node_id="spark-local-music",
        node_role="worker",
        busy=True,
        holder="worker",
    )

    assert len(stats) == 1
    gpu = stats[0]
    assert gpu.label == "spark-local-music"
    assert gpu.utilization_gpu_percent == 95
    assert gpu.utilization_memory_percent == 0
    assert gpu.vram_used_mb is None
    assert gpu.vram_total_mb is None
    assert gpu.power_draw_w == 47.28
    assert gpu.power_limit_w is None
    assert gpu.busy is True
    assert gpu.holder == "worker"


def test_parse_apple_ioreg_performance_statistics() -> None:
    stats = parse_apple_ioreg_performance_statistics(
        '"PerformanceStatistics" = {"Tiler Utilization %"=31,'
        '"Renderer Utilization %"=44,"Device Utilization %"=52}'
    )

    assert stats == {
        "gpu_utilization_percent": 52.0,
        "renderer_utilization_percent": 44.0,
        "tiler_utilization_percent": 31.0,
    }


@pytest.mark.asyncio
async def test_read_local_gpu_stats_uses_mlx_and_ioreg(monkeypatch) -> None:
    import sys
    import types

    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")

    class _FakeMetal:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_info() -> dict[str, object]:
            return {
                "device_name": "Apple M3",
                "max_recommended_working_set_size": 16 * 1024 * 1024 * 1024,
            }

    mlx_core.metal = _FakeMetal()
    mlx_core.get_active_memory = lambda: 2 * 1024 * 1024 * 1024
    mlx_core.get_cache_memory = lambda: 512 * 1024 * 1024
    mlx_core.get_peak_memory = lambda: 3 * 1024 * 1024 * 1024
    mlx_pkg.core = mlx_core

    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setattr(gpu_stats.sys, "platform", "darwin")
    monkeypatch.setattr(
        gpu_stats,
        "_read_apple_ioreg_utilization_sync",
        lambda: {
            "gpu_utilization_percent": 65.0,
            "renderer_utilization_percent": 62.0,
            "tiler_utilization_percent": 59.0,
        },
    )

    response = await gpu_stats.read_local_gpu_stats(
        node_id="mac-studio",
        node_role="standalone",
        device="mps",
        busy=True,
        holder="generation",
    )

    assert response.provider == "mlx"
    assert response.memory_type == "unified"
    assert response.vram_used_mb == 2048.0
    assert response.vram_total_mb == 16384.0
    assert response.vram_percent == 12.5
    assert response.memory_cache_mb == 512.0
    assert response.memory_peak_mb == 3072.0
    assert response.gpu_utilization_percent == 65.0
    assert response.renderer_utilization_percent == 62.0
    assert response.tiler_utilization_percent == 59.0
    assert response.gpus[0].provider == "mlx"
    assert response.gpus[0].busy is True
    assert response.gpus[0].holder == "generation"


@pytest.mark.asyncio
async def test_read_local_gpu_stats_falls_back_to_mps(monkeypatch) -> None:
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True),
    )
    torch.mps = types.SimpleNamespace(
        current_allocated_memory=lambda: 256 * 1024 * 1024,
    )

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(gpu_stats.sys, "platform", "darwin")
    monkeypatch.setattr(gpu_stats, "_read_apple_mlx_stats_sync", lambda **_kwargs: None)
    monkeypatch.setattr(gpu_stats, "_read_nvidia_smi_stats_sync", lambda **_kwargs: None)

    response = await gpu_stats.read_local_gpu_stats(
        node_id="mac-mini",
        node_role="standalone",
        device="mps",
    )

    assert response.provider == "torch.mps"
    assert response.memory_type == "allocated"
    assert response.vram_used_mb == 256.0
    assert response.vram_total_mb is None


@pytest.mark.asyncio
async def test_read_local_gpu_stats_returns_partial_on_provider_errors(monkeypatch) -> None:
    def fail(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gpu_stats, "_read_local_gpu_stats_sync", fail)

    response = await gpu_stats.read_local_gpu_stats(
        node_id="local",
        node_role="standalone",
        device="unknown",
    )

    assert response.device == "unknown"
    assert response.error == "boom"


@pytest.mark.asyncio
async def test_models_gpu_stats_returns_device_list(client, monkeypatch) -> None:
    from bangers.config import settings
    from bangers.routers import models as models_router

    monkeypatch.setattr(settings, "DISTRIBUTED_ROLE", "standalone")
    monkeypatch.setattr(settings, "DISTRIBUTED_WORKERS", ())

    async def fake_read_local_gpu_stats(**_kwargs):
        return GpuStatsResponse(
            device="cuda",
            vram_used_mb=1024.0,
            vram_total_mb=4096.0,
            vram_percent=25.0,
            gpus=[
                GpuDeviceStats(
                    node_id="spark-local-music",
                    label="spark-local-music",
                    utilization_gpu_percent=88.0,
                    power_draw_w=44.5,
                )
            ],
            updated_at=123.0,
        )

    monkeypatch.setattr(models_router, "read_local_gpu_stats", fake_read_local_gpu_stats)

    response = await client.get("/api/models/gpu-stats")

    assert response.status_code == 200
    data = response.json()
    assert data["device"] == "cuda"
    assert data["vram_percent"] == 25.0
    assert data["gpus"][0]["node_id"] == "spark-local-music"
    assert data["gpus"][0]["utilization_gpu_percent"] == 88.0


@pytest.mark.asyncio
async def test_models_gpu_stats_endpoint_returns_partial_on_read_error(client, monkeypatch) -> None:
    from bangers.config import settings
    from bangers.routers import models as models_router

    monkeypatch.setattr(settings, "DISTRIBUTED_ROLE", "standalone")
    monkeypatch.setattr(settings, "DISTRIBUTED_WORKERS", ())

    async def fake_read_local_gpu_stats(**_kwargs):
        raise RuntimeError("stats exploded")

    monkeypatch.setattr(models_router, "read_local_gpu_stats", fake_read_local_gpu_stats)

    response = await client.get("/api/models/gpu-stats")

    assert response.status_code == 200
    data = response.json()
    assert data["device"] == "unknown"
    assert data["error"] == "stats exploded"
