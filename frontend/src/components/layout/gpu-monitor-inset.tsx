"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, ChevronDown, ChevronUp, Cpu, Zap } from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchGpuStats } from "@/lib/api/client";
import type { GpuDeviceStats, GpuStats } from "@/types/api";

const GPU_MONITOR_COLLAPSED_KEY = "bangers-gpu-monitor-collapsed";

type GpuMonitorInsetProps = {
  hasMiniPlayer: boolean;
};

function pct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "--";
  return `${Math.round(value)}%`;
}

function watts(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "-- W";
  return `${value.toFixed(0)} W`;
}

function fallbackGpus(stats: GpuStats | undefined): GpuDeviceStats[] {
  if (!stats) return [];
  if (stats.gpus?.length) return stats.gpus;
  return [
    {
      node_id: "local",
      node_role: "",
      label: "local",
      device_index: 0,
      name: stats.device || "GPU",
      uuid: "",
      provider: stats.provider ?? "",
      memory_type: stats.memory_type ?? "",
      utilization_gpu_percent: stats.gpu_utilization_percent ?? null,
      utilization_memory_percent: null,
      renderer_utilization_percent: stats.renderer_utilization_percent ?? null,
      tiler_utilization_percent: stats.tiler_utilization_percent ?? null,
      vram_used_mb: stats.vram_used_mb ?? null,
      vram_total_mb: stats.vram_total_mb ?? null,
      vram_percent: stats.vram_percent ?? null,
      memory_cache_mb: stats.memory_cache_mb ?? null,
      memory_peak_mb: stats.memory_peak_mb ?? null,
      power_draw_w: null,
      power_limit_w: null,
      busy: false,
      holder: null,
      error: stats.error ?? "",
    },
  ];
}

function shortLabel(gpu: GpuDeviceStats): string {
  const label = gpu.label || gpu.node_id || gpu.name || "GPU";
  return label
    .replace(/^spark-/, "")
    .replace(/-music$/, "")
    .replace(/-lm-chat$/, "");
}

function utilizationWidth(gpu: GpuDeviceStats): string {
  const value = gpu.utilization_gpu_percent;
  if (value == null || Number.isNaN(value)) return "0%";
  return `${Math.max(0, Math.min(100, value))}%`;
}

export function GpuMonitorInset({ hasMiniPlayer }: GpuMonitorInsetProps) {
  const [collapsed, setCollapsed] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.localStorage.getItem(GPU_MONITOR_COLLAPSED_KEY) === "true";
  });
  const { data, isLoading, isError } = useQuery({
    queryKey: ["gpu-stats", "inset"],
    queryFn: fetchGpuStats,
    refetchInterval: collapsed ? 5_000 : 2_000,
    refetchIntervalInBackground: true,
  });

  useEffect(() => {
    window.localStorage.setItem(GPU_MONITOR_COLLAPSED_KEY, collapsed ? "true" : "false");
  }, [collapsed]);

  const gpus = fallbackGpus(data);
  const primaryGpu = gpus[0];
  const hasData = gpus.length > 0;
  const hasGpuError = gpus.some((gpu) => gpu.error);
  const statusLabel = isLoading && !hasData
    ? "loading"
    : isError && !hasData
      ? "offline"
      : hasGpuError
        ? "degraded"
        : "live";
  const statusPct = primaryGpu?.error ? "error" : pct(primaryGpu?.utilization_gpu_percent);
  const positionClass = hasMiniPlayer ? "bottom-24" : "bottom-3 md:bottom-6";

  if (collapsed) {
    return (
      <aside
        className={cn(
          "fixed right-3 z-30 max-w-[calc(100vw-1.5rem)] rounded-lg border bg-background/95 p-2 shadow-lg backdrop-blur md:right-6",
          positionClass,
        )}
        aria-label="GPU utilization"
      >
        <button
          type="button"
          className="flex h-9 items-center gap-2 rounded-md px-2 text-left text-sm transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          onClick={() => setCollapsed(false)}
          aria-expanded="false"
          aria-label="Expand GPU monitor"
        >
          <Activity className="h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
          <span className="font-extrabold uppercase tracking-normal">GPU</span>
          <span className="font-mono text-xs text-muted-foreground">{statusPct}</span>
          <span className="text-xs text-muted-foreground">{statusLabel}</span>
          <ChevronUp className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
        </button>
      </aside>
    );
  }

  return (
    <aside
      className={cn(
        "fixed right-3 z-30 w-[min(440px,calc(100vw-1.5rem))] rounded-lg border bg-background/95 p-4 shadow-lg backdrop-blur md:right-6",
        positionClass,
      )}
      aria-label="GPU utilization"
    >
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <Activity className="h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
          <span className="truncate text-sm font-extrabold uppercase tracking-normal">
            GPU
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span className="text-xs text-muted-foreground">{statusLabel}</span>
          <button
            type="button"
            className="inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            onClick={() => setCollapsed(true)}
            aria-expanded="true"
            aria-label="Collapse GPU monitor"
          >
            <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {gpus.length === 0 ? (
          <div className="rounded-md border border-dashed p-3 text-sm text-muted-foreground">
            GPU stats unavailable
          </div>
        ) : (
          gpus.map((gpu, index) => (
            <div key={gpu.uuid || `${gpu.node_id}-${index}`} className="space-y-2">
              <div className="flex items-center justify-between gap-2 text-sm">
                <span className="flex min-w-0 items-center gap-1.5 font-medium">
                  <Cpu className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true" />
                  <span className="truncate">{shortLabel(gpu)}</span>
                </span>
                <span className="shrink-0 font-mono text-sm">
                  {gpu.error ? "error" : pct(gpu.utilization_gpu_percent)}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-500",
                    gpu.error ? "bg-destructive" : "bg-primary",
                  )}
                  style={{ width: gpu.error ? "100%" : utilizationWidth(gpu) }}
                />
              </div>
              <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                <span className="min-w-0 truncate">
                  {gpu.holder ? gpu.holder : gpu.busy ? "busy" : gpu.name || "idle"}
                </span>
                <span className="inline-flex shrink-0 items-center gap-1 font-mono">
                  <Zap className="h-3.5 w-3.5" aria-hidden="true" />
                  pwr {watts(gpu.power_draw_w)}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
