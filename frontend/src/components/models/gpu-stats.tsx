"use client";

import { useQuery } from "@tanstack/react-query";
import { Cpu } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchGpuStats } from "@/lib/api/client";
import type { GpuDeviceStats } from "@/types/api";

function shortLabel(gpu: GpuDeviceStats): string {
  return (gpu.label || gpu.node_id || gpu.name || "GPU")
    .replace(/^spark-/, "")
    .replace(/-music$/, "")
    .replace(/-lm-chat$/, "");
}

function pct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "--";
  return `${Math.round(value)}%`;
}

function barWidth(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "0%";
  return `${Math.max(0, Math.min(100, value))}%`;
}

function formatMb(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "--";
  if (value >= 1024) {
    const gb = value / 1024;
    return `${gb >= 10 ? gb.toFixed(0) : gb.toFixed(1)} GB`;
  }
  return `${value.toFixed(0)} MB`;
}

function memoryLabel(memoryType: string | null | undefined): string {
  if (memoryType === "unified") return "Unified memory";
  if (memoryType === "allocated") return "Allocated";
  return "VRAM";
}

export function GpuStats() {
  const { data: stats, isLoading, isError } = useQuery({
    queryKey: ["gpu-stats"],
    queryFn: fetchGpuStats,
    refetchInterval: 5000,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Cpu className="h-4 w-4" />
            GPU
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-2 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (isError || !stats) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Cpu className="h-4 w-4" />
            GPU
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">GPU stats unavailable</p>
        </CardContent>
      </Card>
    );
  }

  const gpus = stats.gpus ?? [];

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Cpu className="h-4 w-4" />
          GPU
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Device</span>
          <Badge variant="secondary">{stats.device || "unknown"}</Badge>
        </div>

        {stats.provider && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Provider</span>
            <Badge variant="outline">{stats.provider}</Badge>
          </div>
        )}

        {gpus.length > 0 ? (
          <div className="space-y-2">
            {gpus.map((gpu, index) => (
              <div
                key={gpu.uuid || `${gpu.node_id}-${index}`}
                className="rounded-md border border-border p-2"
              >
                <div className="flex items-center justify-between gap-2 text-sm">
                  <span className="min-w-0 truncate font-medium">{shortLabel(gpu)}</span>
                  {gpu.provider && (
                    <Badge variant="outline" className="shrink-0 text-[10px]">
                      {gpu.provider}
                    </Badge>
                  )}
                </div>
                <div className="mt-2 flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Utilization</span>
                  <span className="font-mono">
                    {gpu.error ? "error" : pct(gpu.utilization_gpu_percent)}
                  </span>
                </div>
                <div className="mt-1 h-2 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary transition-all duration-300"
                    style={{ width: barWidth(gpu.utilization_gpu_percent) }}
                  />
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {gpu.holder ? gpu.holder : gpu.busy ? "busy" : gpu.name || "idle"}
                </div>
                {gpu.vram_used_mb != null && (
                  <div className="mt-2 flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">
                      {memoryLabel(gpu.memory_type || stats.memory_type)}
                    </span>
                    <span className="font-mono">
                      {formatMb(gpu.vram_used_mb)}
                      {gpu.vram_total_mb != null ? ` / ${formatMb(gpu.vram_total_mb)}` : ""}
                    </span>
                  </div>
                )}
                {(gpu.memory_cache_mb != null || gpu.memory_peak_mb != null) && (
                  <div className="mt-1 flex items-center justify-between text-xs text-muted-foreground">
                    <span>MLX</span>
                    <span className="font-mono">
                      cache {formatMb(gpu.memory_cache_mb)} / peak {formatMb(gpu.memory_peak_mb)}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <>
            {stats.gpu_utilization_percent != null && (
              <>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Utilization</span>
                  <span className="font-mono">{pct(stats.gpu_utilization_percent)}</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary transition-all duration-300"
                    style={{ width: barWidth(stats.gpu_utilization_percent) }}
                  />
                </div>
              </>
            )}
            {stats.vram_total_mb != null && (
              <>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">{memoryLabel(stats.memory_type)}</span>
                  <span>
                    {formatMb(stats.vram_used_mb)} / {formatMb(stats.vram_total_mb)}
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary transition-all duration-300"
                    style={{ width: barWidth(stats.vram_percent) }}
                  />
                </div>
              </>
            )}
          </>
        )}

        {stats.vram_used_mb != null && stats.vram_total_mb == null && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">{memoryLabel(stats.memory_type)}</span>
            <span>{formatMb(stats.vram_used_mb)}</span>
          </div>
        )}

        {(stats.memory_cache_mb != null || stats.memory_peak_mb != null) && gpus.length === 0 && (
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>MLX</span>
            <span className="font-mono">
              cache {formatMb(stats.memory_cache_mb)} / peak {formatMb(stats.memory_peak_mb)}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
