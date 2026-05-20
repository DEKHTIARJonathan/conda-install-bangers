import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GpuStats } from "../gpu-stats";
import { fetchGpuStats } from "@/lib/api/client";

vi.mock("@/lib/api/client", () => ({
  fetchGpuStats: vi.fn(),
}));

function renderWithQueryClient() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <GpuStats />
    </QueryClientProvider>,
  );
}

describe("GpuStats", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    vi.mocked(fetchGpuStats).mockReset();
  });

  it("renders Apple MLX utilization and unified memory", async () => {
    vi.mocked(fetchGpuStats).mockResolvedValue({
      device: "mps",
      provider: "mlx",
      memory_type: "unified",
      gpu_utilization_percent: 65,
      renderer_utilization_percent: 62,
      tiler_utilization_percent: 59,
      vram_used_mb: 2048,
      vram_total_mb: 16384,
      vram_percent: 12.5,
      memory_cache_mb: 512,
      memory_peak_mb: 3072,
      gpus: [
        {
          node_id: "mac-studio",
          node_role: "standalone",
          label: "mac-studio",
          device_index: 0,
          name: "Apple M3",
          uuid: "",
          provider: "mlx",
          memory_type: "unified",
          utilization_gpu_percent: 65,
          utilization_memory_percent: null,
          renderer_utilization_percent: 62,
          tiler_utilization_percent: 59,
          vram_used_mb: 2048,
          vram_total_mb: 16384,
          vram_percent: 12.5,
          memory_cache_mb: 512,
          memory_peak_mb: 3072,
          power_draw_w: null,
          power_limit_w: null,
          busy: true,
          holder: "generation",
          error: "",
        },
      ],
    });

    renderWithQueryClient();

    expect(await screen.findByText("Provider")).toBeInTheDocument();
    expect(screen.getAllByText("mlx").length).toBeGreaterThan(0);
    expect(screen.getByText("Utilization")).toBeInTheDocument();
    expect(screen.getByText("65%")).toBeInTheDocument();
    expect(screen.getByText("Unified memory")).toBeInTheDocument();
    expect(screen.getByText("2.0 GB / 16 GB")).toBeInTheDocument();
    expect(screen.getByText("cache 512 MB / peak 3.0 GB")).toBeInTheDocument();
  });

  it("renders unknown stats without crashing", async () => {
    vi.mocked(fetchGpuStats).mockResolvedValue({
      device: "unknown",
      vram_used_mb: null,
      vram_total_mb: null,
      vram_percent: null,
      gpus: [],
    });

    renderWithQueryClient();

    expect(await screen.findByText("Device")).toBeInTheDocument();
    expect(screen.getByText("unknown")).toBeInTheDocument();
  });
});
