import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GpuMonitorInset } from "../gpu-monitor-inset";
import { fetchGpuStats } from "@/lib/api/client";

vi.mock("@/lib/api/client", () => ({
  fetchGpuStats: vi.fn(),
}));

const COLLAPSED_KEY = "bangers-gpu-monitor-collapsed";

function renderWithQueryClient() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <GpuMonitorInset hasMiniPlayer={false} />
    </QueryClientProvider>,
  );
}

describe("GpuMonitorInset", () => {
  afterEach(() => {
    cleanup();
    window.localStorage.clear();
  });

  beforeEach(() => {
    vi.mocked(fetchGpuStats).mockReset();
    vi.mocked(fetchGpuStats).mockResolvedValue({
      device: "mps",
      provider: "mlx",
      memory_type: "unified",
      gpu_utilization_percent: 42,
      gpus: [
        {
          node_id: "local",
          node_role: "standalone",
          label: "local",
          device_index: 0,
          name: "Apple GPU",
          uuid: "",
          provider: "mlx",
          memory_type: "unified",
          utilization_gpu_percent: 42,
          utilization_memory_percent: null,
          renderer_utilization_percent: null,
          tiler_utilization_percent: null,
          vram_used_mb: null,
          vram_total_mb: null,
          vram_percent: null,
          memory_cache_mb: null,
          memory_peak_mb: null,
          power_draw_w: null,
          power_limit_w: null,
          busy: false,
          holder: null,
          error: "",
        },
      ],
    });
  });

  it("collapses and expands the floating GPU monitor", async () => {
    const user = userEvent.setup();
    renderWithQueryClient();

    expect(await screen.findByText("local")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /collapse gpu monitor/i }));

    expect(screen.getByRole("button", { name: /expand gpu monitor/i })).toBeInTheDocument();
    expect(screen.queryByText("local")).not.toBeInTheDocument();
    expect(window.localStorage.getItem(COLLAPSED_KEY)).toBe("true");

    await user.click(screen.getByRole("button", { name: /expand gpu monitor/i }));

    expect(await screen.findByText("local")).toBeInTheDocument();
    expect(window.localStorage.getItem(COLLAPSED_KEY)).toBe("false");
  });

  it("starts collapsed when saved that way", () => {
    window.localStorage.setItem(COLLAPSED_KEY, "true");

    renderWithQueryClient();

    expect(screen.getByRole("button", { name: /expand gpu monitor/i })).toBeInTheDocument();
  });
});
