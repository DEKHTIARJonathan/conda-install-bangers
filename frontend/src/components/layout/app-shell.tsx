"use client";

import { type ReactNode, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Menu, Terminal } from "lucide-react";
import { cn } from "@/lib/utils";
import { useSidebarStore } from "@/stores/sidebar-store";
import { usePlayerStore } from "@/stores/player-store";
import { useGenerationStore } from "@/stores/generation-store";
import { useKeyboardShortcuts } from "@/hooks/use-keyboard-shortcuts";
import { useMediaSession } from "@/hooks/use-media-session";
import { fetchSettings } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { useRadioPlayback } from "@/hooks/use-radio-playback";
import { useGenerationWs } from "@/hooks/use-generation-ws";
import { useHealthWs } from "@/hooks/use-health-ws";
import { Sidebar } from "./sidebar";
import { MiniPlayer } from "./mini-player";
import { MiniPlayerQueue } from "./mini-player-queue";
import { FullPlayer } from "./full-player";
import { ConnectionBanner } from "./connection-banner";
import { PageTransition } from "./page-transition";
import { GpuMonitorInset } from "./gpu-monitor-inset";

export function AppShell({ children }: { children: ReactNode }) {
  useKeyboardShortcuts();
  useMediaSession();
  useRadioPlayback();
  useGenerationWs();
  useHealthWs();
  const serverSettingsQuery = useQuery({
    queryKey: ["settings", "startup-defaults"],
    queryFn: fetchSettings,
    retry: 12,
    retryDelay: 2_000,
  });

  useEffect(() => {
    const settings = serverSettingsQuery.data?.settings;
    if (!settings) return;
    const batchSize = Number.parseInt(settings.batch_size ?? "", 10);
    const inferenceSteps = Number.parseInt(settings.inference_steps ?? "", 10);
    const guidanceScale = Number.parseFloat(settings.guidance_scale ?? "");
    const defaultDuration = Number.parseFloat(settings.default_duration ?? "");
    const store = useGenerationStore.getState();
    store.updateAdvancedSettings({
      ...(Number.isFinite(batchSize) ? { batchSize } : {}),
      ...(Number.isFinite(inferenceSteps) ? { inferenceSteps } : {}),
      ...(Number.isFinite(guidanceScale) ? { guidanceScale } : {}),
      ...(Number.isFinite(defaultDuration) ? { defaultDuration } : {}),
      ...(settings.audio_format ? { audioFormat: settings.audio_format } : {}),
      ...(settings.thinking ? { thinking: settings.thinking === "true" } : {}),
    });
  }, [serverSettingsQuery.data]);
  const collapsed = useSidebarStore((s) => s.collapsed);
  const toggleMobile = useSidebarStore((s) => s.toggleMobile);
  const currentSong = usePlayerStore((s) => s.currentSong);

  return (
    <div className="min-h-screen">
      <Sidebar />

      {/* Mobile header */}
      <div className="fixed left-0 right-0 top-0 z-20 flex h-12 items-center border-b bg-background px-3 md:hidden">
        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={toggleMobile}>
          <Menu className="h-5 w-5" />
        </Button>
        <span className="bangers-brand-mark bangers-brand-mark-sm ml-2" aria-hidden="true">
          <Terminal />
        </span>
        <span className="ml-2 min-w-0 flex-1 truncate text-sm font-extrabold">
          conda install bangers
        </span>
        <span className="bangers-mobile-tag shrink-0 px-2 py-0.5 text-[10px] font-extrabold uppercase">
          Long Beach
        </span>
      </div>

      <div
        className={cn(
          "flex min-h-screen flex-col transition-all duration-250 mt-12 md:mt-0",
          collapsed ? "md:ml-16" : "md:ml-60",
          currentSong && "pb-[72px]",
        )}
      >
        <ConnectionBanner />
        <main className="flex-1 bg-background p-3 sm:p-6">
          <PageTransition>{children}</PageTransition>
        </main>
      </div>
      <MiniPlayer />
      <MiniPlayerQueue />
      <FullPlayer />
      <GpuMonitorInset hasMiniPlayer={Boolean(currentSong)} />
    </div>
  );
}
