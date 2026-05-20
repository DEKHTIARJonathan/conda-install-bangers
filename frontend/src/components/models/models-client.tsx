"use client";

import { useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient, useIsMutating } from "@tanstack/react-query";
import { Check, Cpu, Download, ExternalLink, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import {
  fetchModels,
  fetchAvailableModels,
  fetchHealth,
  downloadModel,
  switchChatLlmModel,
} from "@/lib/api/client";
import { fetchDJInfo } from "@/lib/api/dj-client";
import type { AvailableModel, ModelInfo } from "@/types/api";
import { ModelCards } from "./model-cards";
import { GpuStats } from "./gpu-stats";

type ChatLlmRow =
  | {
      state: "installed";
      key: string;
      name: string;
      compatibility: string[];
      format: string;
      quantization: string;
      description: string;
      isActive: boolean;
      isLoading: boolean;
      loadedOn: string[];
    }
  | {
      state: "downloadable";
      key: string;
      name: string;
      model: AvailableModel;
    }
  | {
      state: "unsupported";
      key: string;
      name: string;
      compatibility: string[];
      format: string;
      quantization: string;
      description: string;
      isActive: boolean;
      isLoading: boolean;
      loadedOn: string[];
    };

const IS_MAC =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad|iPod/.test(navigator.platform);
const MODEL_INIT_STAGES = new Set(["downloading", "loading_dit", "loading_lm", "loading_chat_llm"]);
const MODEL_TERMINAL_INIT_STAGES = new Set(["ready", "error"]);

function isMlxOnly(compatibility: readonly string[]): boolean {
  return compatibility.length > 0 && compatibility.every((c) => c === "mlx");
}

function hasLoadingModel(models?: {
  dit_models: ModelInfo[];
  lm_models: ModelInfo[];
  chat_llm_models: ModelInfo[];
}) {
  return Boolean(
    models &&
      [...models.dit_models, ...models.lm_models, ...models.chat_llm_models].some(
        (model) => model.is_loading,
      ),
  );
}

function ModelCompatibilityBadges({
  model,
}: {
  model: Pick<ModelInfo, "compatibility" | "quantization">;
}) {
  const compatibility = (model.compatibility ?? []).filter((runtime) => runtime === "mlx");
  return (
    <>
      {compatibility.map((runtime) => (
        <Badge key={runtime} variant="outline" className="text-[10px] uppercase text-muted-foreground">
          {runtime}
        </Badge>
      ))}
      {model.quantization ? (
        <Badge variant="outline" className="text-[10px] text-muted-foreground">
          {model.quantization}
        </Badge>
      ) : null}
    </>
  );
}

export function ModelsClient() {
  const queryClient = useQueryClient();
  const modelLoadMutationCount = useIsMutating({ mutationKey: ["model-load"] });

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    retry: false,
  });
  const initStage = health?.init_stage ?? "";
  const isModelInitializing = MODEL_INIT_STAGES.has(initStage);

  const { data: models, isLoading } = useQuery({
    queryKey: ["models"],
    queryFn: fetchModels,
    refetchInterval: (query) =>
      isModelInitializing || hasLoadingModel(query.state.data) ? 1_000 : false,
  });

  const { data: available } = useQuery({
    queryKey: ["available-models"],
    queryFn: fetchAvailableModels,
    refetchInterval: (query) =>
      query.state.data?.models.some((m) => m.downloading) ? 3_000 : 30_000,
  });

  const djInfoQuery = useQuery({
    queryKey: ["dj-info"],
    queryFn: fetchDJInfo,
    retry: false,
  });

  const chatLlmSwitchMutation = useMutation({
    mutationKey: ["model-load"],
    mutationFn: switchChatLlmModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["dj-info"] });
      queryClient.invalidateQueries({ queryKey: ["radio-settings"] });
      queryClient.invalidateQueries({ queryKey: ["settings"] });
      toast.success("Chat model loaded");
    },
    onError: (err: Error) => {
      toast.error(`Failed to load chat model: ${err.message}`);
    },
  });

  const djDownloadMutation = useMutation({
    mutationFn: downloadModel,
    onSuccess: (_data, name) => {
      toast.success(`Download started for ${name}`);
      queryClient.invalidateQueries({ queryKey: ["available-models"] });
    },
    onError: (err: Error) => toast.error(`Download failed: ${err.message}`),
  });

  const prevDownloadingRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (isModelInitializing || MODEL_TERMINAL_INIT_STAGES.has(initStage)) {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    }
  }, [initStage, isModelInitializing, queryClient]);

  useEffect(() => {
    if (!available) return;

    const currentDownloading = new Set(
      available.models.filter((m) => m.downloading).map((m) => m.name),
    );
    const prev = prevDownloadingRef.current;

    for (const name of prev) {
      if (!currentDownloading.has(name)) {
        const model = available.models.find((m) => m.name === name);
        if (model?.installed) {
          toast.success(`Downloaded ${name} successfully`);
          queryClient.invalidateQueries({ queryKey: ["models"] });
          queryClient.invalidateQueries({ queryKey: ["dj-info"] });
        } else {
          toast.error(`Download failed for ${name}`);
        }
      }
    }

    prevDownloadingRef.current = currentDownloading;
  }, [available, queryClient]);

  const ditAvailable = available?.models.filter((m) => m.model_type === "dit") ?? [];
  const lmAvailable = available?.models.filter((m) => m.model_type === "lm") ?? [];
  const chatLlmAvailable = available?.models.filter((m) => m.model_type === "chat_llm") ?? [];
  const hasAnyModelLoading = isModelInitializing || hasLoadingModel(models) || modelLoadMutationCount > 0;
  const hasChatLlmLoadedOrLoading =
    models?.chat_llm_models.some((model) => model.is_active || model.is_loading) ?? false;
  const hasChatLlmLoading =
    models?.chat_llm_models.some((model) => model.is_loading) ?? false;

  const chatLlmRows: ChatLlmRow[] = (() => {
    const installedScanned = models?.chat_llm_models ?? [];
    const availableByName = new Map(
      chatLlmAvailable.map((model) => [model.name, model]),
    );
    const seenNames = new Set<string>();
    const rows: ChatLlmRow[] = [];

    for (const installed of installedScanned) {
      seenNames.add(installed.name);
      const compatibility = installed.compatibility ?? [];
      const isUnsupported = !IS_MAC && isMlxOnly(compatibility);
      rows.push({
        state: isUnsupported ? "unsupported" : "installed",
        key: `${installed.name}:${isUnsupported ? "unsupported" : "installed"}`,
        name: installed.name,
        compatibility,
        format: installed.format ?? "",
        quantization: installed.quantization ?? "",
        description: availableByName.get(installed.name)?.description ?? "",
        isActive: installed.is_active,
        isLoading: Boolean(installed.is_loading),
        loadedOn: installed.loaded_on ?? [],
      });
    }

    for (const candidate of chatLlmAvailable) {
      if (seenNames.has(candidate.name)) continue;
      const compatibility = candidate.compatibility ?? [];
      const isUnsupported = !IS_MAC && isMlxOnly(compatibility);
      if (isUnsupported && !candidate.downloading) {
        rows.push({
          state: "unsupported",
          key: `${candidate.name}:unsupported`,
          name: candidate.name,
          compatibility,
          format: candidate.format ?? "",
          quantization: candidate.quantization ?? "",
          description: candidate.description ?? "",
          isActive: false,
          isLoading: false,
          loadedOn: [],
        });
      } else {
        rows.push({
          state: "downloadable",
          key: `${candidate.name}:downloadable`,
          name: candidate.name,
          model: candidate,
        });
      }
    }

    const loadedModel = models?.chat_llm_models.find((model) => model.is_active)?.name ?? "";
    const rowRank = (row: ChatLlmRow) => {
      if (row.state === "installed") return 0;
      if (row.state === "downloadable")
        return "model" in row && row.model.downloading ? 1 : 2;
      return 3;
    };
    const isLoaded = (row: ChatLlmRow) =>
      row.state === "installed" && row.name === loadedModel;

    return rows.sort((a, b) => {
      const loadedDelta = Number(isLoaded(b)) - Number(isLoaded(a));
      if (loadedDelta !== 0) return loadedDelta;
      const rankDelta = rowRank(a) - rowRank(b);
      if (rankDelta !== 0) return rankDelta;
      return a.name.localeCompare(b.name);
    });
  })();

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Cpu className="h-6 w-6 text-primary" />
        <h1 className="text-2xl font-semibold">Models</h1>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          {isLoading ? (
            <>
              <Skeleton className="h-48 w-full rounded-xl" />
              <Skeleton className="h-48 w-full rounded-xl" />
            </>
          ) : (
            <>
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center justify-between text-lg">
                    ACE-Step
                    <a href="https://github.com/ace-step/ACE-Step-1.5" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-xs font-normal text-primary hover:underline">
                      GitHub <ExternalLink className="h-3 w-3" />
                    </a>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <ModelCards
                    title="DiT Models"
                    models={models?.dit_models ?? []}
                    availableModels={ditAvailable}
                    modelType="dit"
                    loadDisabled={hasAnyModelLoading}
                  />
                  <ModelCards
                    title="Language Models"
                    models={models?.lm_models ?? []}
                    availableModels={lmAvailable}
                    modelType="lm"
                    loadDisabled={hasAnyModelLoading}
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">Chat LLM</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {djInfoQuery.isLoading ? (
                    <Skeleton className="h-32 w-full rounded-lg" />
                  ) : (
                    <div className="space-y-2">
                      {chatLlmRows.length === 0 ? (
                        <p className="text-sm text-muted-foreground">No models found</p>
                      ) : (
                        chatLlmRows.map((row) => {
                          if (row.state === "downloadable") {
                            const model = row.model;
                            const tooltipText = [
                              model.description,
                              model.size_mb > 0
                                ? `~${model.size_mb >= 1000 ? `${(model.size_mb / 1000).toFixed(1)} GB` : `${model.size_mb} MB`} download.`
                                : "",
                            ].filter(Boolean).join(" ");
                            const pct = Math.round(model.download_progress * 100);
                            const loading =
                              model.downloading ||
                              (djDownloadMutation.isPending && djDownloadMutation.variables === model.name);

                            return (
                              <div
                                key={row.key}
                                className="rounded-lg border border-dashed border-border p-3 opacity-75"
                              >
                                <div className="flex items-center justify-between gap-3">
                                  <div className="flex min-w-0 flex-wrap items-center gap-2">
                                    {tooltipText ? (
                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <span className="text-sm font-medium cursor-help">{model.name}</span>
                                        </TooltipTrigger>
                                        <TooltipContent side="top" className="max-w-xs">
                                          {tooltipText}
                                        </TooltipContent>
                                      </Tooltip>
                                    ) : (
                                      <span className="text-sm font-medium">{model.name}</span>
                                    )}
                                    <ModelCompatibilityBadges model={model} />
                                    {model.downloading ? (
                                      <Badge variant="outline" className="text-yellow-600 dark:text-yellow-400">
                                        Downloading{pct > 0 ? ` ${pct}%` : ""}
                                      </Badge>
                                    ) : null}
                                  </div>
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => djDownloadMutation.mutate(model.name)}
                                    disabled={loading}
                                  >
                                    {loading ? (
                                      <>
                                        <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                        Downloading...
                                      </>
                                    ) : (
                                      <>
                                        <Download className="mr-1 h-3 w-3" />
                                        Download
                                      </>
                                    )}
                                  </Button>
                                </div>
                                {model.downloading && (
                                  <div className="mt-2 h-1.5 w-full rounded-full bg-muted overflow-hidden">
                                    <div
                                      className="h-full rounded-full bg-yellow-400 transition-all duration-500"
                                      style={{ width: `${Math.max(pct, 2)}%` }}
                                    />
                                  </div>
                                )}
                              </div>
                            );
                          }

                          const isLoadingModel =
                            row.state === "installed" && row.isLoading;
                          const isLoaded =
                            row.state === "installed" && row.isActive && !row.isLoading;
                          const badgesModel = {
                            compatibility: row.compatibility,
                            quantization: row.quantization,
                          };

                          return (
                            <div
                              key={row.key}
                              className={`flex items-center justify-between gap-3 rounded-lg border border-border p-3 ${
                                row.state === "unsupported" ? "opacity-50" : ""
                              }`}
                            >
                              <div className="flex min-w-0 flex-wrap items-center gap-2">
                                {row.description ? (
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <span className="text-sm font-medium cursor-help">{row.name}</span>
                                    </TooltipTrigger>
                                    <TooltipContent side="top" className="max-w-xs">
                                      {row.description}
                                    </TooltipContent>
                                  </Tooltip>
                                ) : (
                                  <span className="text-sm font-medium">{row.name}</span>
                                )}
                                <ModelCompatibilityBadges model={badgesModel} />
                                {row.loadedOn.map((node) => (
                                  <Badge key={node} variant="outline" className="text-[10px] text-muted-foreground">
                                    {node.replace(/^spark-/, "")}
                                  </Badge>
                                ))}
                              </div>
                              {row.state === "unsupported" ? null : isLoadingModel ? (
                                <Button variant="outline" size="sm" disabled>
                                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                  Loading...
                                </Button>
                              ) : isLoaded ? (
                                <Button variant="outline" size="sm" className="pointer-events-none text-green-500 border-green-500/30">
                                  Loaded
                                  <Check className="ml-1 h-3 w-3" />
                                </Button>
                              ) : (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => chatLlmSwitchMutation.mutate(row.name)}
                                  disabled={hasAnyModelLoading || hasChatLlmLoading || chatLlmSwitchMutation.isPending}
                                >
                                  {chatLlmSwitchMutation.isPending &&
                                    chatLlmSwitchMutation.variables === row.name && (
                                      <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                    )}
                                  {hasChatLlmLoadedOrLoading ? "Switch" : "Load"}
                                </Button>
                              )}
                            </div>
                          );
                        })
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </div>

        <div>
          <GpuStats />
        </div>
      </div>
    </div>
  );
}
