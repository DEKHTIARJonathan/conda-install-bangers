"use client";

import { useEffect, useCallback } from "react";
import { toast } from "sonner";
import {
  submitGeneration,
  cancelJob,
  fetchJobStatus,
  formatCaption as formatCaptionApi,
  createSample as createSampleApi,
} from "@/lib/api/client";
import { useGenerationStore } from "@/stores/generation-store";
import { useGpuStore } from "@/stores/gpu-store";
import { useActiveModel } from "@/hooks/use-active-model";
import { MODE_TO_TASK_TYPE } from "@/lib/constants";
import { registerAutoGenSubmit, unregisterAutoGenSubmit } from "./use-generation-ws";
import type { GenerateRequest } from "@/types/api";
import type {
  AdvancedSettings,
  GenerationJob,
  SimpleForm,
} from "@/stores/generation-store";

/** Modes where thinking is supported (matches official ACE-Step Gradio). */
const thinkingModes = new Set(["Simple", "Custom"]);

type SubmittedJobState = "cancel" | "swap" | "already-swapped";

/** Reconciliation-only: never owns toasts, isGenerating, or GPU store side
 * effects. The websocket handler drives user-visible completion behavior; this
 * function just merges authoritative server state into the store so a missed
 * websocket message can't strand a job in `running` forever. */
export function reconcilePolledStatus(
  status: Awaited<ReturnType<typeof fetchJobStatus>>,
  serverJobId: string,
): "terminal" | "active" | "missing" {
  const store = useGenerationStore.getState();
  const current = store.activeJobs.find((j) => j.jobId === serverJobId);
  if (!current) return "missing";
  if (current.status === "cancelling") return "terminal";
  if (current.status === "completed" || current.status === "failed") return "terminal";

  if (status.status === "completed") {
    store.updateJob(serverJobId, {
      status: "completed",
      progress: 1,
      stage: status.stage || current.stage,
      results: status.results ?? current.results,
      error: null,
      timings: status.timings,
      historyId: status.history_id ?? current.historyId,
    });
    return "terminal";
  }

  if (status.status === "cancelled") {
    // Match websocket handler: silently drop cancelled jobs from the queue.
    store.removeJob(serverJobId);
    return "terminal";
  }

  if (status.status === "failed") {
    store.updateJob(serverJobId, {
      status: "failed",
      progress: status.progress ?? current.progress,
      stage: status.stage || current.stage,
      error: status.error ?? "Generation failed",
      timings: status.timings,
      historyId: status.history_id ?? current.historyId,
    });
    return "terminal";
  }

  store.updateJob(serverJobId, {
    status: status.status === "running" ? "running" : current.status,
    progress: Math.max(status.progress ?? 0, current.progress),
    stage: status.stage || current.stage,
    results: status.results ?? current.results,
    error: status.error ?? current.error,
    timings: status.timings,
    historyId: status.history_id ?? current.historyId,
  });
  return "active";
}

/** Polls /generate/{id} with exponential backoff until terminal or N
 * consecutive errors. Errors don't kill the loop -- transient network blips
 * shouldn't strand reconciliation. */
export async function runReconcilePoller(
  serverJobId: string,
  options: {
    fetchStatus?: typeof fetchJobStatus;
    minDelayMs?: number;
    maxDelayMs?: number;
    maxConsecutiveErrors?: number;
  } = {},
): Promise<void> {
  const fetchStatus = options.fetchStatus ?? fetchJobStatus;
  const hasExplicitMin = options.minDelayMs !== undefined;
  const minDelay = options.minDelayMs ?? 1500;
  const maxDelay = options.maxDelayMs ?? 8000;
  const maxErrors = options.maxConsecutiveErrors ?? 6;

  // Default cadence is ~2s; tests can override via minDelayMs.
  let delay = hasExplicitMin ? minDelay : 2000;
  let consecutiveErrors = 0;

  while (true) {
    await new Promise((resolve) => setTimeout(resolve, delay));
    try {
      const status = await fetchStatus(serverJobId);
      consecutiveErrors = 0;
      const result = reconcilePolledStatus(status, serverJobId);
      if (result === "terminal" || result === "missing") return;
      delay = minDelay;
    } catch {
      consecutiveErrors += 1;
      if (consecutiveErrors >= maxErrors) return;
      delay = Math.min(maxDelay, Math.max(minDelay, delay * 2));
    }
  }
}

export function resolveSubmittedJobState(
  activeJobs: GenerationJob[],
  tempJobId: string,
  serverJobId: string,
): SubmittedJobState {
  const tempJob = activeJobs.find((j) => j.jobId === tempJobId);
  const serverJob = activeJobs.find((j) => j.jobId === serverJobId);

  if (tempJob?.status === "cancelling" || serverJob?.status === "cancelling") {
    return "cancel";
  }
  if (tempJob) {
    return "swap";
  }
  if (serverJob) {
    return "already-swapped";
  }
  return "cancel";
}

export function useGeneration() {
  const {
    activeMode,
    simpleForm,
    customForm,
    remixForm,
    advancedSettings,
    isGenerating,
    isFormatting,
    addJob,
    swapJobId,
    setIsGenerating,
    setIsFormatting,
    updateCustomForm,
    preFormatSnapshot,
    setPreFormatSnapshot,
    autoTitleEnabled,
    customTitle,
  } = useGenerationStore();

  const activeModel = useActiveModel();

  const buildRequest = useCallback((): GenerateRequest => {
    const taskType = MODE_TO_TASK_TYPE[activeMode] ?? "text2music";
    const adv = advancedSettings;
    const base: GenerateRequest = {
      task_type: taskType,
      inference_steps: adv.inferenceSteps,
      guidance_scale: adv.guidanceScale,
      shift: adv.shift,
      infer_method: adv.inferMethod,
      seed: adv.seed,
      thinking: thinkingModes.has(activeMode) && adv.thinking,
      lm_temperature: adv.lmTemperature,
      batch_size: adv.batchSize,
      audio_format: adv.audioFormat,
      auto_title: autoTitleEnabled && !customTitle,
      use_cot_caption: thinkingModes.has(activeMode) && adv.thinking && adv.useCotCaption,
      use_cot_metas: thinkingModes.has(activeMode) && adv.thinking && adv.useCotMetas,
      use_cot_language: thinkingModes.has(activeMode) && adv.thinking && adv.useCotLanguage,
    };

    if (activeMode === "Custom") {
      return {
        ...base,
        caption: customForm.caption,
        lyrics: customForm.instrumental ? "" : customForm.lyrics,
        instrumental: customForm.instrumental,
        vocal_language: "en",
        bpm: customForm.bpm ?? undefined,
        keyscale: customForm.keyscale,
        timesignature: customForm.timesignature,
        duration: adv.defaultDuration,
      };
    }

    if (activeMode === "Remix") {
      return {
        ...base,
        task_type: "music2music",
        caption: remixForm.caption,
        lyrics: remixForm.lyrics,
        vocal_language: "en",
        audio_cover_strength: remixForm.coverStrength,
      };
    }

    // Simple mode — should not reach here (handled by submit)
    return base;
  }, [activeMode, customForm, remixForm, advancedSettings, autoTitleEnabled, customTitle]);

  const submit = useCallback(async () => {
    // Read fresh from store to avoid stale closure (auto-gen keeps isGenerating=true)
    if (useGenerationStore.getState().isGenerating) return;
    setIsGenerating(true);

    // Notify user if GPU is already busy — backend will queue via await_acquire
    const gpuHolder = useGpuStore.getState().holder;
    if (gpuHolder) {
      toast.info(`Queuing — GPU in use by ${gpuHolder}...`);
    }
    useGpuStore.getState().setHolder("generation");

    // Add job to queue immediately so sidebar shows it right away
    // (before any API calls that might block on GPU lock)
    const tempJobId = crypto.randomUUID();
    addJob({
      jobId: tempJobId,
      status: "queued",
      progress: 0,
      stage: gpuHolder ? `Waiting for GPU (${gpuHolder})...` : "",
      results: [],
      error: null,
      historyId: null,
      savedVariants: [],
      generatedTitle: null,
      submittedCustomForm: activeMode === "Custom" ? { ...customForm } : undefined,
    });

    try {
      let serverJobId: string;

      if (activeMode === "Simple") {
        // ACE-Step Simple mode: two-step create sample then auto-generate
        const sample = await createSampleApi({
          query: simpleForm.prompt,
          instrumental: simpleForm.instrumental,
          vocal_language: "en",
          temperature: advancedSettings.lmTemperature,
        });

        // Check if job was cancelled during sample creation
        const jobAfterSample = useGenerationStore.getState().activeJobs.find((j) => j.jobId === tempJobId);
        if (!jobAfterSample || jobAfterSample.status === "cancelling") {
          if (jobAfterSample) {
            setTimeout(() => useGenerationStore.getState().removeJob(tempJobId), 1000);
          }
          setIsGenerating(false);
          useGpuStore.getState().clear();
          return;
        }

        if (!sample.success) {
          toast.error(sample.error ?? "Sample creation failed");
          useGenerationStore.getState().removeJob(tempJobId);
          setIsGenerating(false);
          useGpuStore.getState().clear();
          return;
        }

        const request: GenerateRequest = {
          task_type: "text2music",
          caption: sample.caption,
          lyrics: sample.instrumental ? "" : sample.lyrics,
          instrumental: sample.instrumental,
          vocal_language: "en",
          bpm: sample.bpm ?? undefined,
          keyscale: sample.keyscale,
          timesignature: sample.timesignature,
          duration: advancedSettings.defaultDuration,
          inference_steps: advancedSettings.inferenceSteps,
          guidance_scale: advancedSettings.guidanceScale,
          seed: advancedSettings.seed,
          thinking: advancedSettings.thinking,
          lm_temperature: advancedSettings.lmTemperature,
          batch_size: advancedSettings.batchSize,
          audio_format: advancedSettings.audioFormat,
          auto_title: autoTitleEnabled && !customTitle,
          quality_profile: sample.quality_profile,
          spec_source: sample.spec_source,
          source_prompt: sample.source_prompt || simpleForm.prompt,
        };

        const response = await submitGeneration(request);
        serverJobId = response.job_id;
      } else {
        const request = buildRequest();
        const response = await submitGeneration(request);
        serverJobId = response.job_id;
      }

      // Cancelled during submitGeneration? Cancel the real backend job.
      // WebSocket progress can also arrive before the POST resolves and swap
      // the temp ID to the server ID, which should continue normally.
      const submittedJobState = resolveSubmittedJobState(
        useGenerationStore.getState().activeJobs,
        tempJobId,
        serverJobId,
      );
      if (submittedJobState === "cancel") {
        cancelJob(serverJobId).catch(() => {});
        if (useGenerationStore.getState().activeJobs.some((j) => j.jobId === tempJobId)) {
          setTimeout(() => useGenerationStore.getState().removeJob(tempJobId), 1000);
        }
        setIsGenerating(false);
        useGpuStore.getState().clear();
        return;
      }

      if (submittedJobState === "swap") {
        swapJobId(tempJobId, serverJobId);
      }

      void runReconcilePoller(serverJobId);
    } catch (err) {
      useGenerationStore.getState().removeJob(tempJobId);
      const message = err instanceof Error ? err.message : "Generation failed";
      toast.error(message);
      setIsGenerating(false);
      useGpuStore.getState().clear();
    }
  }, [
    activeMode,
    simpleForm,
    customForm,
    advancedSettings,
    buildRequest,
    addJob,
    swapJobId,
    setIsGenerating,
    autoTitleEnabled,
    customTitle,
  ]);

  // Register submit for auto-gen re-submit from the global WS handler
  useEffect(() => {
    registerAutoGenSubmit(submit);
    return () => unregisterAutoGenSubmit();
  }, [submit]);

  const formatCaptionAction = useCallback(async () => {
    if (isFormatting) return;
    setIsFormatting(true);

    setPreFormatSnapshot({ ...customForm });

    try {
      const result = await formatCaptionApi({
        caption: customForm.caption,
        lyrics: customForm.lyrics,
        bpm: customForm.bpm,
        keyscale: customForm.keyscale,
        timesignature: customForm.timesignature,
        duration: advancedSettings.defaultDuration,
        vocal_language: "en",
      });

      if (result.success) {
        updateCustomForm({
          caption: result.caption || customForm.caption,
          lyrics: result.lyrics || customForm.lyrics,
          bpm: result.bpm ?? customForm.bpm,
          keyscale: result.keyscale || customForm.keyscale,
          timesignature: result.timesignature || customForm.timesignature,
        });
      } else {
        toast.error(result.error ?? "Formatting failed");
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Formatting failed";
      toast.error(message);
    } finally {
      setIsFormatting(false);
    }
  }, [customForm, isFormatting, setIsFormatting, setPreFormatSnapshot, updateCustomForm]);

  const canSubmit =
    !isGenerating &&
    activeModel.isLoaded &&
    (activeMode === "Simple"
      ? simpleForm.prompt.trim().length > 0
      : activeMode === "Custom"
        ? customForm.caption.trim().length > 0
        : activeMode === "Remix"
          ? remixForm.audioFilePath.length > 0
          : false);

  const canFormat =
    !isFormatting &&
    activeModel.lmLoaded &&
    customForm.caption.trim().length > 0;

  const canUndoFormat = preFormatSnapshot !== null;

  const undoFormat = useCallback(() => {
    if (!preFormatSnapshot) return;
    updateCustomForm(preFormatSnapshot);
    setPreFormatSnapshot(null);
  }, [preFormatSnapshot, updateCustomForm, setPreFormatSnapshot]);

  return {
    submit,
    formatCaption: formatCaptionAction,
    canSubmit,
    canFormat,
    canUndoFormat,
    undoFormat,
    isGenerating,
    isFormatting,
  };
}
