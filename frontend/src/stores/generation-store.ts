import { create } from "zustand";
import { DURATION_DEFAULT } from "@/lib/constants";
import type { AudioResult } from "@/types/api";

export type GenerationMode = "Simple" | "Custom" | "Remix";

export interface SimpleForm {
  prompt: string;
  instrumental: boolean;
}

export interface CustomForm {
  caption: string;
  lyrics: string;
  instrumental: boolean;
  bpm: number | null;
  keyscale: string;
  timesignature: string;
}

export interface RemixForm {
  audioFilePath: string;
  audioFileName: string;
  caption: string;
  lyrics: string;
  coverStrength: number;
  sourceSongId: string;
}

export interface AdvancedSettings {
  inferenceSteps: number;
  guidanceScale: number;
  shift: number;
  inferMethod: string;
  seed: number;
  thinking: boolean;
  lmTemperature: number;
  batchSize: number;
  audioFormat: string;
  defaultDuration: number;
  useCotCaption: boolean;
  useCotMetas: boolean;
  useCotLanguage: boolean;
}

const DEFAULT_ADVANCED_SETTINGS: AdvancedSettings = {
  inferenceSteps: 8,
  guidanceScale: 7,
  shift: 3,
  inferMethod: "ode",
  seed: -1,
  thinking: true,
  lmTemperature: 0.85,
  batchSize: 2,
  audioFormat: "flac",
  defaultDuration: DURATION_DEFAULT,
  useCotCaption: false,
  useCotMetas: true,
  useCotLanguage: true,
};

export interface GenerationJob {
  jobId: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelling";
  progress: number;
  stage: string;
  results: AudioResult[];
  error: string | null;
  historyId: string | null;
  savedVariants: number[];
  generatedTitle: string | null;
  timings?: Record<string, number>;
  hiddenFromQueue?: boolean;
  submittedCustomForm?: CustomForm;
}

interface GenerationState {
  activeMode: GenerationMode;
  simpleForm: SimpleForm;
  customForm: CustomForm;
  remixForm: RemixForm;
  advancedSettings: AdvancedSettings;
  activeJobs: GenerationJob[];
  isGenerating: boolean;
  isFormatting: boolean;

  // AutoGen
  autoGenEnabled: boolean;
  autoSaveEnabled: boolean;
  autoGenCount: number;
  autoGenMaxRuns: number; // 0 = unlimited

  // Undo Format
  preFormatSnapshot: CustomForm | null;

  // Auto-Title
  autoTitleEnabled: boolean;
  customTitle: string;

  setActiveMode: (mode: GenerationMode) => void;
  updateSimpleForm: (partial: Partial<SimpleForm>) => void;
  updateCustomForm: (partial: Partial<CustomForm>) => void;
  updateRemixForm: (partial: Partial<RemixForm>) => void;
  updateAdvancedSettings: (partial: Partial<AdvancedSettings>) => void;
  addJob: (job: GenerationJob) => void;
  updateJob: (jobId: string, partial: Partial<GenerationJob>) => void;
  markVariantSaved: (jobId: string, index: number) => void;
  setIsGenerating: (v: boolean) => void;
  setIsFormatting: (v: boolean) => void;
  setAutoGenEnabled: (v: boolean) => void;
  setAutoSaveEnabled: (v: boolean) => void;
  incrementAutoGenCount: () => void;
  resetAutoGenCount: () => void;
  setAutoGenMaxRuns: (v: number) => void;
  setPreFormatSnapshot: (snapshot: CustomForm | null) => void;
  resetCustomForm: () => void;
  setAutoTitleEnabled: (v: boolean) => void;
  setCustomTitle: (v: string) => void;
  swapJobId: (oldId: string, newId: string) => void;
  removeJob: (jobId: string) => void;
  hideJob: (jobId: string) => void;
  clearJobs: () => void;
  setJobTitle: (jobId: string, title: string) => void;
  loadFromHistoryParams: (params: Record<string, unknown>) => void;
}

export const useGenerationStore = create<GenerationState>()((set) => ({
  activeMode: "Custom",

  simpleForm: {
    prompt: "",
    instrumental: false,
  },

  customForm: {
    caption: "",
    lyrics: "",
    instrumental: false,
    bpm: null,
    keyscale: "",
    timesignature: "",
  },

  remixForm: {
    audioFilePath: "",
    audioFileName: "",
    caption: "",
    lyrics: "",
    coverStrength: 1.0,
    sourceSongId: "",
  },

  advancedSettings: { ...DEFAULT_ADVANCED_SETTINGS },

  activeJobs: [],
  isGenerating: false,
  isFormatting: false,

  preFormatSnapshot: null,

  autoGenEnabled: false,
  autoSaveEnabled: false,
  autoGenCount: 0,
  autoGenMaxRuns: 0,

  autoTitleEnabled: true,
  customTitle: "",

  setActiveMode: (mode) => set({ activeMode: mode }),

  updateSimpleForm: (partial) =>
    set((s) => ({ simpleForm: { ...s.simpleForm, ...partial } })),

  updateCustomForm: (partial) =>
    set((s) => ({ customForm: { ...s.customForm, ...partial } })),

  updateRemixForm: (partial) =>
    set((s) => ({ remixForm: { ...s.remixForm, ...partial } })),

  updateAdvancedSettings: (partial) =>
    set((s) => ({ advancedSettings: { ...s.advancedSettings, ...partial } })),

  addJob: (job) =>
    set((s) => ({ activeJobs: [job, ...s.activeJobs] })),

  updateJob: (jobId, partial) =>
    set((s) => ({
      activeJobs: s.activeJobs.map((j) =>
        j.jobId === jobId ? { ...j, ...partial } : j,
      ),
    })),

  markVariantSaved: (jobId, index) =>
    set((s) => ({
      activeJobs: s.activeJobs.map((j) =>
        j.jobId === jobId && !j.savedVariants.includes(index)
          ? { ...j, savedVariants: [...j.savedVariants, index] }
          : j,
      ),
    })),

  setIsGenerating: (v) => set({ isGenerating: v }),
  setIsFormatting: (v) => set({ isFormatting: v }),
  setAutoGenEnabled: (v) => set({ autoGenEnabled: v }),
  setAutoSaveEnabled: (v) => set({ autoSaveEnabled: v }),
  incrementAutoGenCount: () => set((s) => ({ autoGenCount: s.autoGenCount + 1 })),
  resetAutoGenCount: () => set({ autoGenCount: 0 }),
  setAutoGenMaxRuns: (v) => set({ autoGenMaxRuns: v }),
  setPreFormatSnapshot: (snapshot) => set({ preFormatSnapshot: snapshot }),
  resetCustomForm: () =>
    set({
      customForm: {
        caption: "",
        lyrics: "",
        instrumental: false,
        bpm: null,
        keyscale: "",
        timesignature: "",
      },
      preFormatSnapshot: null,
    }),
  setAutoTitleEnabled: (v) => set({ autoTitleEnabled: v }),
  setCustomTitle: (v) => set({ customTitle: v }),
  swapJobId: (oldId, newId) =>
    set((s) => ({
      activeJobs: s.activeJobs.map((j) =>
        j.jobId === oldId ? { ...j, jobId: newId } : j,
      ),
    })),

  removeJob: (jobId) =>
    set((s) => ({
      activeJobs: s.activeJobs.filter((j) => j.jobId !== jobId),
    })),

  hideJob: (jobId) =>
    set((s) => ({
      activeJobs: s.activeJobs.map((j) =>
        j.jobId === jobId ? { ...j, hiddenFromQueue: true } : j,
      ),
    })),

  clearJobs: () =>
    set((s) => ({
      activeJobs: s.activeJobs.map((j) =>
        j.status === "queued" || j.status === "running" || j.status === "cancelling"
          ? j
          : { ...j, hiddenFromQueue: true },
      ),
    })),

  setJobTitle: (jobId, title) =>
    set((s) => ({
      activeJobs: s.activeJobs.map((j) =>
        j.jobId === jobId ? { ...j, generatedTitle: title } : j,
      ),
    })),

  loadFromHistoryParams: (params) => {
    const taskType = (params.task_type as string) ?? "text2music";
    const fallback = useGenerationStore.getState().advancedSettings;

    const advancedSettings: AdvancedSettings = {
      inferenceSteps: (params.inference_steps as number) ?? fallback.inferenceSteps,
      guidanceScale: (params.guidance_scale as number) ?? fallback.guidanceScale,
      shift: (params.shift as number) ?? fallback.shift,
      inferMethod: (params.infer_method as string) ?? fallback.inferMethod,
      seed: (params.seed as number) ?? fallback.seed,
      thinking: (params.thinking as boolean) ?? fallback.thinking,
      lmTemperature: (params.lm_temperature as number) ?? fallback.lmTemperature,
      batchSize: (params.batch_size as number) ?? fallback.batchSize,
      audioFormat: (params.audio_format as string) ?? fallback.audioFormat,
      defaultDuration: (params.default_duration as number) ?? fallback.defaultDuration,
      useCotCaption: (params.use_cot_caption as boolean) ?? fallback.useCotCaption,
      useCotMetas: (params.use_cot_metas as boolean) ?? fallback.useCotMetas,
      useCotLanguage: (params.use_cot_language as boolean) ?? fallback.useCotLanguage,
    };

    if (taskType === "music2music") {
      set({
        activeMode: "Remix",
        remixForm: {
          audioFilePath: "",
          audioFileName: "",
          caption: (params.caption as string) ?? "",
          lyrics: (params.lyrics as string) ?? "",
          coverStrength: (params.audio_cover_strength as number) ?? 1.0,
          sourceSongId: "",
        },
        advancedSettings,
      });
    } else {
      set({
        activeMode: "Custom",
        customForm: {
          caption: (params.caption as string) ?? "",
          lyrics: (params.lyrics as string) ?? "",
          instrumental: (params.instrumental as boolean) ?? false,
          bpm: (params.bpm as number) ?? null,
          keyscale: (params.keyscale as string) ?? "",
          timesignature: (params.timesignature as string) ?? "",
        },
        advancedSettings,
      });
    }
  },
}));
