import { describe, expect, it, beforeEach, vi } from "vitest";
import { toast } from "sonner";
import {
  reconcilePolledStatus,
  resolveSubmittedJobState,
  runReconcilePoller,
} from "../use-generation";
import { applyPreparedParamsToSubmittedCustomJob } from "../use-generation-ws";
import { useGenerationStore } from "@/stores/generation-store";
import type { CustomForm, GenerationJob } from "@/stores/generation-store";
import type { JobStatusResponse } from "@/types/api";

vi.mock("sonner", () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
}));

function job(jobId: string, status: GenerationJob["status"]): GenerationJob {
  return {
    jobId,
    status,
    progress: 0,
    stage: "",
    results: [],
    error: null,
    historyId: null,
    savedVariants: [],
    generatedTitle: null,
  };
}

describe("resolveSubmittedJobState", () => {
  it("continues when websocket already swapped temp id to server id", () => {
    expect(resolveSubmittedJobState([job("server", "running")], "temp", "server")).toBe(
      "already-swapped",
    );
  });

  it("swaps when the temp job is still queued locally", () => {
    expect(resolveSubmittedJobState([job("temp", "queued")], "temp", "server")).toBe("swap");
  });

  it("cancels when either local job is cancelling", () => {
    expect(resolveSubmittedJobState([job("temp", "cancelling")], "temp", "server")).toBe(
      "cancel",
    );
    expect(resolveSubmittedJobState([job("server", "cancelling")], "temp", "server")).toBe(
      "cancel",
    );
  });
});

function makeStatus(overrides: Partial<JobStatusResponse> = {}): JobStatusResponse {
  return {
    job_id: "job-1",
    status: "running",
    progress: 0,
    stage: "",
    results: [],
    error: null,
    timings: {},
    history_id: null,
    ...overrides,
  };
}

function seedJob(overrides: Partial<GenerationJob> = {}) {
  const initial: GenerationJob = {
    jobId: "job-1",
    status: "running",
    progress: 0.1,
    stage: "running",
    results: [],
    error: null,
    historyId: null,
    savedVariants: [],
    generatedTitle: null,
    ...overrides,
  };
  useGenerationStore.setState({ activeJobs: [initial], isGenerating: true });
}

describe("reconcilePolledStatus", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useGenerationStore.setState({ activeJobs: [], isGenerating: false });
  });

  it("merges progress without firing toasts or touching isGenerating", () => {
    seedJob();
    const result = reconcilePolledStatus(
      makeStatus({ status: "running", progress: 0.5, stage: "DiT" }),
      "job-1",
    );
    expect(result).toBe("active");
    expect(toast.success).not.toHaveBeenCalled();
    expect(toast.error).not.toHaveBeenCalled();
    const job = useGenerationStore.getState().activeJobs[0];
    expect(job.progress).toBe(0.5);
    expect(job.stage).toBe("DiT");
    expect(useGenerationStore.getState().isGenerating).toBe(true);
  });

  it("flips status to completed without owning toasts or GPU state", () => {
    seedJob();
    const result = reconcilePolledStatus(
      makeStatus({ status: "completed", progress: 1, history_id: "h1" }),
      "job-1",
    );
    expect(result).toBe("terminal");
    expect(toast.success).not.toHaveBeenCalled();
    const job = useGenerationStore.getState().activeJobs[0];
    expect(job.status).toBe("completed");
    expect(job.historyId).toBe("h1");
    // isGenerating left intact - WS handler owns it.
    expect(useGenerationStore.getState().isGenerating).toBe(true);
  });

  it("removes the job entirely when polling sees cancelled", () => {
    seedJob();
    const result = reconcilePolledStatus(
      makeStatus({ status: "cancelled", error: "Cancelled by user" }),
      "job-1",
    );
    expect(result).toBe("terminal");
    expect(toast.error).not.toHaveBeenCalled();
    expect(useGenerationStore.getState().activeJobs).toHaveLength(0);
  });

  it("treats failed without crashing and without toasting", () => {
    seedJob();
    const result = reconcilePolledStatus(
      makeStatus({ status: "failed", error: "boom" }),
      "job-1",
    );
    expect(result).toBe("terminal");
    expect(toast.error).not.toHaveBeenCalled();
    const job = useGenerationStore.getState().activeJobs[0];
    expect(job.status).toBe("failed");
    expect(job.error).toBe("boom");
  });

  it("returns missing when the job vanished from the store", () => {
    useGenerationStore.setState({ activeJobs: [] });
    const result = reconcilePolledStatus(makeStatus({ status: "running" }), "job-1");
    expect(result).toBe("missing");
  });
});

describe("runReconcilePoller", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useGenerationStore.setState({ activeJobs: [], isGenerating: false });
  });

  it("retries past transient fetch errors and converges on terminal status", async () => {
    seedJob();
    let calls = 0;
    const fetchStatus = vi.fn(async () => {
      calls += 1;
      if (calls < 3) throw new Error("network");
      return makeStatus({ status: "completed", progress: 1 });
    });

    await runReconcilePoller("job-1", {
      fetchStatus,
      minDelayMs: 0,
      maxDelayMs: 0,
      maxConsecutiveErrors: 6,
    });

    expect(fetchStatus).toHaveBeenCalledTimes(3);
    expect(useGenerationStore.getState().activeJobs[0].status).toBe("completed");
  });

  it("gives up only after maxConsecutiveErrors", async () => {
    seedJob();
    const fetchStatus = vi.fn(async () => {
      throw new Error("offline");
    });

    await runReconcilePoller("job-1", {
      fetchStatus,
      minDelayMs: 0,
      maxDelayMs: 0,
      maxConsecutiveErrors: 4,
    });

    expect(fetchStatus).toHaveBeenCalledTimes(4);
    // Job left untouched (still running) so the WS handler can finish the story.
    expect(useGenerationStore.getState().activeJobs[0].status).toBe("running");
  });

  it("stops as soon as the job is missing from the store", async () => {
    useGenerationStore.setState({ activeJobs: [] });
    const fetchStatus = vi.fn(async () => makeStatus({ status: "running" }));

    await runReconcilePoller("job-gone", {
      fetchStatus,
      minDelayMs: 0,
      maxDelayMs: 0,
    });

    expect(fetchStatus).toHaveBeenCalledTimes(1);
  });
});

describe("applyPreparedParamsToSubmittedCustomJob", () => {
  const submittedCustomForm: CustomForm = {
    caption: "rough caption",
    lyrics: "",
    instrumental: false,
    bpm: null,
    keyscale: "",
    timesignature: "",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    useGenerationStore.setState({
      ...useGenerationStore.getInitialState(),
      customForm: { ...submittedCustomForm },
      activeJobs: [
        {
          ...job("job-1", "running"),
          submittedCustomForm: { ...submittedCustomForm },
        },
      ],
    });
  });

  it("copies prepared generated lyrics into the unchanged submitted form", () => {
    const applied = applyPreparedParamsToSubmittedCustomJob("job-1", {
      caption: "polished caption",
      lyrics: "[verse]\nPrepared lyric",
      bpm: 92,
      keyscale: "A minor",
      timesignature: "3/4",
    });

    expect(applied).toBe(true);
    expect(useGenerationStore.getState().customForm).toMatchObject({
      caption: "polished caption",
      lyrics: "[verse]\nPrepared lyric",
      bpm: 92,
      keyscale: "A minor",
      timesignature: "3/4",
    });
  });

  it("does not overwrite a form edited after submit", () => {
    useGenerationStore.getState().updateCustomForm({ lyrics: "user typed more" });

    const applied = applyPreparedParamsToSubmittedCustomJob("job-1", {
      lyrics: "[verse]\nPrepared lyric",
    });

    expect(applied).toBe(false);
    expect(useGenerationStore.getState().customForm.lyrics).toBe("user typed more");
  });
});
