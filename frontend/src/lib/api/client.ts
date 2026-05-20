import { getBaseUrl, getWsUrl, request } from "./base";
import { getClientId } from "@/lib/client-session";
import type {
  HealthResponse,
  GenerateRequest,
  GenerateResponse,
  JobStatusResponse,
  FormatRequest,
  FormatResponse,
  SampleRequest,
  SampleResponse,
  SongResponse,
  SongListResponse,
  SettingsResponse,
  WsProgressMessage,
  ModelsResponse,
  AvailableModelsResponse,
  UploadResponse,
  GenerationHistoryEntry,
  GenerationHistoryListResponse,
  SongVariationsResponse,
  GpuStats,
  GenerateTitleRequest,
  GenerateTitleResponse,
} from "@/types/api";

// Health
export const fetchHealth = () => request<HealthResponse>("/health");

// Settings
export const fetchSettings = () => request<SettingsResponse>("/settings");
export const updateSettings = (settings: Record<string, string>) =>
  request<SettingsResponse>("/settings", {
    method: "PATCH",
    body: JSON.stringify({ settings }),
  });

// Models
export const fetchModels = () => request<ModelsResponse>("/models");

export const fetchAvailableModels = () =>
  request<AvailableModelsResponse>("/models/available");

export const downloadModel = (modelName: string) =>
  request<{ status: string }>("/models/download", {
    method: "POST",
    body: JSON.stringify({ model_name: modelName }),
  });

export const fetchModelDownloadStatus = () =>
  request<Record<string, string>>("/models/download-status");

// Upload
export async function uploadAudio(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const url = `${getBaseUrl()}/api/upload`;
  const res = await fetch(url, { method: "POST", body: formData });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Upload error ${res.status}: ${body}`);
  }
  return res.json();
}

// Generation
export const submitGeneration = (params: GenerateRequest) =>
  request<GenerateResponse>("/generate", {
    method: "POST",
    body: JSON.stringify({ ...params, client_id: getClientId() }),
  });

export const fetchJobStatus = (jobId: string) =>
  request<JobStatusResponse>(`/generate/${jobId}`);

export const cancelJob = (jobId: string) =>
  request<{ message: string }>(`/generate/${jobId}/cancel`, { method: "POST" });

// Format & Sample
export const formatCaption = (params: FormatRequest) =>
  request<FormatResponse>("/format", {
    method: "POST",
    body: JSON.stringify(params),
  });

export const createSample = (params: SampleRequest) =>
  request<SampleResponse>("/sample", {
    method: "POST",
    body: JSON.stringify(params),
  });

// Title generation
export const generateTitle = (params: GenerateTitleRequest) =>
  request<GenerateTitleResponse>("/generate-title", {
    method: "POST",
    body: JSON.stringify(params),
  });

// Songs
export const fetchSongs = (params?: {
  search?: string;
  sort?: string;
  order?: string;
  favorite?: boolean;
  vocal_language?: string;
  file_format?: string;
  instrumental?: boolean;
  timesignature?: string;
  tag?: string;
  limit?: number;
  offset?: number;
}) => {
  const searchParams = new URLSearchParams();
  if (params?.search) searchParams.set("search", params.search);
  if (params?.sort) searchParams.set("sort", params.sort);
  if (params?.order) searchParams.set("order", params.order);
  if (params?.favorite !== undefined)
    searchParams.set("favorite", String(params.favorite));
  if (params?.vocal_language)
    searchParams.set("vocal_language", params.vocal_language);
  if (params?.file_format)
    searchParams.set("file_format", params.file_format);
  if (params?.instrumental !== undefined)
    searchParams.set("instrumental", String(params.instrumental));
  if (params?.timesignature)
    searchParams.set("timesignature", params.timesignature);
  if (params?.tag) searchParams.set("tag", params.tag);
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.offset) searchParams.set("offset", String(params.offset));
  const qs = searchParams.toString();
  return request<SongListResponse>(`/songs${qs ? `?${qs}` : ""}`);
};

export const saveSongToLibrary = (song: {
  title: string;
  file_path: string;
  file_format?: string;
  duration_seconds?: number | null;
  caption?: string;
  lyrics?: string;
  bpm?: number | null;
  keyscale?: string;
  timesignature?: string;
  vocal_language?: string;
  instrumental?: boolean;
  generation_history_id?: string | null;
  variation_index?: number;
  parent_song_id?: string | null;
}) =>
  request<SongResponse>("/songs", {
    method: "POST",
    body: JSON.stringify(song),
  });

export const fetchSong = (id: string) =>
  request<SongResponse>(`/songs/${id}`);

export const updateSong = (
  id: string,
  updates: Partial<
    Pick<
      SongResponse,
      | "title"
      | "caption"
      | "lyrics"
      | "bpm"
      | "keyscale"
      | "timesignature"
      | "vocal_language"
      | "is_favorite"
      | "rating"
      | "tags"
      | "notes"
    >
  >,
) =>
  request<SongResponse>(`/songs/${id}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });

export const deleteSong = (id: string) =>
  request<{ deleted: boolean }>(`/songs/${id}`, { method: "DELETE" });

export const bulkDeleteSongs = (songIds: string[]) =>
  request<{ deleted: number }>("/songs/bulk-delete", {
    method: "POST",
    body: JSON.stringify({ song_ids: songIds }),
  });

export const bulkUpdateSongs = (
  songIds: string[],
  updates: Partial<
    Pick<
      SongResponse,
      | "title"
      | "caption"
      | "lyrics"
      | "bpm"
      | "keyscale"
      | "timesignature"
      | "vocal_language"
      | "is_favorite"
      | "rating"
      | "tags"
      | "notes"
    >
  >,
) =>
  request<{ updated: number }>("/songs/bulk", {
    method: "PATCH",
    body: JSON.stringify({ song_ids: songIds, updates }),
  });

export function getSongAudioUrl(songId: string): string {
  return `${getBaseUrl()}/api/songs/${songId}/audio`;
}

export function getSongDownloadUrl(songId: string): string {
  return `${getBaseUrl()}/api/songs/${songId}/audio?download=true`;
}

export function getStationExportUrl(stationId: string): string {
  return `${getBaseUrl()}/api/radio/stations/${stationId}/export`;
}

// History
export const fetchHistory = (params?: {
  search?: string;
  status?: string;
  task_type?: string;
  sort?: string;
  order?: string;
  limit?: number;
  offset?: number;
}) => {
  const searchParams = new URLSearchParams();
  if (params?.search) searchParams.set("search", params.search);
  if (params?.status) searchParams.set("status", params.status);
  if (params?.task_type) searchParams.set("task_type", params.task_type);
  if (params?.sort) searchParams.set("sort", params.sort);
  if (params?.order) searchParams.set("order", params.order);
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.offset) searchParams.set("offset", String(params.offset));
  const qs = searchParams.toString();
  return request<GenerationHistoryListResponse>(`/history${qs ? `?${qs}` : ""}`);
};

export const fetchHistoryEntry = (id: string) =>
  request<GenerationHistoryEntry>(`/history/${id}`);

export const updateHistoryEntry = (id: string, updates: { title?: string }) =>
  request<GenerationHistoryEntry>(`/history/${id}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });

export const deleteHistoryEntry = (id: string) =>
  request<{ deleted: boolean }>(`/history/${id}`, { method: "DELETE" });

// WebSocket
export function createGenerationWebSocket(
  onMessage: (msg: WsProgressMessage) => void,
  onError?: (err: Event) => void,
): WebSocket {
  const wsUrl = getWsUrl();
  const url = new URL(`${wsUrl}/api/ws/generate`);
  const clientId = getClientId();
  if (clientId) url.searchParams.set("client_id", clientId);
  const ws = new WebSocket(url.toString());
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data) as WsProgressMessage;
    onMessage(msg);
  };
  if (onError) ws.onerror = onError;
  return ws;
}

export function createHealthWebSocket(
  onMessage: (msg: HealthResponse) => void,
  onError?: (err: Event) => void,
): WebSocket {
  const wsUrl = getWsUrl();
  const ws = new WebSocket(`${wsUrl}/api/ws/health`);
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data) as HealthResponse;
    onMessage(msg);
  };
  if (onError) ws.onerror = onError;
  return ws;
}

// Song helpers
export const getSongSourcePath = (songId: string) =>
  request<{ file_path: string }>(`/songs/${songId}/source-path`);

export const fetchSongVariations = (songId: string) =>
  request<SongVariationsResponse>(`/songs/${songId}/variations`);

// Models (enhanced)
export const switchDitModel = (modelName: string) =>
  request<{ message: string }>("/models/switch-dit", {
    method: "POST",
    body: JSON.stringify({ model_name: modelName }),
  });

export const switchLmModel = (modelName: string, runtime?: string) =>
  request<{ message: string }>("/models/switch-lm", {
    method: "POST",
    body: JSON.stringify({ model_name: modelName, runtime }),
  });

export const switchChatLlmModel = (modelName: string) =>
  request<{ message: string }>("/models/switch-chat-llm", {
    method: "POST",
    body: JSON.stringify({ model_name: modelName }),
  });

export const fetchGpuStats = () => request<GpuStats>("/models/gpu-stats");
