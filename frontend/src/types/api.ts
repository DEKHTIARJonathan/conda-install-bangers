export interface HealthResponse {
  status: string;
  dit_model_loaded: boolean;
  lm_model_loaded: boolean;
  dit_model: string;
  lm_model: string;
  device: string;
  version: string;
  init_stage: string;
  init_error: string;
  download_progress: number;
  instance_id: string;
}

export interface SongResponse {
  id: string;
  title: string;
  file_path: string;
  file_format: string;
  duration_seconds: number | null;
  sample_rate: number;
  file_size_bytes: number | null;
  caption: string;
  lyrics: string;
  bpm: number | null;
  keyscale: string;
  timesignature: string;
  vocal_language: string;
  instrumental: boolean;
  is_favorite: boolean;
  rating: number;
  tags: string;
  notes: string;
  parent_song_id: string | null;
  generation_history_id: string | null;
  variation_index: number;
  created_at: string;
  updated_at: string;
}

export interface SongListResponse {
  items: SongResponse[];
  total: number;
}

export interface BulkDeleteRequest {
  song_ids: string[];
}

export interface BulkUpdateRequest {
  song_ids: string[];
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
  >;
}

export interface GenerateRequest {
  task_type?: string;
  caption?: string;
  lyrics?: string;
  instrumental?: boolean;
  vocal_language?: string;
  bpm?: number | null;
  keyscale?: string;
  timesignature?: string;
  duration?: number;
  inference_steps?: number;
  seed?: number;
  guidance_scale?: number;
  shift?: number;
  infer_method?: string;
  batch_size?: number;
  audio_format?: string;
  thinking?: boolean;
  lm_temperature?: number;
  lm_cfg_scale?: number;
  use_cot_metas?: boolean;
  use_cot_caption?: boolean;
  use_cot_language?: boolean;
  audio_cover_strength?: number;
  auto_title?: boolean;
  src_audio_path?: string;
  quality_profile?: string;
  spec_source?: string;
  source_prompt?: string;
  client_id?: string;
}

export interface GenerateResponse {
  job_id: string;
  status: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: string;
  progress: number;
  stage: string;
  results: AudioResult[];
  error: string | null;
  timings: Record<string, number>;
  history_id: string | null;
}

export interface AudioResult {
  path: string;
  key: string;
  sample_rate: number;
  params: Record<string, unknown>;
}

export interface WsProgressMessage {
  type: "progress" | "prepared" | "completed" | "failed" | "title";
  job_id: string;
  progress?: number;
  stage?: string;
  step?: number;
  total_steps?: number;
  params?: Record<string, unknown>;
  results?: AudioResult[];
  error?: string;
  history_id?: string;
  title?: string;
}

export interface FormatRequest {
  caption?: string;
  lyrics?: string;
  bpm?: number | null;
  keyscale?: string;
  timesignature?: string;
  duration?: number | null;
  vocal_language?: string;
}

export interface FormatResponse {
  caption: string;
  lyrics: string;
  bpm: number | null;
  duration: number | null;
  keyscale: string;
  language: string;
  timesignature: string;
  success: boolean;
  error: string | null;
}

export interface SampleRequest {
  query: string;
  instrumental?: boolean;
  vocal_language?: string | null;
  temperature?: number;
}

export interface SampleResponse {
  caption: string;
  lyrics: string;
  bpm: number | null;
  duration: number | null;
  keyscale: string;
  language: string;
  timesignature: string;
  instrumental: boolean;
  quality_profile: string;
  spec_source: string;
  source_prompt: string;
  success: boolean;
  error: string | null;
}

export interface SettingsResponse {
  settings: Record<string, string>;
}

export interface ModelInfo {
  name: string;
  model_type: string;
  is_active: boolean;
  is_loading?: boolean;
  loaded_on?: string[];
  compatibility?: string[];
  format?: string;
  quantization?: string;
}

export interface ModelsResponse {
  dit_models: ModelInfo[];
  lm_models: ModelInfo[];
  chat_llm_models: ModelInfo[];
}

export interface AvailableModel {
  name: string;
  model_type: string;
  repo_id: string;
  installed: boolean;
  description: string;
  downloading: boolean;
  download_progress: number;
  size_mb: number;
  compatibility: string[];
  format: string;
  quantization: string;
}

export interface AvailableModelsResponse {
  models: AvailableModel[];
}

export interface UploadResponse {
  file_path: string;
}

export interface GenerationHistoryEntry {
  id: string;
  task_type: string;
  status: "pending" | "running" | "completed" | "failed";
  title: string | null;
  params: Record<string, unknown>;
  results: AudioResult[];
  audio_count: number;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  created_at: string;
  saved_song_count: number;
}

export interface GenerationHistoryListResponse {
  items: GenerationHistoryEntry[];
  total: number;
}

export interface SongVariationsResponse {
  song: SongResponse;
  ancestors: SongResponse[];
  children: SongResponse[];
}

export interface GpuStats {
  device: string;
  provider?: string;
  memory_type?: string;
  gpu_utilization_percent?: number | null;
  renderer_utilization_percent?: number | null;
  tiler_utilization_percent?: number | null;
  vram_used_mb?: number | null;
  vram_total_mb?: number | null;
  vram_percent?: number | null;
  memory_cache_mb?: number | null;
  memory_peak_mb?: number | null;
  gpus?: GpuDeviceStats[];
  updated_at?: number;
  error?: string;
}

export interface GpuDeviceStats {
  node_id: string;
  node_role: string;
  label: string;
  device_index: number;
  name: string;
  uuid: string;
  provider?: string;
  memory_type?: string;
  utilization_gpu_percent: number | null;
  utilization_memory_percent: number | null;
  renderer_utilization_percent?: number | null;
  tiler_utilization_percent?: number | null;
  vram_used_mb: number | null;
  vram_total_mb: number | null;
  vram_percent: number | null;
  memory_cache_mb?: number | null;
  memory_peak_mb?: number | null;
  power_draw_w: number | null;
  power_limit_w: number | null;
  busy: boolean;
  holder: string | null;
  error: string;
}

export interface SwitchModelRequest {
  model_name: string;
  runtime?: string;
}

// Radio types
export interface StationResponse {
  id: string;
  name: string;
  description: string;
  is_preset: boolean;
  caption_template: string;
  genre: string;
  mood: string;
  instrumental: boolean;
  vocal_language: string;
  bpm_min: number | null;
  bpm_max: number | null;
  keyscale: string;
  timesignature: string;
  advanced_params_json: string;
  total_plays: number;
  last_played_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface StationDetailResponse extends StationResponse {
  recent_songs: SongResponse[];
}

export interface RadioStatusResponse {
  active_station_id: string | null;
  is_generating: boolean;
  songs_generated: number;
}

export interface CreateStationRequest {
  name: string;
  description?: string;
  caption_template?: string;
  genre?: string;
  mood?: string;
  instrumental?: boolean;
  vocal_language?: string;
  bpm_min?: number | null;
  bpm_max?: number | null;
  keyscale?: string;
  timesignature?: string;
  advanced_params_json?: string;
}

export interface UpdateStationRequest {
  name?: string;
  description?: string;
  caption_template?: string;
  genre?: string;
  mood?: string;
  instrumental?: boolean;
  vocal_language?: string;
  bpm_min?: number | null;
  bpm_max?: number | null;
  keyscale?: string;
  timesignature?: string;
  advanced_params_json?: string;
}

// Radio LLM settings types
export interface RadioSettingsResponse {
  active_model: string;
  loaded_model?: string;
  installed_models: string[];
  system_prompt: string;
  default_system_prompt: string;
}

export interface RadioSettingsUpdate {
  model?: string;
  system_prompt?: string;
}

// Title generation types
export interface GenerateTitleRequest {
  caption?: string;
  genre?: string;
  mood?: string;
  fallback?: string;
}

export interface GenerateTitleResponse {
  title: string;
  success: boolean;
  error: string | null;
}

// DJ types
export interface DJConversationResponse {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface DJMessageResponse {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  generation_params_json: string | null;
  generation_job_id: string | null;
  created_at: string;
}

export interface DJConversationDetailResponse extends DJConversationResponse {
  messages: DJMessageResponse[];
}

export interface DJInfoResponse {
  active_model: string;
  loaded_model?: string;
  installed_models: string[];
  system_prompt: string;
  default_system_prompt: string;
}
