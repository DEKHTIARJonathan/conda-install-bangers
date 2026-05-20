from pydantic import BaseModel, Field
from typing import Optional


class SongResponse(BaseModel):
    id: str
    title: str
    file_path: str
    file_format: str = "flac"
    duration_seconds: Optional[float] = None
    sample_rate: int = 48000
    file_size_bytes: Optional[int] = None
    caption: str = ""
    lyrics: str = ""
    bpm: Optional[int] = None
    keyscale: str = ""
    timesignature: str = ""
    vocal_language: str = "unknown"
    instrumental: bool = False
    is_favorite: bool = False
    rating: int = 0
    tags: str = ""
    notes: str = ""
    parent_song_id: Optional[str] = None
    generation_history_id: Optional[str] = None
    variation_index: int = 0
    created_at: str = ""
    updated_at: str = ""


class SongUpdate(BaseModel):
    title: Optional[str] = None
    caption: Optional[str] = None
    lyrics: Optional[str] = None
    bpm: Optional[int] = None
    keyscale: Optional[str] = None
    timesignature: Optional[str] = None
    vocal_language: Optional[str] = None
    is_favorite: Optional[bool] = None
    rating: Optional[int] = None
    tags: Optional[str] = None
    notes: Optional[str] = None


class SaveToLibraryRequest(BaseModel):
    """Request to save a generated audio to the library."""
    title: str
    file_path: str
    file_format: str = "flac"
    duration_seconds: Optional[float] = None
    sample_rate: int = 48000
    caption: str = ""
    lyrics: str = ""
    bpm: Optional[int] = None
    keyscale: str = ""
    timesignature: str = ""
    vocal_language: str = "unknown"
    instrumental: bool = False
    generation_history_id: Optional[str] = None
    variation_index: int = 0
    parent_song_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    dit_model_loaded: bool = False
    lm_model_loaded: bool = False
    dit_model: str = ""
    lm_model: str = ""
    device: str = ""
    version: str = "0.1.0"
    init_stage: str = "idle"
    init_error: str = ""
    download_progress: float = 0.0
    instance_id: str = ""


class AvailableModel(BaseModel):
    name: str
    model_type: str  # "dit", "lm", or "chat_llm"
    repo_id: str
    installed: bool
    description: str
    downloading: bool = False
    download_progress: float = 0.0
    size_mb: int = 0
    compatibility: list[str] = Field(default_factory=list)
    format: str = ""
    quantization: str = ""


class AvailableModelsResponse(BaseModel):
    models: list[AvailableModel]


class DownloadModelRequest(BaseModel):
    model_name: str


class SettingsResponse(BaseModel):
    settings: dict[str, str]


class SettingsUpdate(BaseModel):
    settings: dict[str, str]


class SongListResponse(BaseModel):
    items: list[SongResponse] = Field(default_factory=list)
    total: int = 0


class BulkDeleteRequest(BaseModel):
    song_ids: list[str]


class BulkUpdateRequest(BaseModel):
    song_ids: list[str]
    updates: SongUpdate


class SwitchModelRequest(BaseModel):
    model_name: str
    runtime: Optional[str] = None


class GpuDeviceStats(BaseModel):
    node_id: str = ""
    node_role: str = ""
    label: str = ""
    device_index: int = 0
    name: str = ""
    uuid: str = ""
    provider: str = ""
    memory_type: str = ""
    utilization_gpu_percent: Optional[float] = None
    utilization_memory_percent: Optional[float] = None
    renderer_utilization_percent: Optional[float] = None
    tiler_utilization_percent: Optional[float] = None
    vram_used_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None
    vram_percent: Optional[float] = None
    memory_cache_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    busy: bool = False
    holder: Optional[str] = None
    error: str = ""


class GpuStatsResponse(BaseModel):
    device: str = ""
    provider: str = ""
    memory_type: str = ""
    gpu_utilization_percent: Optional[float] = None
    renderer_utilization_percent: Optional[float] = None
    tiler_utilization_percent: Optional[float] = None
    vram_used_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None
    vram_percent: Optional[float] = None
    memory_cache_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    gpus: list[GpuDeviceStats] = Field(default_factory=list)
    updated_at: float = 0.0
    error: str = ""
