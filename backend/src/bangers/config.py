import os
import socket
import sys
from pathlib import Path

from bangers.model_registry import is_lm_disabled


DEFAULT_LM_BACKEND = "mlx" if sys.platform == "darwin" else "vllm"
DEFAULT_DEVICE = "auto"
DEFAULT_AUDIO_FORMAT = "flac"
DEFAULT_BATCH_SIZE = 2
DEFAULT_GENERATION_DURATION = 200.0
DEFAULT_THINKING = True
DEFAULT_INFERENCE_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 7.0
DEFAULT_VAE_CHUNK_SIZE = 128
DEFAULT_VAE_SLEEP_MS = 200
DEFAULT_DIT_SLEEP_MS = 200
DEFAULT_THROTTLE_RADIO_ONLY = True
DEFAULT_KEEP_ACTIVE_MODELS_RESIDENT = True
DEFAULT_PARALLEL_PIPELINE_ENABLED = False
DEFAULT_LYRICS_GUARDRAILS_ENABLED = True

DISTRIBUTED_CAPABILITY_MUSIC = "music"
DISTRIBUTED_CAPABILITY_ACE_LM = "ace_lm"
DISTRIBUTED_CAPABILITY_CHAT_LLM = "chat_llm"
DISTRIBUTED_CAPABILITIES = frozenset({
    DISTRIBUTED_CAPABILITY_MUSIC,
    DISTRIBUTED_CAPABILITY_ACE_LM,
    DISTRIBUTED_CAPABILITY_CHAT_LLM,
})


def _bool_string(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "true" if value.strip().lower() in {"1", "true", "yes", "on"} else "false"


def _getenv(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _has_env_value(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value != ""


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_capabilities(value: str) -> frozenset[str]:
    requested = {part.lower() for part in _parse_csv(value)}
    return frozenset(requested & DISTRIBUTED_CAPABILITIES)


def normalize_lm_backend(backend: str | None) -> str:
    """Return the ACE-Step backend name for a configured LM runtime."""
    value = (backend or DEFAULT_LM_BACKEND).strip().lower()
    if not value:
        value = DEFAULT_LM_BACKEND
    if value in {"nano-vllm", "nano_vllm", "nanovllm"}:
        value = "vllm"
    if value == "mlx" and sys.platform != "darwin":
        value = "vllm"
    return value


class Settings:
    """Application settings loaded from environment variables."""

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    REPO_ROOT: Path = BASE_DIR.parent

    HOST: str
    PORT: int

    DATA_DIR: Path
    DB_PATH: Path
    AUDIO_DIR: Path
    UPLOADS_DIR: Path

    MODEL_CACHE_DIR: Path
    ACESTEP_PROJECT_ROOT: str
    HF_HOME_DIR: Path
    HF_HUB_CACHE_DIR: Path

    DEFAULT_LM_BACKEND: str
    DEFAULT_DEVICE: str
    DEFAULT_AUDIO_FORMAT: str
    DEFAULT_BATCH_SIZE: int
    DEFAULT_DURATION: float
    DEFAULT_INFERENCE_STEPS: int
    DEFAULT_GUIDANCE_SCALE: float
    DEFAULT_THINKING: bool

    DISTRIBUTED_ROLE: str
    DISTRIBUTED_NODE_ID: str
    DISTRIBUTED_WORKERS: tuple[str, ...]
    DISTRIBUTED_CAPABILITIES: frozenset[str]
    DISTRIBUTED_TOKEN: str
    DISTRIBUTED_REQUEST_TIMEOUT_SECONDS: float

    def apply_runtime_overrides(self) -> None:
        """Refresh environment-backed settings."""
        self.HOST = _getenv("BANGERS_HOST", "0.0.0.0")
        self.PORT = int(_getenv("BANGERS_PORT", "8000"))

        self.DATA_DIR = Path(
            _getenv("BANGERS_DATA_DIR", str(self.BASE_DIR / "data"))
        ).expanduser()
        self.DB_PATH = self.DATA_DIR / "conda-install-bangers.db"
        self.AUDIO_DIR = self.DATA_DIR / "audio"
        self.UPLOADS_DIR = self.DATA_DIR / "uploads"

        self.MODEL_CACHE_DIR = Path(
            _getenv(
                "BANGERS_MODEL_CACHE_DIR",
                str(self.REPO_ROOT / ".cache" / "models"),
            )
        ).expanduser()
        self.ACESTEP_PROJECT_ROOT = _getenv("ACESTEP_PROJECT_ROOT", str(self.MODEL_CACHE_DIR))
        self.HF_HOME_DIR = Path(_getenv("HF_HOME", str(self.MODEL_CACHE_DIR / "huggingface"))).expanduser()
        self.HF_HUB_CACHE_DIR = Path(
            _getenv("HF_HUB_CACHE", str(self.HF_HOME_DIR / "hub"))
        ).expanduser()
        os.environ.setdefault("HF_HOME", str(self.HF_HOME_DIR))
        os.environ.setdefault("HF_HUB_CACHE", str(self.HF_HUB_CACHE_DIR))

        self.DEFAULT_LM_BACKEND = normalize_lm_backend(
            _getenv("BANGERS_LM_BACKEND", DEFAULT_LM_BACKEND)
        )
        self.DEFAULT_DEVICE = _getenv("BANGERS_DEVICE", DEFAULT_DEVICE)

        self.DEFAULT_AUDIO_FORMAT = _getenv("BANGERS_AUDIO_FORMAT", DEFAULT_AUDIO_FORMAT)
        self.DEFAULT_BATCH_SIZE = int(_getenv("BANGERS_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
        self.DEFAULT_DURATION = DEFAULT_GENERATION_DURATION
        self.DEFAULT_INFERENCE_STEPS = int(_getenv("BANGERS_INFERENCE_STEPS", str(DEFAULT_INFERENCE_STEPS)))
        self.DEFAULT_GUIDANCE_SCALE = float(_getenv("BANGERS_GUIDANCE_SCALE", str(DEFAULT_GUIDANCE_SCALE)))
        self.DEFAULT_THINKING = _bool_string(
            _getenv("BANGERS_THINKING", _bool_string(DEFAULT_THINKING))
        ) == "true"

        role = _getenv("BANGERS_DISTRIBUTED_ROLE", "standalone").strip().lower()
        if role not in {"standalone", "coordinator", "worker"}:
            role = "standalone"
        self.DISTRIBUTED_ROLE = role
        self.DISTRIBUTED_NODE_ID = _getenv("BANGERS_NODE_ID", socket.gethostname())
        self.DISTRIBUTED_WORKERS = _parse_csv(_getenv("BANGERS_WORKERS", ""))
        raw_capabilities = os.getenv("BANGERS_WORKER_CAPABILITIES")
        if raw_capabilities is None:
            raw_capabilities = (
                ""
                if role == "coordinator"
                else ",".join(sorted(DISTRIBUTED_CAPABILITIES))
            )
        self.DISTRIBUTED_CAPABILITIES = _parse_capabilities(raw_capabilities)
        self.DISTRIBUTED_TOKEN = _getenv("BANGERS_WORKER_TOKEN", "")
        self.DISTRIBUTED_REQUEST_TIMEOUT_SECONDS = float(
            _getenv("BANGERS_WORKER_TIMEOUT_SECONDS", "900")
        )

    @staticmethod
    def is_lm_disabled(model_name: str | None) -> bool:
        return is_lm_disabled(model_name)

    def startup_setting_overrides(self) -> dict[str, str]:
        """Settings that env should force over persisted DB values.

        Model selection is intentionally NOT exposed via env vars - the in-app
        Models page is the single source of truth for which models load.
        """
        overrides: dict[str, str] = {}
        if _has_env_value("BANGERS_BATCH_SIZE"):
            overrides["batch_size"] = str(self.DEFAULT_BATCH_SIZE)
        if _has_env_value("BANGERS_INFERENCE_STEPS"):
            overrides["inference_steps"] = str(self.DEFAULT_INFERENCE_STEPS)
        if _has_env_value("BANGERS_GUIDANCE_SCALE"):
            overrides["guidance_scale"] = str(self.DEFAULT_GUIDANCE_SCALE)
        if _has_env_value("BANGERS_THINKING"):
            overrides["thinking"] = _bool_string(self.DEFAULT_THINKING)
        if _has_env_value("BANGERS_LM_BACKEND"):
            overrides["lm_backend"] = self.DEFAULT_LM_BACKEND
        if _has_env_value("BANGERS_DEVICE"):
            overrides["device"] = self.DEFAULT_DEVICE
        if _has_env_value("BANGERS_AUDIO_FORMAT"):
            overrides["audio_format"] = self.DEFAULT_AUDIO_FORMAT
        return overrides

    def db_default_overrides(self) -> dict[str, str]:
        """Environment defaults for newly created databases.

        Note: dit_model and lm_model are intentionally absent - they are seeded
        as empty strings via DEFAULT_SETTINGS and only set when the user picks
        a model on the Models page.
        """
        return {
            "lm_backend": self.DEFAULT_LM_BACKEND,
            "device": self.DEFAULT_DEVICE,
            "audio_format": self.DEFAULT_AUDIO_FORMAT,
            "batch_size": str(self.DEFAULT_BATCH_SIZE),
            "default_duration": str(self.DEFAULT_DURATION),
            "inference_steps": str(self.DEFAULT_INFERENCE_STEPS),
            "guidance_scale": str(self.DEFAULT_GUIDANCE_SCALE),
            "thinking": _bool_string(self.DEFAULT_THINKING),
            "keep_active_models_resident": _bool_string(DEFAULT_KEEP_ACTIVE_MODELS_RESIDENT),
            "parallel_pipeline_enabled": _bool_string(DEFAULT_PARALLEL_PIPELINE_ENABLED),
            "lyrics_guardrails_enabled": _bool_string(DEFAULT_LYRICS_GUARDRAILS_ENABLED),
        }

    def ensure_dirs(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        project_root = Path(self.ACESTEP_PROJECT_ROOT)
        (project_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (project_root / "chat-llm").mkdir(parents=True, exist_ok=True)
        (project_root / ".cache" / "acestep").mkdir(parents=True, exist_ok=True)
        self.HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
        self.HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def has_distributed_capability(self, capability: str) -> bool:
        return capability in self.DISTRIBUTED_CAPABILITIES

    @property
    def delegates_to_workers(self) -> bool:
        return self.DISTRIBUTED_ROLE == "coordinator" and bool(self.DISTRIBUTED_WORKERS)


settings = Settings()
settings.apply_runtime_overrides()
