"""Chat-LLM runtime adapters.

Each chat model carries a `compatible_runtimes` tuple in
:mod:`bangers.model_registry`. This module exposes a single
:func:`get_chat_runtime` that returns the correct backend instance
(MLX on Apple Silicon, Transformers elsewhere, or an external
TensorRT-LLM OpenAI-compatible server) based on that metadata.

There is no user-facing 'provider' selection any more: the user only
picks a chat model on the Models page, and the runtime is derived
from the model itself.
"""

import asyncio
import json
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from loguru import logger

from bangers.model_registry import CHAT_LLM_BY_NAME, CHAT_LLM_REGISTRY, chat_runtime_for


class ChatRuntime(ABC):
    """Loads and runs a chat LLM checkpoint."""

    def loaded_model_name(self) -> str:
        """Return the model currently resident in this runtime, if any."""
        return ""

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True iff this runtime can load `model_name`."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        *,
        allow_holders: frozenset[str] | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant response.

        ``allow_holders`` lets callers that are themselves the GPU lock
        holder bypass the "GPU busy" defense. For example, the music
        services own the lock for the duration of a job and need to call
        the chat LLM (titles, lyric specs) without raising
        ``ChatRuntimeBusy`` against themselves.
        """

    def unload(self) -> None:
        """Release any resident local model resources held by this runtime."""


class ChatRuntimeBusy(RuntimeError):
    """Raised when a GPU-backed chat model should be retried later."""


class TrtLlmServerUnavailable(RuntimeError):
    """Raised when an external TRT-LLM server cannot serve a selected model."""

    def __init__(
        self,
        model_name: str,
        command: str,
        server_url: str,
        reason: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.command = command
        self.server_url = server_url
        message = (
            f"TRT-LLM server is not running {model_name} at {server_url}. "
            f"Start the matching server with `{command}` and try again."
        )
        if reason:
            message = f"{message} ({reason})"
        super().__init__(message)


def trtllm_serve_command_for(model_name: str) -> str:
    metadata = CHAT_LLM_BY_NAME.get(model_name)
    if metadata is not None and metadata.serve_command:
        return metadata.serve_command
    task_name = model_name.lower().replace("_", "-")
    return f"mise run trtllm:{task_name}"


class TrtLlmChatRuntime(ChatRuntime):
    """External TensorRT-LLM OpenAI-compatible chat runtime."""

    def __init__(self) -> None:
        self._loaded_model_name = ""
        self._lock = threading.Lock()

    @staticmethod
    def runtime_supported() -> bool:
        return sys.platform.startswith("linux")

    @property
    def _chat_llm_dir(self) -> Path:
        from bangers.config import settings

        return Path(settings.ACESTEP_PROJECT_ROOT) / "chat-llm"

    def _model_installed(self, model_name: str) -> bool:
        if not model_name:
            return False
        config = self._chat_llm_dir / model_name / "config.json"
        return config.exists()

    def loaded_model_name(self) -> str:
        return self._loaded_model_name

    async def is_available(self) -> bool:
        return self.runtime_supported()

    def is_model_loadable(self, model_name: str) -> bool:
        return self.runtime_supported() and self._model_installed(model_name)

    def _unavailable(self, model_name: str, reason: str | None = None) -> TrtLlmServerUnavailable:
        from bangers.config import settings

        return TrtLlmServerUnavailable(
            model_name=model_name,
            command=trtllm_serve_command_for(model_name),
            server_url=settings.TRTLLM_SERVER_URL,
            reason=reason,
        )

    @staticmethod
    def _endpoint(path: str) -> str:
        from bangers.config import settings

        return f"{settings.TRTLLM_SERVER_URL}{path}"

    @staticmethod
    def _timeout() -> float:
        from bangers.config import settings

        return settings.TRTLLM_TIMEOUT_SECONDS

    def _request_json(
        self,
        model_name: str,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urlrequest.Request(
            self._endpoint(path),
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlrequest.urlopen(request, timeout=self._timeout()) as response:
                body = response.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            reason = f"HTTP {exc.code}" + (f": {detail}" if detail else "")
            raise self._unavailable(model_name, reason) from exc
        except (TimeoutError, OSError, urlerror.URLError) as exc:
            raise self._unavailable(model_name, str(exc)) from exc
        if not body:
            return {}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            if path == "/health":
                return {"raw": body}
            raise RuntimeError(f"TRT-LLM server returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("TRT-LLM server returned an invalid JSON payload.")
        return parsed

    def _chat_sync(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        with self._lock:
            self._request_json(model_name, "/health")
            payload: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            response = self._request_json(
                model_name,
                "/v1/chat/completions",
                method="POST",
                payload=payload,
            )
            try:
                choice = response["choices"][0]
                message = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(message, dict):
                    content = message.get("content", "")
                else:
                    content = choice.get("text", "")
            except (KeyError, IndexError, AttributeError) as exc:
                raise RuntimeError("TRT-LLM server returned no chat completion.") from exc
            if not isinstance(content, str):
                raise RuntimeError("TRT-LLM server returned non-text chat content.")
            self._loaded_model_name = model_name
            return content

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        *,
        allow_holders: frozenset[str] | None = None,
    ) -> str:
        del allow_holders
        if not self._model_installed(model):
            raise RuntimeError(f"Chat model '{model}' is not installed.")
        logger.info(f"LLM chat [trtllm] model={model} temp={temperature}")
        return await asyncio.to_thread(
            self._chat_sync, messages, model, max_tokens, temperature
        )


class MLXChatRuntime(ChatRuntime):
    """MLX-based chat runtime (macOS only)."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded_model_name: str = ""
        self._lock = threading.Lock()

        from bangers.config import settings
        self._chat_llm_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "chat-llm"

    @staticmethod
    def runtime_supported() -> bool:
        if sys.platform != "darwin":
            return False
        try:
            import mlx_lm  # noqa: F401
            return True
        except ImportError:
            return False

    def _model_installed(self, model_name: str) -> bool:
        if not model_name:
            return False
        config = self._chat_llm_dir / model_name / "config.json"
        return config.exists()

    def loaded_model_name(self) -> str:
        return self._loaded_model_name if self._model is not None else ""

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded_model_name = ""

    async def is_available(self) -> bool:
        return self.runtime_supported()

    def is_model_loadable(self, model_name: str) -> bool:
        return self.runtime_supported() and self._model_installed(model_name)

    def _load(self, model_name: str) -> None:
        if not self.runtime_supported():
            raise RuntimeError(
                "MLX chat runtime is only available on Apple Silicon Macs with mlx_lm installed."
            )
        if self._model is not None and self._loaded_model_name == model_name:
            return

        model_path = str(self._chat_llm_dir / model_name)
        logger.info(f"Loading MLX chat model from {model_path}")

        from mlx_lm import load  # type: ignore[import-untyped]

        self._model, self._tokenizer = load(model_path)
        self._loaded_model_name = model_name
        logger.info(f"MLX chat model '{model_name}' loaded")

    def _generate_sync(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        with self._lock:
            self._load(model_name)

            from mlx_lm import generate  # type: ignore[import-untyped]
            from mlx_lm.sample_utils import make_sampler  # type: ignore[import-untyped]

            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            sampler = make_sampler(temp=temperature)
            return generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            )

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        *,
        allow_holders: frozenset[str] | None = None,
    ) -> str:
        if not self._model_installed(model):
            raise RuntimeError(f"Chat model '{model}' is not installed.")
        lock_holder = "chat-llm"
        acquired_lock = False
        from bangers.services.gpu_lock import gpu_lock

        allowed = allow_holders or frozenset()
        holder = gpu_lock.holder
        if gpu_lock.is_locked:
            if holder not in allowed:
                raise ChatRuntimeBusy(
                    f"MLX chat model '{model}' deferred because GPU is busy with {holder}"
                )
        else:
            await gpu_lock.await_acquire(lock_holder)
            acquired_lock = True

        try:
            logger.info(f"LLM chat [mlx] model={model} temp={temperature}")
            return await asyncio.to_thread(
                self._generate_sync, messages, model, max_tokens, temperature
            )
        finally:
            if acquired_lock:
                await gpu_lock.release(lock_holder)


class TransformersChatRuntime(ChatRuntime):
    """Transformers-based chat runtime (Linux/CUDA, also CPU)."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded_model_name = ""
        self._lock = threading.Lock()

        from bangers.config import settings
        self._chat_llm_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "chat-llm"

    @staticmethod
    def runtime_supported() -> bool:
        try:
            import accelerate  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _model_installed(self, model_name: str) -> bool:
        if not model_name:
            return False
        config = self._chat_llm_dir / model_name / "config.json"
        return config.exists()

    def loaded_model_name(self) -> str:
        return self._loaded_model_name if self._model is not None else ""

    def unload(self) -> None:
        if self._model is None and self._tokenizer is None:
            self._loaded_model_name = ""
            return
        logger.info("Unloading Transformers chat model")
        self._model = None
        self._tokenizer = None
        self._loaded_model_name = ""
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug(f"Failed to clear CUDA cache after chat unload: {exc}")

    async def is_available(self) -> bool:
        return self.runtime_supported()

    def is_model_loadable(self, model_name: str) -> bool:
        return self.runtime_supported() and self._model_installed(model_name)

    def _load(self, model_name: str) -> None:
        if not self.runtime_supported():
            raise RuntimeError(
                "Transformers chat runtime requires torch, transformers, and accelerate."
            )
        if self._model is not None and self._loaded_model_name == model_name:
            return

        meta = CHAT_LLM_BY_NAME.get(model_name)
        model_path = str(self._chat_llm_dir / model_name)
        trust_remote_code = bool(meta and meta.trust_remote_code)
        logger.info(f"Loading Transformers chat model from {model_path}")

        import gc

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype="auto",
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
            self._model.eval()
            self._loaded_model_name = model_name
        except Exception:
            # A partial load (e.g. unknown model_type, dtype mismatch, OOM mid-shard)
            # can strand accelerate stub/offload buffers on the GPU. Drop references
            # and reclaim VRAM so the next attempt — or whatever runs next on the
            # same device — starts from a clean slate instead of inheriting a
            # zombie allocation.
            self._model = None
            self._tokenizer = None
            self._loaded_model_name = ""
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        logger.info(f"Transformers chat model '{model_name}' loaded")

    def _generate_sync(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        with self._lock:
            self._load(model_name)
            import torch

            inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            generation_kwargs: dict = {
                **inputs,
                "max_new_tokens": max_tokens,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs.update({"do_sample": True, "temperature": temperature})
            else:
                generation_kwargs.update({"do_sample": False})

            with torch.inference_mode():
                outputs = self._model.generate(**generation_kwargs)

            prompt_len = inputs["input_ids"].shape[-1]
            return self._tokenizer.decode(
                outputs[0][prompt_len:],
                skip_special_tokens=True,
            )

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        *,
        allow_holders: frozenset[str] | None = None,
    ) -> str:
        # Transformers (CUDA/CPU) doesn't share Metal with the music engine,
        # so the gpu_lock holder check doesn't apply. allow_holders is accepted
        # only to keep the ChatRuntime signature consistent across runtimes.
        del allow_holders
        if not self._model_installed(model):
            raise RuntimeError(f"Chat model '{model}' is not installed.")
        logger.info(f"LLM chat [transformers] model={model} temp={temperature}")
        return await asyncio.to_thread(
            self._generate_sync, messages, model, max_tokens, temperature
        )


_mlx_runtime = MLXChatRuntime()
_transformers_runtime = TransformersChatRuntime()
_trtllm_runtime = TrtLlmChatRuntime()


def get_chat_runtime(model_name: str) -> Optional[ChatRuntime]:
    """Return the runtime that should load `model_name`, or None if unsupported.

    Selection is based on `chat_runtime_for(model_name)` from the model registry.
    Returns None when the chosen runtime isn't installed/usable on this machine
    (e.g. asking for an MLX model on Linux).
    """
    if not model_name:
        return None
    runtime_kind = chat_runtime_for(model_name)
    if runtime_kind == "trtllm":
        return _trtllm_runtime if _trtllm_runtime.runtime_supported() else None
    if runtime_kind == "mlx":
        return _mlx_runtime if _mlx_runtime.runtime_supported() else None
    return _transformers_runtime if _transformers_runtime.runtime_supported() else None


def installed_chat_models() -> list[str]:
    """Return chat-LLM checkpoints found on disk that this machine can run."""
    from bangers.config import settings
    chat_llm_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "chat-llm"
    if not chat_llm_dir.exists():
        return []
    found: list[str] = []
    for entry in sorted(chat_llm_dir.iterdir()):
        if not entry.is_dir() or not (entry / "config.json").exists():
            continue
        name = entry.name
        if name not in CHAT_LLM_REGISTRY:
            continue
        runtime = get_chat_runtime(name)
        if runtime is None:
            continue
        found.append(name)
    return found


def loaded_chat_model_name() -> str:
    """Return the Chat LLM currently loaded or externally warmed, if any."""
    for runtime in (_mlx_runtime, _transformers_runtime, _trtllm_runtime):
        loaded = runtime.loaded_model_name()
        if loaded:
            return loaded
    return ""


def unload_embedded_chat_models() -> None:
    """Release local chat runtimes after switching to external TRT-LLM."""
    _transformers_runtime.unload()
