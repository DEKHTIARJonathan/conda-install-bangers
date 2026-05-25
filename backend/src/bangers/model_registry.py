from dataclasses import dataclass


LM_DISABLED_ALIASES = frozenset({"", "none", "disabled", "off", "false", "0"})


@dataclass(frozen=True)
class DownloadableModel:
    name: str
    repo_id: str
    model_type: str
    size_mb: int = 0
    description: str = ""
    compatible_runtimes: tuple[str, ...] = ()
    format: str = ""
    quantization: str = ""
    trust_remote_code: bool = False
    serve_command: str = ""


def _chat_model(
    name: str,
    repo_id: str,
    size_mb: int,
    description: str,
    compatible_runtimes: tuple[str, ...],
    format: str,
    quantization: str,
    trust_remote_code: bool = False,
    serve_command: str = "",
) -> DownloadableModel:
    return DownloadableModel(
        name=name,
        repo_id=repo_id,
        model_type="chat_llm",
        size_mb=size_mb,
        description=description,
        compatible_runtimes=compatible_runtimes,
        format=format,
        quantization=quantization,
        trust_remote_code=trust_remote_code,
        serve_command=serve_command,
    )


CHAT_LLM_MODELS: tuple[DownloadableModel, ...] = (
    # ========================= [MLX] Qwen3 models ========================= #
    _chat_model(
        name="Qwen3-0.6B-4bit",
        repo_id="mlx-community/Qwen3-0.6B-4bit",
        size_mb=335,
        description="Ultra-lightweight alternative. 335 MB.",
        compatible_runtimes=("mlx",),
        format="MLX",
        quantization="4bit",
    ),
    _chat_model(
        name="Qwen3-1.7B-4bit",
        repo_id="mlx-community/Qwen3-1.7B-4bit",
        size_mb=968,
        description="Balanced Qwen3 option. 968 MB.",
        compatible_runtimes=("mlx",),
        format="MLX",
        quantization="4bit",
    ),
    _chat_model(
        name="Qwen3-4B-4bit",
        repo_id="mlx-community/Qwen3-4B-4bit",
        size_mb=2_260,
        description="Larger Qwen3 option for better responses. ~2.3 GB.",
        compatible_runtimes=("mlx",),
        format="MLX",
        quantization="4bit",
    ),
    _chat_model(
        name="Qwen3-8B-4bit",
        repo_id="mlx-community/Qwen3-8B-4bit",
        size_mb=4_610,
        description="Qwen3 8B at 4-bit. Stronger local chat model. ~4.6 GB.",
        compatible_runtimes=("mlx",),
        format="MLX",
        quantization="4bit",
    ),
    # # ========================= [Transformers] Qwen3 small models ========================= #
    # # Small Transformers-runtime models for tight VRAM budgets (e.g. when the
    # # ACE-Step already holds a large model resident on the same GPU).
    # # These are plain BF16 checkpoints - no quantization, no trust_remote_code.
    # # ========================= [Transformers] NVIDIA Nemotron models ========================= #
    # _chat_model(
    #     name="NVIDIA-Nemotron-3-Nano-4B-FP8",
    #     repo_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
    #     size_mb=5_280,
    #     description="NVIDIA Nemotron 3 Nano 4B FP8 for Transformers/vLLM. ~5.3 GB.",
    #     compatible_runtimes=(),
    #     format="Transformers",
    #     quantization="FP8",
    # ),
    # ========================= [Transformers] Qwen3 models ========================= #
    _chat_model(
        name="Qwen3-1.7B",
        repo_id="Qwen/Qwen3-1.7B",
        size_mb=3_400,
        description="Qwen3 1.7B BF16 for Transformers. ~3.4 GB VRAM.",
        compatible_runtimes=(),
        format="Transformers",
        quantization="BF16",
    ),
    _chat_model(
        name="Qwen3-4B-Instruct-2507",
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        size_mb=8_100,
        description="Qwen3 4B Instruct BF16 for Transformers. ~8 GB VRAM. Stronger small option.",
        compatible_runtimes=(),
        format="Transformers",
        quantization="BF16",
    ),
    _chat_model(
        name="Qwen3-8B-FP8",
        repo_id="Qwen/Qwen3-8B-FP8",
        size_mb=9_450,
        description="Qwen3 8B FP8 for Transformers. ~9.45 GB VRAM. Stronger small option.",
        compatible_runtimes=(),
        format="Transformers",
        quantization="FP8",
    ),
    _chat_model(
        name="Qwen3-14B-FP8",
        repo_id="Qwen/Qwen3-14B-FP8",
        size_mb=16_300,
        description="Qwen3 14B FP8 for Transformers. ~16.3 GB VRAM. Stronger small option.",
        compatible_runtimes=(),
        format="Transformers",
        quantization="FP8",
    ),
    _chat_model(
        name="Qwen3-30B-A3B-Instruct-2507-FP8",
        repo_id="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        size_mb=31_200,
        description="Qwen3 30B A3B Instruct 2507 FP8 for Transformers. ~31.2 GB VRAM. Stronger small option.",
        compatible_runtimes=(),
        format="Transformers",
        quantization="FP8",
    ),
    # ========================= [TensorRT-LLM] NVIDIA FP4/NVFP4 models ========================= #
    _chat_model(
        name="Qwen3-8B-NVFP4",
        repo_id="nvidia/Qwen3-8B-NVFP4",
        size_mb=5_500,
        description=(
            "NVIDIA Qwen3 8B NVFP4 checkpoint for an external TensorRT-LLM "
            "OpenAI-compatible server. Requires Blackwell-class NVIDIA GPU."
        ),
        compatible_runtimes=("trtllm",),
        format="TensorRT-LLM",
        quantization="NVFP4",
        serve_command="mise run trtllm:qwen3-8b-nvfp4",
    ),
    _chat_model(
        name="Qwen3-14B-NVFP4",
        repo_id="nvidia/Qwen3-14B-NVFP4",
        size_mb=11_000,
        description=(
            "NVIDIA Qwen3 14B NVFP4 checkpoint for an external TensorRT-LLM "
            "OpenAI-compatible server. Requires Blackwell-class NVIDIA GPU."
        ),
        compatible_runtimes=("trtllm",),
        format="TensorRT-LLM",
        quantization="NVFP4",
        serve_command="mise run trtllm:qwen3-14b-nvfp4",
    ),
    _chat_model(
        name="Qwen3-30B-FP4",
        repo_id="nvidia/Qwen3-30B-A3B-NVFP4",
        size_mb=22_000,
        description=(
            "NVIDIA Qwen3 30B A3B FP4 checkpoint for an external "
            "TensorRT-LLM OpenAI-compatible server. Requires Blackwell-class "
            "NVIDIA GPU."
        ),
        compatible_runtimes=("trtllm",),
        format="TensorRT-LLM",
        quantization="FP4",
        serve_command="mise run trtllm:qwen3-30b-fp4",
    ),
    # # ========================= [Transformers] Qwen3.5 models ========================= #
    # _chat_model(
    #     name="Qwen3.5-9B",
    #     repo_id="Qwen/Qwen3.5-9B",
    #     size_mb=19_300,
    #     description="Qwen3.5 9B for Transformers/vLLM. ~19.3 GB.",
    #     compatible_runtimes=(),
    #     format="Transformers",
    #     quantization="",
    # ),
    # # ========================= [Transformers] Qwen3.6 models ========================= #
    # _chat_model(
    #     name="Qwen3.6-27B-FP8",
    #     repo_id="Qwen/Qwen3.6-27B-FP8",
    #     size_mb=30_900,
    #     description="Qwen3.6 27B FP8 for Transformers/vLLM. ~30.9 GB.",
    #     compatible_runtimes=(),
    #     format="Transformers FP8",
    #     quantization="FP8",
    # ),
    # _chat_model(
    #     name="Qwen3.6-35B-A3B-FP8",
    #     repo_id="Qwen/Qwen3.6-35B-A3B-FP8",
    #     size_mb=37_500,
    #     description="Qwen3.6 35B A3B FP8 for Transformers/vLLM. ~37.5 GB.",
    #     compatible_runtimes=(),
    #     format="Transformers FP8",
    #     quantization="FP8",
    # ),
    # _chat_model(
    #     name="Qwen3.6-35B-A3B",
    #     repo_id="Qwen/Qwen3.6-35B-A3B",
    #     size_mb=71_900,
    #     description="Qwen3.6 35B A3B for Transformers/vLLM. ~71.9 GB.",
    #     compatible_runtimes=(),
    #     format="Transformers",
    #     quantization="BF16",
    # ),
    # # ========================= [Transformers] Nemotron models ========================= #
    # _chat_model(
    #     name="Llama-3_3-Nemotron-Super-49B-v1_5-FP8",
    #     repo_id="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8",
    #     size_mb=52_000,
    #     description="NVIDIA Nemotron Super 49B v1.5 FP8 for Transformers/vLLM. ~52 GB.",
    #     compatible_runtimes=(),
    #     format="Transformers FP8",
    #     quantization="FP8",
    #     trust_remote_code=True,
    # ),
)

@dataclass(frozen=True)
class DownloadableAceModel:
    """A curated ACE-Step DiT or LM submodel that we expose in the UI.

    `bundled=True` means the weights live inside the main ACE-Step1.5 repo
    (`ACE_MAIN_BUNDLE.repo_id`) and are pulled together with the support
    assets (vae, text encoder). For bundled entries `repo_id` is informational
    only; downloads route through `download_main_model`.
    """

    name: str
    repo_id: str
    model_type: str  # "dit" | "lm"
    size_mb: int
    description: str
    bundled: bool = False


@dataclass(frozen=True)
class AceMainBundle:
    """The ACE-Step1.5 HuggingFace repo that ships the default weights and
    the two support assets the pipeline expects on disk.

    `provides` is the list of subdirectories the snapshot creates under
    `checkpoints/`. The first two entries are user-pickable models that we
    surface in the UI; the rest are support assets the pipeline loads from
    fixed paths (see acestep/core/generation/handler/init_service_loader.py).
    """

    repo_id: str
    bundle_size_mb: int
    provides: tuple[str, ...]
    visible_components: tuple[str, ...]


ACE_MAIN_BUNDLE = AceMainBundle(
    repo_id="ACE-Step/Ace-Step1.5",
    bundle_size_mb=10_000,
    provides=(
        "acestep-v15-turbo",
        "acestep-5Hz-lm-1.7B",
        "vae",
        "Qwen3-Embedding-0.6B",
    ),
    visible_components=(
        "acestep-v15-turbo",
        "acestep-5Hz-lm-1.7B",
    ),
)


ACE_DIT_MODELS: tuple[DownloadableAceModel, ...] = (
    DownloadableAceModel(
        name="acestep-v15-turbo",
        repo_id=ACE_MAIN_BUNDLE.repo_id,
        model_type="dit",
        size_mb=ACE_MAIN_BUNDLE.bundle_size_mb,
        description=(
            "Default Turbo DiT. Ships in the main ACE-Step bundle alongside "
            "the VAE, text encoder, and 1.7B LM (~10 GB total)."
        ),
        bundled=True,
    ),
    DownloadableAceModel(
        name="acestep-v15-base",
        repo_id="ACE-Step/acestep-v15-base",
        model_type="dit",
        size_mb=4_792,
        description="Base DiT model. 50 steps, CFG-guided.",
    ),
    DownloadableAceModel(
        name="acestep-v15-sft",
        repo_id="ACE-Step/acestep-v15-sft",
        model_type="dit",
        size_mb=4_792,
        description="SFT-tuned DiT. 50 steps, CFG-guided.",
    ),
    DownloadableAceModel(
        name="acestep-v15-turbo-continuous",
        repo_id="ACE-Step/acestep-v15-turbo-continuous",
        model_type="dit",
        size_mb=4_792,
        description="Turbo DiT with continuous noise schedule.",
    ),
    DownloadableAceModel(
        name="acestep-v15-xl-turbo",
        repo_id="ACE-Step/acestep-v15-xl-turbo",
        model_type="dit",
        size_mb=20_000,
        description="XL Turbo DiT (~20 GB). Larger turbo variant.",
    ),
)


ACE_LM_MODELS: tuple[DownloadableAceModel, ...] = (
    DownloadableAceModel(
        name="acestep-5Hz-lm-1.7B",
        repo_id=ACE_MAIN_BUNDLE.repo_id,
        model_type="lm",
        size_mb=ACE_MAIN_BUNDLE.bundle_size_mb,
        description=(
            "Based on Qwen3-1.7B. Balanced default; downloads with the main "
            "ACE-Step bundle (~10 GB)."
        ),
        bundled=True,
    ),
    DownloadableAceModel(
        name="acestep-5Hz-lm-0.6B",
        repo_id="ACE-Step/acestep-5Hz-lm-0.6B",
        model_type="lm",
        size_mb=1_373,
        description="Based on Qwen3-0.6B. Lightweight (6-8 GB VRAM).",
    ),
    DownloadableAceModel(
        name="acestep-5Hz-lm-4B",
        repo_id="ACE-Step/acestep-5Hz-lm-4B",
        model_type="lm",
        size_mb=8_426,
        description="Based on Qwen3-4B. Best quality (24 GB+ VRAM).",
    ),
)

CHAT_LLM_REGISTRY = {model.name: model.repo_id for model in CHAT_LLM_MODELS}
CHAT_LLM_BY_NAME = {model.name: model for model in CHAT_LLM_MODELS}
CHAT_LLM_SIZES = {model.name: model.size_mb for model in CHAT_LLM_MODELS}
CHAT_LLM_DESCRIPTIONS = {model.name: model.description for model in CHAT_LLM_MODELS}
CHAT_LLM_COMPATIBILITY = {
    model.name: model.compatible_runtimes for model in CHAT_LLM_MODELS
}
CHAT_LLM_FORMATS = {model.name: model.format for model in CHAT_LLM_MODELS}
CHAT_LLM_QUANTIZATIONS = {
    model.name: model.quantization for model in CHAT_LLM_MODELS
}
CHAT_LLM_SERVE_COMMANDS = {
    model.name: model.serve_command for model in CHAT_LLM_MODELS
}

ACE_MODELS: tuple[DownloadableAceModel, ...] = ACE_DIT_MODELS + ACE_LM_MODELS
ACE_MODEL_BY_NAME: dict[str, DownloadableAceModel] = {
    model.name: model for model in ACE_MODELS
}
ACE_MODEL_SIZES: dict[str, int] = {model.name: model.size_mb for model in ACE_MODELS}
ACE_MODEL_DESCRIPTIONS: dict[str, str] = {
    model.name: model.description for model in ACE_MODELS
}


def is_lm_disabled(model_name: str | None) -> bool:
    return (model_name or "").strip().lower() in LM_DISABLED_ALIASES


def is_ace_dit(name: str) -> bool:
    model = ACE_MODEL_BY_NAME.get(name)
    return model is not None and model.model_type == "dit"


def is_ace_lm(name: str) -> bool:
    model = ACE_MODEL_BY_NAME.get(name)
    return model is not None and model.model_type == "lm"


def ace_repo_id_for(name: str) -> str | None:
    """Effective HF repo to download `name` from.

    For bundled entries this is always the main bundle repo (the per-model
    `repo_id` is informational). Returns None for unknown names.
    """
    model = ACE_MODEL_BY_NAME.get(name)
    if model is None:
        return None
    return ACE_MAIN_BUNDLE.repo_id if model.bundled else model.repo_id


def is_main_bundle_member(name: str) -> bool:
    """True if `name` is shipped inside ACE_MAIN_BUNDLE.repo_id (including
    support assets like 'vae' and 'Qwen3-Embedding-0.6B')."""
    return name in ACE_MAIN_BUNDLE.provides


def main_bundle_visible_components() -> tuple[str, ...]:
    """The bundle entries that are user-pickable (turbo DiT + 1.7B LM)."""
    return ACE_MAIN_BUNDLE.visible_components


def chat_runtime_for(model_name: str) -> str:
    """Return which chat runtime should load `model_name`.

    Looks up `compatible_runtimes` in CHAT_LLM_BY_NAME. If 'mlx' is in the
    tuple, returns 'trtllm' or 'mlx'; otherwise returns 'transformers'.
    Unknown models default to 'transformers' so user-installed model folders
    still work.
    """
    metadata = CHAT_LLM_BY_NAME.get(model_name)
    if metadata is None:
        return "transformers"
    if "trtllm" in metadata.compatible_runtimes:
        return "trtllm"
    return "mlx" if "mlx" in metadata.compatible_runtimes else "transformers"
