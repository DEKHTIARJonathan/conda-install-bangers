#!/usr/bin/env python3
"""conda install bangers Launcher — starts backend + frontend in one command."""

import os
import argparse
import csv
import errno
import json
import re
import signal
import subprocess
import sys
import shutil
import shlex
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.error import URLError

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
CONDA_ENV_DIR = BACKEND_DIR / ".conda"
BACKEND_ENV_FILE = BACKEND_DIR / "environment.yml"
BACKEND_PYPROJECT_FILE = BACKEND_DIR / "pyproject.toml"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu130"
DEFAULT_DATA_DIR = BACKEND_DIR / "data"
DEFAULT_MODEL_CACHE_DIR = ROOT / ".cache" / "models"
RUNTIME_LOG_FILE = ROOT / "runtime.log"

_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_RUNTIME_LOG_LOCK = threading.Lock()
_RUNTIME_LOG_HANDLE = None

# Self-signed cert + key shared by both Next.js and uvicorn. Generated once
# and reused on every restart (regenerated only if missing or expired). Lives
# under .cache so it's gitignored alongside other runtime
# artifacts and survives `mise run clean`'s narrower data-dir wipe.
CERT_DIR = ROOT / ".cache" / "tls"
CERT_FILE = CERT_DIR / "dev.pem"
KEY_FILE = CERT_DIR / "dev-key.pem"
FRONTEND_PORT = 3000
BACKEND_PORT = 8000
DEFAULT_LOCAL_WORKER_CAPABILITIES = "music,ace_lm,chat_llm"

# Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass(frozen=True)
class LocalGpu:
    index: str
    uuid: str
    name: str
    selector: str


@dataclass(frozen=True)
class LocalWorkerSpec:
    gpu: LocalGpu
    node_id: str
    port: int
    url: str


@dataclass(frozen=True)
class RemoteWorkerSpec:
    url: str
    ssh_host: str
    port: int
    root: str


def _write_runtime_log_bytes(data: bytes) -> None:
    if _RUNTIME_LOG_HANDLE is None:
        return
    with _RUNTIME_LOG_LOCK:
        _RUNTIME_LOG_HANDLE.write(data)
        _RUNTIME_LOG_HANDLE.flush()


class _RuntimeLogTextTee:
    """Text stream that keeps console TTY behavior while appending to runtime.log."""

    def __init__(self, stream):
        self._stream = stream

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._stream, "errors", "replace")

    def write(self, text) -> int:
        if not isinstance(text, str):
            text = str(text)
        written = self._stream.write(text)
        _write_runtime_log_bytes(
            text.encode(self.encoding or "utf-8", errors=self.errors or "replace")
        )
        return written

    def flush(self) -> None:
        self._stream.flush()
        if _RUNTIME_LOG_HANDLE is not None:
            with _RUNTIME_LOG_LOCK:
                _RUNTIME_LOG_HANDLE.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_runtime_logging() -> None:
    global _RUNTIME_LOG_HANDLE
    if _RUNTIME_LOG_HANDLE is not None:
        return
    _RUNTIME_LOG_HANDLE = RUNTIME_LOG_FILE.open("ab", buffering=0)
    sys.stdout = _RuntimeLogTextTee(_ORIGINAL_STDOUT)
    sys.stderr = _RuntimeLogTextTee(_ORIGINAL_STDERR)


def _write_console_bytes(data: bytes) -> None:
    stream = getattr(_ORIGINAL_STDOUT, "buffer", None)
    if stream is not None:
        stream.write(data)
        stream.flush()
    else:
        _ORIGINAL_STDOUT.write(data.decode("utf-8", errors="replace"))
        _ORIGINAL_STDOUT.flush()


def _pipe_pty_to_console_and_log(master_fd: int) -> None:
    try:
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            _write_console_bytes(chunk)
            _write_runtime_log_bytes(chunk)
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass


def log(msg: str, color: str = GREEN) -> None:
    print(f"{color}{BOLD}[conda install bangers]{RESET} {msg}")


def ensure_data_dirs() -> None:
    """Create data directories if they don't exist."""
    data_dir = Path(os.getenv("BANGERS_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
    model_cache_dir = Path(
        os.getenv("BANGERS_MODEL_CACHE_DIR", str(DEFAULT_MODEL_CACHE_DIR))
    ).expanduser()

    (data_dir / "audio").mkdir(parents=True, exist_ok=True)
    (data_dir / "uploads").mkdir(parents=True, exist_ok=True)

    checkpoints_dir = model_cache_dir / "checkpoints"
    (model_cache_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (model_cache_dir / "chat-llm").mkdir(parents=True, exist_ok=True)
    (model_cache_dir / "huggingface" / "hub").mkdir(parents=True, exist_ok=True)
    has_models = any(checkpoints_dir.iterdir()) if checkpoints_dir.exists() else False
    if not has_models:
        log("No model checkpoints found.", YELLOW)
        log("Open the Models page once the app starts to download and select a model.", YELLOW)
        log(f"Checkpoint directory: {checkpoints_dir}", YELLOW)
        print()


def build_runtime_env() -> dict[str, str]:
    model_cache_dir = os.environ.get("BANGERS_MODEL_CACHE_DIR", str(DEFAULT_MODEL_CACHE_DIR))
    data_dir = os.environ.get("BANGERS_DATA_DIR", str(DEFAULT_DATA_DIR))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "BANGERS_DATA_DIR": data_dir,
        "BANGERS_MODEL_CACHE_DIR": model_cache_dir,
        "ACESTEP_PROJECT_ROOT": os.environ.get("ACESTEP_PROJECT_ROOT", model_cache_dir),
        "HF_HOME": os.environ.get("HF_HOME", str(Path(model_cache_dir) / "huggingface")),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", str(Path(model_cache_dir) / "huggingface" / "hub")),
    }
    return env


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE lines from .env without overriding real env."""
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip("'\"")
        os.environ[key] = value


def _conda_executable() -> str | None:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe
    return shutil.which("conda")


def _conda_env_python() -> Path:
    if sys.platform == "win32":
        return CONDA_ENV_DIR / "python.exe"
    return CONDA_ENV_DIR / "bin" / "python"


def _conda_env_bin_dir() -> Path:
    return CONDA_ENV_DIR / ("Scripts" if sys.platform == "win32" else "bin")


def _conda_env_executable(name: str) -> Path:
    bin_dir = _conda_env_bin_dir()
    if sys.platform == "win32":
        for suffix in (".exe", ".bat", ".cmd", ""):
            candidate = bin_dir / f"{name}{suffix}"
            if candidate.exists():
                return candidate
        return bin_dir / f"{name}.exe"
    return bin_dir / name


def _with_conda_env_on_path(env: dict[str, str]) -> dict[str, str]:
    bin_dir = str(_conda_env_bin_dir())
    return {
        **env,
        "CONDA_PREFIX": str(CONDA_ENV_DIR),
        "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
    }


def check_prerequisites() -> bool:
    ok = True

    # Launcher Python version. Backend Python is supplied by conda.
    if sys.version_info < (3, 10):
        log(f"Python 3.10+ required to run the launcher, found {sys.version}", RED)
        ok = False

    # conda
    if _conda_executable() is None:
        log("'conda' not found. Run `mise install` or install Miniforge/Miniconda.", RED)
        ok = False

    # Node.js
    if not shutil.which("node"):
        log("'node' not found. Install: https://nodejs.org/", RED)
        ok = False

    # pnpm
    if not shutil.which("pnpm"):
        log("'pnpm' not found. Install: https://pnpm.io/installation", RED)
        ok = False

    # openssl: used to mint a self-signed cert for the dev TLS stack so
    # browsers expose AudioWorklet (and other secure-context APIs) when the
    # app is reached over a LAN IP. Ships with macOS and almost every Linux
    # distro out of the box; only flag if it's genuinely missing.
    if not shutil.which("openssl"):
        log(
            "'openssl' not found — required to generate the dev TLS cert. "
            "Install via your package manager (e.g. `apt install openssl`).",
            RED,
        )
        ok = False

    return ok


def open_browser_when_ready(url: str, timeout: int = 60) -> None:
    """Poll the frontend URL and open the browser once it responds.

    Next.js serves HTTPS with our self-signed dev cert, so we can't let
    urlopen verify it — we only need a 2xx/3xx response to know the dev
    server is up.
    """
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            # Generous per-request timeout so the very first hit doesn't
            # time out while Next.js is cold-compiling the route.
            urlopen(url, timeout=30, context=ctx)
            log(f"Opening {CYAN}{BOLD}{url}{RESET} in your browser...")
            webbrowser.open(url)
            return
        except (URLError, OSError):
            time.sleep(1)
    log(f"Frontend didn't respond in time — open {url} manually.", YELLOW)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, using shell=True on Windows so .cmd wrappers are found."""
    if sys.platform == "win32":
        return subprocess.run(cmd, shell=True, **kwargs)
    return subprocess.run(cmd, **kwargs)


def _popen(cmd: list[str], **kwargs) -> subprocess.Popen:
    """Open a subprocess, using shell=True on Windows so .cmd wrappers are found."""
    if sys.platform == "win32":
        return subprocess.Popen(cmd, shell=True, **kwargs)
    kwargs.setdefault("start_new_session", True)
    if _RUNTIME_LOG_HANDLE is None:
        return subprocess.Popen(cmd, **kwargs)

    import pty

    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=slave_fd,
            stderr=slave_fd,
            **kwargs,
        )
    finally:
        os.close(slave_fd)

    reader = threading.Thread(
        target=_pipe_pty_to_console_and_log,
        args=(master_fd,),
        daemon=True,
    )
    reader.start()
    return proc


def _cert_is_fresh(cert_path: Path, sans: list[str]) -> bool:
    """True iff the cert exists, is valid for >7 days, and covers every SAN.

    We regenerate proactively when the LAN IP changes (laptop moved networks)
    so the browser doesn't reject the cert with NET::ERR_CERT_COMMON_NAME_INVALID.
    """
    if not cert_path.exists():
        return False
    try:
        result = subprocess.run(
            ["openssl", "x509", "-in", str(cert_path), "-noout",
             "-checkend", str(7 * 24 * 3600), "-text"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    text = result.stdout
    for san in sans:
        # openssl prints SANs as "DNS:foo" or "IP Address:1.2.3.4"
        token = f"IP Address:{san}" if san.replace(".", "").isdigit() else f"DNS:{san}"
        if token not in text:
            return False
    return True


def ensure_dev_cert(extra_hosts: list[str]) -> tuple[Path, Path]:
    """Mint (or reuse) a self-signed cert covering localhost + LAN IP(s).

    The same cert/key pair is fed to both Next.js (`--experimental-https-cert`)
    and uvicorn (`--ssl-certfile`) so the frontend and backend share an origin
    scheme — without that the browser blocks the cross-origin fetch as
    mixed content.

    No CA install, no system trust store, no sudo. The browser will warn
    once about the self-signed cert; click through it.
    """
    sans = ["localhost", "127.0.0.1", *extra_hosts]
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    if _cert_is_fresh(CERT_FILE, sans) and KEY_FILE.exists():
        return CERT_FILE, KEY_FILE

    log("Generating self-signed dev TLS cert...")
    dns_idx = 0
    ip_idx = 0
    san_entries: list[str] = []
    for h in sans:
        if h.replace(".", "").isdigit():
            ip_idx += 1
            san_entries.append(f"IP.{ip_idx} = {h}")
        else:
            dns_idx += 1
            san_entries.append(f"DNS.{dns_idx} = {h}")
    san_lines = "\n".join(san_entries)
    config = f"""
[req]
distinguished_name = dn
x509_extensions = v3_req
prompt = no

[dn]
CN = conda install bangers dev

[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt

[alt]
{san_lines}
""".strip()

    config_path = CERT_DIR / "openssl.cnf"
    config_path.write_text(config)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-nodes",
            "-newkey", "rsa:2048",
            "-keyout", str(KEY_FILE),
            "-out", str(CERT_FILE),
            "-days", "365",
            "-config", str(config_path),
        ],
        check=True, capture_output=True,
    )
    KEY_FILE.chmod(0o600)
    return CERT_FILE, KEY_FILE


def _detect_lan_ip() -> str | None:
    """Best-effort guess at the LAN/VPN IP others would use to reach this box.

    Opens a UDP socket to a non-routable address; the kernel then picks the
    interface it would use, and getsockname() reveals the local IP without
    actually sending a packet.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        ip = sock.getsockname()[0]
        return ip if ip and not ip.startswith("127.") else None
    except OSError:
        return None
    finally:
        sock.close()


def _detect_lan_hostnames() -> list[str]:
    """Return stable hostnames worth putting in the dev cert SAN list."""
    names = [socket.gethostname(), socket.getfqdn()]
    result: list[str] = []
    seen = {"", "localhost", "localhost.localdomain"}
    for name in names:
        name = name.strip()
        if name and name not in seen:
            result.append(name)
            seen.add(name)
    return result


def _has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is available (Windows/Linux)."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def _cuda_visible_devices() -> tuple[str, ...] | None:
    """Return the explicit CUDA device allow-list, or None for all devices."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.lower() == "all":
        return None
    if value.lower() in {"none", "void"} or value == "-1":
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _detect_local_gpus() -> list[LocalGpu]:
    """Detect local NVIDIA GPUs, respecting CUDA_VISIBLE_DEVICES when set."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []

    gpus: list[LocalGpu] = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 3:
            continue
        index = row[0].strip()
        uuid = row[1].strip()
        name = row[2].strip()
        gpus.append(LocalGpu(index=index, uuid=uuid, name=name, selector=index))

    visible = _cuda_visible_devices()
    if visible is None:
        return gpus
    if not visible:
        return []

    by_index = {gpu.index: gpu for gpu in gpus}
    by_uuid = {gpu.uuid: gpu for gpu in gpus if gpu.uuid}
    selected: list[LocalGpu] = []
    for ordinal, token in enumerate(visible):
        gpu = by_index.get(token) or by_uuid.get(token)
        if gpu is None:
            gpu = LocalGpu(
                index=str(ordinal),
                uuid=token if token.startswith("GPU-") else "",
                name=f"CUDA device {token}",
                selector=token,
            )
        else:
            gpu = LocalGpu(
                index=gpu.index,
                uuid=gpu.uuid,
                name=gpu.name,
                selector=token,
            )
        selected.append(gpu)
    return selected


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip().rstrip("/") for part in value.split(",") if part.strip())


def _configured_worker_urls(env: dict[str, str]) -> list[str]:
    return list(_parse_csv(env.get("BANGERS_WORKERS", "")))


def _merge_worker_urls(local_worker_specs: list[LocalWorkerSpec], configured_urls: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for url in [*(spec.url for spec in local_worker_specs), *configured_urls]:
        normalized = url.rstrip("/")
        if normalized and normalized not in seen:
            merged.append(normalized)
            seen.add(normalized)
    return merged


def _is_loopback_host(host: str) -> bool:
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _remote_worker_autostart_enabled(env: dict[str, str]) -> bool:
    value = env.get("BANGERS_REMOTE_WORKER_AUTOSTART", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _remote_worker_root(env: dict[str, str]) -> str:
    return env.get("BANGERS_REMOTE_PROJECT_ROOT", str(ROOT)).strip() or str(ROOT)


def _remote_worker_sync_enabled(env: dict[str, str]) -> bool:
    value = env.get("BANGERS_REMOTE_WORKER_SYNC", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _remote_worker_specs(env: dict[str, str], configured_urls: list[str]) -> list[RemoteWorkerSpec]:
    if not _remote_worker_autostart_enabled(env):
        return []

    root = _remote_worker_root(env)
    specs: list[RemoteWorkerSpec] = []
    for url in configured_urls:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host or _is_loopback_host(host):
            continue
        port = parsed.port or (443 if parsed.scheme == "https" else BACKEND_PORT)
        ssh_host = env.get(f"BANGERS_REMOTE_SSH_HOST_{host.replace('.', '_').replace('-', '_')}", host)
        specs.append(
            RemoteWorkerSpec(
                url=url.rstrip("/"),
                ssh_host=ssh_host,
                port=port,
                root=root,
            )
        )
    return specs


def _sync_remote_project(spec: RemoteWorkerSpec) -> None:
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync is required for remote worker source sync")
    target = f"{spec.ssh_host}:{spec.root.rstrip('/')}/"
    subprocess.run(
        [
            "rsync",
            "-az",
            "--exclude", ".git/",
            "--exclude", ".env",
            "--exclude", ".env.*",
            "--exclude", ".cache/",
            "--exclude", "runtime.log",
            "--exclude", "backend/.conda/",
            "--exclude", "backend/data/",
            "--exclude", "frontend/node_modules/",
            "--exclude", "frontend/.next/",
            "--exclude", "node_modules/",
            str(ROOT) + "/",
            target,
        ],
        cwd=ROOT,
        check=True,
    )


def _shell_export_lines(exports: dict[str, str]) -> str:
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in exports.items())


def _read_startup_model_settings(env: dict[str, str]) -> dict[str, str]:
    import sqlite3

    db_path = Path(env.get("BANGERS_DATA_DIR", str(DEFAULT_DATA_DIR))) / "conda-install-bangers.db"
    if not db_path.exists():
        return {}
    keys = ("dit_model", "lm_model", "lm_backend", "dj_model")
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                f"SELECT key, value FROM settings WHERE key IN ({','.join('?' for _ in keys)})",
                keys,
            ).fetchall()
    except sqlite3.Error:
        return {}
    return {str(key): str(value) for key, value in rows if value is not None}


def _remote_worker_command(
    spec: RemoteWorkerSpec,
    *,
    capabilities: str,
    token: str,
    timeout_seconds: str,
    startup_model_settings: dict[str, str],
) -> list[str]:
    root = spec.root.rstrip("/")
    backend_dir = f"{root}/backend"
    python_bin_dir = f"{backend_dir}/.conda/bin"
    entrypoint = f"{python_bin_dir}/conda-install-bangers"
    model_cache_dir = f"{root}/.cache/models"
    data_dir = f"{backend_dir}/data"
    env = {
        "PYTHONUNBUFFERED": "1",
        "BANGERS_DISTRIBUTED_ROLE": "worker",
        "BANGERS_WORKER_CAPABILITIES": capabilities,
        "BANGERS_WORKER_TOKEN": token,
        "BANGERS_WORKER_TIMEOUT_SECONDS": timeout_seconds,
        "BANGERS_HOST": "0.0.0.0",
        "BANGERS_PORT": str(spec.port),
        "BANGERS_DEVICE": "cuda",
        "BANGERS_DATA_DIR": data_dir,
        "BANGERS_MODEL_CACHE_DIR": model_cache_dir,
        "ACESTEP_PROJECT_ROOT": model_cache_dir,
        "HF_HOME": f"{model_cache_dir}/huggingface",
        "HF_HUB_CACHE": f"{model_cache_dir}/huggingface/hub",
        "CONDA_PREFIX": f"{backend_dir}/.conda",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "BANGERS_STARTUP_MODEL_SETTINGS_JSON": json.dumps(
            startup_model_settings,
            separators=(",", ":"),
        ),
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    export_lines = _shell_export_lines(env)

    remote_script = f"""
set -eu
cd {shlex.quote(backend_dir)}
export PATH={shlex.quote(python_bin_dir)}:"$PATH"
if [ ! -x {shlex.quote(entrypoint)} ]; then
  echo "Remote worker entrypoint missing: {entrypoint}" >&2
  exit 127
fi
{export_lines}
python - <<'PY'
import json
import os
import sqlite3

data_dir = os.environ["BANGERS_DATA_DIR"]
settings = json.loads(os.environ.get("BANGERS_STARTUP_MODEL_SETTINGS_JSON") or "{{}}")
if settings:
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "conda-install-bangers.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings ("
            "key TEXT PRIMARY KEY, "
            "value TEXT NOT NULL, "
            "updated_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        for key, value in settings.items():
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value, updated_at) "
                "VALUES (?, ?, datetime('now'))",
                (key, str(value)),
            )
        conn.commit()
PY
cleanup() {{
  if [ "${{worker_pid:-}}" ]; then
    kill "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" 2>/dev/null || true
  fi
}}
trap cleanup INT TERM HUP EXIT
{shlex.quote(entrypoint)} &
worker_pid=$!
wait "$worker_pid"
""".strip()
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=2",
        spec.ssh_host,
        remote_script,
    ]


def _remote_worker_cleanup_command(spec: RemoteWorkerSpec) -> list[str]:
    root = spec.root.rstrip("/")
    entrypoint = f"{root}/backend/.conda/bin/conda-install-bangers"
    pattern = re.escape(entrypoint).replace(
        "conda\\-install\\-bangers",
        "[c]onda\\-install\\-bangers",
    )
    remote_script = f"""
set +e
pkill -TERM -f {shlex.quote(pattern)} 2>/dev/null
sleep 1
pkill -KILL -f {shlex.quote(pattern)} 2>/dev/null
exit 0
""".strip()
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        spec.ssh_host,
        remote_script,
    ]


def _stop_remote_worker(spec: RemoteWorkerSpec) -> None:
    try:
        subprocess.run(
            _remote_worker_cleanup_command(spec),
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _process_cmdline(pid: int) -> list[str]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [
        part.decode("utf-8", errors="replace")
        for part in raw.split(b"\0")
        if part
    ]


def _process_cwd(pid: int) -> Path | None:
    try:
        return Path(os.readlink(Path("/proc") / str(pid) / "cwd")).resolve()
    except OSError:
        return None


def _path_is_relative_to(path: Path | None, parent: Path) -> bool:
    if path is None:
        return False
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _stale_local_dev_pids() -> list[int]:
    if sys.platform != "linux":
        return []

    current_pid = os.getpid()
    root = ROOT.resolve()
    frontend_dir = FRONTEND_DIR.resolve()
    backend_entry = str(_conda_env_executable("conda-install-bangers"))
    pids: list[int] = []

    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == current_pid:
            continue
        cmdline = _process_cmdline(pid)
        if not cmdline:
            continue
        joined = " ".join(cmdline)
        cwd = _process_cwd(pid)
        in_project = _path_is_relative_to(cwd, root)
        in_frontend = _path_is_relative_to(cwd, frontend_dir)

        is_launcher = (
            in_project
            and Path(cmdline[0]).name.startswith("python")
            and any(Path(arg).name == "start.py" for arg in cmdline[1:])
        )
        is_backend = backend_entry in joined or (in_project and "bangers.main" in joined)
        is_remote_ssh = (
            in_project
            and Path(cmdline[0]).name == "ssh"
            and "/backend/.conda/bin/conda-install-bangers" in joined
        )
        is_frontend = in_frontend and (
            "next dev" in joined
            or "next-server" in joined
            or (Path(cmdline[0]).name in {"node", "pnpm"} and "next" in joined)
        )

        if is_launcher or is_backend or is_remote_ssh or is_frontend:
            pids.append(pid)

    return sorted(set(pids))


def _signal_process_group_or_pid(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
        return
    except ProcessLookupError:
        return
    except OSError:
        pass
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        pass


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _stop_stale_local_dev_processes() -> None:
    pids = _stale_local_dev_pids()
    if not pids:
        return
    log(
        "Stopping stale local dev process(es): "
        + ", ".join(str(pid) for pid in pids),
        YELLOW,
    )
    for pid in pids:
        _signal_process_group_or_pid(pid, signal.SIGTERM)
    time.sleep(1)
    for pid in pids:
        if _pid_is_alive(pid):
            _signal_process_group_or_pid(pid, signal.SIGKILL)


def _explicit_distributed_role(env: dict[str, str]) -> str:
    return env.get("BANGERS_DISTRIBUTED_ROLE", "").strip().lower()


def _local_gpu_mode(env: dict[str, str]) -> str:
    return env.get("BANGERS_DEV_GPU_MODE", "workers").strip().lower()


def _local_worker_port_base(env: dict[str, str], backend_port: int) -> int:
    raw = env.get("BANGERS_DEV_WORKER_PORT_BASE", "")
    if raw:
        try:
            return int(raw)
        except ValueError:
            log(
                f"Ignoring invalid BANGERS_DEV_WORKER_PORT_BASE={raw!r}; using {backend_port + 1}.",
                YELLOW,
            )
    return int(env.get("BANGERS_WORKER_PORT_BASE", str(backend_port + 1)))


def _short_host_id() -> str:
    return socket.gethostname().split(".", 1)[0] or "local"


def _plan_local_gpu_workers(
    env: dict[str, str],
    gpus: list[LocalGpu],
    *,
    backend_port: int,
) -> list[LocalWorkerSpec]:
    mode = _local_gpu_mode(env)
    role = _explicit_distributed_role(env)
    disabled_modes = {"0", "false", "no", "off", "standalone", "single"}
    forced_modes = {"1", "true", "yes", "on", "workers", "local-workers", "all"}

    if mode in disabled_modes:
        return []
    if role and role != "coordinator":
        return []
    if not gpus:
        return []
    if mode == "auto" and len(gpus) < 2:
        return []

    port_base = _local_worker_port_base(env, backend_port)
    host_id = _short_host_id()
    specs: list[LocalWorkerSpec] = []
    for offset, gpu in enumerate(gpus):
        port = port_base + offset
        node_id = f"{host_id}-gpu{gpu.index or offset}"
        specs.append(
            LocalWorkerSpec(
                gpu=gpu,
                node_id=node_id,
                port=port,
                url=f"http://127.0.0.1:{port}",
            )
        )
    return specs


def _torch_has_cuda() -> bool:
    """Check if the installed torch build has CUDA support."""
    env_python = _conda_env_python()
    if not env_python.exists():
        return False
    try:
        result = subprocess.run(
            [str(env_python), "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=30,
        )
        return "True" in result.stdout
    except Exception:
        return False


def _install_cuda_torch() -> None:
    """Replace CPU-only torch with the CUDA build for NVIDIA GPUs."""
    log("NVIDIA GPU detected but torch has no CUDA support.", YELLOW)
    log("Installing PyTorch with CUDA (this may take a few minutes)...")
    _run(
        [str(_conda_env_python()), "-m", "pip", "install",
         "--prefer-binary",
         "--force-reinstall",
         "torch", "torchvision", "torchaudio",
         "--index-url", PYTORCH_CUDA_INDEX_URL],
        cwd=BACKEND_DIR, check=True,
    )
    log("PyTorch with CUDA installed successfully.")


def _deps_changed() -> bool:
    """Check if dependency inputs are newer than the last successful install."""
    backend_stamp = CONDA_ENV_DIR / ".deps-stamp"
    frontend_stamp = FRONTEND_DIR / "node_modules" / ".deps-stamp"
    backend_inputs = [
        BACKEND_ENV_FILE,
        BACKEND_PYPROJECT_FILE,
    ]
    frontend_inputs = [
        FRONTEND_DIR / "package.json",
        FRONTEND_DIR / "pnpm-lock.yaml",
    ]

    if not _conda_env_python().exists() or not backend_stamp.exists():
        return True

    for source in backend_inputs:
        if source.exists() and source.stat().st_mtime > backend_stamp.stat().st_mtime:
            return True

    if not frontend_stamp.exists():
        return True

    for source in frontend_inputs:
        if source.exists() and source.stat().st_mtime > frontend_stamp.stat().st_mtime:
            return True

    return False


def _touch_dep_stamps() -> None:
    """Record a successful install so we can detect future dependency changes."""
    backend_stamp = CONDA_ENV_DIR / ".deps-stamp"
    frontend_stamp = FRONTEND_DIR / "node_modules" / ".deps-stamp"
    backend_stamp.parent.mkdir(parents=True, exist_ok=True)
    frontend_stamp.parent.mkdir(parents=True, exist_ok=True)
    backend_stamp.touch()
    frontend_stamp.touch()


def _ensure_conda_env() -> None:
    """Create or update the project-local conda environment."""
    conda = _conda_executable()
    if conda is None:
        raise RuntimeError("'conda' not found. Run `mise install` or install Miniforge/Miniconda.")
    if not BACKEND_ENV_FILE.exists():
        raise RuntimeError(f"Missing backend conda environment file: {BACKEND_ENV_FILE}")

    if _conda_env_python().exists():
        log("Updating backend conda environment...")
        _run(
            [conda, "env", "update", "--prefix", str(CONDA_ENV_DIR),
             "--file", str(BACKEND_ENV_FILE), "--prune"],
            cwd=ROOT, check=True,
        )
    else:
        log("Creating backend conda environment...")
        _run(
            [conda, "env", "create", "--yes", "--prefix", str(CONDA_ENV_DIR),
             "--file", str(BACKEND_ENV_FILE)],
            cwd=ROOT, check=True,
        )


def install_dependencies() -> None:
    _ensure_conda_env()

    log("Installing backend dependencies...")
    _run(
        [str(_conda_env_python()), "-m", "pip", "install", "--prefer-binary",
         "--extra-index-url", PYTORCH_CUDA_INDEX_URL,
         "-e", ".[dev]"],
        cwd=BACKEND_DIR, check=True,
    )

    # On Windows/Linux with NVIDIA GPU, ensure torch has CUDA support
    if sys.platform != "darwin" and _has_nvidia_gpu() and not _torch_has_cuda():
        _install_cuda_torch()

    log("Installing frontend dependencies...")
    _run(["pnpm", "install", "--frozen-lockfile"], cwd=FRONTEND_DIR, check=True)

    _touch_dep_stamps()


def main() -> None:
    _install_runtime_logging()

    parser = argparse.ArgumentParser(description="Run conda install bangers locally")
    parser.add_argument("--install", action="store_true", help="Force reinstall all dependencies")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}  conda install bangers  {RESET}")
    print(f"  {CYAN}Local AI Music Generation{RESET}\n")

    _load_dotenv(ROOT / ".env")

    if not check_prerequisites():
        log("Missing prerequisites. See above.", RED)
        sys.exit(1)

    ensure_data_dirs()

    # Install deps if needed
    if args.install:
        log("Forcing dependency install...", YELLOW)
        install_dependencies()
    elif not _conda_env_python().exists() or not (FRONTEND_DIR / "node_modules").exists():
        install_dependencies()
    elif _deps_changed():
        log("Dependencies changed — updating...", YELLOW)
        install_dependencies()
    else:
        log("Dependencies up to date.", YELLOW)

    runtime_env = build_runtime_env()
    backend_port = int(runtime_env.get("BANGERS_PORT", str(BACKEND_PORT)))
    local_gpus = _detect_local_gpus()
    local_worker_specs = _plan_local_gpu_workers(
        runtime_env,
        local_gpus,
        backend_port=backend_port,
    )
    configured_worker_urls = _configured_worker_urls(runtime_env)
    worker_urls = _merge_worker_urls(local_worker_specs, configured_worker_urls)
    remote_worker_specs = _remote_worker_specs(runtime_env, configured_worker_urls)
    lan_ip = _detect_lan_ip()
    lan_hostnames = _detect_lan_hostnames()
    extra_sans = [*lan_hostnames, *([lan_ip] if lan_ip else [])]
    cert_path, key_path = ensure_dev_cert(extra_sans)

    local_url = f"https://localhost:{FRONTEND_PORT}"
    lan_url = f"https://{lan_ip}:{FRONTEND_PORT}" if lan_ip else None
    hostname_urls = [f"https://{name}:{FRONTEND_PORT}" for name in lan_hostnames]

    if local_worker_specs:
        log(
            f"Local GPU worker dev mode: coordinator on {backend_port}, "
            f"{len(local_worker_specs)} worker(s) on localhost.",
            YELLOW,
        )
        for spec in local_worker_specs:
            gpu_label = f"GPU {spec.gpu.index}: {spec.gpu.name}".strip()
            log(f"Worker {spec.node_id} on {spec.url} -> {gpu_label}", YELLOW)
    if configured_worker_urls:
        for url in configured_worker_urls:
            log(f"Remote worker configured: {url}", YELLOW)
    if remote_worker_specs:
        for spec in remote_worker_specs:
            log(f"Remote worker autostart via SSH: {spec.ssh_host} -> {spec.url}", YELLOW)

    log(f"Backend  HTTPS on {CYAN}https://localhost:{backend_port}{RESET}")
    log(f"Frontend HTTPS on {CYAN}{local_url}{RESET}")
    for url in hostname_urls:
        log(f"Frontend HTTPS on {CYAN}{url}{RESET} (hostname)")
    if lan_url:
        log(f"Frontend HTTPS on {CYAN}{lan_url}{RESET} (for other devices on the LAN)")
    log(
        f"TLS cert is self-signed — your browser will warn once, click through it.",
        YELLOW,
    )
    print()

    procs: list[tuple[str, subprocess.Popen]] = []

    def terminate_process(proc: subprocess.Popen, *, force: bool = False) -> None:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
            )
            return
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        except OSError:
            if force:
                proc.kill()
            else:
                proc.terminate()

    def shutdown(sig=None, frame=None):
        print()
        log("Shutting down...")
        for spec in remote_worker_specs:
            _stop_remote_worker(spec)
        for _name, p in procs:
            terminate_process(p)
        for _name, p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                terminate_process(p, force=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, shutdown)

    _stop_stale_local_dev_processes()

    worker_capabilities = runtime_env.get(
        "BANGERS_DEV_WORKER_CAPABILITIES",
        DEFAULT_LOCAL_WORKER_CAPABILITIES,
    )
    startup_model_settings = _read_startup_model_settings(runtime_env)
    sync_remote_workers = _remote_worker_sync_enabled(runtime_env)

    for spec in remote_worker_specs:
        if sync_remote_workers:
            log(f"Syncing source to remote worker {spec.ssh_host}...", YELLOW)
            _sync_remote_project(spec)
        _stop_remote_worker(spec)
        remote_worker = _popen(
            _remote_worker_command(
                spec,
                capabilities=worker_capabilities,
                token=runtime_env.get("BANGERS_WORKER_TOKEN", ""),
                timeout_seconds=runtime_env.get("BANGERS_WORKER_TIMEOUT_SECONDS", "900"),
                startup_model_settings=startup_model_settings,
            ),
            cwd=ROOT,
            env=runtime_env,
        )
        procs.append((f"Remote worker {spec.url}", remote_worker))

    # Start backend with TLS — uvicorn loads the same self-signed cert that
    # we feed to Next.js below, so the frontend's cross-origin fetch to the
    # API doesn't get blocked as mixed content.
    backend_env = {
        **_with_conda_env_on_path(runtime_env),
        "BANGERS_SSL_CERTFILE": str(cert_path),
        "BANGERS_SSL_KEYFILE": str(key_path),
    }
    if worker_urls:
        backend_env.update({
            "BANGERS_DISTRIBUTED_ROLE": "coordinator",
            "BANGERS_WORKERS": ",".join(worker_urls),
            "BANGERS_WORKER_CAPABILITIES": "",
            # Keep the coordinator out of CUDA memory. Worker processes own
            # inference devices and the coordinator talks to them over HTTP.
            "CUDA_VISIBLE_DEVICES": "-1",
        })
    backend_entry = _conda_env_executable("conda-install-bangers")
    backend_cmd = (
        [str(backend_entry)]
        if backend_entry.exists()
        else [str(_conda_env_python()), "-m", "bangers.main"]
    )
    backend = _popen(
        backend_cmd,
        cwd=BACKEND_DIR,
        env=backend_env,
    )
    procs.append(("Backend", backend))

    for spec in local_worker_specs:
        worker_env = {
            **_with_conda_env_on_path(runtime_env),
            "BANGERS_DISTRIBUTED_ROLE": "worker",
            "BANGERS_NODE_ID": spec.node_id,
            "BANGERS_WORKER_CAPABILITIES": worker_capabilities,
            "BANGERS_HOST": "127.0.0.1",
            "BANGERS_PORT": str(spec.port),
            "BANGERS_DEVICE": "cuda",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": spec.gpu.selector,
        }
        worker = _popen(
            backend_cmd,
            cwd=BACKEND_DIR,
            env=worker_env,
        )
        procs.append((f"Worker {spec.node_id}", worker))

    # Start frontend. We pass the pre-generated cert via `--experimental-https-*`
    # so Next.js skips its mkcert step entirely — that's important on headless
    # boxes where mkcert would block on a sudo password prompt trying to
    # install its CA into the system trust store.
    #
    # We call `pnpm exec next` rather than `pnpm dev` because pnpm forwards
    # the `--` separator itself as a literal arg to the `next dev` script,
    # which then misreads `--experimental-https` as a project directory.
    frontend_cmd = [
        "pnpm", "exec", "next", "dev",
        "--hostname", "0.0.0.0",
        "--experimental-https",
        "--experimental-https-cert", str(cert_path),
        "--experimental-https-key", str(key_path),
    ]
    frontend = _popen(
        frontend_cmd,
        cwd=FRONTEND_DIR,
        env={
            **runtime_env,
            "NEXT_TELEMETRY_DISABLED": "1",
            "NEXT_PUBLIC_BANGERS_BACKEND_PORT": str(backend_port),
            # SSR/RSC code paths in Next.js dial the backend with Node's
            # built-in fetch, which by default rejects our self-signed cert.
            # Disable the check inside the dev process only — never ship this
            # to production.
            "NODE_TLS_REJECT_UNAUTHORIZED": "0",
        },
    )
    procs.append(("Frontend", frontend))

    if not args.no_open:
        opener = threading.Thread(
            target=open_browser_when_ready,
            args=(local_url,),
            daemon=True,
        )
        opener.start()

    log("Press Ctrl+C to stop.\n")

    # Wait for either process to exit
    try:
        while True:
            for name, p in procs:
                ret = p.poll()
                if ret is not None:
                    log(f"{name} exited with code {ret}", RED if ret != 0 else YELLOW)
                    shutdown()
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
