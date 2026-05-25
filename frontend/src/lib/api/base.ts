const BACKEND_PORT = process.env.NEXT_PUBLIC_BANGERS_BACKEND_PORT || "8000";
const ENV_BASE_URL = process.env.NEXT_PUBLIC_BANGERS_BACKEND_URL || "";

function inferBrowserBaseUrl(): string {
  // The dev launcher (`start.py`) issues a single self-signed cert covering
  // localhost AND the LAN IP, then runs Next.js on :3000 and FastAPI on :8000
  // both serving HTTPS off that cert. We just dial the backend on the same
  // hostname/protocol as the page — no reverse proxy in the loop.
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:${BACKEND_PORT}`;
}

export function getBaseUrl(): string {
  if (typeof window !== "undefined") {
    const override = localStorage.getItem("conda-install-bangers-backend-url");
    if (override) return override;
    if (ENV_BASE_URL) return ENV_BASE_URL;
    return inferBrowserBaseUrl();
  }
  // Server-side (SSR/RSC) — backend serves HTTPS off our self-signed dev
  // cert. The launcher sets NODE_TLS_REJECT_UNAUTHORIZED=0 in the Next.js
  // process env so Node's fetch accepts it; never override that in prod.
  return ENV_BASE_URL || `https://localhost:${BACKEND_PORT}`;
}

export function getWsUrl(): string {
  return getBaseUrl().replace(/^http/, "ws");
}

export function getAudioUrl(path: string): string {
  const filename = path.split(/[/\\]/).pop() ?? "";
  return `${getBaseUrl()}/audio/${filename}`;
}

export async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const baseUrl = getBaseUrl();
  const url = `${baseUrl}/api${path}`;
  const headers: Record<string, string> = { ...(options.headers as Record<string, string>) };
  if (options.body) {
    headers["Content-Type"] = "application/json";
  }
  let res: Response;
  try {
    res = await fetch(url, { ...options, headers });
  } catch (err) {
    const reason = err instanceof Error ? err.message : "network request failed";
    throw new Error(
      `Unable to reach bangers backend at ${baseUrl}. Start the backend or update the backend URL in Settings. (${reason})`,
    );
  }
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(extractErrorMessage(res.status, body));
  }
  return res.json();
}

function extractErrorMessage(status: number, body: string): string {
  if (!body) return `API error ${status}`;
  try {
    const parsed = JSON.parse(body) as unknown;
    if (parsed && typeof parsed === "object" && "detail" in parsed) {
      const detail = (parsed as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (detail && typeof detail === "object" && "message" in detail) {
        const detailRecord = detail as { message: unknown; command?: unknown };
        const message = detailRecord.message;
        if (typeof message === "string") {
          const command = detailRecord.command;
          return typeof command === "string" && command
            ? `${message} Run: ${command}`
            : message;
        }
      }
    }
  } catch {
    // not JSON, fall through to raw body
  }
  return `API error ${status}: ${body}`;
}
