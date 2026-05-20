const CLIENT_ID_KEY = "conda-install-bangers-client-id";

function createClientId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `client-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export function getClientId(): string {
  if (typeof window === "undefined") return "";
  const existing = window.sessionStorage.getItem(CLIENT_ID_KEY);
  if (existing) return existing;
  const next = createClientId();
  window.sessionStorage.setItem(CLIENT_ID_KEY, next);
  return next;
}
