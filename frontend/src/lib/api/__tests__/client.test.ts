import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

// Mock localStorage
const mockLocalStorage: Record<string, string> = {};
vi.stubGlobal("localStorage", {
  getItem: (key: string) => mockLocalStorage[key] ?? null,
  setItem: (key: string, val: string) => {
    mockLocalStorage[key] = val;
  },
  removeItem: (key: string) => {
    delete mockLocalStorage[key];
  },
});

const mockSessionStorage: Record<string, string> = {};
const sessionStorageMock = {
  getItem: (key: string) => mockSessionStorage[key] ?? null,
  setItem: (key: string, val: string) => {
    mockSessionStorage[key] = val;
  },
  removeItem: (key: string) => {
    delete mockSessionStorage[key];
  },
};
vi.stubGlobal("sessionStorage", sessionStorageMock);
Object.defineProperty(window, "sessionStorage", {
  value: sessionStorageMock,
  configurable: true,
});

describe("api client", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    for (const key of Object.keys(mockSessionStorage)) delete mockSessionStorage[key];
  });

  it("request returns parsed JSON on 200", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "ok" }),
    });

    const { fetchHealth } = await import("../client");
    const result = await fetchHealth();
    expect(result).toEqual({ status: "ok" });
  });

  it("request throws on non-200", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "Internal Server Error",
    });

    const { fetchHealth } = await import("../client");
    await expect(fetchHealth()).rejects.toThrow("API error 500");
  });

  it("request throws a backend reachability error when fetch fails", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("Failed to fetch"));

    const { fetchHealth } = await import("../client");
    await expect(fetchHealth()).rejects.toThrow(
      "Unable to reach bangers backend at http://localhost:8000",
    );
  });

  it("submitGeneration sends correct POST body", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ job_id: "test-123", status: "queued" }),
    });

    const { submitGeneration } = await import("../client");
    await submitGeneration({
      task_type: "text2music",
      caption: "test song",
      inference_steps: 8,
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/generate",
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining('"caption":"test song"'),
      }),
    );
    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string);
    expect(body.client_id).toEqual(expect.any(String));
    expect(body.client_id.length).toBeGreaterThan(0);
  });

  it("switchChatLlmModel loads model through switch endpoint", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ message: "Chat LLM loaded: Qwen3-1.7B" }),
    });

    const { switchChatLlmModel } = await import("../client");
    await switchChatLlmModel("Qwen3-1.7B");

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/models/switch-chat-llm",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ model_name: "Qwen3-1.7B" }),
      }),
    );
  });

  it("fetchSongs builds correct query string", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    });

    const { fetchSongs } = await import("../client");
    await fetchSongs({ search: "rock", sort: "title", limit: 10 });

    const calledUrl = mockFetch.mock.calls[0][0] as string;
    expect(calledUrl).toContain("search=rock");
    expect(calledUrl).toContain("sort=title");
    expect(calledUrl).toContain("limit=10");
  });

  it("fetchSongs includes filter params in query string", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ items: [], total: 0 }),
    });

    const { fetchSongs } = await import("../client");
    await fetchSongs({
      favorite: true,
      vocal_language: "ja",
      file_format: "flac",
      instrumental: true,
      timesignature: "4/4",
      offset: 10,
    });

    const calledUrl = mockFetch.mock.calls[0][0] as string;
    expect(calledUrl).toContain("favorite=true");
    expect(calledUrl).toContain("vocal_language=ja");
    expect(calledUrl).toContain("file_format=flac");
    expect(calledUrl).toContain("instrumental=true");
    expect(calledUrl).toContain("timesignature=4%2F4");
    expect(calledUrl).toContain("offset=10");
  });

  it("bulkDeleteSongs sends POST with song_ids", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ deleted: 2 }),
    });

    const { bulkDeleteSongs } = await import("../client");
    const result = await bulkDeleteSongs(["id-1", "id-2"]);

    expect(result).toEqual({ deleted: 2 });
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/songs/bulk-delete",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ song_ids: ["id-1", "id-2"] }),
      }),
    );
  });

  it("bulkUpdateSongs sends PATCH with song_ids and updates", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ updated: 3 }),
    });

    const { bulkUpdateSongs } = await import("../client");
    const result = await bulkUpdateSongs(
      ["id-1", "id-2", "id-3"],
      { is_favorite: true },
    );

    expect(result).toEqual({ updated: 3 });
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/songs/bulk",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({
          song_ids: ["id-1", "id-2", "id-3"],
          updates: { is_favorite: true },
        }),
      }),
    );
  });

  it("getSongAudioUrl returns correct URL", async () => {
    const { getSongAudioUrl } = await import("../client");
    const url = getSongAudioUrl("abc-123");
    expect(url).toBe("http://localhost:8000/api/songs/abc-123/audio");
  });

  it("fetchSongVariations calls correct endpoint", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        song: {},
        ancestors: [],
        children: [],
      }),
    });

    const { fetchSongVariations } = await import("../client");
    await fetchSongVariations("song-42");

    const calledUrl = mockFetch.mock.calls[0][0] as string;
    expect(calledUrl).toBe("http://localhost:8000/api/songs/song-42/variations");
  });
});
