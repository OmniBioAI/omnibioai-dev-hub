import { describe, expect, it, vi } from "vitest";
import { getStatus, ragQuery, ragStream } from "./client";

describe("API client", () => {
  it("sends a JSON query and returns the decoded response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ answer: "ok" }), { status: 200 }));
    await expect(ragQuery("hello")).resolves.toEqual({ answer: "ok" });
    expect(fetch).toHaveBeenCalledWith("/rag/query", expect.objectContaining({
      method: "POST", body: JSON.stringify({ query: "hello" }),
    }));
  });

  it("returns status payloads and preserves HTTP error payloads for callers", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response(JSON.stringify({ index_vectors: 4 }), { status: 200 }));
    await expect(getStatus()).resolves.toEqual({ index_vectors: 4 });
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response(JSON.stringify({ detail: "unauthorized" }), { status: 401 }));
    await expect(ragQuery("private")).resolves.toEqual({ detail: "unauthorized" });
  });

  it("parses token, response, done, and malformed SSE events", async () => {
    const chunks = [
      "data: {\"type\":\"token\",\"content\":\"Hel\"}\n\n",
      "data: not-json\n\n",
      "data: {\"type\":\"response\",\"content\":\"Hello\"}\n\n",
      "data: {\"type\":\"done\"}\n\n",
    ];
    const reader = { read: vi.fn()
      .mockResolvedValueOnce({ value: new TextEncoder().encode(chunks[0] + chunks[1]), done: false })
      .mockResolvedValueOnce({ value: new TextEncoder().encode(chunks[2] + chunks[3]), done: false })
      .mockResolvedValueOnce({ value: undefined, done: true }) };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 200 }));
    vi.mocked(fetch).mockResolvedValueOnce({ body: { getReader: () => reader } } as Response);
    const tokens: string[] = []; const done: unknown[] = []; const errors: unknown[] = [];
    await ragStream("q", (token) => tokens.push(token), (value) => done.push(value), (error) => errors.push(error));
    expect(tokens).toEqual(["Hel", "not-json"]);
    expect(done).toEqual(["Hello", undefined, undefined]);
    expect(errors).toEqual([]);
  });

  it("reports network and missing-stream failures", async () => {
    const onError = vi.fn();
    vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new Error("offline"));
    await ragStream("q", vi.fn(), vi.fn(), onError);
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: "offline" }));
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response(null, { status: 200 }));
    await ragStream("q", vi.fn(), vi.fn(), onError);
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: "No stream" }));
  });
});
