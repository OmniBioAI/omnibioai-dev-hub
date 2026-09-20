import { describe, expect, it, vi } from "vitest";
import { AskError, ASK_ERROR_MESSAGES, describeAskError, getStatus, ragQuery, ragStream } from "./client";

describe("API client", () => {
  it("sends a JSON query and returns the decoded response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ answer: "ok" }), { status: 200 }));
    await expect(ragQuery("hello")).resolves.toEqual({ answer: "ok" });
    expect(fetch).toHaveBeenCalledWith("/rag/query", expect.objectContaining({
      method: "POST", body: JSON.stringify({ query: "hello" }),
    }));
  });

  it("returns status payloads", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response(JSON.stringify({ index_vectors: 4 }), { status: 200 }));
    await expect(getStatus()).resolves.toEqual({ index_vectors: 4 });
  });

  it("turns an HTTP error from /rag/query into a typed error with a concise message, not the response body", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response(JSON.stringify({ detail: { error: "Traceback ... secret" } }), { status: 500 }));
    const err = await ragQuery("q").catch((e) => e);
    expect(err).toBeInstanceOf(AskError);
    expect(err.kind).toBe("server");
    expect(err.message).toBe(ASK_ERROR_MESSAGES.server);
    expect(err.message).not.toMatch(/HTTP|Traceback|secret|Error:/);
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

  it("reports a browser network failure as a concise network error, never 'TypeError: Failed to fetch'", async () => {
    const onError = vi.fn();
    vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await ragStream("q", vi.fn(), vi.fn(), onError);
    const err = onError.mock.calls[0][0];
    expect(err).toBeInstanceOf(AskError);
    expect(err.kind).toBe("network");
    expect(err.message).toBe(ASK_ERROR_MESSAGES.network);
    expect(err.message).not.toMatch(/TypeError|Failed to fetch/);
  });

  it("reports a missing stream body as a server error with a fixed message", async () => {
    const onError = vi.fn();
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response(null, { status: 200 }));
    await ragStream("q", vi.fn(), vi.fn(), onError);
    expect(onError.mock.calls[0][0]).toMatchObject({ kind: "server", message: ASK_ERROR_MESSAGES.server });
  });

  it("maps an in-stream error event to a fixed message and never forwards server text", async () => {
    const reader = { read: vi.fn()
      .mockResolvedValueOnce({ value: new TextEncoder().encode('data: {"type":"error","code":"internal_error","message":"boom /srv/app.py line 4"}\n\n'), done: false })
      .mockResolvedValueOnce({ value: undefined, done: true }) };
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({ ok: true, body: { getReader: () => reader } } as unknown as Response);
    const onError = vi.fn();
    await ragStream("q", vi.fn(), vi.fn(), onError);
    const err = onError.mock.calls[0][0];
    expect(err).toMatchObject({ kind: "server", code: "internal_error", message: ASK_ERROR_MESSAGES.server });
    expect(err.message).not.toContain("app.py");
  });

  it("passes the verified response event, with citations, to onResult and never surfaces tokens for it", async () => {
    const result = { type: "response", content: "Answer [1].", grounded: true, answer_status: "GROUNDED", citations: [{ index: 1 }], context_used: 1 };
    const reader = { read: vi.fn()
      .mockResolvedValueOnce({ value: new TextEncoder().encode(`data: ${JSON.stringify({ type: "status", stage: "retrieved", context_used: 1 })}\n\ndata: ${JSON.stringify(result)}\n\ndata: {"type":"done"}\n\n`), done: false })
      .mockResolvedValueOnce({ value: undefined, done: true }) };
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({ ok: true, body: { getReader: () => reader } } as unknown as Response);
    const tokens: string[] = []; const results: unknown[] = [];
    await ragStream("q", (t) => tokens.push(t), undefined, undefined, (r) => results.push(r));
    expect(tokens).toEqual([]);
    expect(results).toEqual([result]);
  });

  it.each([
    [401, "auth"], [403, "auth"], [400, "invalid"], [422, "invalid"], [429, "busy"], [500, "server"], [502, "server"], [503, "server"],
  ])("maps HTTP %i to a concise '%s' message without the status code or the words 'Request failed'", async (status, kind) => {
    const onError = vi.fn();
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response(JSON.stringify({ detail: "x" }), { status }));
    await ragStream("q", vi.fn(), vi.fn(), onError);
    const err = onError.mock.calls[0][0];
    expect(err.kind).toBe(kind);
    expect(err.message).toBe(ASK_ERROR_MESSAGES[kind as keyof typeof ASK_ERROR_MESSAGES]);
    expect(err.message).not.toMatch(/HTTP|Request failed|Error: Error|\d{3}/);
  });
});

describe("describeAskError", () => {
  it("shows a concise message and keeps diagnostics in the developer console only", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const text = describeAskError(new AskError("server", 500, "internal_error"));
    expect(text).toBe(ASK_ERROR_MESSAGES.server);
    expect(warn).toHaveBeenCalledWith("[ask-omnibioai] request failed", { kind: "server", status: 500, code: "internal_error" });
    warn.mockRestore();
  });

  it("never echoes an arbitrary error's text or stack", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const e = new Error("db password=hunter2"); e.stack = "at /srv/secret.ts:1";
    const text = describeAskError(e);
    expect(text).toBe(ASK_ERROR_MESSAGES.server);
    expect(JSON.stringify(warn.mock.calls)).not.toMatch(/hunter2|secret\.ts/);
    expect(describeAskError("just a string")).toBe(ASK_ERROR_MESSAGES.server);
    expect(describeAskError(new TypeError("Failed to fetch"))).toBe(ASK_ERROR_MESSAGES.network);
    warn.mockRestore();
  });
});

