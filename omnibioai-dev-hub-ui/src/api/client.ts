const API_BASE = "";

// ------------------ ASK OMNIBIOAI ANSWER CONTRACT (ask.v1) ------------------
export interface AskCitation {
  index: number;
  repository: string | null;
  relative_path: string | null;
  source_revision: string | null;
  document_id: string | null;
  chunk_id: string | null;
  content_state: string | null;
  verification_state: string | null;
  title?: string | null;
  relevance?: number | null;
}

export interface AskResult {
  content: string;
  grounded: boolean;
  answer_status: string;
  citations: AskCitation[];
  context_used: number;
  llm_invoked?: boolean;
}

// ------------------ ERRORS ------------------
// One error type for every Ask request, so the UI never shows a raw browser or
// exception string ("TypeError: Failed to fetch", "Error: Error: ...").
export type AskErrorKind = "network" | "auth" | "invalid" | "busy" | "server";

export class AskError extends Error {
  readonly kind: AskErrorKind;
  readonly status?: number;
  readonly code?: string;

  constructor(kind: AskErrorKind, status?: number, code?: string) {
    super(ASK_ERROR_MESSAGES[kind]);
    this.name = "AskError";
    this.kind = kind;
    this.status = status;
    this.code = code;
  }
}

export const ASK_ERROR_MESSAGES: Record<AskErrorKind, string> = {
  network: "Couldn't reach Ask OmniBioAI. Check your connection and try again.",
  auth: "You're not signed in, or your session has expired. Sign in again and retry.",
  invalid: "That question couldn't be processed. Try rewording it.",
  busy: "Ask OmniBioAI is busy right now. Please try again in a moment.",
  server: "Ask OmniBioAI hit a problem answering that. Please try again in a moment.",
};

export const errorFromStatus = (status: number): AskError => {
  if (status === 401 || status === 403) return new AskError("auth", status);
  if (status === 400 || status === 422) return new AskError("invalid", status);
  if (status === 429) return new AskError("busy", status);
  return new AskError("server", status);
};

/** Normalise anything thrown/reported into an AskError. A rejected fetch() is a TypeError => network. */
export const toAskError = (e: unknown): AskError => {
  if (e instanceof AskError) return e;
  if (e instanceof TypeError || (e instanceof Error && e.name === "AbortError")) return new AskError("network");
  return new AskError("server");
};

/**
 * The concise message to show the user. Diagnostics (kind, HTTP status, server error code -- never message
 * text, stack traces or response bodies) go to the developer console only.
 */
export const describeAskError = (e: unknown): string => {
  const err = toAskError(e);
  console.warn("[ask-omnibioai] request failed", { kind: err.kind, status: err.status, code: err.code });
  return err.message;
};

// ------------------ RAG QUERY ------------------
export const ragQuery = async (query: string) => {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/rag/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
  } catch (e) {
    throw toAskError(e);
  }
  if (res.ok === false) throw errorFromStatus(res.status);

  return res.json();
};

// ------------------ STREAMING ------------------
export const ragStream = async (
  query: string,
  onToken: (t: string) => void,
  onDone?: (fullContent?: string) => void,
  onError?: (e: unknown) => void,
  onResult?: (result: AskResult) => void
) => {
  try {
    const res = await fetch(`${API_BASE}/rag/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (res.ok === false) throw errorFromStatus(res.status);

    const reader = res.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new AskError("server");

    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const p of parts) {
        const match = p.match(/data:\s*(.*)/);
        if (match?.[1]) {
          try {
            const json = JSON.parse(match[1]);
            if (json.type === "token" && json.content) onToken(json.content);
            if (json.type === "response") {
              onResult?.(json as AskResult);
              if (json.content) onDone?.(json.content);
            }
            if (json.type === "done") onDone?.();
            // The server sends a stable code and a generic message; never forward its text to the user.
            if (json.type === "error") onError?.(new AskError("server", undefined, typeof json.code === "string" ? json.code : undefined));
          } catch {
            onToken(match[1]);
          }
        }
      }
    }

    onDone?.();
  } catch (e) {
    onError?.(toAskError(e));
  }
};

// ------------------ STATUS ------------------
export const getStatus = async () => {
  const res = await fetch(`${API_BASE}/status`);
  return res.json();
};
