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

// ------------------ RAG QUERY ------------------
export const ragQuery = async (query: string) => {
  const res = await fetch(`${API_BASE}/rag/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  return res.json();
};

// ------------------ STREAMING ------------------
export const ragStream = async (
  query: string,
  onToken: (t: string) => void,
  onDone?: (fullContent?: string) => void,
  onError?: (e: any) => void,
  onResult?: (result: AskResult) => void
) => {
  try {
    const res = await fetch(`${API_BASE}/rag/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (res.ok === false) throw new Error(`Request failed (HTTP ${res.status})`);

    const reader = res.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error("No stream");

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
            if (json.type === "error") onError?.(json.message);
          } catch {
            onToken(match[1]);
          }
        }
      }
    }

    onDone?.();
  } catch (e) {
    onError?.(e);
  }
};

// ------------------ STATUS ------------------
export const getStatus = async () => {
  const res = await fetch(`${API_BASE}/status`);
  return res.json();
};