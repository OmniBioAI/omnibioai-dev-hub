import { useState, useEffect } from "react";
import { ragQuery, getStatus, describeAskError } from "../api/client";
import Citations from "../components/Citations";
import { displaySource } from "../lib/docLinks";

export default function SearchPage() {
  const [q, setQ] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [indexVectors, setIndexVectors] = useState<number | null>(null);

  useEffect(() => {
    getStatus()
      .then((s) => setIndexVectors(s.index_vectors))
      .catch(() => {});
  }, []);

  const search = async () => {
    if (!q.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await ragQuery(q);
      setResult(res);
    } catch (e) {
      setResult({ error: describeAskError(e) });
    } finally {
      setLoading(false);
    }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") search();
  };

  const contexts: any[] = result?.context || [];

  return (
    <>
      <div className="page-header">
        <div className="page-title">Vector Search</div>
        <div className="page-sub">Semantic search across {indexVectors != null ? indexVectors.toLocaleString() : "…"} embeddings · FAISS IndexFlatIP</div>
      </div>

      <div className="search-bar-row">
        <input
          className="search-input"
          placeholder="Search embeddings... e.g. 'CRISPR delivery mechanisms'"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={onKey}
        />
        <button className="btn-primary" onClick={search} disabled={loading}>
          {loading ? <><span className="spinner" />Searching</> : "Search"}
        </button>
      </div>

      {result && !result.error && (
        <div className="surface" style={{ marginBottom: 12 }}>
          <div style={{ display: "flex", gap: 16, fontSize: 12, fontFamily: "var(--mono)", color: "var(--text-secondary)" }}>
            <span>Query: <span style={{ color: "var(--teal)" }}>{result.query}</span></span>
            {result.context_used && <span>Context chunks: <span style={{ color: "var(--text-primary)" }}>{result.context_used}</span></span>}
            {result.version && <span>Engine: <span style={{ color: "var(--text-primary)" }}>{result.version}</span></span>}
          </div>
        </div>
      )}

      {result?.answer && result.grounded === false && (
        <div className="surface no-answer-card" role="status" data-answer-status={result.answer_status} style={{ marginBottom: 12 }}>
          <div className="section-header" style={{ marginBottom: 8 }}>
            <div className="section-title">No trusted documentation answer</div>
          </div>
          <div style={{ fontSize: 13, color: "var(--text-primary)", lineHeight: 1.7 }}>{result.answer}</div>
        </div>
      )}

      {result?.answer && result.grounded !== false && (
        <div className="surface" style={{ marginBottom: 12 }}>
          <div className="section-header" style={{ marginBottom: 8 }}>
            <div className="section-title">{result.grounded ? "Answer (grounded in documentation)" : "Generated Answer"}</div>
          </div>
          <div style={{ fontSize: 13, color: "var(--text-primary)", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
            {result.answer}
          </div>
          {result.grounded && <Citations citations={result.citations} />}
        </div>
      )}

      {contexts.length > 0 && (
        <div>
          <div className="section-header">
            <div className="section-title">Retrieved Chunks ({contexts.length})</div>
          </div>
          {contexts.map((ctx: any, i: number) => (
            <div key={i} className="result-card">
              <div className="result-source">
                {displaySource(ctx.source || ctx.file, ctx.repository ?? ctx.citation?.repository) || `chunk_${i + 1}`}
              </div>
              <div className="result-text">
                {typeof ctx === "string" ? ctx : ctx.text || ctx.content || ""}
              </div>
            </div>
          ))}
        </div>
      )}

      {result?.error && (
        <div className="result-card" style={{ borderLeftColor: "var(--red)" }}>
          <div className="result-source" style={{ color: "var(--red)" }} role="alert">Search unavailable</div>
          <div className="result-text">{result.error}</div>
        </div>
      )}

      {!result && !loading && (
        <div className="surface" style={{ textAlign: "center", padding: "40px 20px" }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>⌕</div>
          <div style={{ color: "var(--text-secondary)", fontSize: 14 }}>
            Enter a query to search across your indexed documents
          </div>
          <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 6, fontFamily: "var(--mono)" }}>
            768-dimensional FAISS vector search · nomic-embed-text
          </div>
        </div>
      )}
    </>
  );
}