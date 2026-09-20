import type { AskCitation } from "../api/client";

const stateClass = (kind: "content" | "verification", value: string | null) => {
  if (!value) return "info";
  if (kind === "content") return value === "CURRENT" ? "success" : "warn"; // TARGET / HISTORICAL are not "current"
  return value === "VERIFIED" ? "success" : "warn"; // CONFIGURED / UNKNOWN are not "verified"
};

/** Numbered sources for a grounded answer. The [n] matches the [n] in the answer text. */
export default function Citations({ citations }: { citations: AskCitation[] }) {
  if (!citations || citations.length === 0) return null;
  return (
    <ol className="citation-list" aria-label="Sources">
      {citations.map((c) => (
        <li
          key={c.chunk_id ?? c.index}
          className="citation-item"
          data-document-id={c.document_id ?? undefined}
          data-chunk-id={c.chunk_id ?? undefined}
          data-revision={c.source_revision ?? undefined}
          title={[
            `revision ${c.source_revision ?? "unknown"}`,
            `document ${c.document_id ?? "unknown"}`,
            `chunk ${c.chunk_id ?? "unknown"}`,
          ].join("\n")}
        >
          <span className="citation-index">[{c.index}]</span>{" "}
          <span className="citation-path">{c.repository}/{c.relative_path}</span>{" "}
          {c.content_state && <span className={`badge ${stateClass("content", c.content_state)}`}>{c.content_state.toLowerCase()}</span>}{" "}
          {c.verification_state && (
            <span className={`badge ${stateClass("verification", c.verification_state)}`}>
              {c.verification_state === "VERIFIED" ? "verified" : `not verified · ${c.verification_state.toLowerCase()}`}
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}
