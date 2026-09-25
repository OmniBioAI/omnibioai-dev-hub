import type { AskCitation } from "../api/client";
import { groupCitationsByDocument } from "../lib/docLinks";

const stateClass = (kind: "content" | "verification", value: string | null) => {
  if (!value) return "info";
  if (kind === "content") return value === "CURRENT" ? "success" : "warn"; // TARGET / HISTORICAL are not "current"
  return value === "VERIFIED" ? "success" : "warn"; // CONFIGURED / UNKNOWN are not "verified"
};

/**
 * Sources for a grounded answer, one row per DOCUMENT. Several cited excerpts of the same file are listed on
 * one row ([1] [4]); different documents are always separate rows, so a multi-document answer is never shown as
 * single-sourced. The full per-chunk provenance stays in the API response; the browser only renders what a
 * reader can act on. Revision SHAs and internal ids of non-public repositories are never rendered (not even in
 * tooltips or data attributes).
 */
export default function Citations({ citations }: { citations: AskCitation[] }) {
  if (!citations || citations.length === 0) return null;
  const groups = groupCitationsByDocument(citations);
  return (
    <ol className="citation-list" aria-label="Sources">
      {groups.map((g) => {
        const excerpts = g.indices.length;
        const tip = [`${g.repository}/${g.relative_path}`, g.publicRevision ? `revision ${g.publicRevision}` : null,
          `${excerpts} cited excerpt${excerpts === 1 ? "" : "s"}`].filter(Boolean).join("\n");
        return (
          <li key={g.key} className="citation-item" data-excerpts={g.indices.join(",")} title={tip}>
            <span className="citation-index">{g.indices.map((n) => `[${n}]`).join(" ")}</span>{" "}
            <span className="citation-path">{g.repository}/{g.relative_path}</span>{" "}
            {g.content_state && <span className={`badge ${stateClass("content", g.content_state)}`}>{g.content_state.toLowerCase()}</span>}{" "}
            {g.verification_state && (
              <span className={`badge ${stateClass("verification", g.verification_state)}`}>
                {g.verification_state === "VERIFIED" ? "verified" : `not verified · ${g.verification_state.toLowerCase()}`}
              </span>
            )}
            {excerpts > 1 && <span className="citation-count"> · {excerpts} excerpts</span>}
          </li>
        );
      })}
    </ol>
  );
}
