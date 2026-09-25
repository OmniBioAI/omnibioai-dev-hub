import ReactMarkdown from "react-markdown";
import type { AskCitation } from "../api/client";
import { isExternalHref, resolveDocLink } from "../lib/docLinks";

/**
 * Markdown for a grounded answer. Links are resolved intentionally (see resolveDocLink): documentation-relative
 * paths go to the canonical docs site, http(s) URLs are kept as written, and anything unsafe or ambiguous is
 * shown as plain text instead of a link that would resolve against this host and land on a nonexistent page.
 */
export default function AnswerMarkdown({ text, citations = [] }: { text: string; citations?: AskCitation[] }) {
  return (
    <ReactMarkdown
      disallowedElements={["img"]}
      components={{
        a: ({ href, children }) => {
          const resolved = resolveDocLink(href, citations);
          if (!resolved) return <span className="unlinked-doc">{children}</span>;
          return (
            <a href={resolved} target="_blank" rel="noopener noreferrer" data-external={isExternalHref(resolved) ? "true" : undefined}>
              {children}
            </a>
          );
        },
      }}
    >
      {text}
    </ReactMarkdown>
  );
}
