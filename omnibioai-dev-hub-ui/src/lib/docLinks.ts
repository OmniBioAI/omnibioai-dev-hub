import type { AskCitation } from "../api/client";

/** Canonical home of the published documentation. Docusaurus: routeBasePath "/", trailing slashes. */
export const DOCS_ORIGIN = "https://docs.omnibioai.org";

/** The repository the published docs are built from, and the folder its pages live in. */
const DOCS_REPOSITORY = "omnibioai-docs";
const DOCS_ROOT = "site/docs/";

/**
 * Repositories that are public on GitHub. Anything not listed is treated as private, so its revision SHAs
 * and internal identifiers are never rendered in the browser (default-deny). Backend/API provenance is
 * unaffected: the full citation stays in the API response.
 */
export const PUBLIC_REPOSITORIES: ReadonlySet<string> = new Set([
  "omnibioai-dev-hub",
  "omnibioai-studio",
  "omnibioai-model-registry",
  "omnibioai-tool-runtime",
]);

export const isPublicRepository = (repo: string | null | undefined): boolean => !!repo && PUBLIC_REPOSITORIES.has(repo);

const isDocsPage = (c: AskCitation): boolean =>
  c.repository === DOCS_REPOSITORY && !!c.relative_path && c.relative_path.startsWith(DOCS_ROOT);

/** Docs-site URL for a docs source file path such as "site/docs/admin/deployment.md", or null. */
export const docsUrlForPath = (filePath: string, suffix = ""): string | null => {
  if (!filePath.startsWith(DOCS_ROOT)) return null;
  const m = filePath.slice(DOCS_ROOT.length).match(/^(.*?)(?:\.mdx?)?$/);
  let route = m ? m[1] : "";
  if (!/\.mdx?$/.test(filePath)) return null; // only pages map to routes (not images, csv, ...)
  if (route === "index") route = "";
  else if (route.endsWith("/index")) route = route.slice(0, -"index".length);
  if (route && !route.endsWith("/")) route += "/";
  return `${DOCS_ORIGIN}/${route}${suffix}`;
};

/** Resolve a relative markdown href against a docs FILE path (not a URL: "./x.md" is relative to the file). */
const resolveAgainstFile = (filePath: string, href: string): string | null => {
  const m = href.match(/^([^?#]*)([?#].*)?$/);
  if (!m) return null;
  const [, rawPath, suffix = ""] = m;
  const siteAbsolute = rawPath.startsWith("/"); // "/admin/deployment": already a docs-site route, not a file path
  let segments: string[];
  if (rawPath === "") segments = filePath.split("/"); // "#section" -> same page
  else if (siteAbsolute) segments = rawPath.split("/");
  else segments = [...filePath.split("/").slice(0, -1), ...rawPath.split("/")];

  const out: string[] = [];
  for (const raw of segments) {
    let seg: string;
    try {
      seg = decodeURIComponent(raw);
    } catch {
      return null; // malformed percent-encoding
    }
    if (seg.includes("/") || seg.includes("\\")) return null; // encoded separator smuggling
    if (seg === "" || seg === ".") continue;
    if (seg === "..") {
      if (out.length === 0) return null;
      out.pop();
      continue;
    }
    out.push(seg);
  }
  const target = out.join("/");
  if (siteAbsolute) {
    if (/\.mdx?$/.test(target)) return docsUrlForPath(DOCS_ROOT + target, suffix);
    if (target === "") return `${DOCS_ORIGIN}/${suffix}`;
    const isAsset = /\.[A-Za-z0-9]+$/.test(out[out.length - 1]);
    return `${DOCS_ORIGIN}/${target}${isAsset ? "" : "/"}${suffix}`;
  }
  if (!target.startsWith(DOCS_ROOT)) return null; // "../../README.md" escapes the docs folder: do not guess
  return docsUrlForPath(target, suffix);
};

/**
 * Turn a link found in an answer into a safe destination, or null (render as plain text).
 *
 *  - http(s) URLs (docs.omnibioai.org or any external site) are kept exactly as written -- never rewritten.
 *  - Every other scheme (javascript:, data:, vbscript:, mailto:, file: ...) is dropped.
 *  - Protocol-relative ("//host"), backslash forms and control characters are dropped: they can point anywhere.
 *  - Relative paths are docs-repo file paths. They are resolved against the cited docs page(s) and mapped to the
 *    canonical docs site. If the citations disagree about where the link points, or it escapes site/docs,
 *    it is not linked (no guessing).
 *  - Nothing is ever routed through a redirect endpoint, and every constructed URL is re-verified to sit on
 *    DOCS_ORIGIN, so a crafted href cannot turn this into an open redirect.
 */
export const resolveDocLink = (href: string | null | undefined, citations: AskCitation[]): string | null => {
  if (!href) return null;
  const raw = href.trim();
  if (!raw || raw.length > 2048 || /[\u0000-\u001f\u007f\\]/.test(raw)) return null;

  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(raw)) {
    try {
      const u = new URL(raw);
      if (u.protocol !== "https:" && u.protocol !== "http:") return null;
      if (u.username || u.password) return null;
      return u.href;
    } catch {
      return null;
    }
  }
  if (raw.startsWith("//")) return null;

  const candidates = new Set<string>();
  for (const c of citations) {
    if (!isDocsPage(c)) continue;
    const url = resolveAgainstFile(c.relative_path as string, raw);
    if (!url) continue;
    try {
      if (new URL(url).origin !== DOCS_ORIGIN) return null;
    } catch {
      return null;
    }
    candidates.add(url);
  }
  return candidates.size === 1 ? [...candidates][0] : null;
};

/** True when the link points outside this app (opened in a new tab with rel=noopener noreferrer). */
export const isExternalHref = (href: string): boolean => /^https?:\/\//i.test(href);

/** A source string like "repo:path@<sha>" for browser display: the revision is shown only for public repos. */
export const displaySource = (source: string | null | undefined, repo?: string | null): string => {
  const s = source ?? "";
  if (isPublicRepository(repo)) return s;
  return s.replace(/@[0-9a-f]{7,64}$/i, "");
};

export interface SourceGroup {
  key: string;
  repository: string | null;
  relative_path: string | null;
  content_state: string | null;
  verification_state: string | null;
  /** The [n] excerpt numbers cited from this one document, ascending. */
  indices: number[];
  /** Short revision, only for public repositories. */
  publicRevision: string | null;
}

/**
 * One row per DOCUMENT, not per chunk. Two excerpts of the same file read as one source cited twice;
 * different documents stay separate rows, so a multi-document answer never looks single-sourced.
 */
export const groupCitationsByDocument = (citations: AskCitation[]): SourceGroup[] => {
  const groups = new Map<string, SourceGroup>();
  for (const c of citations) {
    const key = c.document_id ?? `${c.repository}/${c.relative_path}`;
    const g = groups.get(key);
    if (g) {
      g.indices.push(c.index);
      continue;
    }
    groups.set(key, {
      key,
      repository: c.repository,
      relative_path: c.relative_path,
      content_state: c.content_state,
      verification_state: c.verification_state,
      indices: [c.index],
      publicRevision: isPublicRepository(c.repository) && c.source_revision ? c.source_revision.slice(0, 10) : null,
    });
  }
  return [...groups.values()].map((g) => ({ ...g, indices: [...g.indices].sort((a, b) => a - b) }));
};
