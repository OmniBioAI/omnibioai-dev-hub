import { describe, expect, it } from "vitest";
import type { AskCitation } from "../api/client";
import {
  DOCS_ORIGIN, displaySource, docsUrlForPath, groupCitationsByDocument, isExternalHref, isPublicRepository, resolveDocLink,
} from "./docLinks";

const cite = (over: Partial<AskCitation> = {}): AskCitation => ({
  index: 1, repository: "omnibioai-docs", relative_path: "site/docs/admin/installation.md",
  source_revision: "217b75ad0c7aa3e0fc505b31711ba946d6fc4dd5", document_id: "doc-a", chunk_id: "chunk-a",
  content_state: "CURRENT", verification_state: "CONFIGURED", ...over,
});

describe("docs URL mapping (matches the real Docusaurus routing: routeBasePath '/', trailing slash)", () => {
  it.each([
    ["site/docs/admin/deployment.md", "https://docs.omnibioai.org/admin/deployment/"],
    ["site/docs/index.md", "https://docs.omnibioai.org/"],
    ["site/docs/admin/index.md", "https://docs.omnibioai.org/admin/"],
    ["site/docs/integrations/details/vault.md", "https://docs.omnibioai.org/integrations/details/vault/"],
  ])("%s -> %s", (path, url) => expect(docsUrlForPath(path)).toBe(url));

  it("does not map non-page files or files outside the docs folder", () => {
    expect(docsUrlForPath("site/docs/img/logo.png")).toBeNull();
    expect(docsUrlForPath("README.md")).toBeNull();
    expect(docsUrlForPath("tutorials/ml/README.md")).toBeNull();
  });
});

describe("resolveDocLink", () => {
  const one = [cite()];

  it("resolves a relative documentation path against the cited page, at file level", () => {
    expect(resolveDocLink("./deployment.md", one)).toBe(`${DOCS_ORIGIN}/admin/deployment/`);
    expect(resolveDocLink("deployment.md", one)).toBe(`${DOCS_ORIGIN}/admin/deployment/`);
    expect(resolveDocLink("../user/getting-started.md", one)).toBe(`${DOCS_ORIGIN}/user/getting-started/`);
    expect(resolveDocLink("./security.md#tls", one)).toBe(`${DOCS_ORIGIN}/admin/security/#tls`);
    expect(resolveDocLink("#prerequisites", one)).toBe(`${DOCS_ORIGIN}/admin/installation/#prerequisites`);
  });

  it("resolves a Docusaurus site-absolute route onto the docs site", () => {
    expect(resolveDocLink("/admin/deployment", one)).toBe(`${DOCS_ORIGIN}/admin/deployment/`);
    expect(resolveDocLink("/admin/deployment.md", one)).toBe(`${DOCS_ORIGIN}/admin/deployment/`);
    expect(resolveDocLink("/", one)).toBe(`${DOCS_ORIGIN}/`);
  });

  it("keeps an absolute documentation URL exactly as written", () => {
    expect(resolveDocLink("https://docs.omnibioai.org/admin/security/", one)).toBe("https://docs.omnibioai.org/admin/security/");
  });

  it("keeps an external https/http URL as written and never rewrites it to the docs site", () => {
    expect(resolveDocLink("https://github.com/OmniBioAI/omnibioai-studio/releases/latest", one)).toBe(
      "https://github.com/OmniBioAI/omnibioai-studio/releases/latest");
    expect(resolveDocLink("http://example.org/a?b=c#d", one)).toBe("http://example.org/a?b=c#d");
  });

  it.each([
    "javascript:alert(1)", "JaVaScRiPt:alert(1)", "data:text/html,<script>1</script>", "vbscript:x", "file:///etc/passwd",
    "mailto:a@b.c", "ftp://example.org/x",
  ])("drops the non-http(s) scheme %s", (href) => expect(resolveDocLink(href, one)).toBeNull());

  it.each([
    "//evil.example/x", "/\\evil.example/x", "\\\\evil.example\\x", "https://user:pw@evil.example/", "  ", "", undefined, null,
    "java\nscript:alert(1)", "./a\u0000b.md", "https://exa mple.org", `./${"a".repeat(3000)}.md`,
  ])("treats a malformed / dangerous href %j as plain text", (href) => expect(resolveDocLink(href as string, one)).toBeNull());

  it("cannot be used as an open redirect: relative forms never leave the docs origin or the docs folder", () => {
    for (const href of ["../../../etc/passwd", "../../README.md", "../../../../evil.md", "%2e%2e/%2e%2e/%2e%2e/x.md", "..%2fsecret.md", "./%2e%2e%2f%2e%2e%2fx.md", "..\\x.md"]) {
      const out = resolveDocLink(href, one);
      expect(out === null || new URL(out).origin === DOCS_ORIGIN).toBe(true);
      expect(out).toBeNull();
    }
    for (const href of ["/admin/../../evil.md", "/../evil.md"]) {
      const out = resolveDocLink(href, one);
      expect(out === null || new URL(out).origin === DOCS_ORIGIN).toBe(true);
    }
  });

  it("does not guess when the cited pages disagree about where a relative link points", () => {
    const two = [cite(), cite({ document_id: "doc-b", relative_path: "site/docs/user/getting-started.md" })];
    expect(resolveDocLink("./deployment.md", two)).toBeNull(); // admin/deployment vs user/deployment
    // a link that both cited pages resolve to the SAME page is unambiguous, so it is linked
    expect(resolveDocLink("../admin/deployment.md", two)).toBe(`${DOCS_ORIGIN}/admin/deployment/`);
    // ...and it also agrees when both pages are in the same folder
    const same = [cite(), cite({ document_id: "doc-c", relative_path: "site/docs/admin/monitoring.md" })];
    expect(resolveDocLink("./deployment.md", same)).toBe(`${DOCS_ORIGIN}/admin/deployment/`);
  });

  it("does not link relative paths when no cited page is on the docs site", () => {
    expect(resolveDocLink("./deployment.md", [])).toBeNull();
    expect(resolveDocLink("./x.md", [cite({ repository: "omnibioai-rag", relative_path: "README.md" })])).toBeNull();
  });

  it("does not map relative links to non-page files", () => {
    expect(resolveDocLink("./logo.png", one)).toBeNull();
    expect(resolveDocLink("./data.csv", one)).toBeNull();
  });
});

describe("isExternalHref", () => {
  it("is true only for absolute http(s) links", () => {
    expect(isExternalHref("https://docs.omnibioai.org/x/")).toBe(true);
    expect(isExternalHref("./x.md")).toBe(false);
  });
});

describe("private provenance in the browser", () => {
  it("treats only known-public repositories as public (default deny)", () => {
    expect(isPublicRepository("omnibioai-docs")).toBe(false);
    expect(isPublicRepository("omnibioai-rag")).toBe(false);
    expect(isPublicRepository("something-new")).toBe(false);
    expect(isPublicRepository(null)).toBe(false);
    expect(isPublicRepository("omnibioai-model-registry")).toBe(true);
  });

  it("strips the @revision from a displayed source unless the repository is public", () => {
    const sha = "217b75ad0c7aa3e0fc505b31711ba946d6fc4dd5";
    expect(displaySource(`omnibioai-docs:site/docs/x.md@${sha}`, "omnibioai-docs")).toBe("omnibioai-docs:site/docs/x.md");
    expect(displaySource(`omnibioai-docs:site/docs/x.md@${sha}`, undefined)).toBe("omnibioai-docs:site/docs/x.md");
    expect(displaySource(`omnibioai-model-registry:README.md@${sha}`, "omnibioai-model-registry")).toContain(sha);
    expect(displaySource(null)).toBe("");
  });
});

describe("groupCitationsByDocument", () => {
  it("shows one row per document and keeps every cited excerpt number", () => {
    const groups = groupCitationsByDocument([
      cite({ index: 4, chunk_id: "c4" }), cite({ index: 1, chunk_id: "c1" }),
      cite({ index: 2, document_id: "doc-b", relative_path: "site/docs/admin/deployment.md", chunk_id: "c2" }),
    ]);
    expect(groups.map((g) => g.relative_path)).toEqual(["site/docs/admin/installation.md", "site/docs/admin/deployment.md"]);
    expect(groups[0].indices).toEqual([1, 4]);
    expect(groups[1].indices).toEqual([2]);
  });

  it("never merges different documents", () => {
    const g = groupCitationsByDocument([cite({ index: 1 }), cite({ index: 2, document_id: "doc-b", relative_path: "site/docs/user/projects.md" })]);
    expect(g).toHaveLength(2);
  });

  it("exposes a short revision only for public repositories", () => {
    expect(groupCitationsByDocument([cite()])[0].publicRevision).toBeNull();
    const pub = groupCitationsByDocument([cite({ repository: "omnibioai-model-registry", relative_path: "README.md" })])[0];
    expect(pub.publicRevision).toBe("217b75ad0c");
  });
});
