import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { AskCitation } from "../api/client";
import AnswerMarkdown from "./AnswerMarkdown";

const cite: AskCitation = {
  index: 1, repository: "omnibioai-docs", relative_path: "site/docs/admin/installation.md", source_revision: "217b75ad0c7aa3e0fc505b31711ba946d6fc4dd5",
  document_id: "doc-a", chunk_id: "chunk-a", content_state: "CURRENT", verification_state: "CONFIGURED",
};

describe("AnswerMarkdown link handling", () => {
  it("links a relative documentation path to docs.omnibioai.org, opening safely in a new tab", () => {
    render(<AnswerMarkdown text="See [deployment](./deployment.md)." citations={[cite]} />);
    const a = screen.getByRole("link", { name: "deployment" });
    expect(a).toHaveAttribute("href", "https://docs.omnibioai.org/admin/deployment/");
    expect(a).toHaveAttribute("target", "_blank");
    expect(a.getAttribute("rel")).toBe("noopener noreferrer");
  });

  it("keeps an absolute docs URL and an external URL exactly as written", () => {
    render(<AnswerMarkdown text="[docs](https://docs.omnibioai.org/admin/security/) and [gh](https://github.com/OmniBioAI/omnibioai-studio)" citations={[cite]} />);
    expect(screen.getByRole("link", { name: "docs" })).toHaveAttribute("href", "https://docs.omnibioai.org/admin/security/");
    expect(screen.getByRole("link", { name: "gh" })).toHaveAttribute("href", "https://github.com/OmniBioAI/omnibioai-studio");
  });

  it("renders a malformed, dangerous or escaping link as plain text, not a link", () => {
    render(<AnswerMarkdown text="[a](javascript:alert(1)) [b](//evil.example/x) [c](../../../etc/passwd) [d](https://u:p@evil.example/)" citations={[cite]} />);
    expect(screen.queryAllByRole("link")).toHaveLength(0);
    for (const label of ["a", "b", "c", "d"]) expect(screen.getByText(label)).toHaveClass("unlinked-doc");
  });

  it("does not link a relative path when no docs page was cited, so it cannot resolve against this host", () => {
    render(<AnswerMarkdown text="[x](./deployment.md)" citations={[]} />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("never renders images from an answer", () => {
    const { container } = render(<AnswerMarkdown text="![x](https://evil.example/t.png)" citations={[cite]} />);
    expect(container.querySelector("img")).toBeNull();
  });
});
