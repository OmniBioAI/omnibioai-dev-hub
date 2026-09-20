import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AskError } from "../api/client";
import SearchPage from "./SearchPage";

const { getStatus, ragQuery } = vi.hoisted(() => ({ getStatus: vi.fn(), ragQuery: vi.fn() }));
vi.mock("../api/client", async (orig) => ({ ...(await orig<typeof import("../api/client")>()), getStatus, ragQuery }));

describe("SearchPage", () => {
  beforeEach(() => { getStatus.mockResolvedValue({ index_vectors: 12 }); ragQuery.mockReset(); });

  it("shows the empty prompt, then renders answer and retrieved chunks", async () => {
    ragQuery.mockResolvedValue({ query: "gene", answer: "Found it", context_used: 1, version: "v6", context: [{ source: "doc.md", text: "chunk text" }] });
    render(<SearchPage />);
    await waitFor(() => expect(screen.getByText(/across 12 embeddings/)).toBeInTheDocument());
    expect(screen.getByText(/Enter a query to search/)).toBeInTheDocument();
    const input = screen.getByPlaceholderText(/Search embeddings/);
    fireEvent.change(input, { target: { value: "gene" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(screen.getByText("Found it")).toBeInTheDocument());
    expect(screen.getByText("doc.md")).toBeInTheDocument();
    expect(screen.getByText("chunk text")).toBeInTheDocument();
  });

  it("ignores blank searches and displays a concise query error", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    ragQuery.mockRejectedValue(new AskError("network"));
    render(<SearchPage />);
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(ragQuery).not.toHaveBeenCalled();
    fireEvent.change(screen.getByPlaceholderText(/Search embeddings/), { target: { value: "bad" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(screen.getByText(/Couldn't reach Ask OmniBioAI/)).toBeInTheDocument());
    expect(screen.queryByText(/Error:|TypeError|Failed to fetch|HTTP/)).not.toBeInTheDocument();
  });

  it("never shows a raw exception or HTTP status for an unexpected failure", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    ragQuery.mockRejectedValue(new Error("Request failed (HTTP 500) at /srv/x.py"));
    render(<SearchPage />);
    fireEvent.change(screen.getByPlaceholderText(/Search embeddings/), { target: { value: "bad" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(document.body.textContent).not.toMatch(/HTTP 500|x\.py|Request failed/);
  });

  it("does not render private revision SHAs, ids, or raw chunk objects in retrieved chunks", async () => {
    const sha = "217b75ad0c7aa3e0fc505b31711ba946d6fc4dd5";
    ragQuery.mockResolvedValue({ query: "q", context: [
      { source: `omnibioai-docs:site/docs/a.md@${sha}`, repository: "omnibioai-docs", chunk_id: "chunk-secret", document_id: "doc-secret", text: "body text" },
      { source: `omnibioai-rag:README.md@${sha}`, chunk_id: "chunk-secret-2" },
    ] });
    const { container } = render(<SearchPage />);
    fireEvent.change(screen.getByPlaceholderText(/Search embeddings/), { target: { value: "q" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(screen.getByText("omnibioai-docs:site/docs/a.md")).toBeInTheDocument());
    expect(container.innerHTML).not.toContain(sha);
    expect(container.innerHTML).not.toContain("217b75ad");
    expect(container.innerHTML).not.toMatch(/chunk-secret|doc-secret/);
  });

  it("renders string context results and optional metadata branches", async () => {
    ragQuery.mockResolvedValue({ query: "gene", context: ["plain result"] });
    render(<SearchPage />);
    fireEvent.change(screen.getByPlaceholderText(/Search embeddings/), { target: { value: "gene" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(screen.getByText("plain result")).toBeInTheDocument());
    expect(screen.getByText("Retrieved Chunks (1)")).toBeInTheDocument();
  });

  it("labels a grounded answer, shows its citations, and never labels an ungrounded one as an answer", async () => {
    ragQuery.mockResolvedValue({ query: "q", answer: "Do X [1].", grounded: true, answer_status: "GROUNDED", context_used: 1,
      citations: [{ index: 1, repository: "omnibioai-docs", relative_path: "a.md", source_revision: "r", document_id: "d", chunk_id: "c", content_state: "CURRENT", verification_state: "CONFIGURED" }],
      context: [{ source: "a.md", text: "t" }] });
    render(<SearchPage />);
    fireEvent.change(screen.getByPlaceholderText(/Search embeddings/), { target: { value: "q" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(screen.getByText("Answer (grounded in documentation)")).toBeInTheDocument());
    expect(screen.getByRole("list", { name: "Sources" })).toHaveTextContent("omnibioai-docs/a.md");

    ragQuery.mockResolvedValue({ query: "q2", answer: "No sufficiently relevant trusted OmniBioAI documentation was found.", grounded: false, answer_status: "NO_TRUSTED_CONTEXT", context_used: 0, citations: [], context: [] });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(screen.getByText("No trusted documentation answer")).toBeInTheDocument());
    expect(screen.queryByText("Generated Answer")).not.toBeInTheDocument();
    expect(screen.queryByText("Answer (grounded in documentation)")).not.toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "Sources" })).not.toBeInTheDocument();
  });
});

