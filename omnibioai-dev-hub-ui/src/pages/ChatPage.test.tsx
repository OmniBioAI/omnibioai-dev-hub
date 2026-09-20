import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChatPage from "./ChatPage";

const { ragStream } = vi.hoisted(() => ({ ragStream: vi.fn() }));
vi.mock("../api/client", () => ({ ragStream }));
vi.mock("react-markdown", () => ({ default: ({ children }: { children: string }) => <div>{children}</div> }));

const citation = {
  index: 1, repository: "omnibioai-docs", relative_path: "site/docs/admin/disaster-recovery.md",
  source_revision: "217b75ad0c7aa3e0fc505b31711ba946d6fc4dd5", document_id: "doc-1", chunk_id: "chunk-1",
  content_state: "CURRENT", verification_state: "CONFIGURED",
};

// ragStream(query, onToken, onDone, onError, onResult)
const respondWith = (result: Record<string, unknown>) =>
  ragStream.mockImplementation(async (_q: string, _t: unknown, onDone: (v?: string) => void, _e: unknown, onResult: (r: unknown) => void) => {
    onResult(result);
    onDone(result.content as string);
  });

const ask = (text: string) => {
  const input = screen.getByPlaceholderText(/Ask OmniBioAI/);
  fireEvent.change(input, { target: { value: text } });
  fireEvent.keyDown(input, { key: "Enter" });
};

describe("Ask OmniBioAI (ChatPage)", () => {
  beforeEach(() => { ragStream.mockReset(); });

  it("greets with the documentation-only promise and asks through the streaming client", async () => {
    respondWith({ content: "Back up nightly [1].", grounded: true, answer_status: "GROUNDED", citations: [citation], context_used: 1 });
    render(<ChatPage />);
    expect(screen.getByText("Ask OmniBioAI")).toBeInTheDocument();
    expect(screen.getByText(/I answer only from OmniBioAI's published documentation/)).toBeInTheDocument();
    ask("How do I back up?");
    await waitFor(() => expect(screen.getByText("How do I back up?")).toBeInTheDocument());
    expect(ragStream).toHaveBeenCalledWith("How do I back up?", expect.any(Function), expect.any(Function), expect.any(Function), expect.any(Function));
  });

  it("renders a grounded answer with its numbered citations and full provenance available", async () => {
    respondWith({ content: "Back up nightly [1].", grounded: true, answer_status: "GROUNDED", citations: [citation], context_used: 1 });
    render(<ChatPage />);
    ask("How do I back up?");
    await waitFor(() => expect(screen.getByText("Back up nightly [1].")).toBeInTheDocument());
    const sources = screen.getByRole("list", { name: "Sources" });
    expect(sources).toHaveTextContent("[1]");
    expect(sources).toHaveTextContent("omnibioai-docs/site/docs/admin/disaster-recovery.md");
    expect(sources).toHaveTextContent("current");
    expect(sources).toHaveTextContent("not verified · configured"); // CONFIGURED must not read as verified
    const item = sources.querySelector("li")!;
    expect(item.getAttribute("data-document-id")).toBe("doc-1");
    expect(item.getAttribute("data-chunk-id")).toBe("chunk-1");
    expect(item.getAttribute("data-revision")).toBe("217b75ad0c7aa3e0fc505b31711ba946d6fc4dd5");
    expect(item.getAttribute("title")).toContain("217b75ad0c7aa3e0fc505b31711ba946d6fc4dd5");
    expect(screen.queryByText("No trusted documentation answer")).not.toBeInTheDocument();
  });

  it("shows a distinct 'no trusted documentation' state, never an answer bubble, when nothing grounded was found", async () => {
    respondWith({ content: "No sufficiently relevant trusted OmniBioAI documentation was found for this question, so no answer is given.",
      grounded: false, answer_status: "NO_TRUSTED_CONTEXT", citations: [], context_used: 0 });
    render(<ChatPage />);
    ask("What does the OmniBioAI iOS app do?");
    const status = await screen.findByRole("status");
    expect(status).toHaveTextContent("No trusted documentation answer");
    expect(status).toHaveTextContent("No sufficiently relevant trusted OmniBioAI documentation was found");
    expect(status.getAttribute("data-answer-status")).toBe("NO_TRUSTED_CONTEXT");
    expect(screen.queryByRole("list", { name: "Sources" })).not.toBeInTheDocument();
    expect(status.closest(".chat-bubble")).toHaveClass("no-answer");
  });

  it("never shows citations for an ungrounded result, even if the payload carried some", async () => {
    respondWith({ content: "Documentation was found, but it does not contain a supported answer.", grounded: false,
      answer_status: "INSUFFICIENT_CONTEXT", citations: [citation], context_used: 3 });
    render(<ChatPage />);
    ask("q");
    await screen.findByRole("status");
    expect(screen.queryByRole("list", { name: "Sources" })).not.toBeInTheDocument();
  });

  it("ignores raw model tokens: an unverified answer is never rendered", async () => {
    ragStream.mockImplementation(async (_q: string, onToken: (t: string) => void, onDone: (v?: string) => void, _e: unknown, onResult: (r: unknown) => void) => {
      onToken("UNVERIFIED HALLUCINATION");
      onResult({ content: "No sufficiently relevant trusted OmniBioAI documentation was found.", grounded: false, answer_status: "NO_TRUSTED_CONTEXT", citations: [], context_used: 0 });
      onDone("No sufficiently relevant trusted OmniBioAI documentation was found.");
    });
    render(<ChatPage />);
    ask("q");
    await screen.findByRole("status");
    expect(screen.queryByText(/UNVERIFIED HALLUCINATION/)).not.toBeInTheDocument();
  });

  it("does not submit blank input and renders streaming errors", async () => {
    render(<ChatPage />);
    fireEvent.click(screen.getByRole("button", { name: /ask/i }));
    expect(ragStream).not.toHaveBeenCalled();
    ragStream.mockImplementation(async (_q: string, _t: unknown, _d: unknown, onError: (e: unknown) => void) => onError("backend failed"));
    ask("query");
    await waitFor(() => expect(screen.getByText("Error: backend failed")).toBeInTheDocument());
  });
});
