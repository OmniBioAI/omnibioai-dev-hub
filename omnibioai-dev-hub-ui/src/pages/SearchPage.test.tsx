import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SearchPage from "./SearchPage";

const { getStatus, ragQuery } = vi.hoisted(() => ({ getStatus: vi.fn(), ragQuery: vi.fn() }));
vi.mock("../api/client", () => ({ getStatus, ragQuery }));

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

  it("ignores blank searches and displays query errors", async () => {
    ragQuery.mockRejectedValue(new Error("offline"));
    render(<SearchPage />);
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(ragQuery).not.toHaveBeenCalled();
    fireEvent.change(screen.getByPlaceholderText(/Search embeddings/), { target: { value: "bad" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(screen.getByText("Error: offline")).toBeInTheDocument());
  });

  it("renders string context results and optional metadata branches", async () => {
    ragQuery.mockResolvedValue({ query: "gene", context: ["plain result"] });
    render(<SearchPage />);
    fireEvent.change(screen.getByPlaceholderText(/Search embeddings/), { target: { value: "gene" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(screen.getByText("plain result")).toBeInTheDocument());
    expect(screen.getByText("Retrieved Chunks (1)")).toBeInTheDocument();
  });
});
