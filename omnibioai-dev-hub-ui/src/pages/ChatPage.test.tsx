import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChatPage from "./ChatPage";

const { ragStream } = vi.hoisted(() => ({ ragStream: vi.fn() }));
vi.mock("../api/client", () => ({ ragStream }));
vi.mock("react-markdown", () => ({ default: ({ children }: { children: string }) => <div>{children}</div> }));

describe("ChatPage", () => {
  beforeEach(() => ragStream.mockReset());

  it("renders the greeting and streams a response after submitting a query", async () => {
    ragStream.mockImplementation(async (...args: unknown[]) => {
      const callbacks = args.filter((arg): arg is (value?: string) => void => typeof arg === "function");
      if (callbacks.length < 2) return;
      callbacks[0]("Hel"); callbacks[1]("Hello there");
    });
    render(<ChatPage />);
    expect(screen.getByText(/Hello! I'm the OmniBioAI/)).toBeInTheDocument();
    const input = screen.getByPlaceholderText(/Ask OmniBioAI/);
    fireEvent.change(input, { target: { value: "What is RAG?" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(screen.getByText("What is RAG?")).toBeInTheDocument());
    expect(screen.getByText("Hello there")).toBeInTheDocument();
    expect(ragStream).toHaveBeenCalledWith("What is RAG?", expect.any(Function), expect.any(Function), expect.any(Function));
  });

  it("does not submit blank input and renders streaming errors", async () => {
    render(<ChatPage />);
    fireEvent.click(screen.getByRole("button", { name: /send/i }));
    expect(ragStream).not.toHaveBeenCalled();
    ragStream.mockImplementation(async (...args: unknown[]) => {
      const callbacks = args.filter((arg): arg is (value?: string) => void => typeof arg === "function");
      if (callbacks.length < 3) return;
      callbacks[2]("backend failed");
    });
    fireEvent.change(screen.getByPlaceholderText(/Ask OmniBioAI/), { target: { value: "query" } });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));
    await waitFor(() => expect(screen.getByText("Error: backend failed")).toBeInTheDocument());
  });
});
