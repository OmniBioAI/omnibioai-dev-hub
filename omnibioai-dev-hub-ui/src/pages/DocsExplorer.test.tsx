import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DocsExplorer from "./DocsExplorer";

const { getStatus } = vi.hoisted(() => ({ getStatus: vi.fn() }));
vi.mock("../api/client", () => ({ getStatus }));

describe("DocsExplorer", () => {
  beforeEach(() => getStatus.mockReset());

  it("renders loading and status payloads", async () => {
    getStatus.mockResolvedValue({ total_docs: 8, total_vectors: 12, index_size: "4 MB", model: "nomic", version: "v6", uptime: "2h" });
    render(<DocsExplorer />);
    expect(screen.getByText("Connecting to backend...")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Raw Status Payload")).toBeInTheDocument());
    expect(screen.getByText("8")).toBeInTheDocument();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: /refresh/i })); });
    expect(getStatus).toHaveBeenCalledTimes(2);
  });

  it("shows an error and can retry", async () => {
    getStatus.mockRejectedValueOnce(new Error("unavailable")).mockResolvedValueOnce({ version: "ok" });
    render(<DocsExplorer />);
    await waitFor(() => expect(screen.getByText("Backend unreachable")).toBeInTheDocument());
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Retry" })); });
    await waitFor(() => expect(screen.getByText("Raw Status Payload")).toBeInTheDocument());
  });

  it("handles a valid but empty status payload", async () => {
    getStatus.mockResolvedValue({});
    render(<DocsExplorer />);
    await waitFor(() => expect(screen.getByText("Raw Status Payload")).toBeInTheDocument());
    expect(screen.getByText("{}", { exact: false })).toBeInTheDocument();
  });
});
