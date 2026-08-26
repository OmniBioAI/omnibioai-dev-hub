import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import GraphView from "./GraphView";

const { getStatus } = vi.hoisted(() => ({ getStatus: vi.fn() }));
vi.mock("../api/client", () => ({ getStatus }));

describe("GraphView", () => {
  it("shows loading and then graph summary values", async () => {
    getStatus.mockResolvedValue({ graph_edges: 3, plugins_loaded: 2, repos_loaded: 4 });
    render(<GraphView />);
    expect(screen.getByText("Loading graph...")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Edges: 3")).toBeInTheDocument());
    expect(screen.getByText("Plugins: 2")).toBeInTheDocument();
    expect(screen.getByText("Repos: 4")).toBeInTheDocument();
  });
});
