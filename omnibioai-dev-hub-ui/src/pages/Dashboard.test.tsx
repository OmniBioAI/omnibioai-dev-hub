import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DashboardPage from "./Dashboard";

const { getStatus } = vi.hoisted(() => ({ getStatus: vi.fn() }));
vi.mock("../api/client", () => ({ getStatus }));

const status = { control_plane: { status: "READY", uptime_sec: 3661, engine_ready: true, error: null, metrics: { init_time: 1, build_time_ms: 12.34 } }, index_vectors: 128, graph_edges: 42, plugins_loaded: 7 };

describe("DashboardPage", () => {
  beforeEach(() => getStatus.mockReset());

  it("shows a loading state before rendering status metrics and navigation actions", async () => {
    getStatus.mockResolvedValue(status);
    const onNavigate = vi.fn();
    render(<DashboardPage onNavigate={onNavigate} />);
    expect(screen.getByText(/Connecting to backend/)).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("128").length).toBeGreaterThan(0));
    expect(screen.getAllByText("READY").length).toBeGreaterThan(0);
    expect(screen.getAllByText("1.0h").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByText("Try Query →"));
    expect(onNavigate).toHaveBeenCalledWith("chat");
  });

  it("shows a recoverable backend error and retries", async () => {
    getStatus.mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(status);
    render(<DashboardPage onNavigate={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Backend unreachable")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
    await waitFor(() => expect(screen.getAllByText("READY").length).toBeGreaterThan(0));
    expect(getStatus).toHaveBeenCalledTimes(2);
  });

  it("renders initializing and zero-resource states", async () => {
    getStatus.mockResolvedValue({ control_plane: { status: "STARTING", uptime_sec: 25, engine_ready: false, error: "booting", metrics: {} }, index_vectors: 0, graph_edges: 0, plugins_loaded: 0 });
    render(<DashboardPage onNavigate={vi.fn()} />);
    await waitFor(() => expect(screen.getAllByText("STARTING").length).toBeGreaterThan(0));
    expect(screen.getAllByText("25s").length).toBeGreaterThan(0);
    expect(screen.getByText("Initializing")).toBeInTheDocument();
  });
});
