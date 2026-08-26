import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("./pages/Dashboard", () => ({ default: () => <div>Dashboard page</div> }));
vi.mock("./pages/ChatPage", () => ({ default: () => <div>Chat page</div> }));
vi.mock("./pages/SearchPage", () => ({ default: () => <div>Search page</div> }));
vi.mock("./pages/DocsExplorer", () => ({ default: () => <div>Docs page</div> }));
vi.mock("./pages/GraphView", () => ({ default: () => <div>Graph page</div> }));

describe("App", () => {
  it("starts on the dashboard inside the application shell", () => {
    render(<App />);
    expect(screen.getByText("Dashboard page")).toBeInTheDocument();
    expect(screen.getAllByText("Overview").length).toBeGreaterThan(1);
  });

  it("switches between all user-visible application pages", () => {
    render(<App />);
    fireEvent.click(screen.getByText("Query Assistant"));
    expect(screen.getByText("Chat page")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Vector Search"));
    expect(screen.getByText("Search page")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Knowledge Graph"));
    expect(screen.getByText("Graph page")).toBeInTheDocument();
    fireEvent.click(screen.getByText("System Status"));
    expect(screen.getByText("Docs page")).toBeInTheDocument();
  });
});
