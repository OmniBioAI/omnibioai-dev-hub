import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import KnowledgeGraph from "./KnowledgeGraph";

vi.mock("react-force-graph-3d", () => ({ default: ({ graphData }: { graphData: { nodes: unknown[]; links: unknown[] } }) => <div data-testid="force-graph">{graphData.nodes.length} nodes / {graphData.links.length} links</div> }));

describe("KnowledgeGraph", () => {
  it("loads graph data and passes the derived graph to the visualization", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ graph_edges: 3 }), { status: 200 }));
    render(<KnowledgeGraph />);
    await waitFor(() => expect(screen.getByTestId("force-graph")).toHaveTextContent("4 nodes / 3 links"));
    expect(fetch).toHaveBeenCalledWith("/status");
  });
});
