import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MainLayout from "./MainLayout";

describe("MainLayout", () => {
  it("renders shell status, user identity, breadcrumb, and navigates from the sidebar", () => {
    localStorage.setItem("access_token", `x.${btoa(JSON.stringify({ email: "ada.lovelace@example.test", roles: ["admin"] }))}.x`);
    const setPage = vi.fn();
    render(<MainLayout page="dashboard" setPage={setPage} breadcrumb="Overview"><div>content</div></MainLayout>);
    expect(screen.getByText("OmniBioAI")).toBeInTheDocument();
    expect(screen.getByText("System Healthy")).toBeInTheDocument();
    expect(screen.getAllByText("Overview").length).toBeGreaterThan(1);
    expect(screen.getByText("Ada.lovelace")).toBeInTheDocument();
    expect(screen.getByText("Admin")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Vector Search"));
    expect(setPage).toHaveBeenCalledWith("search");
  });
});
