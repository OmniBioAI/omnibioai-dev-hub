import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useCurrentUser } from "./useCurrentUser";

const tokenFor = (payload: object) => `header.${btoa(JSON.stringify(payload))}.signature`;

describe("useCurrentUser", () => {
  it("returns a guest for missing or malformed tokens", () => {
    expect(renderHook(() => useCurrentUser()).result.current).toEqual({ name: "User", role: "Guest", initials: "?" });
    localStorage.setItem("access_token", "bad-token");
    expect(renderHook(() => useCurrentUser()).result.current).toEqual({ name: "User", role: "Guest", initials: "?" });
  });

  it("decodes email, role, and initials from a JWT payload", () => {
    localStorage.setItem("access_token", tokenFor({ email: "ada.lovelace@example.test", roles: ["admin"] }));
    expect(renderHook(() => useCurrentUser()).result.current).toEqual({ name: "Ada.lovelace", role: "Admin", initials: "AL" });
  });

  it("falls back safely for partial claims", () => {
    localStorage.setItem("access_token", tokenFor({ email: "", roles: [] }));
    expect(renderHook(() => useCurrentUser()).result.current).toEqual({ name: "User", role: "Guest", initials: "U" });
  });

  it("handles an invalid encoded payload and claims without an email", () => {
    localStorage.setItem("access_token", "header.%not-json%.signature");
    expect(renderHook(() => useCurrentUser()).result.current.initials).toBe("?");
    localStorage.setItem("access_token", tokenFor({ roles: ["reviewer"] }));
    expect(renderHook(() => useCurrentUser()).result.current).toMatchObject({ name: "User", role: "Reviewer", initials: "U" });
  });
});
