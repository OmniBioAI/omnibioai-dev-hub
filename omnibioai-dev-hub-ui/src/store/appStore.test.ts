import { describe, expect, it } from "vitest";
import { useAppStore } from "./appStore";

describe("app store", () => {
  it("starts empty and exposes meaningful query/result transitions", () => {
    const store = useAppStore.getState();
    expect(store.query).toBe("");
    expect(store.answer).toBe("");
    store.setQuery("CRISPR");
    store.setAnswer("result");
    store.setTrace({ steps: 2 });
    expect(useAppStore.getState()).toMatchObject({ query: "CRISPR", answer: "result", trace: { steps: 2 } });
  });
});
