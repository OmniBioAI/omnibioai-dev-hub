import "@testing-library/jest-dom/vitest";

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
  Element.prototype.scrollIntoView = vi.fn();
});
