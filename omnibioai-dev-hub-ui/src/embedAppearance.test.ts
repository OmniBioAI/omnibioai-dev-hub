import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import vm from "node:vm";
import {
  appearanceFromSearch,
  connectAppearanceMessages,
  EMBED_READY_EVENT,
  EMBED_THEME_EVENT,
  parseThemeMessage,
} from "./embedAppearance";

const html = readFileSync(resolve(process.cwd(), "index.html"), "utf8");
const bootstrap = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)?.[1];
if (!bootstrap) throw new Error("index.html must run the appearance bootstrap before styles and app scripts");
const tokenPackage = JSON.parse(readFileSync(resolve(process.cwd(), "vendor/omnibioai-design-tokens/package.json"), "utf8")) as { version: string };

function fakeEmbeddedDocument(referrer = "https://webstudio.omnibioai.org/workbench") {
  const listeners = new Map<string, EventListener>();
  const sent: Array<[unknown, string]> = [];
  const parent = { postMessage: (message: unknown, origin: string) => sent.push([message, origin]) };
  const win = {
    parent,
    addEventListener: (type: string, listener: EventListener) => listeners.set(type, listener),
    removeEventListener: (type: string) => listeners.delete(type),
  } as unknown as Window;
  const root = { dataset: {} as DOMStringMap, style: { cssText: "untouched" }, innerHTML: "untouched" } as unknown as HTMLElement;
  const doc = { referrer, documentElement: root } as Document;
  return { win, doc, parent, root, listeners, sent };
}

test("pre-paint bootstrap runs before stylesheet links", () => {
  expect(html.indexOf("<script>")).toBeLessThan(html.indexOf("<link"));
});

test("pre-paint bootstrap accepts bounded appearance values, strips exact secrets, and preserves author/username", () => {
  const location = {
    pathname: "/_svc/devhub",
    search: "?view=chat&obai_theme=%3Csvg%3E&obai_accent=purple&obai_density=compact&obai_motion=reduce&obai_owner=x&token=x&access_token=x&refresh_token=x&id_token=x&password=x&api_key=x&api-key=x&session=x&session_id=x&secret=x&client_secret=x&credential=x&authorization=x&auth=x&user_id=x&email=x&organization=x&organization_id=x&account=x&account_id=x&author=Jane&username=reader",
    hash: "#answer",
  };
  let replaced = "";
  const root = { dataset: {} as Record<string, string>, style: { cssText: "untouched" }, innerHTML: "untouched" };
  vm.runInNewContext(bootstrap!, {
    window: {
      location,
      matchMedia: () => ({ matches: false }),
      history: { replaceState: (_state: unknown, _title: string, url: string) => { replaced = url; } },
    },
    document: { documentElement: root },
    URLSearchParams,
    Object,
  });

  expect(root.dataset).toEqual({ theme: "light", accent: "purple", density: "compact", motion: "reduce" });
  const sanitized = new URL(replaced, "https://ask.example");
  const sanitizedParams = sanitized.searchParams;
  for (const key of [
    "token", "access_token", "refresh_token", "id_token", "password", "api_key", "api-key",
    "session", "session_id", "secret", "client_secret", "credential", "authorization", "auth",
    "user_id", "email", "organization", "organization_id", "account", "account_id", "obai_owner",
  ]) expect(sanitizedParams.has(key)).toBe(false);
  expect(sanitizedParams.get("author")).toBe("Jane");
  expect(sanitizedParams.get("username")).toBe("reader");
  expect(sanitizedParams.get("view")).toBe("chat");
  expect(sanitized.hash).toBe("#answer");
  expect(root.style.cssText).toBe("untouched");
  expect(root.innerHTML).toBe("untouched");
});

test("appearance query and invalid parameters only produce allowlisted values", () => {
  expect(appearanceFromSearch("?obai_theme=dark&obai_accent=orange&obai_density=compact&obai_motion=reduce")).toEqual({
    theme: "dark", accent: "orange", density: "compact", motion: "reduce",
  });
  expect(appearanceFromSearch("?obai_theme=evil&obai_accent=data%3Atext%2Fhtml&obai_density=huge&obai_motion=token", "light")).toEqual({
    theme: "light", accent: "teal", density: "comfortable", motion: "system",
  });
});

test("theme message accepts only exact origin, parent source, and expected fields", () => {
  const { win, doc, parent, root, listeners, sent } = fakeEmbeddedDocument();
  const stop = connectAppearanceMessages(win, doc, "http://localhost:5174");
  expect(sent).toEqual([[{ type: EMBED_READY_EVENT }, "https://webstudio.omnibioai.org"]]);
  expect(sent.some(([, origin]) => origin === "*")).toBe(false);

  const listener = listeners.get("message")!;
  const good = { type: EMBED_THEME_EVENT, theme: "light", accent: "blue", density: "compact", motion: "reduce" };
  const deliver = (origin: string, source: unknown, data: unknown) => listener({ origin, source, data } as MessageEvent);
  deliver("https://attacker.example", parent, good);
  deliver("https://webstudio.omnibioai.org", {}, good);
  deliver("https://webstudio.omnibioai.org", parent, { ...good, token: "secret" });
  deliver("https://webstudio.omnibioai.org", parent, { ...good, author: "Jane" });
  deliver("https://webstudio.omnibioai.org", parent, { ...good, accent: "javascript:alert(1)" });
  expect(root.dataset).toEqual({});
  expect(parseThemeMessage({ ...good, username: "reader" })).toBeNull();

  deliver("https://webstudio.omnibioai.org", parent, good);
  expect(root.dataset).toEqual({ theme: "light", accent: "blue", density: "compact", motion: "reduce" });
  expect(root.style.cssText).toBe("untouched");
  expect(root.innerHTML).toBe("untouched");
  stop();
  expect(listeners.has("message")).toBe(false);
});

test("development parent origin is accepted only as the exact configured origin", () => {
  const accepted = fakeEmbeddedDocument("http://localhost:5174/");
  connectAppearanceMessages(accepted.win, accepted.doc, "http://localhost:5174");
  expect(accepted.sent).toEqual([[{ type: EMBED_READY_EVENT }, "http://localhost:5174"]]);
  expect(accepted.listeners.has("message")).toBe(true);

  const hostile = fakeEmbeddedDocument("http://localhost:5174/");
  connectAppearanceMessages(hostile.win, hostile.doc, "http://localhost:5174.attacker.example");
  expect(hostile.sent).toEqual([]);
  expect(hostile.listeners.has("message")).toBe(false);

  const topLevel = fakeEmbeddedDocument("");
  (topLevel.win as unknown as { parent: unknown }).parent = topLevel.win;
  connectAppearanceMessages(topLevel.win, topLevel.doc, "http://localhost:5174");
  expect(topLevel.sent).toEqual([]);
  expect(topLevel.listeners.has("message")).toBe(false);
});

test("nginx sends the explicit frame policy as an always header for error responses too", () => {
  const entrypoint = readFileSync(resolve(process.cwd(), "../docker-entrypoint.sh"), "utf8");
  expect(entrypoint).toContain('_frame_policy="frame-ancestors \'self\' https://webstudio.omnibioai.org"');
  expect(entrypoint).toMatch(/add_header Content-Security-Policy "%s" always;/);
});

test("Ask surface colors come from the shared token package, not hard-coded dark values", () => {
  const stylesheet = readFileSync(resolve(process.cwd(), "src/index.css"), "utf8");
  expect(stylesheet).not.toMatch(/#[0-9a-f]{3,8}\b|rgba?\(/i);
  expect(tokenPackage.version).toBe("1.1.0");
});
