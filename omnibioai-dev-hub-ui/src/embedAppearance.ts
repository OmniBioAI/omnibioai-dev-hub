export const EMBED_READY_EVENT = "omnibioai.embed.ready.v1";
export const EMBED_THEME_EVENT = "omnibioai.theme.set.v1";

const DEFAULTS = Object.freeze({ theme: "dark", accent: "teal", density: "comfortable", motion: "system" });
const ALLOWED = Object.freeze({
  theme: new Set(["light", "dark"]),
  accent: new Set(["teal", "blue", "purple", "orange"]),
  density: new Set(["comfortable", "compact"]),
  motion: new Set(["system", "reduce"]),
});
const PRODUCTION_STUDIO_ORIGIN = "https://webstudio.omnibioai.org";
const MESSAGE_KEYS = ["accent", "density", "motion", "theme", "type"];
const DEV_STUDIO_ORIGIN = typeof __OBAI_STUDIO_ORIGIN__ === "string" ? __OBAI_STUDIO_ORIGIN__ : undefined;

function configuredStudioOrigin(value: string | undefined): string | null {
  if (!value) return null;
  try {
    const parsed = new URL(value);
    if (!/^https?:$/.test(parsed.protocol) || parsed.origin !== value || parsed.pathname !== "/" || parsed.search || parsed.hash) return null;
    return parsed.origin;
  } catch {
    return null;
  }
}

export function normalizeAppearance(values: Record<string, unknown> = {}, systemTheme = "dark") {
  return {
    theme: ALLOWED.theme.has(values.theme as string) ? values.theme as string : (ALLOWED.theme.has(systemTheme) ? systemTheme : DEFAULTS.theme),
    accent: ALLOWED.accent.has(values.accent as string) ? values.accent as string : DEFAULTS.accent,
    density: ALLOWED.density.has(values.density as string) ? values.density as string : DEFAULTS.density,
    motion: ALLOWED.motion.has(values.motion as string) ? values.motion as string : DEFAULTS.motion,
  };
}

export function appearanceFromSearch(search: string, systemTheme = "dark") {
  const params = new URLSearchParams(search);
  return normalizeAppearance({
    theme: params.get("obai_theme"),
    accent: params.get("obai_accent"),
    density: params.get("obai_density"),
    motion: params.get("obai_motion"),
  }, systemTheme);
}

export function applyAppearance(root: HTMLElement, appearance: Record<string, unknown>) {
  const normalized = normalizeAppearance(appearance);
  root.dataset.theme = normalized.theme;
  root.dataset.accent = normalized.accent;
  root.dataset.density = normalized.density;
  root.dataset.motion = normalized.motion;
  return normalized;
}

export function parseThemeMessage(data: unknown) {
  if (!data || typeof data !== "object" || Array.isArray(data)) return null;
  const message = data as Record<string, unknown>;
  const keys = Object.keys(message).sort();
  if (keys.length !== MESSAGE_KEYS.length || keys.some((key, index) => key !== [...MESSAGE_KEYS].sort()[index])) return null;
  if (message.type !== EMBED_THEME_EVENT) return null;
  if (!Object.entries(ALLOWED).every(([key, values]) => values.has(message[key] as string))) return null;
  return { theme: message.theme as string, accent: message.accent as string, density: message.density as string, motion: message.motion as string };
}

export function connectAppearanceMessages(
  win: Window = window,
  doc: Document = document,
  devStudioOrigin: string | undefined = DEV_STUDIO_ORIGIN,
) {
  if (win.parent === win || !doc.referrer) return () => {};
  let parentOrigin: string;
  try {
    parentOrigin = new URL(doc.referrer).origin;
  } catch {
    return () => {};
  }
  const allowedOrigins = new Set([PRODUCTION_STUDIO_ORIGIN]);
  const configuredOrigin = configuredStudioOrigin(devStudioOrigin);
  if (configuredOrigin) allowedOrigins.add(configuredOrigin);
  if (!allowedOrigins.has(parentOrigin)) return () => {};

  const expectedSource = win.parent;
  const onMessage = (event: MessageEvent) => {
    if (event.origin !== parentOrigin || event.source !== expectedSource) return;
    const appearance = parseThemeMessage(event.data);
    if (appearance) applyAppearance(doc.documentElement, appearance);
  };
  win.addEventListener("message", onMessage);
  win.parent.postMessage({ type: EMBED_READY_EVENT }, parentOrigin);
  return () => win.removeEventListener("message", onMessage);
}
