// AI settings and API keys. Both live only in this browser:
//  - settings (provider, model, base URLs) in localStorage
//  - keys in localStorage ("remember on this device") or sessionStorage ("until I close the tab")
import { useSyncExternalStore } from "react";
import { createStore, jsonStore, useStore } from "../localstore";
import type { Connection } from "./client";
import { type ProviderId, PROVIDERS, providerById } from "./providers";

export type AISettings = {
  provider: ProviderId;
  models: Partial<Record<ProviderId, string>>;
  baseUrls: Partial<Record<ProviderId, string>>;
  remember: boolean;
  shareContext: boolean;
  contextChars: number | null; // null = provider default
  configured: boolean; // the learner has completed setup (a successful test, or pressed Done)
  azureApiVersion: string;
};

const DEFAULTS: AISettings = { provider: "ollama", models: {}, baseUrls: {}, remember: true, shareContext: true, contextChars: null, configured: false, azureApiVersion: "2024-10-21" };

export const settingsStore = jsonStore<AISettings>("ml100-ai-settings", DEFAULTS);
export const useAISettings = () => useStore(settingsStore);

// ------------------------------------------------------------------------------------------ keys

type Keys = Partial<Record<ProviderId, string>>;
const KEYS = "ml100-ai-keys";
const parseKeys = (raw: string | null): Keys => {
  try {
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
};

const keyStore = createStore<Keys>({
  server: {},
  // session keys win over device keys, so "this session only" can shadow a stored one
  read: () => `${sessionStorage.getItem(KEYS) ?? ""}\u0000${localStorage.getItem(KEYS) ?? ""}`,
  parse(raw) {
    const [session = "", local = ""] = (raw ?? "").split("\u0000");
    return { ...parseKeys(local || null), ...parseKeys(session || null) };
  },
  write: () => {},
});

export const useAIKeys = () => useStore(keyStore);

export function saveKey(provider: ProviderId, key: string, remember: boolean) {
  const session = parseKeys(sessionStorage.getItem(KEYS));
  const local = parseKeys(localStorage.getItem(KEYS));
  delete session[provider];
  delete local[provider];
  if (key) (remember ? local : session)[provider] = key;
  persist(session, local);
}

export function forgetAllKeys() {
  persist({}, {});
}

/** Move every stored key between "this device" and "this session" when the learner flips the switch. */
export function moveKeys(remember: boolean) {
  const all = keyStore.get();
  persist(remember ? {} : all, remember ? all : {});
}

function persist(session: Keys, local: Keys) {
  const put = (s: Storage, v: Keys) => (Object.keys(v).length ? s.setItem(KEYS, JSON.stringify(v)) : s.removeItem(KEYS));
  try {
    put(sessionStorage, session);
    put(localStorage, local);
  } catch {
    /* storage blocked */
  }
  // notify subscribers: the keyStore re-reads storage on set()
  keyStore.set({ ...local, ...session });
}

// ------------------------------------------------------------------------------------- connection

export function buildConnection(s: AISettings, keys: Keys, provider: ProviderId = s.provider): Connection {
  const p = providerById(provider);
  return {
    provider,
    kind: p.kind,
    baseUrl: s.baseUrls[provider] || p.baseUrl,
    apiKey: keys[provider] ?? "",
    model: s.models[provider] ?? p.defaultModel,
    apiVersion: s.azureApiVersion,
  };
}

// ----------------------------------------------------------------------------- local-only providers

const LOCAL_HOSTS = ["localhost", "127.0.0.1", "[::1]", "::1"];
const subscribeNever = () => () => {};

/** True when the app is being served from this machine. Some providers (Azure OpenAI) are only offered then. */
export function useIsLocal(): boolean {
  return useSyncExternalStore(subscribeNever, () => LOCAL_HOSTS.includes(window.location.hostname), () => false);
}

/** Providers this page may offer: everything on localhost, everything except local-only ones elsewhere. */
export function availableProviders(isLocal: boolean) {
  return PROVIDERS.filter((p) => isLocal || !p.localOnly);
}

export function isReady(c: Connection): boolean {
  const p = providerById(c.provider);
  return !!c.model && (!p.needsKey || !!c.apiKey);
}

export const contextBudget = (s: AISettings) => s.contextChars ?? providerById(s.provider).contextChars;

/** Hook: the active connection plus whether it is ready to use. */
export function useConnection() {
  const settings = useAISettings();
  const keys = useAIKeys();
  const isLocal = useIsLocal();
  const connection = buildConnection(settings, keys);
  const allowed = isLocal || !providerById(settings.provider).localOnly; // a saved Azure choice is ignored on a hosted copy
  return { settings, keys, connection, ready: allowed && settings.configured && isReady(connection) };
}

export { PROVIDERS };
