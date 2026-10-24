"use client";
import { useEffect, useState } from "react";
import { clearChats } from "@/lib/ai/chats";
import { listModels } from "@/lib/ai/client";
import type { ProviderId } from "@/lib/ai/providers";
import { availableProviders, buildConnection, forgetAllKeys, isReady, moveKeys, saveKey, settingsStore, useConnection, useIsLocal } from "@/lib/ai/settings";
import Icon from "../ui/Icon";

type Status = { tone: "idle" | "busy" | "ok" | "bad"; message?: string; hint?: string };

export default function AISettings({ onDone }: { onDone: () => void }) {
  const { settings, keys, connection } = useConnection();
  const isLocal = useIsLocal();
  const offered = availableProviders(isLocal); // Azure OpenAI only shows up when this page is served from localhost
  const provider = offered.find((p) => p.id === settings.provider) ?? offered[0];
  const [apiKey, setApiKey] = useState(keys[provider.id] ?? "");
  const [baseUrl, setBaseUrl] = useState(settings.baseUrls[provider.id] ?? provider.baseUrl);
  const [showKey, setShowKey] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [status, setStatus] = useState<Status>({ tone: "idle" });

  // switching provider loads that provider's saved values
  useEffect(() => {
    setApiKey(keys[provider.id] ?? "");
    setBaseUrl(settings.baseUrls[provider.id] ?? provider.baseUrl);
    setModels([]);
    setStatus({ tone: "idle" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider.id]);

  const update = (patch: Partial<typeof settings>) => settingsStore.set((s) => ({ ...s, ...patch }));

  async function connect() {
    setStatus({ tone: "busy", message: "Connecting…" });
    const key = apiKey.trim();
    const base = baseUrl.trim() || provider.baseUrl;
    saveKey(provider.id, key, settings.remember);
    update({ baseUrls: { ...settings.baseUrls, [provider.id]: base } });
    const probe = buildConnection({ ...settings, baseUrls: { ...settings.baseUrls, [provider.id]: base } }, { [provider.id]: key });
    try {
      const found = await listModels(probe);
      setModels(found);
      const current = settings.models[provider.id] ?? provider.defaultModel;
      const preferred = found.length && !found.includes(current) ? (found.find((m) => m === provider.defaultModel) ?? found[0]) : current;
      update({ models: { ...settings.models, [provider.id]: preferred }, configured: true });
      setStatus({
        tone: "ok",
        message: found.length ? `Connected - ${found.length} models available.` : provider.kind === "azure" ? "Endpoint and key accepted. Now enter your deployment name below." : "Connected, but no models were listed. Type a model name below.",
      });
    } catch (e) {
      const err = e as { message?: string; hint?: string };
      setStatus({ tone: "bad", message: err.message ?? "Could not connect.", hint: err.hint });
    }
  }

  const model = settings.models[provider.id] ?? provider.defaultModel;
  return (
    <div className="ai-settings">
      <div className="ai-section">
        <h3>Choose your AI</h3>
        <div className="provider-grid" role="radiogroup" aria-label="AI provider">
          {offered.map((p) => (
            <button
              key={p.id}
              type="button"
              role="radio"
              aria-checked={p.id === provider.id}
              className={`provider ${p.id === provider.id ? "selected" : ""}`}
              onClick={() => update({ provider: p.id as ProviderId })}
            >
              <span className="provider-top">
                <b>{p.label}</b>
                <span className={`pill ${p.local ? "good" : ""}`}>{p.local ? "local" : "cloud"}</span>
              </span>
              <span className="muted">{p.tagline}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="ai-section">
        <h3>Connect {provider.label}</h3>
        <p className="muted small">{provider.setup}</p>

        {provider.editableBaseUrl && (
          <label className="field">
            <span>{provider.baseLabel ?? "Server address"}</span>
            <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} spellCheck={false} autoComplete="off" placeholder={provider.basePlaceholder ?? provider.baseUrl} />
          </label>
        )}

        {provider.kind === "azure" && (
          <label className="field">
            <span>API version</span>
            <input value={settings.azureApiVersion} onChange={(e) => update({ azureApiVersion: e.target.value })} spellCheck={false} autoComplete="off" placeholder="2024-10-21" />
            <span className="muted small">Leave the default unless your deployment needs a different version.</span>
          </label>
        )}

        {(provider.needsKey || provider.id === "custom") && (
          <label className="field">
            <span>
              {provider.keyLabel}
              {provider.keyUrl && (
                <a href={provider.keyUrl} target="_blank" rel="noreferrer" className="field-link">
                  Get a key <Icon name="external" size={12} />
                </a>
              )}
            </span>
            <span className="input-row">
              <input
                type={showKey ? "text" : "password"}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                spellCheck={false}
                autoComplete="off"
                placeholder={provider.needsKey ? "Paste your key" : "Optional"}
              />
              <button type="button" className="icon-btn" onClick={() => setShowKey(!showKey)} aria-label={showKey ? "Hide key" : "Show key"}>
                <Icon name={showKey ? "eye-off" : "eye"} />
              </button>
            </span>
          </label>
        )}

        <div className="row">
          <button type="button" className="btn primary" onClick={connect} disabled={status.tone === "busy" || (provider.needsKey && !apiKey.trim()) || (provider.kind === "azure" && !baseUrl.trim())}>
            <Icon name="zap" size={16} /> {status.tone === "busy" ? "Connecting…" : provider.needsKey || provider.id === "custom" || provider.editableBaseUrl ? "Save & test connection" : "Test connection"}
          </button>
          {keys[provider.id] && provider.needsKey && (
            <button type="button" className="btn" onClick={() => { saveKey(provider.id, "", settings.remember); setApiKey(""); setStatus({ tone: "idle" }); }}>
              Remove key
            </button>
          )}
        </div>
        {status.message && (
          <div className={`notice ${status.tone}`} role="status">
            <Icon name={status.tone === "ok" ? "circle-check" : status.tone === "bad" ? "alert" : "info"} size={16} />
            <div>
              {status.message}
              {status.hint && <div className="small">{status.hint}</div>}
            </div>
          </div>
        )}

        <label className="field">
          <span>{provider.modelLabel ?? "Model"}</span>
          <input
            list="ai-models"
            value={model}
            onChange={(e) => update({ models: { ...settings.models, [provider.id]: e.target.value } })}
            spellCheck={false}
            autoComplete="off"
            placeholder={provider.id === "custom" ? "Name of a model your server has loaded" : provider.kind === "azure" ? "e.g. my-gpt-4o" : provider.defaultModel}
          />
          <datalist id="ai-models">{models.map((m) => <option key={m} value={m} />)}</datalist>
          <span className="muted small">{provider.modelHelp ?? "Pick from the list after connecting, or type any model name."}</span>
        </label>
      </div>

      <div className="ai-section">
        <h3>Where your key is stored</h3>
        <div className="choice" role="radiogroup" aria-label="Key storage">
          {[
            { v: true, t: "On this device", d: "Stays in this browser until you remove it." },
            { v: false, t: "This session only", d: "Forgotten when you close the tab." },
          ].map((o) => (
            <button
              key={String(o.v)}
              type="button"
              role="radio"
              aria-checked={settings.remember === o.v}
              className={`choice-opt ${settings.remember === o.v ? "selected" : ""}`}
              onClick={() => { update({ remember: o.v }); moveKeys(o.v); }}
            >
              <b>{o.t}</b>
              <span className="muted small">{o.d}</span>
            </button>
          ))}
        </div>
        <div className="notice info">
          <Icon name="shield" size={16} />
          <div className="small">
            Keys are saved only in your browser&apos;s storage and sent straight to the provider you chose - this app&apos;s server never sees them, and they are never written to the repository.
            Anyone who can use this browser profile can read them, so avoid saving keys on a shared computer, and prefer a key with a spending limit.
          </div>
        </div>
      </div>

      <div className="ai-section">
        <h3>What the tutor can see</h3>
        <label className="switch-row">
          <input type="checkbox" className="switch" checked={settings.shareContext} onChange={(e) => update({ shareContext: e.target.checked })} />
          <span>
            <b>Share the current page with my questions</b>
            <span className="muted small block">Sends the lesson text / code you are viewing so answers match this course. {provider.local ? "Stays on your computer." : `Goes to ${provider.label}.`}</span>
          </span>
        </label>
        <label className="field">
          <span>Amount of page text to share</span>
          <select value={settings.contextChars ?? ""} onChange={(e) => update({ contextChars: e.target.value ? Number(e.target.value) : null })}>
            <option value="">Automatic for {provider.label} ({provider.contextChars.toLocaleString()} characters)</option>
            <option value="2500">Small - 2,500 (tiny local models)</option>
            <option value="6000">Medium - 6,000</option>
            <option value="16000">Large - 16,000</option>
            <option value="40000">Very large - 40,000 (big-context models)</option>
          </select>
        </label>
      </div>

      <div className="ai-section">
        <h3>Reset</h3>
        <button
          type="button"
          className="btn danger"
          onClick={() => {
            if (!confirm("Remove every saved API key and all saved AI chats from this browser?")) return;
            forgetAllKeys();
            clearChats();
            setApiKey("");
            setStatus({ tone: "idle" });
          }}
        >
          <Icon name="trash" size={16} /> Forget all keys &amp; chats
        </button>
      </div>

      <div className="ai-settings-foot">
        <button type="button" className="btn primary" onClick={() => { update({ configured: true }); onDone(); }} disabled={!isReady(connection)}>
          Done
        </button>
      </div>
    </div>
  );
}
