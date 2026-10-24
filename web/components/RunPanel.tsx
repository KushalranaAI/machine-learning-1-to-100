"use client";
import { useRef, useState } from "react";
import AskButton from "./ai/AskButton";
import Icon from "./ui/Icon";

type Line = { kind: "out" | "err"; text: string };

const REPO = process.env.NEXT_PUBLIC_REPO_URL;

/** The hosted (GitHub Pages) copy has no server, so scripts can't run there. Say how to run them. */
function RunLocally({ script }: { script: string }) {
  return (
    <div className="notice info run-local">
      <Icon name="terminal" size={16} />
      <div>
        <b>Running code needs the local app.</b> This hosted copy is read-only. To run <code>{script.split("/").pop()}</code> and see its plots, clone the repository and start the app:
        <pre className="small" style={{ margin: "8px 0 0" }}>{`git clone ${REPO}.git\ncd machine-learning-1-to-100\npython start.py`}</pre>
      </div>
    </div>
  );
}

export default function RunPanel(props: { script: string; args?: string[]; label?: string }) {
  return process.env.NEXT_PUBLIC_STATIC_EXPORT === "1" ? <RunLocally script={props.script} /> : <LiveRunPanel {...props} />;
}

/** Runs a course script with the learner's local Python and shows its output and figures live. */
function LiveRunPanel({ script, args = [], label = "Run" }: { script: string; args?: string[]; label?: string }) {
  const [lines, setLines] = useState<Line[]>([]);
  const [images, setImages] = useState<{ name: string; data: string }[]>([]);
  const [status, setStatus] = useState<"idle" | "running" | "ok" | "failed">("idle");
  const termRef = useRef<HTMLDivElement>(null);

  async function run() {
    setLines([]);
    setImages([]);
    setStatus("running");
    try {
      const res = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ script, args }) });
      if (!res.ok || !res.body) {
        setLines([{ kind: "err", text: await res.text() }]);
        setStatus("failed");
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          if (!part) continue;
          const msg = JSON.parse(part);
          if (msg.type === "out" || msg.type === "err") {
            setLines((prev) => [...prev, { kind: msg.type, text: msg.text }]);
            requestAnimationFrame(() => termRef.current?.scrollTo({ top: termRef.current.scrollHeight }));
          } else if (msg.type === "img") setImages((prev) => [...prev, { name: msg.name, data: msg.data }]);
          else if (msg.type === "exit") setStatus(msg.code === 0 ? "ok" : "failed");
        }
      }
    } catch (e) {
      setLines((prev) => [...prev, { kind: "err", text: String(e) }]);
      setStatus("failed");
    }
  }

  const output = lines.map((l) => l.text).join("");
  return (
    <div className="run-panel">
      <div className="run-bar">
        <button type="button" className="btn primary" onClick={run} disabled={status === "running"}>
          <Icon name="play" size={16} /> {status === "running" ? "Running…" : label}
        </button>
        <code className="run-cmd">python {script.split("/").pop()} {args.join(" ")}</code>
        {status === "ok" && <span className="status-text good"><Icon name="circle-check" size={16} /> finished</span>}
        {status === "failed" && <span className="status-text bad"><Icon name="alert" size={16} /> failed - see output</span>}
        {(status === "ok" || status === "failed") && output && (
          <AskButton
            label={status === "failed" ? "Help me fix this" : "Explain the output"}
            quote={output.slice(-4000)}
            prompt={status === "failed" ? `This script (${script.split("/").pop()}) failed. What went wrong and how do I fix it?` : `Explain what this output of ${script.split("/").pop()} means and what I should learn from it.`}
          />
        )}
      </div>
      {lines.length > 0 && (
        <div className="terminal" ref={termRef} tabIndex={0} aria-label="Script output">
          {lines.map((l, i) => <span key={i} className={l.kind === "err" ? "err" : ""}>{l.text}</span>)}
        </div>
      )}
      {images.length > 0 && (
        <div className="figures">
          {images.map((img) => (
            // eslint-disable-next-line @next/next/no-img-element
            <figure key={img.name}><img src={img.data} alt={img.name} /><figcaption>{img.name}</figcaption></figure>
          ))}
        </div>
      )}
    </div>
  );
}
