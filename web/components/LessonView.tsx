"use client";
import { useState } from "react";
import Markdown from "./Markdown";
import Mermaid from "./Mermaid";
import RunPanel from "./RunPanel";
import { setDone, useProgress } from "./progress";
import { WIDGETS } from "./widgets";

type Snippet = { spec: string; file: string; symbol: string | null; language: string; code: string };

type Props = {
  id: string;
  body: string;
  workflow: string;
  widgets: string[];
  code: Snippet[];
  demo: string | null;
};

export default function LessonView({ id, body, workflow, widgets, code, demo }: Props) {
  const tabs = [
    ["learn", "📖 Learn"],
    ...(workflow ? [["workflow", "🔀 Workflow"]] : []),
    ...(widgets.length ? [["interactive", "🎛️ Interactive"]] : []),
    ...(code.length ? [["code", "🧩 From-scratch code"]] : []),
    ...(demo ? [["run", "▶ Run demo"]] : []),
  ];
  const [tab, setTab] = useState(tabs[0][0]);
  const progress = useProgress();
  const done = !!progress[id];

  return (
    <>
      <div className="tabs" role="tablist">
        {tabs.map(([key, label]) => (
          <button key={key} role="tab" className={`tab ${tab === key ? "active" : ""}`} onClick={() => setTab(key)}>{label}</button>
        ))}
        <span style={{ flex: 1 }} />
        <button className="btn" style={{ alignSelf: "center" }} onClick={() => setDone(id, !done)}>
          {done ? "✓ Completed" : "Mark as complete"}
        </button>
      </div>

      {tab === "learn" && <Markdown>{body}</Markdown>}
      {tab === "workflow" && (
        <div>
          <p className="muted">How this step fits into a real machine-learning workflow:</p>
          <Mermaid chart={workflow} />
        </div>
      )}
      {tab === "interactive" &&
        widgets.map((w) => {
          const Widget = WIDGETS[w];
          return Widget ? <Widget key={w} /> : <p key={w} className="muted">Widget “{w}” is not available.</p>;
        })}
      {tab === "code" && (
        <div>
          <p className="muted">
            The exact code from this repository - no libraries hiding the math. Open the file in your editor to experiment.
          </p>
          {code.map((c) => (
            <div key={c.spec} style={{ marginBottom: 22 }}>
              <div className="stat">{c.file}{c.symbol ? ` → ${c.symbol}` : ""}</div>
              <Markdown>{"```" + c.language + "\n" + c.code + "\n```"}</Markdown>
            </div>
          ))}
        </div>
      )}
      {tab === "run" && demo && (
        <div>
          <p className="muted">
            Runs <code>{demo}</code> with your local Python (the same as running it in a terminal). Printed output and saved figures appear below.
          </p>
          <RunPanel script={demo} />
        </div>
      )}
    </>
  );
}
