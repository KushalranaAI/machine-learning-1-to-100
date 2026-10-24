"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { ALGORITHMS, type Task } from "@/lib/algorithms";
import type { PageContext } from "@/lib/ai/prompts";
import { usePageContext } from "./ai/AIProvider";
import AskButton from "./ai/AskButton";
import Icon from "./ui/Icon";

const TASKS: (Task | "All")[] = ["All", "Regression", "Classification", "Clustering", "Dimensionality reduction", "Anomaly detection", "Rules", "Reinforcement learning"];

function Meter({ n, label }: { n: number; label: string }) {
  return (
    <span className="meter" role="img" aria-label={`${label}: ${n} of 3`}>
      {[1, 2, 3].map((i) => <i key={i} className={i <= n ? "on" : ""} />)}
    </span>
  );
}

function Toggle({ on, onChange, children }: { on: boolean; onChange: (v: boolean) => void; children: React.ReactNode }) {
  return (
    <button type="button" className={`filter-chip ${on ? "on" : ""}`} aria-pressed={on} onClick={() => onChange(!on)}>
      {on && <Icon name="check" size={13} />} {children}
    </button>
  );
}

export default function AlgorithmMap() {
  const [task, setTask] = useState<Task | "All">("All");
  const [needReadable, setNeedReadable] = useState(false);
  const [bigData, setBigData] = useState(false);
  const [nonlinear, setNonlinear] = useState(false);
  const [q, setQ] = useState("");

  const rows = useMemo(
    () =>
      ALGORITHMS.filter(
        (a) =>
          (task === "All" || a.tasks.includes(task)) &&
          (!needReadable || a.interpretable === 3) &&
          (!bigData || a.bigData === 3) &&
          (!nonlinear || a.nonlinear) &&
          (q === "" || `${a.name} ${a.idea} ${a.strengths}`.toLowerCase().includes(q.toLowerCase())),
      ),
    [task, needReadable, bigData, nonlinear, q],
  );

  const filters = [task !== "All" && task, needReadable && "must be explainable", bigData && "millions of rows", nonlinear && "non-linear patterns", q && `matching “${q}”`].filter(Boolean).join(", ");
  const context = useMemo<PageContext>(
    () => ({
      key: "map",
      kind: "map",
      title: "Algorithm map",
      focus: filters ? `the algorithm comparison table, filtered by: ${filters}` : "the full algorithm comparison table",
      text: `Algorithms currently shown (${rows.length} of ${ALGORITHMS.length}):\n` + rows.map((a) => `- ${a.name} [${a.tasks.join(", ")}]: ${a.idea}. Objective: ${a.objective}. Cost: ${a.train}. Strengths: ${a.strengths}. Weaknesses: ${a.weaknesses}. Needs scaling: ${a.scale}. Interpretable ${a.interpretable}/3, big-data ${a.bigData}/3.`).join("\n"),
    }),
    [rows, filters],
  );
  usePageContext(context);

  return (
    <div>
      <div className="filters card">
        <div className="filter-row">
          <span className="filter-label">I want to</span>
          <div className="chips" role="group" aria-label="Task">
            {TASKS.map((t) => (
              <button key={t} type="button" className={`filter-chip ${task === t ? "on" : ""}`} aria-pressed={task === t} onClick={() => setTask(t)}>{t === "All" ? "Anything" : t}</button>
            ))}
          </div>
        </div>
        <div className="filter-row">
          <span className="filter-label">It must be</span>
          <div className="chips">
            <Toggle on={needReadable} onChange={setNeedReadable}>explainable</Toggle>
            <Toggle on={bigData} onChange={setBigData}>fast on millions of rows</Toggle>
            <Toggle on={nonlinear} onChange={setNonlinear}>able to capture non-linear patterns</Toggle>
          </div>
        </div>
        <div className="filter-row">
          <label className="lab-search wide">
            <Icon name="search" size={15} />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search, e.g. trees, margin, probability" aria-label="Search algorithms" />
          </label>
          <span className="muted small">{rows.length} of {ALGORITHMS.length} algorithms</span>
          <AskButton label="Help me choose" prompt={`Based on the algorithms currently shown${filters ? ` (filtered by ${filters})` : ""}, how would I pick between them? Ask me about my data if you need to.`} />
        </div>
      </div>

      <div className="algo-grid">
        {rows.map((a) => (
          <article key={a.name} className="card algo" data-askable="">
            <header>
              <Link href={`/learn/${a.lesson}`} className="algo-name">{a.name}</Link>
              <div className="chips">{a.tasks.map((t) => <span key={t} className="pill">{t}</span>)}</div>
            </header>
            <p className="algo-idea">{a.idea}</p>
            <code className="algo-math">{a.objective}</code>
            <dl className="algo-stats">
              <div><dt>Explainable</dt><dd><Meter n={a.interpretable} label="Explainability" /></dd></div>
              <div><dt>Scales</dt><dd><Meter n={a.bigData} label="Scalability" /></dd></div>
              <div><dt>Non-linear</dt><dd>{a.nonlinear ? "yes" : "no"}</dd></div>
              <div><dt>Needs scaling</dt><dd>{a.scale ? "yes" : "no"}</dd></div>
              <div className="wide"><dt>Training cost</dt><dd><code>{a.train}</code></dd></div>
            </dl>
            <ul className="proscons">
              <li className="pro"><Icon name="check" size={15} label="Strength" /> {a.strengths}</li>
              <li className="con"><Icon name="alert" size={15} label="Weakness" /> {a.weaknesses}</li>
            </ul>
            <footer>
              <Link href={`/learn/${a.lesson}`} className="btn sm">Open lesson <Icon name="arrow-right" size={14} /></Link>
              <AskButton label="Ask AI" prompt={`Tell me when I should (and shouldn't) use ${a.name}, compared with its closest alternatives in this course.`} />
            </footer>
          </article>
        ))}
      </div>
      {rows.length === 0 && (
        <div className="card empty">
          <Icon name="info" size={20} />
          <p>No algorithm matches every filter - loosen one. That&apos;s a real trade-off, not a bug!</p>
        </div>
      )}
    </div>
  );
}
