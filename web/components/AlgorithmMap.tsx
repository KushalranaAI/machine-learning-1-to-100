"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { ALGORITHMS, type Task } from "@/lib/algorithms";

const TASKS: (Task | "All")[] = ["All", "Regression", "Classification", "Clustering", "Dimensionality reduction", "Anomaly detection", "Rules", "Reinforcement learning"];
const dots = (n: number) => "●".repeat(n) + "○".repeat(3 - n);

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

  return (
    <div className="card">
      <div className="controls" style={{ flexWrap: "wrap", gap: 10 }}>
        <label>
          Task{" "}
          <select value={task} onChange={(e) => setTask(e.target.value as Task | "All")}>
            {TASKS.map((t) => <option key={t}>{t}</option>)}
          </select>
        </label>
        <label><input type="checkbox" checked={needReadable} onChange={(e) => setNeedReadable(e.target.checked)} /> must be explainable</label>
        <label><input type="checkbox" checked={bigData} onChange={(e) => setBigData(e.target.checked)} /> millions of rows</label>
        <label><input type="checkbox" checked={nonlinear} onChange={(e) => setNonlinear(e.target.checked)} /> non-linear patterns</label>
        <label>Search <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. trees" /></label>
        <span className="muted">{rows.length} of {ALGORITHMS.length}</span>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>Algorithm</th><th>Core idea &amp; maths</th><th>Training cost</th>
              <th title="Interpretability">Explain</th><th title="Scales to large data">Scale</th><th>Scaling?</th>
              <th>Strengths</th><th>Weaknesses</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((a) => (
              <tr key={a.name}>
                <td style={{ minWidth: 150 }}><Link href={`/learn/${a.lesson}`}><b>{a.name}</b></Link><br /><span className="muted" style={{ fontSize: "0.8rem" }}>{a.tasks.join(" · ")}</span></td>
                <td style={{ minWidth: 240 }}>{a.idea}<br /><code style={{ fontSize: "0.78rem" }}>{a.objective}</code></td>
                <td style={{ minWidth: 130 }}><code style={{ fontSize: "0.78rem" }}>{a.train}</code></td>
                <td style={{ whiteSpace: "nowrap" }}>{dots(a.interpretable)}</td>
                <td style={{ whiteSpace: "nowrap" }}>{dots(a.bigData)}</td>
                <td>{a.scale ? "yes" : "no"}</td>
                <td style={{ minWidth: 200 }}>{a.strengths}</td>
                <td style={{ minWidth: 200 }}>{a.weaknesses}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length === 0 && <p className="muted">No algorithm matches every filter - loosen one. (That is a real trade-off, not a bug!)</p>}
    </div>
  );
}
