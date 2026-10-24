"use client";
import { useEffect, useMemo, useState } from "react";
import type { PageContext } from "@/lib/ai/prompts";
import { useResolvedTheme } from "@/lib/theme";
import { usePageContext } from "./ai/AIProvider";
import AskButton from "./ai/AskButton";
import Icon from "./ui/Icon";
import { WIDGETS } from "./widgets";

export const LAB: { id: string; title: string; topic: string }[] = [
  { id: "matrix-transform", title: "Matrices as transformations", topic: "Math" },
  { id: "distributions", title: "Distributions & the central limit theorem", topic: "Math" },
  { id: "gradient-descent", title: "Gradient descent explorer", topic: "Math" },
  { id: "outliers", title: "Outlier detection: z-score vs IQR", topic: "Data" },
  { id: "scaling", title: "Feature scaling", topic: "Features" },
  { id: "pca", title: "Principal component analysis", topic: "Features" },
  { id: "cross-validation", title: "Validation schemes", topic: "Evaluation" },
  { id: "roc", title: "Thresholds, confusion matrix, ROC", topic: "Evaluation" },
  { id: "bias-variance", title: "Bias-variance trade-off", topic: "Evaluation" },
  { id: "linear-regression", title: "Least squares playground", topic: "Regression" },
  { id: "classifier-playground", title: "Classifier playground (6 models)", topic: "Classification" },
  { id: "tree-split", title: "How a tree chooses a split", topic: "Classification" },
  { id: "boosting", title: "Gradient boosting stage by stage", topic: "Ensembles" },
  { id: "kmeans", title: "k-means step by step", topic: "Unsupervised" },
  { id: "dbscan", title: "DBSCAN density clustering", topic: "Unsupervised" },
  { id: "q-learning", title: "Q-learning grid world", topic: "Reinforcement learning" },
];

export default function Lab() {
  const [id, setId] = useState(LAB[0].id);
  const [q, setQ] = useState("");
  const theme = useResolvedTheme();
  const Widget = WIDGETS[id];
  const current = LAB.find((w) => w.id === id)!;

  useEffect(() => {
    const h = window.location.hash.slice(1);
    if (LAB.some((w) => w.id === h)) setId(h);
  }, []);
  const choose = (next: string) => {
    setId(next);
    history.replaceState(null, "", `#${next}`);
  };

  const context = useMemo<PageContext>(
    () => ({
      key: `lab:${id}`,
      kind: "lab",
      title: current.title,
      focus: `the interactive visual “${current.title}” (${current.topic})`,
      text: `Interactive lab visual: ${current.title}\nTopic: ${current.topic}\nThe learner can drag points and move sliders; the visual runs a real implementation of the algorithm in the browser.`,
    }),
    [id, current],
  );
  usePageContext(context);

  const shown = LAB.filter((w) => `${w.title} ${w.topic}`.toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="lab">
      <aside className="lab-nav card" aria-label="Visuals">
        <label className="lab-search">
          <Icon name="search" size={15} />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter visuals" aria-label="Filter visuals" />
        </label>
        <div className="lab-list">
          {shown.map((w, i) => (
            <div key={w.id}>
              {(i === 0 || shown[i - 1].topic !== w.topic) && <div className="lab-topic">{w.topic}</div>}
              <button type="button" className={`lab-item ${id === w.id ? "active" : ""}`} aria-current={id === w.id ? "true" : undefined} onClick={() => choose(w.id)}>
                {w.title}
              </button>
            </div>
          ))}
          {shown.length === 0 && <p className="muted small" style={{ padding: 8 }}>No visual matches.</p>}
        </div>
      </aside>
      <section className="lab-stage" style={{ minWidth: 0 }}>
        <header className="widget-head">
          <div>
            <span className="muted small">{current.topic}</span>
            <h2>{current.title}</h2>
          </div>
          <AskButton label="Ask about this" prompt={`Explain what the interactive visual “${current.title}” demonstrates, and suggest three experiments to try with its controls, with what I should notice each time.`} />
        </header>
        {Widget ? <Widget key={`${id}-${theme}`} /> : null}
      </section>
    </div>
  );
}
