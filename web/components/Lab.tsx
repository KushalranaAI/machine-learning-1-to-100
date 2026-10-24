"use client";
import { useState } from "react";
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
  const Widget = WIDGETS[id];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(200px, 240px) 1fr", gap: 18, alignItems: "start" }}>
      <div className="card" style={{ padding: 10, position: "sticky", top: 12 }}>
        {LAB.map((w, i) => (
          <div key={w.id}>
            {(i === 0 || LAB[i - 1].topic !== w.topic) && <div className="stat" style={{ margin: "8px 6px 2px" }}>{w.topic}</div>}
            <button className={`tab ${id === w.id ? "active" : ""}`} style={{ display: "block", width: "100%", textAlign: "left", borderBottom: "none", padding: "5px 8px" }} onClick={() => setId(w.id)}>{w.title}</button>
          </div>
        ))}
      </div>
      <div style={{ minWidth: 0 }}><Widget /></div>
    </div>
  );
}
