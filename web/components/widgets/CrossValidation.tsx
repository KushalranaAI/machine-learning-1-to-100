"use client";
import { useState } from "react";

export default function CrossValidation() {
  const [k, setK] = useState(5);
  const [fold, setFold] = useState(0);
  const [mode, setMode] = useState<"kfold" | "holdout" | "timeseries">("kfold");
  const n = 30;
  const role = (i: number): "train" | "val" | "test" | "unused" => {
    if (i >= 25) return "test";
    if (mode === "holdout") return i >= 20 ? "val" : "train";
    if (mode === "timeseries") {
      const size = Math.floor(25 / (k + 1));
      const end = size * (fold + 1);
      return i < end ? "train" : i < end + size ? "val" : "unused";
    }
    const size = 25 / k;
    return i >= fold * size && i < (fold + 1) * size ? "val" : "train";
  };
  const color = { train: "#4263eb", val: "#e8590c", test: "#2b8a3e", unused: "rgba(128,128,128,0.25)" };
  const folds = mode === "holdout" ? 1 : k;
  return (
    <div className="widget">
      <h3>Hold-out, k-fold and time-series validation</h3>
      <div className="controls">
        <label>scheme <select value={mode} onChange={(e) => { setMode(e.target.value as typeof mode); setFold(0); }}>
          <option value="holdout">train / validation / test split</option><option value="kfold">k-fold cross-validation</option><option value="timeseries">time-series (expanding window)</option>
        </select></label>
        {mode !== "holdout" && <label>k = {k} <input type="range" min={2} max={10} value={k} onChange={(e) => { setK(Number(e.target.value)); setFold(0); }} /></label>}
        {mode !== "holdout" && <label>fold {fold + 1} of {folds} <input type="range" min={0} max={folds - 1} value={fold} onChange={(e) => setFold(Number(e.target.value))} /></label>}
      </div>
      <div style={{ display: "flex", gap: 3, margin: "14px 0" }}>
        {Array.from({ length: n }, (_, i) => (
          <div key={i} title={`sample block ${i + 1}: ${role(i)}`} style={{ flex: 1, height: 44, borderRadius: 4, background: color[role(i)] }} />
        ))}
      </div>
      <div className="controls" style={{ fontSize: "0.85rem" }}>
        {Object.entries(color).map(([k2, c]) => <span key={k2}><span style={{ display: "inline-block", width: 12, height: 12, background: c, borderRadius: 3, marginRight: 4 }} />{k2}</span>)}
      </div>
      <p className="muted" style={{ fontSize: "0.88rem" }}>
        The <b>test set</b> (green) is locked away until the very end. In k-fold CV every sample is used for validation exactly once, and the score is the average of the k folds - a far more stable estimate than one split.
        For time series, never train on the future: the window only grows forward.
      </p>
    </div>
  );
}
