"use client";
import { useMemo, useState } from "react";
import { fmt, rng, textColor, useCanvas } from "./lib";

export default function ROC() {
  const [sep, setSep] = useState(1.5);
  const [ratio, setRatio] = useState(0.3);
  const [t, setT] = useState(0.5);
  const data = useMemo(() => {
    const r = rng(5);
    const n = 1000;
    const nPos = Math.round(n * ratio);
    const scores = Array.from({ length: n }, (_, i) => (i < nPos ? sep / 2 : -sep / 2) + r.normal());
    return { scores: scores.map((s) => 1 / (1 + Math.exp(-s))), y: scores.map((_, i): number => (i < nPos ? 1 : 0)) };
  }, [sep, ratio]);
  const stats = useMemo(() => {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    data.scores.forEach((s, i) => { const p = s >= t; if (p && data.y[i]) tp++; else if (p) fp++; else if (data.y[i]) fn++; else tn++; });
    const precision = tp / Math.max(tp + fp, 1), recall = tp / Math.max(tp + fn, 1);
    return { tp, fp, fn, tn, precision, recall, f1: (2 * precision * recall) / Math.max(precision + recall, 1e-12), acc: (tp + tn) / data.y.length, fpr: fp / Math.max(fp + tn, 1) };
  }, [data, t]);
  const curve = useMemo(() => {
    const order = data.scores.map((s, i) => [s, data.y[i]]).sort((a, b) => b[0] - a[0]);
    const P = data.y.reduce((a: number, b) => a + b, 0), N = data.y.length - P;
    let tp = 0, fp = 0, auc = 0, prevFpr = 0, prevTpr = 0;
    const pts: [number, number][] = [[0, 0]];
    for (const [, y] of order) {
      if (y) tp++; else fp++;
      const fpr = fp / N, tpr = tp / P;
      auc += (fpr - prevFpr) * (tpr + prevTpr) / 2;
      prevFpr = fpr; prevTpr = tpr;
      pts.push([fpr, tpr]);
    }
    return { pts, auc };
  }, [data]);

  const distRef = useCanvas((ctx, w, h) => {
    // Histogram of the model's predicted probabilities, split by true class.
    const bins = 40, toX = (v: number) => 20 + v * (w - 40);
    const counts = [new Array(bins).fill(0), new Array(bins).fill(0)];
    data.scores.forEach((v, i) => counts[data.y[i]][Math.min(bins - 1, Math.floor(v * bins))]++);
    const hi = Math.max(...counts[0], ...counts[1], 1);
    [["rgba(66,99,235,0.55)", 0], ["rgba(232,89,12,0.55)", 1]].forEach(([color, c]) => {
      ctx.fillStyle = color as string;
      counts[c as number].forEach((k, b) => { const bh = (k / hi) * (h - 40); ctx.fillRect(toX(b / bins), h - 20 - bh, (w - 40) / bins - 1, bh); });
    });
    ctx.strokeStyle = textColor(); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(toX(t), 10); ctx.lineTo(toX(t), h - 20); ctx.stroke();
    ctx.fillStyle = textColor(); ctx.font = "12px ui-sans-serif";
    ctx.fillText("predicted probability → (blue = negatives, orange = positives)", 20, h - 4);
    ctx.fillText(`threshold ${t.toFixed(2)}`, toX(t) + 6, 22);
  }, [data, t], 460, 200);

  const rocRef = useCanvas((ctx, w, h) => {
    const s = (x: number) => 30 + x * (w - 40), u = (y: number) => h - 25 - y * (h - 40);
    ctx.strokeStyle = "rgba(128,128,128,0.4)"; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(s(0), u(0)); ctx.lineTo(s(1), u(1)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = "#2b8a3e"; ctx.lineWidth = 2; ctx.beginPath();
    curve.pts.forEach(([x, y], i) => (i ? ctx.lineTo(s(x), u(y)) : ctx.moveTo(s(x), u(y)))); ctx.stroke();
    ctx.fillStyle = "#e8590c"; ctx.beginPath(); ctx.arc(s(stats.fpr), u(stats.recall), 6, 0, 7); ctx.fill();
    ctx.fillStyle = textColor(); ctx.font = "11px ui-sans-serif";
    ctx.fillText("FPR →", w - 50, h - 8); ctx.fillText("TPR", 2, 16); ctx.fillText(`AUC = ${curve.auc.toFixed(3)}`, s(0.45), u(0.15));
  }, [curve, stats], 260, 240);

  return (
    <div className="widget">
      <h3>Threshold, confusion matrix and ROC curve</h3>
      <div className="controls">
        <label>class separation {sep.toFixed(1)} <input type="range" min={0} max={5} step={0.1} value={sep} onChange={(e) => setSep(Number(e.target.value))} /></label>
        <label>positive rate {(100 * ratio).toFixed(0)}% <input type="range" min={0.02} max={0.5} step={0.01} value={ratio} onChange={(e) => setRatio(Number(e.target.value))} /></label>
        <label>threshold {t.toFixed(2)} <input type="range" min={0.01} max={0.99} step={0.01} value={t} onChange={(e) => setT(Number(e.target.value))} /></label>
      </div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-start" }}>
        <canvas ref={distRef} />
        <canvas ref={rocRef} />
        <table style={{ display: "table" }}>
          <thead><tr><th></th><th>pred +</th><th>pred −</th></tr></thead>
          <tbody>
            <tr><th>actual +</th><td style={{ color: "var(--good)" }}>TP {stats.tp}</td><td style={{ color: "var(--bad)" }}>FN {stats.fn}</td></tr>
            <tr><th>actual −</th><td style={{ color: "var(--bad)" }}>FP {stats.fp}</td><td style={{ color: "var(--good)" }}>TN {stats.tn}</td></tr>
          </tbody>
        </table>
      </div>
      <div className="stat">accuracy {fmt(stats.acc)} · precision {fmt(stats.precision)} · recall (TPR) {fmt(stats.recall)} · F1 {fmt(stats.f1)} · FPR {fmt(stats.fpr)}</div>
      <p className="muted" style={{ fontSize: "0.85rem" }}>Lower the positive rate to 2%: accuracy stays high even when the model finds almost no positives - that&apos;s why accuracy is misleading on imbalanced data. Moving the threshold trades precision against recall; the ROC curve shows every threshold at once.</p>
    </div>
  );
}
