"use client";
import { useMemo, useState } from "react";
import { CLASS_COLORS, Plane, dot, fmt, makeDataset, textColor, useCanvas } from "./lib";

const W = 440, H = 400;
const P = new Plane(W, H, -1.3, 1.3, -1.3, 1.3);
const giniOf = (c: number[]) => { const n = c[0] + c[1]; return n ? 1 - (c[0] / n) ** 2 - (c[1] / n) ** 2 : 0; };
const entropyOf = (c: number[]) => { const n = c[0] + c[1]; return c.reduce((s, k) => (k ? s - (k / n) * Math.log2(k / n) : s), 0); };

export default function TreeSplit() {
  const [feature, setFeature] = useState<0 | 1>(0);
  const [t, setT] = useState(0);
  const [crit, setCrit] = useState<"gini" | "entropy">("gini");
  const [seed, setSeed] = useState(2);
  const data = useMemo(() => makeDataset("blobs", 120, 0.35, seed), [seed]);
  const imp = crit === "gini" ? giniOf : entropyOf;
  const evalSplit = (f: number, thr: number) => {
    const L = [0, 0], R = [0, 0];
    data.X.forEach((p, i) => (p[f] <= thr ? L : R)[data.y[i]]++);
    const n = data.X.length, nl = L[0] + L[1], nr = R[0] + R[1];
    const parent = imp([L[0] + R[0], L[1] + R[1]]);
    const child = (nl * imp(L) + nr * imp(R)) / n;
    return { L, R, parent, child, gain: parent - child };
  };
  const s = evalSplit(feature, t);
  const best = useMemo(() => {
    let b = { f: 0, t: 0, gain: -1 };
    for (const f of [0, 1]) for (let k = -120; k <= 120; k++) { const thr = k / 100; const g = evalSplit(f, thr).gain; if (g > b.gain) b = { f, t: thr, gain: g }; }
    return b;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, crit]);
  const sweep = useMemo(() => Array.from({ length: 241 }, (_, k) => evalSplit(feature, -1.2 + k / 100).gain), [data, feature, crit]); // eslint-disable-line react-hooks/exhaustive-deps

  const ref = useCanvas((ctx) => {
    ctx.fillStyle = "rgba(66,99,235,0.07)";
    if (feature === 0) ctx.fillRect(0, 0, P.px(t), H); else ctx.fillRect(0, P.py(t), W, H - P.py(t));
    ctx.fillStyle = "rgba(232,89,12,0.07)";
    if (feature === 0) ctx.fillRect(P.px(t), 0, W - P.px(t), H); else ctx.fillRect(0, 0, W, P.py(t));
    data.X.forEach((p, i) => dot(ctx, P.px(p[0]), P.py(p[1]), CLASS_COLORS[data.y[i]]));
    ctx.strokeStyle = textColor(); ctx.lineWidth = 2.5; ctx.beginPath();
    if (feature === 0) { ctx.moveTo(P.px(t), 0); ctx.lineTo(P.px(t), H); } else { ctx.moveTo(0, P.py(t)); ctx.lineTo(W, P.py(t)); }
    ctx.stroke();
    ctx.fillStyle = textColor(); ctx.font = "12px ui-sans-serif";
    ctx.fillText(`split: x${feature + 1} ≤ ${t.toFixed(2)}`, 8, 16);
  }, [data, feature, t], W, H);

  const sweepRef = useCanvas((ctx, w, h) => {
    const hi = Math.max(...sweep, 1e-6);
    ctx.strokeStyle = "#2b8a3e"; ctx.lineWidth = 2; ctx.beginPath();
    sweep.forEach((g, k) => { const x = 10 + (k / 240) * (w - 20), y = h - 20 - (g / hi) * (h - 40); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    const x = 10 + ((t + 1.2) / 2.4) * (w - 20);
    ctx.strokeStyle = textColor(); ctx.beginPath(); ctx.moveTo(x, 10); ctx.lineTo(x, h - 20); ctx.stroke();
    ctx.fillStyle = textColor(); ctx.font = "11px ui-sans-serif"; ctx.fillText(`information gain for every threshold on x${feature + 1}`, 10, h - 4);
  }, [sweep, t, feature], 300, 150);

  return (
    <div className="widget">
      <h3>How a decision tree chooses a split</h3>
      <p className="muted" style={{ marginTop: 0 }}>A tree tries every feature and threshold and keeps the one with the largest drop in impurity. Move the threshold yourself, then compare with the best split.</p>
      <div className="controls">
        <label>feature <select value={feature} onChange={(e) => setFeature(Number(e.target.value) as 0 | 1)}><option value={0}>x1 (horizontal)</option><option value={1}>x2 (vertical)</option></select></label>
        <label>threshold {t.toFixed(2)} <input type="range" min={-1.2} max={1.2} step={0.01} value={t} onChange={(e) => setT(Number(e.target.value))} /></label>
        <label>impurity <select value={crit} onChange={(e) => setCrit(e.target.value as "gini" | "entropy")}><option value="gini">Gini</option><option value="entropy">Entropy</option></select></label>
        <button className="btn" onClick={() => { setFeature(best.f as 0 | 1); setT(best.t); }}>Jump to best split</button>
        <button className="btn" onClick={() => setSeed(seed + 1)}>New data</button>
      </div>
      <div className="stat">left: {s.L[0]} blue / {s.L[1]} orange (impurity {fmt(imp(s.L))}) · right: {s.R[0]} / {s.R[1]} (impurity {fmt(imp(s.R))})</div>
      <div className="stat">parent impurity {fmt(s.parent)} − weighted children {fmt(s.child)} = <b>gain {fmt(s.gain)}</b> (best possible: {fmt(best.gain)} at x{best.f + 1} ≤ {best.t.toFixed(2)})</div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-start", marginTop: 8 }}>
        <canvas ref={ref} />
        <canvas ref={sweepRef} />
      </div>
    </div>
  );
}
