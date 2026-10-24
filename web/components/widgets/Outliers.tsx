"use client";
import { useMemo, useState } from "react";
import { fmt, rng, textColor, useCanvas } from "./lib";

const quantile = (s: number[], q: number) => { const i = (s.length - 1) * q; const lo = Math.floor(i); return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo); };

export default function Outliers() {
  const [k, setK] = useState(1.5);
  const [z, setZ] = useState(3);
  const [nOut, setNOut] = useState(4);
  const [skewed, setSkewed] = useState(false);
  const data = useMemo(() => {
    const r = rng(11);
    const base = Array.from({ length: 200 }, () => (skewed ? Math.exp(0.6 * r.normal()) * 20 : 50 + 8 * r.normal()));
    const outs = Array.from({ length: nOut }, (_, i) => (i % 2 ? 110 + 20 * r.next() : -5 - 10 * r.next()));
    return [...base, ...outs];
  }, [nOut, skewed]);
  const s = useMemo(() => [...data].sort((a, b) => a - b), [data]);
  const q1 = quantile(s, 0.25), q3 = quantile(s, 0.75), iqr = q3 - q1, med = quantile(s, 0.5);
  const lo = q1 - k * iqr, hi = q3 + k * iqr;
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  const sd = Math.sqrt(data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length);
  const iqrOut = data.filter((v) => v < lo || v > hi).length;
  const zOut = data.filter((v) => Math.abs((v - mean) / sd) > z).length;

  const ref = useCanvas((ctx, w, h) => {
    const min = Math.min(...data, lo) - 5, max = Math.max(...data, hi) + 5;
    const X = (v: number) => 20 + ((v - min) / (max - min)) * (w - 40);
    const r = rng(3);
    data.forEach((v) => {
      const flagI = v < lo || v > hi, flagZ = Math.abs((v - mean) / sd) > z;
      ctx.fillStyle = flagI && flagZ ? "#c92a2a" : flagI ? "#e8590c" : flagZ ? "#ae3ec9" : "rgba(66,99,235,0.6)";
      ctx.beginPath(); ctx.arc(X(v), 60 + 40 * r.next(), 3.5, 0, 7); ctx.fill();
    });
    ctx.strokeStyle = textColor(); ctx.lineWidth = 2;
    ctx.strokeRect(X(q1), 140, X(q3) - X(q1), 40);
    ctx.beginPath(); ctx.moveTo(X(med), 140); ctx.lineTo(X(med), 180);
    const wl = Math.max(lo, s[0]), wh = Math.min(hi, s[s.length - 1]);
    ctx.moveTo(X(wl), 160); ctx.lineTo(X(q1), 160); ctx.moveTo(X(q3), 160); ctx.lineTo(X(wh), 160); ctx.stroke();
    ctx.setLineDash([4, 4]); ctx.strokeStyle = "#e8590c";
    [lo, hi].forEach((v) => { ctx.beginPath(); ctx.moveTo(X(v), 30); ctx.lineTo(X(v), 190); ctx.stroke(); });
    ctx.strokeStyle = "#ae3ec9";
    [mean - z * sd, mean + z * sd].forEach((v) => { ctx.beginPath(); ctx.moveTo(X(v), 30); ctx.lineTo(X(v), 190); ctx.stroke(); });
    ctx.setLineDash([]);
    ctx.fillStyle = textColor(); ctx.font = "11px ui-sans-serif";
    ctx.fillText("orange dashed = Tukey fences (IQR rule)   purple dashed = mean ± z·σ", 20, 16);
    ctx.fillText("box plot: Q1 · median · Q3, whiskers to the fences", 20, 205);
  }, [data, k, z], 560, 215);

  return (
    <div className="widget">
      <h3>Detecting outliers: z-score vs IQR</h3>
      <div className="controls">
        <label>IQR multiplier k = {k.toFixed(1)} <input type="range" min={0.5} max={4} step={0.1} value={k} onChange={(e) => setK(Number(e.target.value))} /></label>
        <label>z threshold = {z.toFixed(1)} <input type="range" min={1} max={5} step={0.1} value={z} onChange={(e) => setZ(Number(e.target.value))} /></label>
        <label>injected outliers = {nOut} <input type="range" min={0} max={30} value={nOut} onChange={(e) => setNOut(Number(e.target.value))} /></label>
        <label><input type="checkbox" checked={skewed} onChange={(e) => setSkewed(e.target.checked)} /> skewed data</label>
      </div>
      <div className="stat">IQR rule flags {iqrOut} · z-score flags {zOut} · mean {fmt(mean, 1)} vs median {fmt(med, 1)} · σ {fmt(sd, 1)}</div>
      <canvas ref={ref} style={{ marginTop: 8 }} />
      <p className="muted" style={{ fontSize: "0.85rem" }}>Inject 30 outliers: they inflate σ, so the z-score rule <b>stops seeing them</b> (masking). The median and IQR barely move, so the IQR rule keeps working. On skewed data both rules flag many legitimate values in the long tail - transform first (log), then detect.</p>
    </div>
  );
}
