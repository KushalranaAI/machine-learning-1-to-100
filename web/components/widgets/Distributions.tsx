"use client";
import { useMemo, useState } from "react";
import { fmt, rng, textColor, useCanvas } from "./lib";

type Dist = "normal" | "uniform" | "exponential" | "binomial" | "poisson" | "bimodal";

function sampler(d: Dist, a: number, b: number, r: ReturnType<typeof rng>) {
  return () => {
    if (d === "normal") return a + b * r.normal();
    if (d === "uniform") return a - b + 2 * b * r.next();
    if (d === "exponential") return -Math.log(1 - r.next()) / Math.max(b, 0.05);
    if (d === "binomial") { let k = 0; const n = Math.round(a); for (let i = 0; i < n; i++) if (r.next() < b) k++; return k; }
    if (d === "poisson") { const L = Math.exp(-a); let k = 0, p = 1; do { k++; p *= r.next(); } while (p > L); return k - 1; }
    return r.next() < 0.5 ? a - 2 + 0.6 * r.normal() : a + 2 + 0.6 * r.normal();
  };
}

export default function Distributions() {
  const [dist, setDist] = useState<Dist>("exponential");
  const [a, setA] = useState(10);
  const [b, setB] = useState(1);
  const [sampleSize, setSampleSize] = useState(1);
  const [seed, setSeed] = useState(1);
  const result = useMemo(() => {
    const r = rng(seed);
    const draw = sampler(dist, a, b, r);
    // Each plotted value is the MEAN of `sampleSize` draws → the central limit theorem in action.
    const vals = Array.from({ length: 4000 }, () => { let s = 0; for (let i = 0; i < sampleSize; i++) s += draw(); return s / sampleSize; });
    const mean = vals.reduce((x, y) => x + y, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((x, y) => x + (y - mean) ** 2, 0) / vals.length);
    const skew = vals.reduce((x, y) => x + ((y - mean) / sd) ** 3, 0) / vals.length;
    return { vals, mean, sd, skew };
  }, [dist, a, b, sampleSize, seed]);

  const ref = useCanvas((ctx, w, h) => {
    const { vals, mean, sd } = result;
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const bins = 50, width = (hi - lo) / bins || 1;
    const counts = new Array(bins).fill(0);
    vals.forEach((v) => counts[Math.min(bins - 1, Math.floor((v - lo) / width))]++);
    const top = Math.max(...counts);
    const X = (v: number) => 30 + ((v - lo) / (hi - lo || 1)) * (w - 50);
    ctx.fillStyle = "rgba(66,99,235,0.6)";
    counts.forEach((c, i) => { const bh = (c / top) * (h - 50); ctx.fillRect(X(lo + i * width), h - 25 - bh, (w - 50) / bins - 1, bh); });
    // normal curve with the same mean / sd for comparison
    ctx.strokeStyle = "#e8590c"; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const v = lo + ((hi - lo) * i) / 200;
      const dens = Math.exp(-0.5 * ((v - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
      const y = h - 25 - ((dens * vals.length * width) / top) * (h - 50);
      i ? ctx.lineTo(X(v), y) : ctx.moveTo(X(v), y);
    }
    ctx.stroke();
    ctx.fillStyle = textColor(); ctx.font = "11px ui-sans-serif";
    ctx.fillText(fmt(lo, 2), 24, h - 8); ctx.fillText(fmt(hi, 2), w - 50, h - 8);
    ctx.fillText("histogram of 4000 values · orange = normal curve with the same mean and sd", 30, 14);
  }, [result], 520, 300);

  const labels: Record<Dist, [string, number, number, number, string, number, number, number]> = {
    normal: ["mean μ", -5, 5, 0.1, "std σ", 0.1, 3, 0.1],
    uniform: ["centre", -5, 5, 0.1, "half-width", 0.1, 5, 0.1],
    exponential: ["(unused)", 0, 1, 1, "rate λ", 0.1, 3, 0.05],
    binomial: ["trials n", 1, 60, 1, "success p", 0.01, 0.99, 0.01],
    poisson: ["rate λ", 0.2, 20, 0.1, "(unused)", 0, 1, 1],
    bimodal: ["centre", -5, 5, 0.1, "(unused)", 0, 1, 1],
  };
  const L = labels[dist];
  return (
    <div className="widget">
      <h3>Distributions and the central limit theorem</h3>
      <div className="controls">
        <label>distribution <select value={dist} onChange={(e) => { const d = e.target.value as Dist; setDist(d); setA(d === "binomial" ? 10 : d === "poisson" ? 3 : 0); setB(d === "binomial" ? 0.3 : 1); }}>
          {["normal", "uniform", "exponential", "binomial", "poisson", "bimodal"].map((d) => <option key={d}>{d}</option>)}
        </select></label>
        <label>{L[0]} = {fmt(a, 2)} <input type="range" min={L[1]} max={L[2]} step={L[3]} value={a} onChange={(e) => setA(Number(e.target.value))} /></label>
        <label>{L[4]} = {fmt(b, 2)} <input type="range" min={L[5]} max={L[6]} step={L[7]} value={b} onChange={(e) => setB(Number(e.target.value))} /></label>
        <label>average of n = {sampleSize} draws <input type="range" min={1} max={50} value={sampleSize} onChange={(e) => setSampleSize(Number(e.target.value))} /></label>
        <button className="btn" onClick={() => setSeed(seed + 1)}>Resample</button>
      </div>
      <div className="stat">mean {fmt(result.mean)} · std {fmt(result.sd)} · skewness {fmt(result.skew)}</div>
      <canvas ref={ref} style={{ marginTop: 8 }} />
      <p className="muted" style={{ fontSize: "0.85rem" }}>Pick a skewed distribution (exponential) and raise n: the distribution of the <b>average</b> becomes a bell curve and its std shrinks like σ/√n. This is why confidence intervals and many tests assume normality.</p>
    </div>
  );
}
